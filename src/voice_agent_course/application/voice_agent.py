"""WebSocket-ready Voice Agent with LangGraph capabilities and real-time streaming."""

import asyncio
import base64
import traceback
from typing import Any

from loguru import logger

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent
from voice_agent_course.infrastructure.audio.realtime_stt_adapter import STTModel
from voice_agent_course.infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter


class VoiceAgentCallbacks:
    """Callback system for coordinating between voice agent components"""

    def __init__(self, message_queue: asyncio.Queue):
        self.message_queue = message_queue
        self.tts_to_client = False
        self.current_generation_id: str | None = None

    async def on_partial_transcription(self, text: str) -> None:
        """Handle partial transcription from STT"""
        await self.message_queue.put({"type": "partial_transcription", "content": text})

    async def on_final_transcription(self, text: str) -> None:
        """Handle final transcription from STT"""
        await self.message_queue.put({"type": "final_transcription", "content": text})

    async def on_recording_start(self) -> None:
        """Handle recording start (interruption)"""
        self.tts_to_client = False
        await self.message_queue.put({"type": "recording_start"})

    async def on_partial_assistant_text(self, text: str) -> None:
        """Handle partial assistant response"""
        await self.message_queue.put({"type": "assistant_partial", "content": text})

    async def on_first_audio_chunk_synthesized(self) -> None:
        """Handle first TTS audio chunk ready"""
        self.tts_to_client = True
        await self.message_queue.put({"type": "tts_start"})

    async def send_final_assistant_answer(self) -> None:
        """Handle assistant response complete"""
        await self.message_queue.put({"type": "assistant_complete"})

    def reset_state(self) -> None:
        """Reset callback state"""
        self.tts_to_client = False
        self.current_generation_id = None


class VoiceAgent:
    """
    WebSocket-ready Voice Agent with LangGraph tool support and real-time streaming.
    Implements the 4-task production architecture for handling real-time voice conversations.
    """

    def __init__(
        self,
        websocket,
        llm_provider: str = "groq",
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
        stt_model: STTModel = STTModel.TINY_EN,
        stt_language: str = "en",
        source_sample_rate: int = 16000,
        source_channels: int = 1,
    ):
        """
        Initialize the WebSocket Voice Agent.

        Args:
            websocket: FastAPI WebSocket connection
            llm_provider: LLM provider (groq, ollama)
            llm_model: Model name (uses provider default if None)
            llm_temperature: Model temperature
            stt_model: STT model for speech recognition
            stt_language: Language for STT processing
            source_sample_rate: Sample rate of incoming audio
            source_channels: Number of channels in incoming audio
        """
        logger.info("🤖 Initializing WebSocket Voice Agent with real STT/TTS...")

        self.websocket = websocket

        # Initialize LangGraph agent for conversation and tool orchestration
        self.langgraph_agent = LangGraphAgent(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )

        # Asyncio queues for task coordination (production pattern)
        self.audio_chunks_queue = asyncio.Queue()  # Raw audio from WebSocket
        self.message_queue = asyncio.Queue()  # Outgoing messages to client
        self.transcription_queue = asyncio.Queue()  # Transcriptions for agent
        self.tts_chunks_queue = asyncio.Queue()  # TTS audio chunks

        # Callback system for real-time coordination
        self.callbacks = VoiceAgentCallbacks(self.message_queue)

        # State management
        self.shutdown_requested = False
        self.current_generation = None
        self.loop = None  # Will be set when run() is called

        # Initialize real audio processing components (chunk-based)
        logger.info("🔄 Initializing chunk-based STT and TTS processors...")

        # Import the production-style audio processor
        from voice_agent_course.infrastructure.audio.audio_input_processor import AudioInputProcessor

        # Initialize chunk-based audio processor (no microphone)
        self.audio_processor = AudioInputProcessor(
            source_sample_rate=source_sample_rate,
            source_channels=source_channels,
            target_sample_rate=16000,
            target_channels=1,
            stt_model=stt_model.value if hasattr(stt_model, "value") else str(stt_model),
            stt_language=stt_language,
            on_partial_transcription=self._on_partial_transcription,
            on_final_transcription=self._on_final_transcription,
            on_recording_start=self._on_stt_recording_start,
        )
        self.audio_processor.transcription_processor.potential_sentence_end = self._on_potential_sentence_end

        # Initialize TTS adapter
        self.tts_adapter = RealtimeTTSAdapter()
        self.tts_processor = self.tts_adapter  # Real TTS processor

        # Audio processing state
        self.current_audio_buffer = bytearray()  # Buffer for audio chunks

        self.agent_stats = self.langgraph_agent.get_stats()

        logger.info("✅ WebSocket Voice Agent initialized with real audio processing!")
        logger.info(f"🛠️  Available tools: {', '.join(self.agent_stats['tool_names'])}")
        logger.info(f"🎤 STT Model: {stt_model.value}")
        logger.info("🔊 TTS Engine: Kokoro")

    def _on_partial_transcription(self, text: str):
        """Callback when STT produces a partial transcription."""
        logger.debug(f"👂📝 Partial Transcription: {text}")
        # Send partial transcription to WebSocket client only - don't process with agent
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.callbacks.on_partial_transcription(text), self.loop)

    def _on_final_transcription(self, text: str):
        """Callback when STT produces a final transcription."""
        logger.info(f"👂✅ Final Transcription: {text}")
        # Put ONLY final transcription in queue for agent processing
        try:
            self.transcription_queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Transcription queue full, dropping message")

        # Send final transcription to WebSocket client
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.callbacks.on_final_transcription(text), self.loop)

    def _on_stt_recording_start(self):
        """Callback when STT detects recording start."""
        logger.info("👂▶️ STT Recording started - stopping TTS")
        # Stop TTS playback
        self.tts_adapter.stop_playing()
        # Send control message to client - schedule from thread-safe context
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.callbacks.on_recording_start(), self.loop)

    # ==================== AUDIO PROCESSING CALLBACK METHODS ====================

    def _on_partial_transcription(self, text: str):
        """Callback for partial transcription updates"""
        try:
            logger.debug(f"👤 Partial: {text}")
            # Send partial transcription to WebSocket client - schedule from thread-safe context
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.callbacks.on_partial_transcription(text), self.loop)
        except Exception as e:
            logger.error(f"❌ Error handling partial transcription: {e}")

    def _on_final_transcription(self, text: str):
        """Callback for final transcription results"""
        try:
            logger.info(f"👤 Final Transcription: {text}")
            # Put transcription in queue for agent processing
            self.transcription_queue.put_nowait(text)
            # Also send to WebSocket client - schedule from thread-safe context
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.callbacks.on_final_transcription(text), self.loop)
        except Exception as e:
            logger.error(f"❌ Error handling final transcription: {e}")

    def _on_potential_sentence_end(self, text: str):
        """Callback for potential sentence end detection"""
        try:
            logger.info(f"👤 Potential sentence end: {text}")
            # Trigger early agent processing for responsiveness
            self.transcription_queue.put_nowait(text)
        except Exception as e:
            logger.error(f"❌ Error handling potential sentence end: {e}")

    def _on_recording_start(self):
        """Callback when recording starts (interruption)"""
        try:
            logger.info("🎤 Recording started - interrupting TTS")
            # Stop TTS playback immediately
            if self.tts_adapter:
                self.tts_adapter.stop_playing()
            # Signal TTS interruption to client - schedule from thread-safe context
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.callbacks.on_recording_start(), self.loop)
        except Exception as e:
            logger.error(f"❌ Error handling recording start: {e}")

    def _on_silence_change(self, is_active: bool):
        """Callback when silence state changes"""
        try:
            logger.debug(f"🤫 Silence: {'ACTIVE' if is_active else 'INACTIVE'}")
            # Could be used for UI feedback or processing optimization
        except Exception as e:
            logger.error(f"❌ Error handling silence change: {e}")

    # ==================== PRODUCTION 4-TASK ARCHITECTURE ====================

    async def process_incoming_data_task(self):
        """
        Task 1: Process incoming WebSocket data (audio chunks and control messages).
        Equivalent to production process_incoming_data().
        """
        logger.info("📥 Starting incoming data processing task...")

        while not self.shutdown_requested:
            try:
                # Wait for WebSocket message
                message = await self.websocket.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio chunk received
                        audio_data = message["bytes"]
                        # logger.info(f"📥 Received audio chunk: {len(audio_data)} bytes")
                        # Extract metadata if needed (sample_rate, etc.)
                        metadata = {
                            "pcm": audio_data,
                            "timestamp": asyncio.get_event_loop().time(),
                            "sample_rate": 16000,  # Default, could be extracted from header
                        }
                        await self.audio_chunks_queue.put(metadata)
                        # logger.debug("📥 Audio chunk queued for processing")

                    elif "text" in message:
                        # Control message received
                        await self._handle_control_message(message["text"])

                elif message["type"] == "websocket.disconnect":
                    logger.info("🔌 WebSocket disconnected")
                    self.shutdown_requested = True
                    break

            except Exception as e:
                logger.error(f"❌ Error processing incoming data: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                break

    async def process_audio_chunks_task(self):
        """
        Task 2: Process audio chunks through production-style AudioInputProcessor.
        Uses the same approach as production: delegate to AudioInputProcessor.
        """
        logger.info("🎵 Starting production-style audio chunk processing task...")

        # Process audio chunks from the queue and feed to AudioInputProcessor
        while not self.shutdown_requested:
            try:
                # Get audio chunk from queue
                logger.debug("🎵 Waiting for audio chunk from queue...")
                audio_data = await self.audio_chunks_queue.get()
                if audio_data is None:
                    logger.info("🎵 Received termination signal for audio processing.")
                    break

                # logger.info(f"🎵 Got audio data from queue: {type(audio_data)}")

                # Extract PCM data
                pcm_data = audio_data.get("pcm", b"")
                if pcm_data:
                    # logger.info(f"🎵 Feeding audio chunk: {len(pcm_data)} bytes")
                    await self.audio_processor.feed_audio_chunk(pcm_data)
                else:
                    logger.warning("🎵 No PCM data found in audio_data")

            except Exception as e:
                logger.error(f"❌ Error in audio chunk processing: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)

    async def send_text_messages_task(self):
        """
        Task 3: Process transcriptions with agent and send text messages to WebSocket client.
        Combines agent processing + message sending to match production pattern.
        """
        logger.info("💬 Starting text message sending task (with agent processing)...")

        while not self.shutdown_requested:
            try:
                # Check for transcriptions to process with agent (non-blocking)
                try:
                    text = self.transcription_queue.get_nowait()
                    logger.info(f"🤖 Agent processing: {text}")
                    await self._process_user_input(text)
                except asyncio.QueueEmpty:
                    pass  # No transcription to process

                # Check for messages to send to client (with timeout)
                try:
                    message_data = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                    await self.websocket.send_json(message_data)
                except asyncio.TimeoutError:
                    pass  # No message to send, continue loop

            except Exception as e:
                logger.error(f"❌ Error in text messages task: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def send_tts_chunks_task(self):
        """
        Task 4: Send real TTS audio chunks to WebSocket client.
        Equivalent to production send_tts_chunks().
        """
        logger.info("🔊 Starting real TTS chunk sending task...")

        while not self.shutdown_requested:
            try:
                await asyncio.sleep(0.001)  # Yield control

                # Check if TTS generation is active
                if not self.callbacks.tts_to_client:
                    continue

                # Try to get TTS audio chunk (non-blocking)
                try:
                    chunk = self.tts_chunks_queue.get_nowait()
                    if chunk:
                        # Convert to base64 and send via WebSocket
                        base64_chunk = base64.b64encode(chunk).decode("utf-8")
                        await self.message_queue.put({"type": "tts_chunk", "content": base64_chunk})
                except asyncio.QueueEmpty:
                    # Handle end-of-generation logic
                    if self._is_generation_complete():
                        await self.callbacks.send_final_assistant_answer()
                        self.callbacks.reset_state()

            except Exception as e:
                logger.error(f"❌ Error sending TTS chunks: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    # ==================== HELPER METHODS ====================

    async def _handle_control_message(self, message_text: str):
        """Handle control messages from WebSocket client"""
        try:
            import json

            control_msg = json.loads(message_text)

            if control_msg.get("type") == "tts_start":
                self.callbacks.tts_to_client = True
            elif control_msg.get("type") == "tts_stop":
                self.callbacks.tts_to_client = False
            elif control_msg.get("type") == "interrupt":
                await self.callbacks.on_recording_start()

        except Exception as e:
            logger.error(f"❌ Error handling control message: {e}")

    def _process_audio_chunk(self, audio_data: bytes) -> bytes:
        """Process raw audio chunk for STT"""
        # Basic audio processing - in production you might add:
        # 1. Convert audio format if needed
        # 2. Apply noise reduction
        # 3. Normalize volume
        # 4. Resample to correct rate
        return audio_data

    async def _feed_audio_to_stt(self, audio_data: bytes) -> None:
        """Feed audio data to STT processor"""
        try:
            # Note: RealtimeSTT typically uses microphone input directly
            # For WebSocket audio, we'd need to implement a custom audio source
            # For now, we'll trigger STT processing via get_text_blocking
            # This is a simplified approach - in production you'd need a custom audio feed

            # Trigger STT processing (this will use microphone by default)
            # The transcription will come through the callback we set up
            logger.debug(f"📡 Received {len(audio_data)} bytes of audio data")

            # In a real implementation, you'd feed this audio to a custom STT pipeline
            # For now, we'll let the STT adapter handle microphone input

        except Exception as e:
            logger.error(f"❌ Error feeding audio to STT: {e}")

    def _is_generation_complete(self) -> bool:
        """Check if TTS generation is complete"""
        return not self.tts_adapter.is_playing if self.tts_adapter else True

    async def _process_user_input(self, user_input: str):
        """
        Process user input through LangGraph agent with WebSocket streaming.
        """
        try:
            # Check for exit command
            if user_input.lower().strip() in ["exit", "quit", "goodbye", "stop"]:
                logger.info("👋 Goodbye!")
                self.shutdown_requested = True
                return

            # Stream response with immediate WebSocket output
            chunk_count = 0

            async for chunk in self.langgraph_agent.stream(user_message=user_input):
                if chunk:
                    chunk_count += 1
                    # Send partial response to client
                    await self.callbacks.on_partial_assistant_text(chunk)

                    # Also generate TTS (would integrate with TTS processor)
                    await self._generate_tts_chunk(chunk)

            # Signal end of stream
            logger.info(f"🔊 Streamed {chunk_count} chunks to client, signaling end...")
            await self.callbacks.send_final_assistant_answer()

        except Exception as e:
            logger.error(f"❌ Error processing user input: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    async def _generate_tts_chunk(self, text: str):
        """Generate TTS audio chunk using real RealtimeTTS"""
        try:
            if text.strip():
                # Feed text to TTS adapter for streaming synthesis
                success = self.tts_adapter.feed_text(text)

                if success:
                    # Start TTS playback if this is the first chunk
                    if not self.callbacks.tts_to_client:
                        # Start async playback
                        if self.tts_adapter.play_stream_async():
                            await self.callbacks.on_first_audio_chunk_synthesized()
                            logger.info(f"🔊 Started TTS playback for: '{text[:50]}...'")
                        else:
                            logger.error("❌ Failed to start TTS playback")
                    else:
                        logger.debug(f"🔊 Fed text to TTS: '{text[:50]}...'")
                else:
                    logger.error(f"❌ Failed to feed text to TTS: '{text}'")
            else:
                # Empty string signals end of stream
                self.tts_adapter.feed_text("")
                logger.info("🔊 Signaled end of TTS stream")

        except Exception as e:
            logger.error(f"❌ Error generating TTS chunk: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    # ==================== MAIN ORCHESTRATOR ====================

    async def run_conversation(self):
        """
        Main conversation orchestrator with 5 concurrent tasks.
        Production-ready WebSocket architecture with real STT/TTS.
        """
        logger.info("🚀 Starting voice conversation with real STT/TTS support...")

        # Store the current event loop for thread-safe callback scheduling
        self.loop = asyncio.get_running_loop()

        # Start the transcription task
        await self.audio_processor.start_transcription_task()

        try:
            # Create the 4 concurrent tasks (production pattern)
            tasks = [
                asyncio.create_task(self.process_incoming_data_task(), name="incoming_data"),
                asyncio.create_task(self.process_audio_chunks_task(), name="audio_chunks"),
                asyncio.create_task(self.send_text_messages_task(), name="text_messages"),
                asyncio.create_task(self.send_tts_chunks_task(), name="tts_chunks"),
            ]

            # Wait for any task to complete (usually client disconnect or shutdown)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"❌ Error in voice conversation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all components including real audio adapters."""
        logger.info("🔄 Shutting down WebSocket Voice Agent...")

        try:
            # Set shutdown flag
            self.shutdown_requested = True

            # Stop TTS playback
            if hasattr(self, "tts_adapter") and self.tts_adapter:
                self.tts_adapter.stop_playing()
                logger.info("🔊 TTS adapter stopped")

            # Shutdown audio processor (includes STT)
            if hasattr(self, "audio_processor") and self.audio_processor:
                await self.audio_processor.shutdown()
                logger.info("🎤 Audio processor shutdown")

            # Signal queues to stop
            await self.audio_chunks_queue.put(None)

            logger.info("✅ WebSocket Voice Agent shutdown complete")

        except Exception as e:
            logger.error(f"⚠️  Error during shutdown: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive agent statistics including real audio adapters."""
        stats = {
            "langgraph_agent_stats": self.langgraph_agent.get_stats(),
            "queue_sizes": {
                "audio_chunks": self.audio_chunks_queue.qsize(),
                "messages": self.message_queue.qsize(),
                "transcriptions": self.transcription_queue.qsize(),
                "tts_chunks": self.tts_chunks_queue.qsize(),
            },
            "callbacks_state": {
                "tts_to_client": self.callbacks.tts_to_client,
                "current_generation_id": self.callbacks.current_generation_id,
            },
        }

        # Add real adapter statistics
        if hasattr(self, "audio_processor") and self.audio_processor:
            stats["audio_processor_stats"] = self.audio_processor.get_stats()

        if hasattr(self, "tts_adapter") and self.tts_adapter:
            stats["tts_adapter_stats"] = self.tts_adapter.get_stats()

        return stats
