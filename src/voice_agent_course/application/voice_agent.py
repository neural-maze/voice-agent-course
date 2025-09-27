"""Voice Agent with LangGraph agent capabilities and tool support."""

import asyncio
import time
import traceback
from typing import Any

from loguru import logger

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent

from ..infrastructure.audio.realtime_stt_adapter import RealtimeSTTAdapter, STTModel
from ..infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter


class VoiceAgent:
    """
    Voice-enabled AI agent with LangGraph tool support.
    Combines STT, LangGraph Agent with tools, and TTS for natural conversation.
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
    ):
        """
        Initialize the Enhanced Voice Agent.

        Args:
            llm_provider: LLM provider (groq, ollama)
            llm_model: Model name (uses provider default if None)
            llm_temperature: Model temperature
        """
        logger.info("🤖 Initializing Enhanced Voice Agent with LangGraph tools...")

        # Initialize TTS adapter first (needed for interruption callback)
        self.tts_adapter = RealtimeTTSAdapter()

        # Initialize STT with interruption support (using improved default timing)
        self.stt_adapter = RealtimeSTTAdapter(
            model=STTModel.TINY_EN,
            language="en",
            enable_realtime=True,
            on_recording_start=self._on_recording_start,
        )

        # Initialize LangGraph agent for conversation and tool orchestration
        self.langgraph_agent = LangGraphAgent(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )

        # Queues for async communication
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        # Timing metrics
        self.transcription_received_time: float | None = None
        self.agent_processing_start_time: float | None = None
        self.first_audio_generated_time: float | None = None

        # Tool execution state
        self.is_tool_executing = False
        self.current_tool_name: str | None = None

        self.agent_stats = self.langgraph_agent.get_stats()

        logger.info("✅ Enhanced Voice Agent initialized with tools!")
        logger.info(f"🛠️  Available tools: {', '.join(self.agent_stats['tool_names'])}")

    def _on_recording_start(self):
        """
        Primary interruption: Triggered immediately when recording starts.
        Stops TTS playback instantly to prevent interference.
        This is the main interruption mechanism for responsiveness.
        """
        self.tts_adapter.stop_playing()

    async def clear_queues(self):
        """Clear all data from the input and response queues."""
        queues = [self.input_queue, self.response_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def stt_loop(self):
        """
        Enhanced speech-to-text loop with tool execution awareness.
        """
        logger.info("🎤 Starting enhanced speech-to-text loop...")

        while True:
            try:
                result = await self.stt_adapter.get_text_blocking()

                if result["success"]:
                    text = result["text"]
                    logger.info(f"👤 User: {text}")

                    # Stop any ongoing TTS playback
                    self.tts_adapter.stop_playing()

                    # Clear queues and add new input
                    await self.clear_queues()
                    self.transcription_received_time = time.time()
                    await self.input_queue.put(text)

                    # Check for exit command
                    if text.lower().strip() in ["exit", "quit", "goodbye", "stop"]:
                        logger.info("👋 Goodbye!")
                        await self.input_queue.put(None)
                        break
                else:
                    logger.info(f"❌ STT Error: {result['error']}")

            except Exception as e:
                logger.error(f"❌ Error in STT loop: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    async def agent_loop(self):
        """
        Enhanced LLM processing loop using LangGraph agent with tools and optimized streaming.
        """
        logger.info("🧠 Starting LangGraph agent processing loop...")

        while True:
            try:
                user_input = await self.input_queue.get()
                if user_input is None:
                    break

                logger.info(f"🤖 Agent processing: {user_input}")
                self.agent_processing_start_time = time.time()

                # Stream response with immediate TTS feeding for low latency
                first_chunk = True
                chunk_count = 0

                # Reset TTS stream to ensure clean state for new response
                # self.tts_adapter.reset_stream()

                # Set timing for TTS
                self.tts_adapter.set_transcription_received_time(self.transcription_received_time)

                async for chunk in self.langgraph_agent.stream(user_message=user_input):
                    if chunk:
                        chunk_count += 1

                        # IMMEDIATELY feed to TTS - true streaming!
                        self.tts_adapter.feed_text(chunk)

                        # Start TTS playback on first chunk
                        if first_chunk:
                            self.tts_adapter.play_stream_async()
                            first_chunk = False

                        print(chunk, end="", flush=True)

                print()  # New line after complete response

                # Signal end of stream to TTS (this is the missing piece!)
                logger.info(f"🔊 Streamed {chunk_count} chunks to TTS, signaling end...")
                self.tts_adapter.feed_text("")  # Signal end of stream

            except Exception as e:
                logger.error(f"❌ Error in agent loop: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                # Don't crash the agent on errors

    async def tts_loop(self):
        """
        Simplified TTS loop - now TTS is fed directly from agent_loop for minimal latency.
        This loop now just handles cleanup and status monitoring.
        """
        logger.info("🔊 TTS now streams directly from agent - minimal latency mode!")

        while True:
            try:
                # Just sleep and monitor - TTS is handled directly in agent_loop now
                await asyncio.sleep(0.1)

                # Reset state when not executing tools
                if not self.is_tool_executing:
                    self.current_tool_name = None

            except Exception as e:
                logger.error(f"❌ Error in TTS monitoring: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    async def run_conversation(self):
        """
        Main conversation loop for enhanced voice agent.
        """
        logger.info("🚀 Starting enhanced voice conversation with tool support...")
        logger.info("💡 Say something to begin, or say 'exit' to quit")

        try:
            await asyncio.gather(
                self.stt_loop(),
                self.agent_loop(),
                self.tts_loop(),
                return_exceptions=True,
            )

        except KeyboardInterrupt:
            logger.info("\n⏹️  Enhanced conversation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in enhanced conversation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("🔄 Shutting down Enhanced Voice Agent...")

        try:
            # Shutdown adapters
            if hasattr(self, "stt_adapter"):
                self.stt_adapter.shutdown()

            # Clear queues
            await self.clear_queues()

            logger.info("✅ Enhanced Voice Agent shutdown complete")

        except Exception as e:
            logger.error(f"⚠️  Error during shutdown: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive enhanced agent statistics."""
        return {
            "stt_stats": self.stt_adapter.get_stats(),
            "tts_stats": self.tts_adapter.get_stats(),
            "langgraph_agent_stats": self.langgraph_agent.get_stats(),
            "timing": {
                "transcription_received_time": self.transcription_received_time,
                "agent_processing_start_time": self.agent_processing_start_time,
                "first_audio_generated_time": self.first_audio_generated_time,
            },
            "tool_execution": {
                "is_tool_executing": self.is_tool_executing,
                "current_tool_name": self.current_tool_name,
            },
        }
