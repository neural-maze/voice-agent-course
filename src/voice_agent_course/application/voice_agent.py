import asyncio
import time
from typing import Any

from ..domain.prompts.system_prompts import DEFAULT_SYSTEM_PROMPT
from ..infrastructure.audio.realtime_stt_adapter import RealtimeSTTAdapter, STTModel
from ..infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter
from ..infrastructure.llm_providers.ollama import OllamaService


class VoiceAgent:
    """
    Voice-enabled AI agent that combines STT, LLM, and TTS for natural conversation.
    Clean architecture implementation using adapters for modular audio processing.
    """

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        stt_model: STTModel = STTModel.TINY_EN,
        system_behavior: str | None = None,
    ):
        """
        Initialize the Voice Agent.

        Args:
            model: Ollama model name
            stt_model: Speech-to-text model to use
            system_behavior: System prompt for the AI assistant
        """
        print("🤖 Initializing Voice Agent...")

        # Initialize TTS adapter first (needed for interruption callback)
        self.tts_adapter = RealtimeTTSAdapter()

        # Use the adapters, not the libraries directly
        self.stt_adapter = RealtimeSTTAdapter(
            model=stt_model,
            language="en",
            enable_realtime=True,  # Enable realtime transcription for interruption
            on_partial_transcription=self._on_partial_speech,  # Re-enable interruption
            sensitivity_config={
                "silero_sensitivity": 0.01,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.1,
                "min_length_of_recording": 0.2,
                "min_gap_between_recordings": 0,
            },
        )
        self.llm_service = OllamaService(model=model)

        # Queues for async communication
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        # Configuration
        self.model = model
        self.system_behavior = system_behavior or DEFAULT_SYSTEM_PROMPT

        # Timing metrics
        self.prompt_start_time: float | None = None
        self.response_start_time: float | None = None
        self.first_audio_byte_time: float | None = None

        print("✅ Voice Agent initialized!")

    def _on_user_speech_detected(self):
        """
        Callback triggered when user starts speaking (recording starts).
        Immediately interrupts any ongoing TTS playback.
        """
        print("🔇 Interrupting AI speech...")
        self.tts_adapter.stop_playing()

    async def _on_partial_speech(self, partial_text: str = ""):
        """
        Callback triggered when partial speech is detected.
        Implements interruption by clearing queues and stopping playback.
        """
        if partial_text.strip():
            print(f"🔇 Interrupting AI speech (heard: '{partial_text[:20]}...')")
            # Clear all queues when user starts speaking
            await self.clear_queues()
            # Also stop TTS playback
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
        Speech-to-text loop using the adapter.
        Continuously listens for speech and puts transcriptions in the input queue.
        """
        print("🎤 Starting speech-to-text loop...")

        while True:
            try:
                # Use the adapter instead of direct recorder access
                result = await self.stt_adapter.get_text_blocking()

                if result["success"]:
                    text = result["text"]
                    print(f"👤 User: {text}")

                    # Stop any ongoing TTS playback (interrupt AI speech)
                    self.tts_adapter.stop_playing()

                    # Clear queues and add new input
                    await self.clear_queues()
                    self.prompt_start_time = time.time()
                    await self.input_queue.put(text)

                    # Check for exit command
                    if text.lower().strip() in ["exit", "quit", "goodbye", "stop"]:
                        print("👋 Goodbye!")
                        await self.input_queue.put(None)  # Signal to exit
                        break
                else:
                    print(f"❌ STT Error: {result['error']}")

            except Exception as e:
                print(f"❌ Error in STT loop: {e}")
                continue

    async def llm_loop(self):
        """
        LLM processing loop.
        Takes input from input_queue, processes with Ollama, streams to response_queue.
        """
        print("🧠 Starting LLM processing loop...")

        while True:
            try:
                # Wait for input
                user_input = await self.input_queue.get()
                if user_input is None:
                    break  # Exit signal

                print(f"🤖 Processing: {user_input}")
                self.response_start_time = time.time()

                # Stream response from LLM
                async for chunk in self.llm_service.stream_chat(
                    user_message=user_input, system_prompt=self.system_behavior
                ):
                    if chunk:  # Send all chunks including whitespace for natural speech
                        await self.response_queue.put(chunk)
                        print(chunk, end="", flush=True)  # Print response as it streams

                print()  # New line after complete response
                await self.response_queue.put(None)  # Signal end of response

            except Exception as e:
                print(f"❌ Error in LLM loop: {e}")
                await self.response_queue.put(None)  # Signal end even on error

    async def tts_loop(self):
        """
        Text-to-speech loop with smart play logic to reduce warnings.
        """
        print("🔊 Starting text-to-speech loop...")
        is_response_active = False

        while True:
            try:
                chunk = await self.response_queue.get()
                if chunk is None:
                    # End of response - reset for next response
                    is_response_active = False
                    self.first_audio_byte_time = None
                    continue

                # Track timing for first audio byte
                if self.first_audio_byte_time is None and self.prompt_start_time:
                    self.first_audio_byte_time = time.time()
                    time_to_first_audio = self.first_audio_byte_time - self.prompt_start_time
                    print(f"\n⏱️  Time from prompt to first audio: {time_to_first_audio:.4f}s")

                # Feed text chunk
                self.tts_adapter.feed_text(chunk)

                # Only call play_async on first chunk of response to reduce warnings
                if not is_response_active:
                    self.tts_adapter.play_stream_async()
                    is_response_active = True

            except Exception as e:
                print(f"❌ Error in TTS loop: {e}")
                is_response_active = False
                continue

    async def run_conversation(self):
        """
        Main conversation loop that coordinates STT, LLM, and TTS.
        Runs all components concurrently.
        """
        print("🚀 Starting voice conversation...")
        print("💡 Say something to begin, or say 'exit' to quit")

        try:
            # Run all loops concurrently
            await asyncio.gather(
                self.stt_loop(),
                self.llm_loop(),
                self.tts_loop(),
                return_exceptions=True,
            )

        except KeyboardInterrupt:
            print("\n⏹️  Conversation interrupted by user")
        except Exception as e:
            print(f"❌ Error in conversation: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of all components"""
        print("🔄 Shutting down Voice Agent...")

        try:
            # Shutdown adapters
            if hasattr(self, "stt_adapter"):
                self.stt_adapter.shutdown()

            # Clear queues
            await self.clear_queues()

            print("✅ Voice Agent shutdown complete")

        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "stt_stats": self.stt_adapter.get_stats(),
            "tts_stats": self.tts_adapter.get_stats(),
            "llm_stats": self.llm_service.get_stats(),
            "timing": {
                "prompt_start_time": self.prompt_start_time,
                "response_start_time": self.response_start_time,
                "first_audio_byte_time": self.first_audio_byte_time,
            },
        }
