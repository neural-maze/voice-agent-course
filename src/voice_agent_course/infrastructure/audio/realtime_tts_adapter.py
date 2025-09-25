import asyncio
import time
from enum import Enum
from typing import Any, Protocol

from RealtimeTTS import KokoroEngine, TextToAudioStream


class TTSMode(Enum):
    """TTS operation modes"""

    STREAMING = "streaming"  # Non-blocking, real-time streaming
    BLOCKING = "blocking"  # Blocking, wait for completion


class TTSEngine(Protocol):
    """Protocol for TTS engines to make testing easier"""

    pass


class TTSStream(Protocol):
    """Protocol for TTS streams to make testing easier"""

    def feed(self, text: str) -> None: ...
    def play(self) -> None: ...
    def play_async(self) -> None: ...


class RealtimeTTSAdapter:
    """
    Simplified adapter for RealtimeTTS Kokoro engine.
    Provides a clean interface for high-quality text-to-speech.
    """

    def __init__(self, engine=None, stream=None, engine_type: str = "kokoro"):
        """
        Initialize Kokoro TTS adapter.

        Args:
            engine: Optional Kokoro engine (for dependency injection in tests)
            stream: Optional TTS stream (for dependency injection in tests)
            engine_type: Engine type (currently only supports "kokoro")
        """
        if engine is None or stream is None:
            self._initialize_kokoro()
        else:
            # Testing: use injected mocks
            self.engine = engine
            self.stream = stream

        # State tracking
        self.total_characters_processed: int = 0
        self.is_playing: bool = False
        self.first_audio_byte_time: float | None = None
        self.prompt_start_time: float | None = None
        self.response_start_time: float | None = None
        self.current_mode: TTSMode = TTSMode.BLOCKING

    def _initialize_kokoro(self):
        """Initialize Kokoro engine for production"""
        try:
            self.engine = KokoroEngine()
            self.stream = TextToAudioStream(self.engine)

        except ImportError as e:
            raise ImportError(
                "RealtimeTTS with Kokoro not available. Install with: uv add 'RealtimeTTS[all]' && uv add pip"
            ) from e

    async def synthesize_and_play(self, text: str, mode: TTSMode = TTSMode.BLOCKING) -> dict[str, Any]:
        """
        Synthesize text to speech and play it.

        Args:
            text: Text to synthesize
            mode: TTSMode.BLOCKING (wait for completion) or TTSMode.STREAMING (non-blocking)
        """
        if not text.strip():
            return {"success": False, "error": "Empty text provided"}

        start_time = time.time()
        self.current_mode = mode

        try:
            self.is_playing = True
            self.stream.feed(text)

            # Track first audio byte time if not set
            if self.first_audio_byte_time is None:
                self.first_audio_byte_time = time.time()
                if self.prompt_start_time:
                    time_to_first_audio = self.first_audio_byte_time - self.prompt_start_time
                    print(f"Time from prompt to first audio byte: {time_to_first_audio:.4f} seconds")

            if mode == TTSMode.BLOCKING:
                # Wait for audio to complete
                await asyncio.to_thread(self.stream.play)
                self.is_playing = False
            else:
                # Non-blocking streaming mode
                self.stream.play_async()
                # Don't set is_playing to False immediately in streaming mode

            self.total_characters_processed += len(text)

            return {
                "success": True,
                "characters_processed": len(text),
                "duration": time.time() - start_time,
                "mode": mode.value,
            }

        except Exception as e:
            self.is_playing = False
            return {"success": False, "error": str(e)}

    def feed_text(self, text: str) -> bool:
        """Feed text to the stream for streaming synthesis."""
        try:
            if text:  # Accept all text including whitespace for natural speech
                self.stream.feed(text)
                self.total_characters_processed += len(text)

                # Track first audio byte time if not set (only for non-whitespace chunks)
                if self.first_audio_byte_time is None and text.strip():
                    self.first_audio_byte_time = time.time()
                    if self.prompt_start_time:
                        time_to_first_audio = self.first_audio_byte_time - self.prompt_start_time
                        print(f"Time from prompt to first audio byte: {time_to_first_audio:.4f} seconds")

                return True
            return False
        except Exception:
            return False

    def play_stream_async(self) -> bool:
        """Play the stream asynchronously (non-blocking)."""
        try:
            self.current_mode = TTSMode.STREAMING
            self.is_playing = True
            self.stream.play_async()  # Let RealtimeTTS handle warnings internally
            return True
        except Exception as e:
            print(f"❌ Error starting TTS playback: {e}")
            self.is_playing = False
            return False

    async def play_async(self) -> dict[str, Any]:
        """Play the previously fed text asynchronously."""
        start_time = time.time()

        try:
            self.is_playing = True
            await asyncio.to_thread(self.stream.play)
            self.is_playing = False

            return {"success": True, "duration": time.time() - start_time}

        except Exception as e:
            self.is_playing = False
            return {"success": False, "error": str(e)}

    def set_voice(self, voice_id: str) -> bool:
        """Set the Kokoro voice (e.g., 'af_heart', 'bf_emma')."""
        try:
            self.engine.set_voice(voice_id)
            return True
        except Exception:
            return False

    def set_speed(self, speed: float) -> bool:
        """Set the speech speed (e.g., 0.7 for slower, 1.5 for faster)."""
        try:
            self.engine.set_speed(speed)
            return True
        except Exception:
            return False

    def set_prompt_start_time(self, start_time: float | None = None) -> None:
        """Set the prompt start time for timing calculations."""
        self.prompt_start_time = start_time or time.time()

    def set_response_start_time(self, start_time: float | None = None) -> None:
        """Set the response start time for timing calculations."""
        self.response_start_time = start_time or time.time()

    def stop_playing(self) -> None:
        """Stop the current playback immediately using official RealtimeTTS API."""
        print("⏹️  Stopping TTS playback...")
        try:
            # Only try to stop if we're actually playing
            if self.is_playing and hasattr(self.stream, "stop"):
                self.stream.stop()
                print("🛑 Called official stream.stop() - should stop immediately")
            else:
                print("ℹ️  TTS not playing or no stop method available")

            # Reset state
            self.is_playing = False
            self.current_mode = TTSMode.BLOCKING

        except Exception as e:
            print(f"⚠️  Error calling stream.stop(): {e}")
            # Only recreate stream if the error isn't just about IDLE state
            if "IDLE" not in str(e):
                try:
                    self.stream = TextToAudioStream(self.engine)
                    print("🔄 Recreated TTS stream as fallback")
                except Exception as e2:
                    print(f"❌ Failed to recreate TTS stream: {e2}")
            else:
                print("ℹ️  Stream was already idle, no action needed")
        finally:
            self.is_playing = False

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        stats = {
            "total_characters_processed": self.total_characters_processed,
            "is_playing": self.is_playing,
            "engine_type": "KokoroEngine",
            "first_audio_byte_time": self.first_audio_byte_time,
            "current_mode": self.current_mode.value,
            "prompt_start_time": self.prompt_start_time,
            "response_start_time": self.response_start_time,
        }

        # Add calculated timing metrics
        if self.prompt_start_time and self.first_audio_byte_time:
            stats["time_to_first_audio"] = self.first_audio_byte_time - self.prompt_start_time

        if self.response_start_time and self.first_audio_byte_time:
            stats["response_to_audio_time"] = self.first_audio_byte_time - self.response_start_time

        return stats

    def reset_stats(self) -> None:
        """Reset adapter statistics."""
        self.total_characters_processed = 0
        self.is_playing = False
        self.first_audio_byte_time = None
        self.prompt_start_time = None
        self.response_start_time = None
        self.current_mode = TTSMode.BLOCKING
