import asyncio
import time
import traceback
from enum import Enum
from typing import Any

from loguru import logger
from RealtimeTTS import KokoroEngine, TextToAudioStream


class TTSMode(Enum):
    """TTS operation modes"""

    STREAMING = "streaming"  # Non-blocking, real-time streaming
    BLOCKING = "blocking"  # Blocking, wait for completion


class RealtimeTTSAdapter:
    """
    Simplified adapter for RealtimeTTS Kokoro engine.
    Provides a clean interface for high-quality text-to-speech.
    """

    def __init__(
        self,
        engine=None,
        stream=None,
        engine_type: str = "kokoro",
        comma_silence_duration: float = 0.05,
        sentence_silence_duration: float = 0.1,
        default_silence_duration: float = 0.0,
        frames_per_buffer: int = 512,
        playout_chunk_size: int = 1024,
        buffer_threshold_seconds: float = 1.0,
    ):
        """
        Initialize Kokoro TTS adapter.

        Args:
            engine: Optional Kokoro engine (for dependency injection in tests)
            stream: Optional TTS stream (for dependency injection in tests)
            engine_type: Engine type (currently only supports "kokoro")
            comma_silence_duration: Silence duration after commas (seconds)
            sentence_silence_duration: Silence duration after sentences (seconds)
            default_silence_duration: Default silence duration (seconds)
            frames_per_buffer: Frames per buffer for the TTS stream
            playout_chunk_size: Playout chunk size for the TTS stream
        """
        # Playback things
        self.buffer_threshold_seconds = buffer_threshold_seconds
        self.comma_silence_duration = comma_silence_duration
        self.sentence_silence_duration = sentence_silence_duration
        self.default_silence_duration = default_silence_duration

        # Stream things
        # NOTE: see https://github.com/KoljaB/RealtimeTTS/releases/tag/v0.4.40
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size

        self.engine = KokoroEngine()
        self.stream = self._initialize_stream()

        # State tracking
        self.total_characters_processed: int = 0
        self.is_playing: bool = False
        self.first_audio_generated_time: float | None = None
        self.transcription_received_time: float | None = None
        self.agent_processing_start_time: float | None = None
        self.current_mode: TTSMode = TTSMode.BLOCKING

    def _initialize_stream(self):
        """Initialize Kokoro engine and stream"""
        return TextToAudioStream(
            self.engine,
            frames_per_buffer=self.frames_per_buffer,
            # playout_chunk_size=self.playout_chunk_size,
        )

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

            if mode == TTSMode.BLOCKING:
                # Wait for audio to complete with silence parameters
                await asyncio.to_thread(
                    self.stream.play,
                    buffer_threshold_seconds=self.buffer_threshold_seconds,
                    comma_silence_duration=self.comma_silence_duration,
                    sentence_silence_duration=self.sentence_silence_duration,
                    default_silence_duration=self.default_silence_duration,
                )
                self.is_playing = False
            else:
                # Non-blocking streaming mode with silence parameters
                self.stream.play_async(
                    buffer_threshold_seconds=self.buffer_threshold_seconds,
                    comma_silence_duration=self.comma_silence_duration,
                    sentence_silence_duration=self.sentence_silence_duration,
                    default_silence_duration=self.default_silence_duration,
                )
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
            if text:
                self.stream.feed(text)
                self.total_characters_processed += len(text)

                # Track first audio byte time if not set (only for non-whitespace chunks)
                if self.first_audio_generated_time is None and text.strip():
                    self.first_audio_generated_time = time.time()
                    if self.transcription_received_time:
                        time_to_first_audio = self.first_audio_generated_time - self.transcription_received_time
                        logger.info(
                            f"⏱️ Time from transcription received to first audio generated: "
                            f"{time_to_first_audio:.4f} seconds"
                        )

                return True
            else:
                # Empty string signals end of stream - still feed it to the stream
                self.stream.feed(text)
                return True
        except Exception:
            return False

    def play_stream_async(self) -> bool:
        """Play the stream asynchronously (non-blocking) with configured silence durations."""
        try:
            self.current_mode = TTSMode.STREAMING
            self.is_playing = True
            self.stream.play_async(
                buffer_threshold_seconds=self.buffer_threshold_seconds,
                comma_silence_duration=self.comma_silence_duration,
                sentence_silence_duration=self.sentence_silence_duration,
                default_silence_duration=self.default_silence_duration,
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error starting TTS playback: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.is_playing = False
            return False

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

    def set_transcription_received_time(self, start_time: float | None = None) -> None:
        """Set the prompt start time for timing calculations."""
        self.transcription_received_time = start_time or time.time()

    def stop_playing(self) -> None:
        """Stop the current playback immediately using official RealtimeTTS API."""
        logger.info("⏹️  Stopping TTS playback...")
        try:
            # Only try to stop if we're actually playing
            if self.is_playing and hasattr(self.stream, "stop"):
                self.stream.stop()
                logger.info("🛑 Speech streaming should stop immediately")
            else:
                logger.info("ℹ️  TTS not playing or no stop method available")

            # Reset state
            self.is_playing = False
            self.current_mode = TTSMode.BLOCKING

        except Exception as e:
            logger.error(f"⚠️  Error calling stream.stop(): {e}")
            # Only recreate stream if the error isn't just about IDLE state
            if "IDLE" not in str(e):
                try:
                    self.stream = self._initialize_stream()
                    logger.info("🔄 Recreated TTS stream as fallback")
                except Exception as e2:
                    logger.info(f"❌ Failed to recreate TTS stream: {e2}")
            else:
                logger.info("ℹ️  Stream was already idle, no action needed")
        finally:
            self.is_playing = False

    def reset_stream(self) -> bool:
        """Reset the TTS stream to ensure it's ready for new text after interruption."""
        try:
            logger.info("🔄 Resetting TTS stream for fresh start...")
            # Stop any current playback first
            self.stop_playing()

            # Reset timing states for new response (this is the key part)
            self.first_audio_generated_time = None

            self.agent_processing_start_time = None

            logger.info("✅ TTS stream reset successfully - reusing existing stream")
            return True
        except Exception as e:
            logger.info(f"❌ Failed to reset TTS stream: {e}")
            # Fallback: recreate stream if there's an issue
            try:
                self.stream = self._initialize_stream()
                self.first_audio_generated_time = None
                logger.info("🔄 Created fresh TTS stream as fallback")
                return True
            except Exception as e2:
                logger.info(f"❌ Failed to create fresh TTS stream: {e2}")
                return False

    def get_silence_durations(self) -> dict[str, float]:
        """Get the current silence durations configuration."""
        return {
            "comma_silence_duration": self.comma_silence_duration,
            "sentence_silence_duration": self.sentence_silence_duration,
            "default_silence_duration": self.default_silence_duration,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        stats = {
            "total_characters_processed": self.total_characters_processed,
            "is_playing": self.is_playing,
            "engine_type": "KokoroEngine",
            "first_audio_generated_time": self.first_audio_generated_time,
            "current_mode": self.current_mode.value,
            "transcription_received_time": self.transcription_received_time,
            "agent_processing_start_time": self.agent_processing_start_time,
            "silence_config": self.get_silence_durations(),
        }

        # Add calculated timing metrics
        if self.transcription_received_time and self.first_audio_generated_time:
            stats["time_to_first_audio"] = self.first_audio_generated_time - self.transcription_received_time

        if self.agent_processing_start_time and self.first_audio_generated_time:
            stats["response_to_audio_time"] = self.first_audio_generated_time - self.agent_processing_start_time

        return stats

    def reset_stats(self) -> None:
        """Reset adapter statistics."""
        self.total_characters_processed = 0
        self.is_playing = False
        self.first_audio_generated_time = None
        self.transcription_received_time = None
        self.agent_processing_start_time = None
        self.current_mode = TTSMode.BLOCKING
