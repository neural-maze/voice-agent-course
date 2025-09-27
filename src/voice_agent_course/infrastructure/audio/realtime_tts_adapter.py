import traceback

from loguru import logger
from RealtimeTTS import KokoroEngine, TextToAudioStream


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
        frames_per_buffer: int = 256,
        playout_chunk_size: int = 1024,
        buffer_threshold_seconds: float = 2.0,
        minimum_first_fragment_length: int = 20,
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
            buffer_threshold_seconds: Buffer threshold seconds for the TTS stream
        """
        # Playback things
        # NOTE: see -> https://github.com/KoljaB/RealtimeTTS?tab=readme-ov-file#methods
        self.buffer_threshold_seconds = buffer_threshold_seconds
        self.comma_silence_duration = comma_silence_duration
        self.sentence_silence_duration = sentence_silence_duration
        self.default_silence_duration = default_silence_duration
        self.minimum_first_fragment_length = minimum_first_fragment_length

        # Stream things
        # NOTE: see https://github.com/KoljaB/RealtimeTTS/releases/tag/v0.4.40
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size

        self.engine = KokoroEngine()
        self.engine.set_voice("af_heart")
        self.engine.set_speed(1.0)
        self.stream = self._initialize_stream()

        # State tracking
        self.is_playing: bool = False

    def _initialize_stream(self) -> TextToAudioStream:
        """Initialize Kokoro engine and stream"""
        return TextToAudioStream(
            self.engine,
            frames_per_buffer=self.frames_per_buffer,
            playout_chunk_size=self.playout_chunk_size,
        )

    def feed_text(self, text: str) -> None:
        """Feed text to the stream for streaming synthesis."""
        self.stream.feed(text)

    def play_stream_async(self) -> bool:
        """Play the stream asynchronously (non-blocking) with configured silence durations."""
        try:
            self.is_playing = True
            logger.info("🔊 Starting TTS playback...")
            self.stream.play_async(
                buffer_threshold_seconds=self.buffer_threshold_seconds,
                comma_silence_duration=self.comma_silence_duration,
                sentence_silence_duration=self.sentence_silence_duration,
                default_silence_duration=self.default_silence_duration,
                minimum_first_fragment_length=self.minimum_first_fragment_length,
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error starting TTS playback: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.is_playing = False
            return False

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
