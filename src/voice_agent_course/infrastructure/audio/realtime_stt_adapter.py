import asyncio
import traceback
from collections.abc import Callable
from enum import Enum
from typing import Any

from loguru import logger
from RealtimeSTT import AudioToTextRecorder


class STTModel(Enum):
    """Available STT models for different use cases"""

    TINY_EN = "tiny.en"  # Fastest, English only
    TINY = "tiny"  # Fast, multilingual
    BASE_EN = "base.en"  # Balanced, English only
    BASE = "base"  # Balanced, multilingual
    SMALL_EN = "small.en"  # Good accuracy, English only
    SMALL = "small"  # Good accuracy, multilingual
    MEDIUM_EN = "medium.en"  # High accuracy, English only
    MEDIUM = "medium"  # High accuracy, multilingual
    LARGE_V1 = "large-v1"  # Highest accuracy, multilingual
    LARGE_V2 = "large-v2"  # Latest large model
    LARGE_V3 = "large-v3"  # Most recent and accurate


class RealtimeSTTAdapter:
    """
    Adapter for RealtimeSTT library.
    Wraps the external dependency and provides a clean interface for speech-to-text.
    """

    def __init__(
        self,
        on_transcription: Callable[[str], None] | None = None,
        on_recording_start: Callable[[], None] | None = None,
        model: STTModel = STTModel.TINY_EN,
        language: str = "en",
        enable_realtime: bool = False,
        sensitivity_config: dict[str, Any] | None = None,
    ):
        """
        Initialize STT adapter.

        Args:
            engine: Optional STT engine (for dependency injection in tests)
            on_transcription: Callback for final transcriptions
            model: STT model to use (affects speed vs accuracy)
            language: Language for transcription
            enable_realtime: Enable real-time partial transcriptions
            sensitivity_config: Custom sensitivity settings
        """
        self.on_transcription = on_transcription
        self.on_recording_start = on_recording_start
        self.model = model
        self.language = language
        self.enable_realtime = enable_realtime

        # Default sensitivity configuration - adjusted for slightly longer wait times
        self.sensitivity_config = sensitivity_config or {
            "silero_sensitivity": 0.6,
            "webrtc_sensitivity": 3,
            "post_speech_silence_duration": 0.3,
            "min_length_of_recording": 0.2,
            "min_gap_between_recordings": 0.1,
        }

        self._initialize_engine_stt()

    def _initialize_engine_stt(self):
        """Initialize real RealtimeSTT components for production"""
        try:
            logger.info(f"🔄 Initializing STT with model '{self.model.value}'...")
            logger.info("   This may take 10-30 seconds on first run (downloading/loading model)")

            # Build recorder configuration
            recorder_config = {
                "model": self.model.value,
                "language": self.language,
                "spinner": False,  # Show progress spinner during initialization
                "use_microphone": True,
                "on_recording_start": self._on_recording_start,
                "on_recording_stop": self._on_recording_stop,
                "on_transcription_start": self._on_transcription_start,
                **self.sensitivity_config,  # Add sensitivity settings
            }

            # Add realtime transcription callbacks if enabled
            if self.enable_realtime:
                recorder_config.update(
                    {
                        "enable_realtime_transcription": True,
                        # add callbacks for realtime transcriptions
                    }
                )

            # Create the recorder with enhanced configuration
            self.engine = AudioToTextRecorder(**recorder_config)
            logger.info("✅ STT engine initialized and ready!")

        except Exception as e:
            logger.error(f"❌ Error creating AudioToTextRecorder: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _on_recording_start(self, *args, **kwargs):
        """Called when recording starts"""
        print()
        logger.info("🎤 Recording started...")

        # Call the user-provided callback for immediate interruption
        if self.on_recording_start:
            self.on_recording_start()

    def _on_recording_stop(self, *args, **kwargs):
        """Called when recording stops"""
        logger.info("⏹️  Recording stopped.")

    def _on_transcription_start(self, *args, **kwargs):
        """Called when transcription processing starts"""
        logger.info("🔄 Transcribing...")

    def _on_partial_transcription(self, text: str, *args, **kwargs):
        """Called for partial/interim transcriptions"""
        if self.on_partial_transcription and text.strip():
            self.on_partial_transcription(text)

    async def get_text_blocking(self) -> str | None:
        """
        Get transcribed text or None if failed.

        Returns:
            str | None: Transcribed text or None if no speech detected or error occurred
        """
        if not hasattr(self, "engine") or self.engine is None:
            logger.error("STT engine not available")
            return None

        try:
            # Use blocking text retrieval with callback - this is the core RealtimeSTT API
            transcribed_text = None

            def capture_text(text):
                nonlocal transcribed_text
                transcribed_text = text

            # Call engine.text() with callback function
            await asyncio.to_thread(self.engine.text, capture_text)

            if transcribed_text and transcribed_text.strip():
                text = transcribed_text.strip()

                # Manually call the callback if one was provided
                if self.on_transcription:
                    self.on_transcription(text)

                return text
            else:
                # No speech detected - this is normal, don't log as error
                return None

        except Exception as e:
            logger.error(f"STT error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None
