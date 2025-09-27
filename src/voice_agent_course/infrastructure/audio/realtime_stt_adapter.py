import asyncio
import time
import traceback
from collections.abc import Callable
from datetime import datetime
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
        enable_realtime: bool = True,
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

        # State tracking
        self.total_transcriptions: int = 0
        self.last_transcription_time: float | None = None
        self.is_active: bool = False

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
                        "on_realtime_transcription_stabilized": self._on_transcription,
                    }
                )

            # Create the recorder with enhanced configuration
            self.engine = AudioToTextRecorder(**recorder_config)
            self.is_active = True
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

    def _on_transcription(self, text: str, *args, **kwargs):
        """Called for final transcriptions"""
        if text.strip():
            self.total_transcriptions += 1
            self.last_transcription_time = time.time()

            if self.on_transcription:
                self.on_transcription(text)

    async def get_text_blocking(self) -> dict[str, Any]:
        """
        Get text using blocking approach (like voice agent example).
        This is the main method that matches RealtimeSTT's .text() API.

        Returns:
            dict: Transcription result with metadata
        """
        if not self.is_active:
            return {"success": False, "error": "STT adapter not initialized"}

        if not hasattr(self, "engine") or self.engine is None:
            return {"success": False, "error": "STT engine not available"}

        start_time = time.time()

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
                self.total_transcriptions += 1
                self.last_transcription_time = time.time()

                # Manually call the callback if one was provided
                if self.on_transcription:
                    self.on_transcription(text)

                return {
                    "success": True,
                    "text": text,
                    "duration": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "method": "blocking",
                }
            else:
                return {
                    "success": False,
                    "error": "No speech detected or empty transcription",
                    "duration": time.time() - start_time,
                }

        except Exception as e:
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    async def listen_once(self, timeout: float = 10.0) -> dict[str, Any]:
        """
        Listen for a single utterance and return the transcription.
        This is a wrapper around get_text_blocking() for compatibility.

        Args:
            timeout: Maximum time to wait for speech (not used in RealtimeSTT)

        Returns:
            dict: Transcription result with metadata
        """
        # RealtimeSTT doesn't support timeout, so we just call the blocking method
        return await self.get_text_blocking()

    async def continuous_listen(
        self, duration: float | None = None, on_speech: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """
        Listen continuously for speech using multiple calls to get_text_blocking().

        Args:
            duration: How long to listen (None = indefinitely)
            on_speech: Callback for each transcription

        Returns:
            dict: Session metadata
        """
        if not self.is_active:
            return {"success": False, "error": "STT adapter not initialized", "transcriptions": []}

        session_start = time.time()
        transcriptions = []

        try:
            # If duration is specified, listen for that long
            if duration:
                end_time = session_start + duration
                while time.time() < end_time:
                    result = await self.get_text_blocking()
                    if result["success"]:
                        text = result["text"]
                        transcription_entry = {
                            "text": text,
                            "timestamp": result["timestamp"],
                            "session_time": time.time() - session_start,
                        }
                        transcriptions.append(transcription_entry)

                        if on_speech:
                            on_speech(text)

                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.01)
            else:
                # Indefinite listening - not practical with blocking API
                return {
                    "success": False,
                    "error": "Indefinite continuous listening not supported with blocking API",
                    "transcriptions": [],
                }

            return {
                "success": True,
                "transcriptions": transcriptions,
                "session_duration": time.time() - session_start,
                "total_utterances": len(transcriptions),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "transcriptions": transcriptions}

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics"""
        return {
            "is_active": self.is_active,
            "total_transcriptions": self.total_transcriptions,
            "last_transcription_time": self.last_transcription_time,
            "engine_type": type(self.engine).__name__ if hasattr(self, "engine") else None,
            "model": self.model.value,
            "language": self.language,
            "enable_realtime": self.enable_realtime,
            "sensitivity_config": self.sensitivity_config,
        }

    def reset_stats(self) -> None:
        """Reset adapter statistics"""
        self.total_transcriptions = 0
        self.last_transcription_time = None

    def shutdown(self) -> None:
        """Clean shutdown of the STT engine"""
        try:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            self.is_active = False

        except Exception as e:
            logger.error(f"⚠️  Error during STT shutdown: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
