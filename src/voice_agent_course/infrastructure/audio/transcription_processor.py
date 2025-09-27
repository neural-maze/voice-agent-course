"""
Transcription Processor for WebSocket Voice Agent.
Based on production implementation - uses RealtimeSTT with chunk feeding.
"""

import asyncio
import copy
from collections.abc import Callable
from typing import Any

from loguru import logger
from RealtimeSTT import AudioToTextRecorder

# Constants
INT16_MAX_ABS_VALUE: float = 32768.0
SAMPLE_RATE: int = 16000

# Default recorder configuration
DEFAULT_RECORDER_CONFIG: dict[str, Any] = {
    "use_microphone": False,  # ← KEY CHANGE: No microphone, chunk feeding only
    "spinner": False,
    "model": "tiny",
    "realtime_model_type": "tiny",
    "use_main_model_for_realtime": False,
    "language": "en",
    "silero_sensitivity": 0.4,  # ← CRITICAL: Much more sensitive for chunk feeding
    "webrtc_sensitivity": 2,  # ← CRITICAL: Lower for better detection
    "post_speech_silence_duration": 0.4,  # ← CRITICAL: Shorter for faster response
    "min_length_of_recording": 0.3,  # ← CRITICAL: Shorter minimum
    "min_gap_between_recordings": 0,
    "enable_realtime_transcription": True,
    "realtime_processing_pause": 0.02,  # ← CRITICAL: Faster processing
    "silero_use_onnx": True,
    "silero_deactivity_detection": True,
    "early_transcription_on_silence": 0,
    "beam_size": 3,
    "beam_size_realtime": 3,
    "no_log_file": True,
    "allowed_latency_limit": 500,
    "debug_mode": True,  # ← ENABLED: More verbose for debugging
    "level": 20,  # ← CRITICAL: Set logging level for VAD debugging
    "initial_prompt_realtime": (
        "The sky is blue. When the sky... She walked home. Because he... Today is sunny. If only I..."
    ),
    "faster_whisper_vad_filter": False,
}


class TranscriptionProcessor:
    """
    Manages audio transcription using RealtimeSTT with chunk feeding (no microphone).

    This class acts as a bridge between raw audio chunks and transcription results,
    coordinating the RealtimeSTT recorder and processing callbacks.
    """

    def __init__(
        self,
        source_language: str = "en",
        realtime_transcription_callback: Callable[[str], None] | None = None,
        full_transcription_callback: Callable[[str], None] | None = None,
        potential_sentence_end: Callable[[str], None] | None = None,
        silence_active_callback: Callable[[bool], None] | None = None,
        on_recording_start_callback: Callable[[], None] | None = None,
        is_orpheus: bool = False,
        pipeline_latency: float = 0.5,
        recorder_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the TranscriptionProcessor for chunk-based processing.
        """
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_sentence_end = potential_sentence_end
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.is_orpheus = is_orpheus
        self.pipeline_latency = pipeline_latency

        self.recorder: AudioToTextRecorder | None = None
        self.realtime_text: str | None = None
        self.final_transcription: str | None = None
        self.shutdown_performed: bool = False
        self.silence_time: float = 0.0
        self.silence_active: bool = False
        self.on_final_callback = None  # Will be set in process_audio_stream

        # Use provided config or default
        self.recorder_config = copy.deepcopy(recorder_config if recorder_config else DEFAULT_RECORDER_CONFIG)
        self.recorder_config["language"] = self.source_language

        self._create_recorder()
        self._start_silence_monitor()  # ← CRITICAL: Start the silence monitor thread
        logger.info("👂✅ TranscriptionProcessor initialized for chunk feeding")

    def _create_recorder(self) -> None:
        """
        Creates the RealtimeSTT recorder with chunk feeding configuration.
        """

        def on_partial(text: str | None):
            """Callback for real-time transcription updates."""
            if text is None:
                logger.debug("👂📝 Partial transcription callback received None")
                return
            self.realtime_text = text
            import time

            partial_time = time.time()
            logger.info(f"👂📝 PARTIAL TRANSCRIPTION: '{text}' (timestamp: {partial_time:.3f})")

            # CRITICAL: Check for sentence endings and trigger early transcription
            self._detect_sentence_end(text)

            if self.realtime_transcription_callback:
                self.realtime_transcription_callback(text)
            else:
                logger.warning("👂⚠️ No realtime_transcription_callback set")

        def start_recording():
            """Callback when recording starts."""
            import time

            start_time = time.time()
            logger.info(f"👂▶️ RECORDING STARTED - Voice detected! (timestamp: {start_time:.3f})")
            if self.on_recording_start_callback:
                self.on_recording_start_callback()

        def stop_recording() -> bool:
            """Callback when recording stops."""
            import time

            stop_time = time.time()
            logger.info(f"👂⏹️ RECORDING STOPPED - Voice ended. (timestamp: {stop_time:.3f})")
            return False

        def on_silence_active(is_active: bool):
            """Callback when silence state changes."""
            import time

            silence_time = time.time()
            if is_active:
                logger.info(f"🔇 Silence detected (timestamp: {silence_time:.3f})")
            else:
                logger.info(f"🔊 Voice activity detected (timestamp: {silence_time:.3f})")

        def on_vad_detect_start():
            """Callback when VAD detects start of silence (end of speech)."""
            import time

            vad_start_time = time.time()
            logger.info(f"🎙️ VAD: Speech ended, silence started (timestamp: {vad_start_time:.3f})")
            # Update silence time for the monitor thread
            self.silence_time = vad_start_time

        def on_vad_detect_stop():
            """Callback when VAD detects end of silence (start of speech)."""
            import time

            vad_stop_time = time.time()
            logger.info(f"🎙️ VAD: Silence ended, speech started (timestamp: {vad_stop_time:.3f})")
            # Reset silence time when speech resumes
            self.silence_time = 0.0

        # Prepare recorder configuration
        active_config = self.recorder_config.copy()
        active_config["on_realtime_transcription_update"] = on_partial
        active_config["on_recording_start"] = start_recording
        active_config["on_recording_stop"] = stop_recording
        # ← CRITICAL: Add VAD callbacks for better silence detection
        active_config["on_vad_detect_start"] = on_vad_detect_start
        active_config["on_vad_detect_stop"] = on_vad_detect_stop

        try:
            self.recorder = AudioToTextRecorder(**active_config)
            logger.info("👂✅ AudioToTextRecorder created for chunk feeding")
            logger.info("👂🔗 Final transcription callback will be set up in process_audio_stream")
        except Exception as e:
            logger.error(f"👂💥 Failed to create recorder: {e}")
            self.recorder = None

    async def process_audio_stream(self):
        """Runs the RealtimeSTT processing loop using the PRODUCTION transcribe_loop pattern."""
        logger.info("👂▶️ Starting RealtimeSTT audio stream processing...")
        if self.recorder:
            # Set up final transcription callback
            def on_final(text: str | None):
                if text is None or text == "":
                    logger.warning("👂❓ Final transcription received None or empty string.")
                    return

                import time

                final_time = time.time()
                self.final_transcription = text
                logger.info(f"👂✅ FINAL TRANSCRIPTION: '{text}' (timestamp: {final_time:.3f})")

                if self.full_transcription_callback:
                    self.full_transcription_callback(text)
                else:
                    logger.warning("👂⚠️ No full_transcription_callback set")

            # Store the callback for use in transcribe_loop
            self.on_final_callback = on_final

            # Use the PRODUCTION transcribe_loop pattern - this is the key difference!
            logger.info("👂🚀 Starting PRODUCTION transcribe_loop...")

            # Run the transcribe_loop method like the production code
            await asyncio.to_thread(self.transcribe_loop)
        else:
            logger.error("👂❌ Cannot start audio stream: Recorder not initialized")

        logger.info("👂🛑 RealtimeSTT audio stream processing stopped.")

    def transcribe_loop(self):
        """
        Production-style transcribe_loop method that runs the recorder's main processing loop.
        This is the blocking method that should be called from a thread.
        """
        logger.info("👂🔄 Starting production transcribe_loop...")

        try:
            # Register the final callback with the recorder - this is the blocking call
            if hasattr(self.recorder, "text") and hasattr(self, "on_final_callback"):
                logger.info("👂🔗 Registering final transcription callback in thread...")
                self.recorder.text(self.on_final_callback)  # This blocks until transcription
                logger.info("👂✅ Final transcription callback completed!")
            else:
                logger.error("👂❌ Recorder missing 'text' method or callback not set")

        except Exception as e:
            logger.error(f"👂💥 Error in transcribe_loop: {e}")

        logger.info("👂✅ Production transcribe_loop completed")

    def _start_silence_monitor(self) -> None:
        """
        CRITICAL: Starts a background thread to monitor silence duration and trigger
        early transcription when silence is detected, preventing 20-second delays.

        This is the KEY component missing from the original implementation!
        """
        import threading
        import time

        def monitor():
            logger.info("👂🔍 Starting silence monitor thread...")
            hot = False

            while not self.shutdown_performed:
                try:
                    if not self.recorder:
                        time.sleep(0.1)
                        continue

                    # Use our tracked silence time instead of recorder's internal state
                    speech_end_silence_start = self.silence_time

                    if speech_end_silence_start and speech_end_silence_start != 0:
                        silence_waiting_time = getattr(self.recorder, "post_speech_silence_duration", 0.4)
                        time_since_silence = time.time() - speech_end_silence_start

                        logger.debug(
                            f"👂⏱️ Silence monitor: {time_since_silence:.3f}s since silence started"
                            f"(limit: {silence_waiting_time:.3f}s)"
                        )

                        # CRITICAL: Trigger early transcription at 70% of silence duration
                        early_trigger_time = silence_waiting_time * 0.7

                        if time_since_silence > early_trigger_time and not hot:
                            # Only trigger if we actually have meaningful speech content
                            if self.realtime_text and len(self.realtime_text.strip()) > 3:
                                hot = True
                                logger.info(
                                    f"🔥 HOT STATE: Triggering early transcription after "
                                    f"{time_since_silence:.3f}s silence"
                                )
                                logger.info(f"👂⚡ EARLY TRANSCRIPTION TRIGGER: '{self.realtime_text}'")

                                # Trigger the final callback directly
                                if hasattr(self, "on_final_callback") and self.on_final_callback:
                                    self.on_final_callback(self.realtime_text)
                            else:
                                logger.debug(
                                    f"👂⏸️ Silence detected but no meaningful speech content yet "
                                    f"(text: '{self.realtime_text}')"
                                )
                    else:
                        if hot:
                            hot = False
                            logger.debug("👂❄️ COLD STATE: Speech resumed")

                except Exception as e:
                    logger.error(f"👂💥 Error in silence monitor: {e}")

                time.sleep(0.01)  # Check every 10ms for responsiveness

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("👂✅ Silence monitor thread started")

    def feed_audio(self, chunk: bytes, audio_meta_data: dict[str, Any] | None = None) -> None:
        """
        Feeds an audio chunk to the recorder for processing.

        Args:
            chunk: Raw audio data as bytes
            audio_meta_data: Optional metadata about the audio
        """
        import time

        feed_time = time.time()
        logger.debug(
            f"👂🔊 TranscriptionProcessor.feed_audio called with {len(chunk)} bytes (timestamp: {feed_time:.3f})"
        )

        if self.recorder and not self.shutdown_performed:
            try:
                logger.debug(f"👂🔊 Feeding to RealtimeSTT recorder: {len(chunk)} bytes (timestamp: {feed_time:.3f})")
                self.recorder.feed_audio(chunk)
                logger.debug(
                    f"👂✅ Successfully fed audio chunk of size {len(chunk)} bytes to RealtimeSTT "
                    f"(timestamp: {feed_time:.3f})"
                )
            except Exception as e:
                logger.error(f"👂💥 Error feeding audio to RealtimeSTT: {e}")
        elif not self.recorder:
            logger.error("👂⚠️ Cannot feed audio: Recorder not initialized")
        elif self.shutdown_performed:
            logger.warning("👂⚠️ Cannot feed audio: Shutdown already performed")

    def abort_generation(self) -> None:
        """
        Aborts any ongoing generation process.
        """
        logger.info("👂⏹️ Generation aborted")

    def shutdown(self) -> None:
        """
        Shuts down the transcription processor.
        """
        if not self.shutdown_performed:
            logger.info("👂🔌 Shutting down TranscriptionProcessor...")
            self.shutdown_performed = True

            if self.recorder:
                try:
                    self.recorder.shutdown()
                    logger.info("👂🔌 Recorder shutdown completed")
                except Exception as e:
                    logger.error(f"👂💥 Error during recorder shutdown: {e}")
                finally:
                    self.recorder = None

            logger.info("👂🔌 TranscriptionProcessor shutdown finished")
        else:
            logger.info("👂ℹ️ Shutdown already performed")

    def get_stats(self) -> dict:
        """Returns statistics from the RealtimeSTT recorder."""
        if self.recorder:
            return self.recorder.get_stats()
        return {}
