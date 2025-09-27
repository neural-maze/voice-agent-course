"""
Audio Input Processor for WebSocket Voice Agent.
Based on production implementation - processes audio chunks instead of blocking microphone.
"""

import asyncio
from collections.abc import Callable

import numpy as np
from loguru import logger
from scipy.signal import resample_poly

from .transcription_processor import TranscriptionProcessor


class AudioInputProcessor:
    """
    Manages audio input, processes it for transcription, and handles related callbacks.

    This class receives raw audio chunks, resamples them to the required format (16kHz),
    feeds them to an underlying `TranscriptionProcessor`, and manages callbacks for
    real-time transcription updates, recording start events, and silence detection.
    It also runs the transcription process in a background task.
    """

    def __init__(
        self,
        source_sample_rate: int,
        source_channels: int,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        stt_model: str = "tiny",
        stt_language: str = "en",
        on_partial_transcription: Callable[[str], None] | None = None,
        on_final_transcription: Callable[[str], None] | None = None,
        on_recording_start: Callable[[], None] | None = None,
    ):
        """
        Initializes the AudioInputProcessor.

        Args:
            source_sample_rate: Sample rate of incoming audio
            source_channels: Number of channels in incoming audio
            target_sample_rate: Target sample rate for STT (default 16kHz)
            target_channels: Target channels for STT (default mono)
            stt_model: STT model to use
            stt_language: Language for STT processing
            on_partial_transcription: Callback for partial transcription results
            on_final_transcription: Callback for final transcription results
            on_recording_start: Callback when recording starts
        """
        logger.info("👂🚀 AudioInputProcessor initialized.")
        self.source_sample_rate = source_sample_rate
        self.source_channels = source_channels
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.on_partial_transcription = on_partial_transcription
        self.on_final_transcription = on_final_transcription
        self.on_recording_start = on_recording_start
        self.shutdown_event = asyncio.Event()
        self.transcription_task: asyncio.Task | None = None

        # Calculate resample ratio dynamically based on actual sample rates
        self.resample_ratio = self.source_sample_rate / self.target_sample_rate
        logger.info(
            f"👂🔄 Audio resampling: {source_sample_rate}Hz → {target_sample_rate}Hz (ratio: {self.resample_ratio:.2f})"
        )

        self.transcription_processor = TranscriptionProcessor(
            source_language=stt_language,
            realtime_transcription_callback=self._on_realtime_transcription,
            full_transcription_callback=self._on_full_transcription,
            on_recording_start_callback=self._on_recording_start_from_stt,
            # recorder_config={"model": stt_model, "language": stt_language},
        )
        logger.info("👂✅ AudioInputProcessor initialized.")

    def _on_realtime_transcription(self, text: str | None):
        """Internal callback for partial transcriptions from STT."""
        if text and self.on_partial_transcription:
            self.on_partial_transcription(text)

    def _on_full_transcription(self, text: str | None):
        """Internal callback for final transcriptions from STT."""
        if text and self.on_final_transcription:
            self.on_final_transcription(text)

    def _on_recording_start_from_stt(self):
        """Internal callback when STT detects recording start."""
        if self.on_recording_start:
            self.on_recording_start()

    async def feed_audio_chunk(self, audio_chunk: bytes):
        """
        Feeds a raw audio chunk to the processor.
        Resamples and converts to the target format before feeding to the transcriber.
        """
        # logger.info(f"👂🔊 AudioInputProcessor received chunk: {len(audio_chunk)} bytes")

        # Convert bytes to numpy array (int16)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        logger.debug(f"👂🔊 Converted to numpy array: {audio_np.shape}")

        # Resample if necessary
        if self.source_sample_rate != self.target_sample_rate:
            num = self.target_sample_rate
            den = self.source_sample_rate
            audio_np = resample_poly(audio_np, num, den)
            audio_np = audio_np.astype(np.int16)  # Convert back to int16
            logger.debug(f"👂🔊 Resampled from {self.source_sample_rate}Hz to {self.target_sample_rate}Hz")

        # Convert to mono if necessary (assuming stereo input, take one channel)
        if self.source_channels > self.target_channels:
            # Assuming interleaved stereo, take every other sample for mono
            audio_np = audio_np[:: self.source_channels]
            logger.debug(f"👂🔊 Converted to mono from {self.source_channels} channels")

        # Feed to transcription processor
        processed_bytes = audio_np.tobytes()
        # logger.info(f"👂🔊 Feeding {len(processed_bytes)} bytes to transcription processor")
        self.transcription_processor.feed_audio(processed_bytes)

    async def start_transcription_task(self):
        """Starts the background task for transcription processing."""
        logger.info("👂▶️ Starting background transcription task (Task-5).")
        self.transcription_task = asyncio.create_task(self.transcription_processor.process_audio_stream())

    def _setup_callbacks(self) -> None:
        """Sets up internal callbacks for the TranscriptionProcessor instance."""

        def partial_transcript_callback(text: str) -> None:
            """Handles partial transcription results from the transcriber."""
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    async def _run_transcription_loop(self) -> None:
        """
        Continuously runs the transcription loop in a background asyncio task.
        """
        task_name = (
            self.transcription_task.get_name() if hasattr(self.transcription_task, "get_name") else "TranscriptionTask"
        )
        logger.info(f"👂▶️ Starting background transcription task ({task_name}).")
        while True:
            try:
                # Run one cycle of the underlying blocking loop
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                logger.debug("👂✅ TranscriptionProcessor.transcribe_loop completed one cycle.")
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 Transcription loop ({task_name}) cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"👂💥 Transcription loop ({task_name}) encountered a fatal error: {e}. Loop terminated.",
                    exc_info=True,
                )
                self._transcription_failed = True
                break

        logger.info(f"👂⏹️ Background transcription task ({task_name}) finished.")

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        Converts raw audio bytes (int16) to target sample rate 16-bit PCM numpy array.

        Args:
            raw_bytes: Raw audio data assumed to be in int16 format.

        Returns:
            A numpy array containing the resampled audio in int16 format at target sample rate.
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # Calculate expected length after resampling for silence
            expected_len = int(np.ceil(len(raw_audio) / self.resample_ratio))
            return np.zeros(expected_len, dtype=np.int16)

        # Skip resampling if source and target sample rates are the same
        if self.source_sample_rate == self.target_sample_rate:
            return raw_audio

        # Convert to float32 for resampling precision
        audio_float32 = raw_audio.astype(np.float32)

        # Resample using scipy's resample_poly with proper up/down sampling
        # resample_poly(signal, up, down) where new_rate = old_rate * up / down
        # We want: new_rate = target_sample_rate, old_rate = source_sample_rate
        # So: target_sample_rate = source_sample_rate * up / down
        # Therefore: up/down = target_sample_rate / source_sample_rate
        up = self.target_sample_rate
        down = self.source_sample_rate
        resampled_float = resample_poly(audio_float32, up, down)

        # Convert back to int16, clipping to ensure validity
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16

    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        Continuously processes audio chunks received from an asyncio Queue.

        Args:
            audio_queue: An asyncio queue expected to yield dictionaries containing
                         'pcm' (raw audio bytes) or None to terminate.
        """
        logger.info("👂▶️ Starting audio chunk processing loop.")
        while True:
            try:
                if self._transcription_failed:
                    logger.error("👂🛑 Transcription task failed previously. Stopping audio processing.")
                    break

                if self.transcription_task and self.transcription_task.done() and not self._transcription_failed:
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        logger.error(
                            f"👂🛑 Transcription task finished with unexpected error: {task_exception}. "
                            "Stopping audio processing.",
                            exc_info=task_exception,
                        )
                        self._transcription_failed = True
                        break
                    else:
                        logger.warning("👂⏹️ Transcription task is no longer running. Stopping audio processing.")
                        break

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal for audio processing.")
                    break

                logger.info(f"👂📥 Processing audio chunk: {len(audio_data.get('pcm', b''))} bytes")

                pcm_data = audio_data.pop("pcm")

                # Process audio chunk
                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    continue

                # Feed audio only if not interrupted and transcriber should be running
                if not self.interrupted and not self._transcription_failed:
                    self.transcriber.feed_audio(processed.tobytes(), audio_data)

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing task cancelled.")
                break
            except Exception as e:
                logger.error(f"👂💥 Audio processing error in queue loop: {e}", exc_info=True)

        logger.info("👂⏹️ Audio chunk processing loop finished.")

    async def shutdown(self):
        """Shuts down the audio input processor and its components."""
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        if self.transcription_task:
            self.transcription_task.cancel()
            try:
                await self.transcription_task
            except asyncio.CancelledError:
                logger.info("👂✅ Transcription task cancelled.")
        self.transcription_processor.shutdown()
        logger.info("👂✅ AudioInputProcessor shutdown complete.")

    def get_stats(self) -> dict:
        """Returns statistics from the underlying transcription processor."""
        return self.transcription_processor.get_stats()
