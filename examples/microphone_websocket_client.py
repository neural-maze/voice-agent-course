"""
Real Microphone WebSocket Client for Voice Agent.
Captures audio from your microphone and sends it to the WebSocket voice agent server.
"""

import asyncio
import json
import threading

import pyaudio
import websockets
from loguru import logger


class MicrophoneWebSocketClient:
    """WebSocket client that captures real microphone audio and sends it to the voice agent"""

    def __init__(self, uri: str = "ws://localhost:8000/voice"):
        self.uri = uri
        self.websocket = None
        self.audio = None
        self.input_stream = None  # For microphone input
        self.is_recording = False
        self.shutdown_event = threading.Event()

        # Audio configuration (16kHz, 16-bit, mono - matches voice agent expectations)
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    async def connect(self):
        """Connect to the WebSocket voice agent server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"🔌 Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False

    async def send_control_message(self, message_type: str, **kwargs):
        """Send control message to server"""
        if self.websocket:
            message = {"type": message_type, **kwargs}
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Sent control: {message}")

    async def send_audio_chunk(self, chunk: bytes):
        """Send audio chunk to server"""
        if self.websocket:
            # Send as binary WebSocket message
            await self.websocket.send(chunk)
            # logger.info(f"🎵 Sent audio chunk: {len(chunk)} bytes")

    def start_microphone_capture(self):
        """Start capturing audio from microphone in a separate thread"""

        def capture_audio():
            logger.info("🎤 Starting microphone capture...")

            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Open microphone input stream
            self.input_stream = self.audio.open(
                format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
            )

            self.is_recording = True
            logger.info("🔴 Recording started - speak into your microphone!")

            # Capture and send audio chunks
            while not self.shutdown_event.is_set():
                try:
                    # Read audio chunk from microphone
                    data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)

                    # Send to WebSocket server (we'll queue this for the async task)
                    if hasattr(self, "_audio_queue"):
                        self._audio_queue.put_nowait(data)

                except Exception as e:
                    logger.error(f"❌ Error capturing audio: {e}")
                    break

            # Cleanup
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.audio:
                self.audio.terminate()

            self.is_recording = False
            logger.info("🔴 Recording stopped")

        # Start capture in separate thread
        capture_thread = threading.Thread(target=capture_audio, daemon=True)
        capture_thread.start()
        return capture_thread

    async def audio_sender_task(self):
        """Task to send audio chunks from queue to WebSocket"""
        self._audio_queue = asyncio.Queue()

        while True:
            try:
                # Get audio chunk from queue
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break

                # Send to server
                await self.send_audio_chunk(chunk)

            except Exception as e:
                logger.error(f"❌ Error sending audio: {e}")
                break

    async def message_listener_task(self):
        """Listen for messages from the server"""
        logger.info("👂 Listening for server messages...")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    # Text message (JSON)
                    try:
                        data = json.loads(message)
                        await self._handle_text_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"📄 Received non-JSON text: {message}")
                elif isinstance(message, bytes):
                    # Binary message (TTS audio)
                    await self._handle_audio_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed by server")
        except Exception as e:
            logger.error(f"❌ Error receiving messages: {e}")

    async def _handle_text_message(self, data: dict):
        """Handle text messages from server"""
        msg_type = data.get("type", "unknown")
        content = data.get("content", "")

        if msg_type == "partial_transcription":
            logger.info(f"🔄 You (partial): {content}")
        elif msg_type == "final_transcription":
            logger.info(f"👤 You (final): {content}")
        elif msg_type == "partial_assistant_text":
            # Print assistant response token by token (streaming)
            print(f"🤖 {content}", end="", flush=True)
        elif msg_type == "final_assistant_answer":
            # Print final complete response
            print(f"\n🤖 Assistant (complete): {content}")
            logger.info(f"🤖 Assistant finished: {content}")
        elif msg_type == "recording_start":
            logger.info("🎤 Server detected speech start")
        elif msg_type == "first_audio_chunk":
            logger.info("🔊 Server started generating response")
        elif msg_type == "tts_chunk":
            # Skip TTS audio for now - just log that we received it
            logger.debug(f"🔊 Received TTS chunk (skipped): {len(content)} chars")
        else:
            logger.info(f"📨 Server message [{msg_type}]: {content}")

    async def _handle_tts_chunk(self, base64_audio: str):
        """Handle TTS audio chunk (base64 encoded) - just log for now"""
        logger.debug(f"🔊 Received TTS chunk: {len(base64_audio)} chars (skipped for text-only mode)")

    async def _handle_audio_message(self, audio_data: bytes):
        """Handle audio messages from server (TTS) - currently just logging"""
        logger.debug(f"🔊 Received binary audio: {len(audio_data)} bytes (skipped for text-only mode)")

    async def run_conversation(self):
        """Run the real-time voice conversation"""
        logger.info("🚀 Starting real-time voice conversation (TEXT-ONLY MODE)...")
        logger.info("💡 Speak into your microphone to interact with the voice agent")
        logger.info("💡 You'll see text responses instead of hearing audio")
        logger.info("💡 Press Ctrl+C to stop")
        logger.info("==================================================")

        # Start microphone capture
        capture_thread = self.start_microphone_capture()

        # Wait a moment for microphone to initialize
        await asyncio.sleep(1)

        # Send initial control message
        await self.send_control_message("tts_start")

        # Start concurrent tasks (no audio playback for now)
        tasks = [
            asyncio.create_task(self.audio_sender_task()),
            asyncio.create_task(self.message_listener_task()),
        ]

        try:
            # Wait for any task to complete or Ctrl+C
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except KeyboardInterrupt:
            logger.info("⏹️  Stopping conversation...")
        finally:
            # Cleanup
            self.shutdown_event.set()

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

            # Wait for tasks to finish
            await asyncio.gather(*pending, return_exceptions=True)

            # Wait for capture thread to finish
            if capture_thread.is_alive():
                capture_thread.join(timeout=2)

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Connection closed")


async def main():
    """Main function to run the microphone WebSocket client"""
    client = MicrophoneWebSocketClient()

    # Connect to server
    if await client.connect():
        try:
            await client.run_conversation()
        except KeyboardInterrupt:
            logger.info("👋 Goodbye!")
        finally:
            await client.close()
    else:
        logger.error("❌ Could not connect to server. Make sure the WebSocket server is running!")
        logger.info("💡 Start the server with: uv run python examples/websocket_server_example.py")


if __name__ == "__main__":
    asyncio.run(main())
