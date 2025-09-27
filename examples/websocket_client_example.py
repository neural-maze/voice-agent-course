"""
Example WebSocket client for testing the WebSocket Voice Agent.
This demonstrates how to interact with the 4-task architecture.
"""

import asyncio
import json

import websockets
from loguru import logger


class VoiceWebSocketClient:
    """Example WebSocket client for voice conversations"""

    def __init__(self, uri: str = "ws://localhost:8000/voice"):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"🔌 Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False

    async def send_control_message(self, message_type: str, **kwargs):
        """Send control message to server"""
        control_msg = {"type": message_type, **kwargs}
        await self.websocket.send(json.dumps(control_msg))
        logger.info(f"📤 Sent control: {control_msg}")

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to server"""
        await self.websocket.send(audio_data)
        logger.info(f"🎵 Sent audio chunk: {len(audio_data)} bytes")

    async def listen_for_messages(self):
        """Listen for messages from server"""
        logger.info("👂 Listening for server messages...")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    # Text message (JSON)
                    try:
                        data = json.loads(message)
                        await self._handle_text_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️  Invalid JSON: {message}")
                else:
                    # Binary message (audio)
                    await self._handle_audio_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed by server")
        except Exception as e:
            logger.error(f"❌ Error listening for messages: {e}")

    async def _handle_text_message(self, data: dict):
        """Handle text message from server"""
        msg_type = data.get("type")
        content = data.get("content", "")

        if msg_type == "partial_transcription":
            logger.info(f"🎤 Partial: {content}")
        elif msg_type == "final_transcription":
            logger.info(f"🎤 Final: {content}")
        elif msg_type == "assistant_partial":
            print(content, end="", flush=True)
        elif msg_type == "assistant_complete":
            print()  # New line
            logger.info("🤖 Assistant response complete")
        elif msg_type == "recording_start":
            logger.info("🎤 Recording started (interruption)")
        elif msg_type == "tts_start":
            logger.info("🔊 TTS playback started")
        else:
            logger.info(f"📨 Received: {data}")

    async def _handle_audio_message(self, audio_data: bytes):
        """Handle audio message from server"""
        logger.info(f"🔊 Received audio: {len(audio_data)} bytes")
        # Here you would play the audio or process it

    async def simulate_conversation(self):
        """Simulate a conversation with the voice agent"""
        logger.info("🎭 Starting simulated conversation...")

        # Simulate sending some control messages
        await self.send_control_message("tts_start")
        await asyncio.sleep(0.1)

        # Simulate sending audio chunks (dummy data)
        for i in range(5):
            dummy_audio = b"dummy_audio_chunk_" + str(i).encode() * 100
            await self.send_audio_chunk(dummy_audio)
            await asyncio.sleep(0.1)

        # Send stop message
        await self.send_control_message("tts_stop")

        # Listen for responses
        await asyncio.sleep(2)

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Connection closed")


async def main():
    """Main client function"""
    client = VoiceWebSocketClient()

    if await client.connect():
        try:
            # Run listening and simulation concurrently
            await asyncio.gather(
                client.listen_for_messages(),
                client.simulate_conversation(),
                return_exceptions=True,
            )
        except KeyboardInterrupt:
            logger.info("⏹️  Interrupted by user")
        finally:
            await client.close()
    else:
        logger.error("❌ Could not connect to server")


if __name__ == "__main__":
    logger.info("🚀 Starting WebSocket Voice Client...")
    asyncio.run(main())
