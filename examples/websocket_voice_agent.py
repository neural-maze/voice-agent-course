"""
Example of how to extend the VoiceAgent for WebSocket support.
This shows the future WebSocket implementation without breaking current functionality.
"""

import asyncio
from collections.abc import Callable

from voice_agent_course.application.voice_agent import (
    VoiceAgent,
)


class WebSocketInputSource:
    """WebSocket audio input source (future implementation)"""

    def __init__(self, websocket):
        self.websocket = websocket
        self.transcription_callback: Callable[[str], None] | None = None
        self.recording_start_callback: Callable[[], None] | None = None
        self.audio_processor = None  # Would be your STT processor
        self.shutdown_requested = False

    async def start_listening(self) -> None:
        """Start listening for WebSocket audio chunks"""
        while not self.shutdown_requested:
            try:
                # Receive audio data from WebSocket
                message = await self.websocket.receive()

                if message.get("type") == "websocket.receive":
                    data = message.get("bytes")
                    if data:
                        # Process audio chunk through STT
                        # This would feed to your audio processor
                        await self._process_audio_chunk(data)

            except Exception as e:
                print(f"Error in WebSocket input: {e}")
                break

    async def _process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk (placeholder for STT processing)"""
        # This is where you'd:
        # 1. Feed audio to STT processor
        # 2. Handle partial transcriptions
        # 3. Detect complete utterances
        # 4. Call transcription_callback when ready

        # For demo purposes, simulate transcription
        if len(audio_data) > 1000:  # Simulate "complete utterance"
            if self.transcription_callback:
                self.transcription_callback("Hello from WebSocket!")

    async def stop_listening(self) -> None:
        """Stop listening for WebSocket input"""
        self.shutdown_requested = True

    def set_transcription_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for when transcription is ready"""
        self.transcription_callback = callback

    def set_recording_start_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when recording starts (for interruption)"""
        self.recording_start_callback = callback


class WebSocketOutputSink:
    """WebSocket audio output sink (future implementation)"""

    def __init__(self, websocket):
        self.websocket = websocket
        self.tts_processor = None  # Would be your TTS processor

    async def stream_text(self, text: str) -> None:
        """Stream text as audio through WebSocket"""
        if text:
            # Convert text to audio and send via WebSocket
            # This would:
            # 1. Generate audio from text using TTS
            # 2. Send audio chunks via WebSocket
            await self._send_text_message(text)
        else:
            # End of stream signal
            await self._send_end_signal()

    async def stream_audio_chunk(self, audio_chunk: bytes) -> None:
        """Stream raw audio chunk via WebSocket"""
        await self.websocket.send_bytes(audio_chunk)

    def stop_playback(self) -> None:
        """Stop current audio playback"""
        # Send stop signal to client
        asyncio.create_task(self._send_stop_signal())

    async def _send_text_message(self, text: str):
        """Send text message to WebSocket client"""
        await self.websocket.send_json({"type": "text_chunk", "content": text})

    async def _send_end_signal(self):
        """Send end of stream signal"""
        await self.websocket.send_json({"type": "stream_end"})

    async def _send_stop_signal(self):
        """Send stop playback signal"""
        await self.websocket.send_json({"type": "stop_playback"})


# Example usage functions
def create_microphone_agent() -> VoiceAgent:
    """Create agent with microphone input (current functionality)"""
    return VoiceAgent(
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
    )


def create_websocket_agent(websocket) -> VoiceAgent:
    """Create agent with WebSocket input/output (future functionality)"""
    websocket_input = WebSocketInputSource(websocket)
    websocket_output = WebSocketOutputSink(websocket)

    return VoiceAgent(
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        audio_input=websocket_input,
        audio_output=websocket_output,
    )


async def main():
    """Example of using both modes"""
    print("🎤 Creating microphone agent...")
    create_microphone_agent()

    # This works exactly as before
    # await mic_agent.run_conversation()

    print("🌐 WebSocket agent would be created like this:")
    print("websocket_agent = create_websocket_agent(websocket)")
    print("await websocket_agent.run_conversation()")


if __name__ == "__main__":
    asyncio.run(main())
