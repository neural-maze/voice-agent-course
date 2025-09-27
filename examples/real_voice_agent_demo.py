"""
Demo of the WebSocket Voice Agent with real STT/TTS adapters.
This version uses microphone input and local speaker output for demonstration.
"""

import asyncio

from loguru import logger

from voice_agent_course.application.voice_agent import VoiceAgent
from voice_agent_course.infrastructure.audio.realtime_stt_adapter import STTModel


class MockWebSocket:
    """Mock WebSocket for demonstration purposes"""

    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        """Mock sending JSON data"""
        logger.info(f"📤 WebSocket would send: {data}")
        self.messages.append(data)

    async def receive(self):
        """Mock receiving data (not used in this demo)"""
        # In a real WebSocket, this would wait for client messages
        await asyncio.sleep(1)
        return {"type": "websocket.disconnect"}


async def main():
    """
    Demo the real voice agent with microphone input and speaker output.
    This shows how the WebSocket architecture works with real audio processing.
    """
    logger.info("🎙️  Real Voice Agent Demo")
    logger.info("=" * 50)
    logger.info("This demo uses:")
    logger.info("• Real microphone input via RealtimeSTT")
    logger.info("• Real speaker output via RealtimeTTS (Kokoro)")
    logger.info("• LangGraph agent with tool support")
    logger.info("• WebSocket-ready architecture")
    logger.info("=" * 50)
    logger.info("💡 Speak into your microphone to interact!")
    logger.info("💡 Say 'exit', 'quit', or 'stop' to end the conversation")
    logger.info("=" * 50)

    # Create mock WebSocket
    mock_websocket = MockWebSocket()

    try:
        # Create voice agent with real STT/TTS
        voice_agent = VoiceAgent(
            websocket=mock_websocket,
            llm_provider="groq",
            llm_model="llama-3.1-8b-instant",
            llm_temperature=0.7,
            stt_model=STTModel.TINY_EN,  # Fast model for demo
            stt_language="en",
        )

        logger.info("✅ Voice agent initialized!")
        logger.info("🎤 Listening for speech...")

        # Start the voice agent
        # Note: This will run the 5-task architecture but use microphone instead of WebSocket audio
        await voice_agent.run_conversation()

    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error in demo: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")

    logger.info("👋 Demo complete!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
