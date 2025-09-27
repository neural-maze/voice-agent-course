"""
Example FastAPI WebSocket server using the new WebSocket-ready VoiceAgent.
This demonstrates the production-ready 4-task architecture.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from voice_agent_course.application.voice_agent import VoiceAgent

app = FastAPI(title="WebSocket Voice Agent Server")


@app.websocket("/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversations.
    Implements the production 4-task architecture.
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection accepted")

    try:
        # Create WebSocket-ready voice agent
        voice_agent = VoiceAgent(
            websocket=websocket,
            llm_provider="groq",
            llm_model="llama-3.1-8b-instant",
            llm_temperature=0.7,
        )

        # Run the conversation with 4 concurrent tasks
        await voice_agent.run_conversation()

    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected")
    except Exception as e:
        logger.error(f"❌ Error in WebSocket endpoint: {e}")
    finally:
        logger.info("🔄 Cleaning up WebSocket connection")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "websocket-voice-agent"}


@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "message": "WebSocket Voice Agent Server",
        "websocket_url": "ws://localhost:8000/voice",
        "usage": "Connect to the WebSocket endpoint to start voice conversations",
        "architecture": "4-task production pattern with real-time streaming",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting WebSocket Voice Agent Server...")
    uvicorn.run(
        "websocket_server_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
