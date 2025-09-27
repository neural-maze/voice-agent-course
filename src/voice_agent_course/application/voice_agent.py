"""Voice Agent with LangGraph agent capabilities and tool support."""

import traceback
from typing import Any

from loguru import logger

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent

from ..infrastructure.audio.realtime_stt_adapter import RealtimeSTTAdapter, STTModel
from ..infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter


class VoiceAgent:
    """
    Voice-enabled AI agent with LangGraph tool support.
    Combines STT, LangGraph Agent with tools, and TTS for natural conversation.
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
    ):
        """
        Initialize the Voice Agent.

        Args:
            llm_provider: LLM provider (groq, ollama)
            llm_model: Model name (uses provider default if None)
            llm_temperature: Model temperature
        """
        logger.info("🤖 Initializing Voice Agent with LangGraph tools...")

        # Initialize TTS adapter first (needed for interruption callback)
        self.tts_adapter = RealtimeTTSAdapter()

        # Initialize STT with interruption callback for immediate TTS stopping
        self.stt_adapter = RealtimeSTTAdapter(
            model=STTModel.TINY_EN,
            language="en",
            on_recording_start=self._on_recording_start,
        )

        # Initialize LangGraph agent for conversation and tool orchestration
        self.langgraph_agent = LangGraphAgent(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )

    def _on_recording_start(self):
        """
        Primary interruption: Triggered immediately when recording starts.
        Stops TTS playback instantly to prevent interference.
        This is the main interruption mechanism for responsiveness.
        """
        self.tts_adapter.stop_playing()

    async def _main_loop(self):
        """
        Speech-to-text task that processes microphone input and triggers agent responses.
        """
        while True:
            try:
                text = await self.stt_adapter.get_text_blocking()

                if text:
                    logger.info(f"👤 User: {text}")
                    # Process with agent directly
                    await self._process_user_input(text)
                # If None, just continue listening (no error spam)

            except Exception as e:
                logger.error(f"❌ Error in STT task: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    async def _process_user_input(self, user_input: str):
        """
        Process user input through LangGraph agent with direct TTS streaming.
        """
        try:
            logger.info(f"🤖 Agent processing: {user_input}")

            # Stop any ongoing TTS playback immediately
            self.tts_adapter.stop_playing()

            # Stream response with immediate TTS feeding for low latency
            first_chunk = True

            async for chunk in self.langgraph_agent.stream(user_message=user_input):
                if chunk:
                    # Immediately feed to TTS - true streaming!
                    self.tts_adapter.feed_text(chunk)
                    # Start TTS playback on first chunk
                    if first_chunk:
                        self.tts_adapter.play_stream_async()
                        first_chunk = False
                    print(chunk, end="", flush=True)
            print()

        except Exception as e:
            logger.error(f"❌ Error processing user input: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    async def run_conversation(self):
        """
        Main conversation task for voice agent.
        """
        logger.info("🚀 Starting voice conversation with tool support...")
        logger.info("💡 Say something to begin, or say 'exit' to quit")

        try:
            # Run STT task
            await self._main_loop()

        except KeyboardInterrupt:
            logger.info("\n⏹️  Conversation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in conversation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def get_info(self) -> dict[str, Any]:
        """Get information about the voice agent."""
        return {
            "langgraph_agent": self.langgraph_agent.get_info(),
        }
