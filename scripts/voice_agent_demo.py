#!/usr/bin/env python3
"""
Voice Agent Demo - Complete voice conversation with AI

This demonstrates the full VoiceAgent functionality:
- Speech-to-Text (STT) using RealtimeSTT
- Large Language Model (LLM) using Ollama
- Text-to-Speech (TTS) using RealtimeTTS with Kokoro

Usage:
    uv run python scripts/voice_agent_demo.py

Requirements:
    - Working microphone and speakers
    - Ollama running locally with qwen3:1.7b model
    - RealtimeSTT and RealtimeTTS libraries

Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull model: ollama pull qwen3:1.7b
    3. Install dependencies: uv sync
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent_course.application.voice_agent import VoiceAgent
from voice_agent_course.domain.prompts.system_prompts import HER_COMPANION
from voice_agent_course.infrastructure.audio.realtime_stt_adapter import STTModel


async def demo_basic_voice_agent():
    """Demo the basic voice agent functionality"""
    print("💬 Voice Companion Demo")
    print("=" * 60)
    print("This will start a natural voice conversation with an AI companion!")
    print()
    print("💡 Instructions:")
    print("   - Speak naturally, like talking to a friend")
    print("   - The AI will respond with warmth and curiosity")
    print("   - Say 'exit', 'quit', or 'goodbye' to end")
    print()
    print("⚠️  Make sure:")
    print("   - Your microphone is working")
    print("   - Your speakers are on")
    print("   - Ollama is running with qwen3:1.7b model")
    print()

    input("Press Enter to start your conversation...")

    try:
        # Create and start the voice agent
        agent = VoiceAgent(
            model="qwen3:1.7b",
            stt_model=STTModel.TINY_EN,
        )

        await agent.run_conversation()

    except KeyboardInterrupt:
        print("\n👋 Voice agent stopped by user")
    except Exception as e:
        print(f"❌ Error running voice agent: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Is Ollama running? (ollama serve)")
        print("   - Is the model available? (ollama pull qwen3:1.7b)")
        print("   - Is your microphone working?")


async def demo_custom_voice_agent():
    """Demo with custom configuration"""
    print("🌟 Thoughtful Companion Demo")
    print("=" * 60)

    # Use the thoughtful companion prompt from domain layer

    print("🌸 Starting thoughtful companion mode...")
    print("💭 Focus: Deep conversations, philosophy, and wonder")
    print()

    try:
        agent = VoiceAgent(
            model="qwen3:1.7b",
            stt_model=STTModel.TINY_EN,
            system_behavior=HER_COMPANION,
        )

        await agent.run_conversation()

        # Show stats after conversation
        print("\n📊 Session Statistics:")
        stats = agent.get_stats()
        print(f"   STT Transcriptions: {stats['stt_stats']['total_transcriptions']}")
        print(f"   TTS Characters: {stats['tts_stats']['total_characters_processed']}")
        print(f"   LLM Conversation Turns: {stats['llm_stats']['conversation_turns']}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_menu():
    """Interactive demo menu"""
    print("🚀 Voice Agent Demo Suite")
    print("=" * 60)
    print()
    print("Choose a demo:")
    print("1. Voice Companion (Warm & empathetic, like 'Her')")
    print("2. Thoughtful Companion (Deep & philosophical)")
    print("3. Exit")
    print()

    while True:
        try:
            await demo_basic_voice_agent()

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break


async def main():
    """Main demo runner"""
    print("🎙️  Voice Companion Demo")
    print("=" * 60)
    print("Natural voice conversations with an empathetic AI companion")
    print()

    # Check if we should run a specific demo
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        if demo_type == "basic":
            await demo_basic_voice_agent()
        elif demo_type == "custom":
            await demo_custom_voice_agent()
        else:
            print(f"❌ Unknown demo type: {demo_type}")
            print("Available: basic, custom")
    else:
        # Interactive menu
        await demo_menu()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
