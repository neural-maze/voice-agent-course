import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.application.voice_agent_enhanced import EnhancedVoiceAgent


async def demo_enhanced_agent():
    """Demo the enhanced voice agent with tools"""
    print("🛠️  Enhanced Voice Agent Demo")
    print("=" * 60)
    print("This will start a voice conversation with an AI agent that can use tools!")
    print()
    print("💡 Instructions:")
    print("   - Speak naturally, like talking to a smart assistant")
    print("   - Ask for tasks that require tools (see examples below)")
    print("   - Try interrupting during tool execution!")
    print("   - Say 'exit', 'quit', or 'goodbye' to end")
    print()
    print("🛠️  Try these commands:")
    print("   - 'Get me a random number'")
    print("   - 'Calculate the 15th Fibonacci number'")
    print("   - 'What's the weather like in Paris?'")
    print("   - 'Give me two random numbers' (multiple tool calls)")
    print()
    print("🎯 Test Interruption:")
    print("   - Ask for a random number, then start talking while it's processing")
    print("   - The agent should handle interruption gracefully")
    print()
    print("⚠️  Make sure:")
    print("   - Your microphone is working")
    print("   - You are using headphones (recommended for best experience)")
    print()

    input("Press Enter to start your enhanced conversation...")

    try:
        # Create and start the enhanced voice agent
        agent = EnhancedVoiceAgent(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
        )

        await agent.run_conversation()

    except KeyboardInterrupt:
        print("\n👋 Enhanced voice agent stopped by user")
    except Exception as e:
        print(f"❌ Error running enhanced voice agent: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Is your microphone working?")
        print("   - Are LangGraph dependencies installed? (uv sync)")


async def main():
    """Main demo runner"""
    print("🎙️  Enhanced Voice Agent Demo")
    print("=" * 60)
    print("Natural voice conversations with an AI agent that can use tools!")
    print()

    while True:
        await demo_enhanced_agent()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Enhanced demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
