import argparse
import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.application.voice_agent import VoiceAgent


async def main():
    """Enhanced Voice Agent Demo with Tools"""
    parser = argparse.ArgumentParser(description="Enhanced Voice Agent Demo with Tools")
    parser.add_argument("--llm-provider", default="groq", help="LLM provider to use (default: groq)")
    parser.add_argument("--llm-model", help="Specific model to use (optional, uses provider default)")

    args = parser.parse_args()

    print("🛠️  Enhanced Voice Agent Demo")
    print("=" * 50)
    print()
    print("💡 Try: 'Get me a random number', 'Weather in Paris'")
    print("🎯 Test interruption by speaking while tools are running")
    print("⏹️  Say 'exit', 'quit', or 'goodbye' to end")
    print()

    try:
        # Create the agent
        print("🔧 Creating Enhanced Voice Agent...")
        voice_agent = VoiceAgent(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )
        print("✅ Agent created successfully!")

        # Show agent info
        try:
            langgraph_agent_info = voice_agent.get_info()["langgraph_agent"]
            print(f"   Provider: {langgraph_agent_info['llm_provider']}")
            print(f"   Model: {langgraph_agent_info['llm_model']}")
            print(f"   Temperature: {langgraph_agent_info['llm_temperature']}")
            print(f"   Tools available: {langgraph_agent_info['tools_count']}")
            print(f"   Tool names: {', '.join(langgraph_agent_info['tool_names'])}")
            print()
        except Exception as e:
            print(f"❌ Failed to get agent stats: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            return

        input("Press Enter to start...")

        await voice_agent.run_conversation()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())
