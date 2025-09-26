import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.application.voice_agent_enhanced import EnhancedVoiceAgent


async def main():
    """Enhanced Voice Agent Demo with Tools"""
    parser = argparse.ArgumentParser(description="Enhanced Voice Agent Demo with Tools")
    parser.add_argument("--llm-provider", default="groq", help="LLM provider to use (default: groq)")
    parser.add_argument("--llm-model", help="Specific model to use (optional, uses provider default)")

    args = parser.parse_args()

    print("🛠️  Enhanced Voice Agent Demo")
    print("=" * 50)
    print(f"🤖 Provider: {args.llm_provider}")
    if args.llm_model:
        print(f"🧠 Model: {args.llm_model}")
    print()
    print("💡 Try: 'Get me a random number', 'Weather in Paris'")
    print("🎯 Test interruption by speaking while tools are running")
    print("⏹️  Say 'exit', 'quit', or 'goodbye' to end")
    print()

    try:
        # Create the agent
        print("🔧 Creating Enhanced Voice Agent...")
        agent = EnhancedVoiceAgent(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )
        print("✅ Agent created successfully!")

        # Show agent stats
        try:
            stats = agent.get_stats()
            langgraph_stats = stats["langgraph_agent_stats"]
            print(f"   Provider: {langgraph_stats['llm_provider']}")
            print(f"   Model: {langgraph_stats['llm_model']}")
            print(f"   Temperature: {langgraph_stats['temperature']}")
            print(f"   Tools available: {langgraph_stats['tools_count']}")
            print(f"   Tool names: {', '.join(langgraph_stats['tool_names'])}")
            print()
        except Exception as e:
            print(f"❌ Failed to get agent stats: {e}")
            return

        input("Press Enter to start...")

        await agent.run_conversation()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
