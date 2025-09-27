#!/usr/bin/env python3
"""LangGraph Agent Demo with invoke and stream testing."""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent


async def main():
    """LangGraph Agent Demo"""
    parser = argparse.ArgumentParser(description="LangGraph Agent Demo")
    parser.add_argument("--llm-provider", default="groq", help="LLM provider to use (default: groq)")
    parser.add_argument("--llm-model", help="Specific model to use (optional, uses provider default)")

    args = parser.parse_args()

    print("🚀 LangGraph Agent Demo")
    print("=" * 50)
    print(f"🤖 Provider: {args.llm_provider}")
    if args.llm_model:
        print(f"🧠 Model: {args.llm_model}")
    print()

    try:
        # Create the agent
        print("🔧 Creating LangGraph agent...")
        try:
            langgraph_agent = LangGraphAgent(
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
            print("✅ Agent created successfully!")
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            traceback.print_exc()
            return

        # Show agent info
        try:
            langgraph_agent_info = langgraph_agent.get_info()
            print(f"   Provider: {langgraph_agent_info['llm_provider']}")
            print(f"   Model: {langgraph_agent_info['llm_model']}")
            print(f"   Temperature: {langgraph_agent_info['llm_temperature']}")
            print(f"   Tools available: {langgraph_agent_info['tools_count']}")
            print(f"   Tool names: {', '.join(langgraph_agent_info['tool_names'])}")
            print()
        except Exception as e:
            print(f"❌ Failed to get agent stats: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            traceback.print_exc()
            return

        print("📝 Testing streaming functionality")
        print("-" * 30)
        user_message = "Get me a random number and the weather in Tokyo"
        print(f"User: {user_message}")
        print("Agent: ", end="")

        async for chunk in langgraph_agent.stream(user_message):
            if chunk:
                print(chunk, end="", flush=True)

        print("\n")

        print("✅ Tests completed successfully!")

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
