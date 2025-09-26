#!/usr/bin/env python3
"""Simple example showing how to use the LangGraph agent."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent


async def main():
    """Main example function."""
    print("🚀 LangGraph Agent Example")
    print("=" * 50)

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY environment variable not set!")
        print("   Please set it with: export GROQ_API_KEY=your_api_key_here")
        print("   You can get a free API key from: https://console.groq.com/")
        return

    try:
        # Create the agent
        print("🔧 Creating LangGraph agent...")
        agent = LangGraphAgent(model="meta-llama/llama-4-maverick-17b-128e-instruct")

        # Show agent stats
        stats = agent.get_stats()
        print(f"✅ Agent created successfully!")
        print(f"   Model: {stats['model']}")
        print(f"   Tools available: {stats['tools_count']}")
        print(f"   Tool names: {', '.join(stats['tool_names'])}")
        print()

        # Example 1: Simple invoke
        print("📝 Example 1: Simple invoke")
        print("-" * 30)
        user_message = "what is the weather in San Francisco?"
        print(f"User: {user_message}")

        response = await agent.invoke(user_message)
        print(f"Agent: {response}")
        print()

        # Example 2: Streaming response
        print("📝 Example 2: Streaming response")
        print("-" * 30)
        user_message = "get me a random number and calculate the 5th fibonacci number"
        print(f"User: {user_message}")
        print("Agent: ", end="")

        async for chunk in agent.stream(user_message):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")

        print("✅ Examples completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
