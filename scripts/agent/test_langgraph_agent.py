#!/usr/bin/env python3
"""Simple test script for LangGraph agent with streaming output."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.domain.agents.langgraph_agent import LangGraphAgent


async def test_agent_invoke():
    """Test the agent with a simple invoke call."""
    print("🤖 Testing LangGraph Agent - Invoke Mode")
    print("=" * 50)

    # Check if API key is available before creating agent
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Cannot test agent - GROQ_API_KEY not set")
        return

    try:
        agent = LangGraphAgent()
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return

    # Test messages
    test_messages = [
        "what is the weather in sf",
        "get me a random number",
        "what's the weather in New York?",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {message}")
        print("-" * 30)

        try:
            response = await agent.invoke(message)
            print(f"🔄 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

        print()


async def test_agent_streaming():
    """Test the agent with streaming output."""
    print("\n🌊 Testing LangGraph Agent - Streaming Mode")
    print("=" * 50)

    # Check if API key is available before creating agent
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Cannot test agent - GROQ_API_KEY not set")
        return

    try:
        agent = LangGraphAgent()
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return

    # Test streaming with a single message
    test_message = "what is the weather in San Francisco and give me a random number?"

    print(f"\n📝 Streaming Test: {test_message}")
    print("-" * 30)
    print("🔄 Streaming Response:")

    try:
        response_parts = []
        async for chunk in agent.stream(test_message):
            if chunk.strip():  # Only print non-empty chunks
                print(chunk, end="", flush=True)
                response_parts.append(chunk)

        print(f"\n\n📊 Total response length: {len(''.join(response_parts))} characters")

    except Exception as e:
        print(f"❌ Streaming Error: {e}")


async def test_agent_stats():
    """Test agent statistics."""
    print("\n📈 Testing Agent Statistics")
    print("=" * 50)

    # Check if API key is available before creating agent
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Cannot test agent - GROQ_API_KEY not set")
        return

    try:
        agent = LangGraphAgent()
        stats = agent.get_stats()

        print("🔧 Agent Configuration:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")


async def main():
    """Main test function."""
    print("🚀 LangGraph Agent Test Suite")
    print("=" * 50)

    # Check for GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY environment variable not set!")
        print("   Please set it with: export GROQ_API_KEY=your_api_key_here")
        print()

    # Run tests
    await test_agent_stats()
    await test_agent_invoke()
    await test_agent_streaming()

    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
