#!/usr/bin/env python3
"""
TTS Usage Examples - Demonstrates both streaming and blocking modes.

This shows how to use the enhanced RealtimeTTS adapter for:
1. Real-time streaming TTS (like in your full implementation)
2. Blocking TTS (for complete utterances)

Usage:
    uv run python scripts/tts_usage_examples.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter, TTSMode


async def demo_blocking_mode():
    """Demonstrate blocking TTS mode - waits for each utterance to complete"""
    print("🎵 Blocking Mode Demo")
    print("=" * 50)
    print("Each sentence will play completely before moving to the next...")
    print()

    adapter = RealtimeTTSAdapter(engine_type="kokoro")
    adapter.set_voice("af_heart")
    adapter.set_speed(0.9)

    sentences = [
        "This is the first sentence in blocking mode.",
        "Now I'm speaking the second sentence.",
        "And finally, here's the third sentence.",
    ]

    for i, sentence in enumerate(sentences, 1):
        print(f"🔊 Playing sentence {i}: '{sentence}'")

        result = await adapter.synthesize_and_play(sentence, mode=TTSMode.BLOCKING)

        if result["success"]:
            print(f"   ✅ Completed in {result['duration']:.2f}s")
        else:
            print(f"   ❌ Failed: {result['error']}")

        print("   Waiting for audio to finish before continuing...")
        await asyncio.sleep(0.5)  # Small pause between sentences

    print("\n✅ Blocking mode demo completed!")
    return adapter


async def demo_streaming_mode():
    """Demonstrate streaming TTS mode - like your real-time conversation system"""
    print("\n🎵 Streaming Mode Demo")
    print("=" * 50)
    print("Simulating real-time conversation with streaming chunks...")
    print()

    adapter = RealtimeTTSAdapter(engine_type="kokoro")
    adapter.set_voice("bf_emma")
    adapter.set_speed(0.9)

    # Simulate conversation chunks arriving in real-time
    conversation_chunks = [
        "Hello there! ",
        "I'm speaking ",
        "in streaming mode, ",
        "which means ",
        "I can start playing ",
        "before all the text ",
        "has been generated. ",
        "This is perfect ",
        "for real-time ",
        "conversations!",
    ]

    # Set timing for metrics (like in your full implementation)
    adapter.set_transcription_received_time()

    print("🔊 Starting streaming conversation...")

    for i, chunk in enumerate(conversation_chunks):
        print(f"   Chunk {i + 1}: '{chunk.strip()}'")

        # Feed text and play immediately (non-blocking)
        if adapter.feed_text(chunk):
            adapter.play_stream_async()

        # Simulate delay between chunks (like LLM generation)
        await asyncio.sleep(0.1)

    # Wait a bit for the last chunks to finish playing
    print("   Waiting for streaming audio to complete...")
    await asyncio.sleep(10.0)

    print("✅ Streaming mode demo completed!")
    return adapter


def show_detailed_stats(adapter: RealtimeTTSAdapter, title: str):
    """Show detailed statistics from the adapter"""
    print(f"\n📊 {title} Statistics:")
    print("-" * 40)

    stats = adapter.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


async def main():
    """Main demo runner"""
    print("🚀 TTS Usage Examples - Streaming vs Blocking")
    print("=" * 60)
    print("This demonstrates the enhanced adapter capabilities")
    print("for both real-time streaming and complete utterances.")
    print("=" * 60)

    try:
        # Demo 1: Blocking mode
        adapter1 = await demo_blocking_mode()
        show_detailed_stats(adapter1, "Blocking Mode")

        # Demo 2: Streaming mode
        adapter2 = await demo_streaming_mode()
        show_detailed_stats(adapter2, "Streaming Mode")

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("You should have heard different TTS patterns:")
        print("  - Blocking: Complete sentences with pauses")
        print("  - Streaming: Continuous speech from chunks")
        print("  - Mixed: Both patterns combined")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed: uv add 'RealtimeTTS[all]'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
