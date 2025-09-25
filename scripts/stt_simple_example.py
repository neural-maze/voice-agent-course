#!/usr/bin/env python3
"""
Simple RealtimeSTT test using the correct API pattern.

This follows the official RealtimeSTT example from GitHub.

Usage:
    uv run python scripts/stt_simple_test.py

Requirements:
    - Working microphone
    - RealtimeSTT library: uv add RealtimeSTT
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent_course.infrastructure.audio.realtime_stt_adapter import RealtimeSTTAdapter, STTModel


async def test_basic_transcription():
    """Test basic transcription using our adapter"""
    print("🎤 Basic STT Test (RealtimeSTT Pattern)")
    print("=" * 50)
    print("Say something and it will be transcribed when you stop speaking...")

    try:
        print("⏳ Initializing STT adapter (this may take 10-30 seconds)...")

        # Initialize adapter with minimal settings (no callbacks)
        adapter = RealtimeSTTAdapter(
            model=STTModel.TINY_EN,
            language="en",
            enable_realtime=False,  # Disable callbacks completely
            sensitivity_config={
                "silero_sensitivity": 0.01,
            },
        )

        print("✅ STT adapter initialized and ready!")
        print("💡 Now the system is listening for voice activity...")

        # Test multiple transcriptions
        for i in range(3):
            print(f"\n[Turn {i + 1}] 🎤 Speak now...")

            result = await adapter.get_text_blocking()

            if result["success"]:
                text = result["text"]
                print(f"✅ You said: '{text}'")
                print(f"   Duration: {result['duration']:.2f}s")

                # Check for exit
                if text.lower().strip() in ["exit", "quit", "stop"]:
                    print("👋 Goodbye!")
                    break
            else:
                print(f"❌ Error: {result['error']}")

        # Show stats
        stats = adapter.get_stats()
        print("\n📊 Session Statistics:")
        print(f"   Total transcriptions: {stats['total_transcriptions']}")
        print(f"   Model: {stats['model']}")
        print(f"   Engine: {stats['engine_type']}")

        adapter.shutdown()

    except Exception as e:
        print(f"❌ Error in basic test: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test runner"""
    print("🚀 RealtimeSTT Simple Tests")
    print("=" * 60)
    print("Testing the corrected STT adapter with proper RealtimeSTT API")
    print("Make sure your microphone is working!")

    try:
        await test_basic_transcription()

        print("\n🎉 All tests completed!")
        print("Your STT adapter is now working correctly with RealtimeSTT!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
