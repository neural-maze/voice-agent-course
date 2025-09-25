#!/usr/bin/env python3
"""
Minimal Kokoro TTS example using our RealtimeTTS adapter.

This demonstrates multilingual TTS with different voices using clean architecture.

Usage:
    uv run python scripts/kokoro_minimal_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent_course.infrastructure.audio.realtime_tts_adapter import RealtimeTTSAdapter

# Voice samples for different languages
VOICE_SAMPLES = {
    "american": {
        "voice_id": "af_heart",
        "text": "Hello, this is an American voice test.",
        "language": "English (American)",
    },
    "british": {
        "voice_id": "bf_emma",
        "text": "Good day, mate! This is a British voice test.",
        "language": "English (British)",
    },
}


async def test_kokoro_voices():
    """Test different Kokoro voices with our adapter"""
    print("🎵 Kokoro TTS Multilingual Test")
    print("=" * 50)
    print("Testing different voices with our TTS adapter...")
    print("=" * 50)

    try:
        # Initialize adapter with Kokoro engine
        adapter = RealtimeTTSAdapter(engine_type="kokoro")
        print("✅ Kokoro TTS adapter initialized!")

        # Test each voice
        for voice_name, config in VOICE_SAMPLES.items():
            print(f"\n🔊 Testing {config['language']} ({voice_name})")
            print(f"   Voice ID: {config['voice_id']}")
            print(f"   Text: '{config['text']}'")

            # Set voice
            voice_set = adapter.set_voice(config["voice_id"])
            if not voice_set:
                print(f"   ⚠️  Failed to set voice {config['voice_id']}")
                continue

            # Set a slower speed for better clarity
            adapter.set_speed(0.8)

            # Synthesize and play
            result = await adapter.synthesize_and_play(config["text"])

            if result["success"]:
                print(f"   ✅ Success! Duration: {result['duration']:.2f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")

            # Short pause between voices
            await asyncio.sleep(1.0)

        # Show final stats
        stats = adapter.get_stats()
        print("\n📊 Final Statistics:")
        print(f"   Total characters processed: {stats['total_characters_processed']}")
        print(f"   Engine type: {stats['engine_type']}")
        print(f"   First audio time: {stats['first_audio_byte_time']}")

        print("\n🎉 Multilingual test completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed: uv add 'RealtimeTTS[all]'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test runner"""
    print("🚀 Starting Kokoro TTS tests...")

    # Test different voices
    await test_kokoro_voices()

    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("You should have heard different voices speaking in multiple languages.")
    print("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
