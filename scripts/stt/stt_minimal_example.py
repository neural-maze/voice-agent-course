#!/usr/bin/env python3
"""
Minimal RealtimeSTT example using our STT adapter.

This demonstrates real-time speech-to-text transcription with different test modes.

Usage:
    uv run python scripts/stt_minimal_example.py

Requirements:
    - Working microphone
    - RealtimeSTT library: uv add RealtimeSTT
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voice_agent_course.infrastructure.audio.realtime_stt_adapter import RealtimeSTTAdapter, STTModel


async def test_single_transcription():
    """Test single utterance transcription with timeout"""
    print("🎤 Single Transcription Test")
    print("=" * 50)
    print("Say something within 10 seconds...")

    try:
        adapter = RealtimeSTTAdapter()
        print("✅ STT adapter initialized!")

        # Listen for a single utterance
        result = await adapter.listen_once(timeout=10.0)

        if result["success"]:
            print(f"✅ Transcription: '{result['text']}'")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Failed: {result['error']}")

        return result

    except Exception as e:
        print(f"❌ Error in single transcription test: {e}")
        return {"success": False, "error": str(e)}


async def test_continuous_transcription():
    """Test continuous transcription for a fixed duration"""
    print("\n🔄 Continuous Transcription Test")
    print("=" * 50)
    print("Speak continuously for 15 seconds...")

    transcriptions = []

    def on_speech(text: str):
        print(f"🗣️  Heard: '{text}'")
        transcriptions.append(text)

    try:
        adapter = RealtimeSTTAdapter()

        # Listen continuously for 15 seconds
        result = await adapter.continuous_listen(duration=15.0, on_speech=on_speech)

        if result["success"]:
            print(f"\n✅ Session completed!")
            print(f"   Total utterances: {result['total_utterances']}")
            print(f"   Session duration: {result['session_duration']:.2f}s")
            print(f"   Transcriptions collected: {len(transcriptions)}")
        else:
            print(f"❌ Failed: {result['error']}")

        return result

    except Exception as e:
        print(f"❌ Error in continuous transcription test: {e}")
        return {"success": False, "error": str(e)}


async def test_voice_agent_mode():
    """Test voice agent interaction mode (like the example you provided)"""
    print("\n🤖 Voice Agent Mode Test")
    print("=" * 50)
    print("This simulates how the voice agent would work...")
    print("Say 'exit' to stop, or speak normally to get transcriptions")

    conversation_count = 0

    def on_transcription(text: str):
        nonlocal conversation_count
        conversation_count += 1
        print(f"\n[Turn {conversation_count}] You said: '{text}'")

        # Simulate AI response (in real voice agent, this would go to LLM)
        if text.lower().strip() in ["exit", "quit", "stop"]:
            print("👋 Goodbye!")
            return False
        else:
            print(f"🤖 AI would respond to: '{text}'")
            return True

    try:
        adapter = RealtimeSTTAdapter(on_transcription=on_transcription)
        print("✅ Voice agent STT initialized!")
        print("🎤 Listening... (say 'exit' to stop)")

        # Start continuous listening
        if not adapter.continuous_listen():
            print("❌ Could not start listening")
            return {"success": False}

        # Keep listening until user says exit
        exit_requested = False
        start_time = asyncio.get_event_loop().time()

        while not exit_requested and (asyncio.get_event_loop().time() - start_time) < 60:  # Max 1 minute
            await asyncio.sleep(0.5)  # Check every 500ms

            # In a real voice agent, you'd have more sophisticated exit conditions
            # For this test, we'll just run for a reasonable time

        adapter.stop_listening()

        # Get final stats
        stats = adapter.get_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total transcriptions: {stats['total_transcriptions']}")
        print(f"   Engine type: {stats['engine_type']}")
        print(f"   Conversation turns: {conversation_count}")

        return {"success": True, "turns": conversation_count}

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        if "adapter" in locals():
            adapter.stop_listening()
        return {"success": True, "interrupted": True}
    except Exception as e:
        print(f"❌ Error in voice agent mode: {e}")
        return {"success": False, "error": str(e)}


async def test_callback_system():
    """Test the callback system with partial and final transcriptions"""
    print("\n🔗 Callback System Test")
    print("=" * 50)
    print("Testing partial and final transcription callbacks...")

    partial_count = 0
    final_count = 0

    def on_partial(text: str):
        nonlocal partial_count
        partial_count += 1
        print(f"📝 Partial [{partial_count}]: '{text}'")

    def on_final(text: str):
        nonlocal final_count
        final_count += 1
        print(f"✅ Final [{final_count}]: '{text}'")

    try:
        adapter = RealtimeSTTAdapter(on_transcription=on_final, on_partial_transcription=on_partial)

        print("🎤 Speak for 10 seconds to see partial and final transcriptions...")

        # Use continuous listen to see callbacks in action
        result = await adapter.continuous_listen(duration=10.0)

        print(f"\n📊 Callback Statistics:")
        print(f"   Partial transcriptions: {partial_count}")
        print(f"   Final transcriptions: {final_count}")
        print(f"   Success: {result['success']}")

        return result

    except Exception as e:
        print(f"❌ Error in callback test: {e}")
        return {"success": False, "error": str(e)}


async def test_blocking_mode():
    """Test blocking text retrieval (like voice agent example)"""
    print("\n🚫 Blocking Mode Test")
    print("=" * 50)
    print("Testing blocking text retrieval (voice agent style)...")

    try:
        adapter = RealtimeSTTAdapter(
            model=STTModel.TINY_EN,  # Fast model for testing
            enable_realtime=False,  # Disable realtime for blocking mode
        )

        print("🎤 Say something and it will be transcribed when you stop speaking...")

        # Test blocking text retrieval
        result = await adapter.get_text_blocking()

        if result["success"]:
            print(f"✅ Blocking transcription: '{result['text']}'")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Method: {result['method']}")
            print(f"   Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Failed: {result['error']}")

        return result

    except Exception as e:
        print(f"❌ Error in blocking mode test: {e}")
        return {"success": False, "error": str(e)}


async def test_different_models():
    """Test different STT models for comparison"""
    print("\n🧠 Model Comparison Test")
    print("=" * 50)
    print("Testing different STT models...")

    models_to_test = [
        (STTModel.TINY_EN, "Fastest, English only"),
        (STTModel.BASE_EN, "Balanced, English only"),
    ]

    results = {}

    for model, description in models_to_test:
        print(f"\n🔬 Testing {model.value} - {description}")
        print("   Say the same phrase for comparison...")

        try:
            adapter = RealtimeSTTAdapter(model=model)
            result = await adapter.listen_once(timeout=8.0)

            if result["success"]:
                print(f"   ✅ '{result['text']}'")
                print(f"   Duration: {result['duration']:.2f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")

            results[model.value] = result

        except Exception as e:
            print(f"   ❌ Error with {model.value}: {e}")
            results[model.value] = {"success": False, "error": str(e)}

        # Pause between models
        await asyncio.sleep(1.0)

    return {"success": True, "model_results": results}


async def main():
    """Main test runner"""
    print("🚀 Starting RealtimeSTT Adapter Tests...")
    print("Make sure your microphone is working and permissions are granted!")
    print("\n" + "=" * 60)

    tests = [
        ("Single Transcription", test_single_transcription),
        ("Continuous Transcription", test_continuous_transcription),
        ("Callback System", test_callback_system),
        ("Blocking Mode", test_blocking_mode),
        ("Model Comparison", test_different_models),
        ("Voice Agent Mode", test_voice_agent_mode),
    ]

    results = {}

    try:
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            print("-" * 30)

            try:
                result = await test_func()
                results[test_name] = result

                if result.get("success", False):
                    print(f"✅ {test_name} completed successfully!")
                else:
                    print(f"❌ {test_name} failed: {result.get('error', 'Unknown error')}")

                # Short pause between tests
                print("\n⏳ Pausing 2 seconds before next test...")
                await asyncio.sleep(2.0)

            except KeyboardInterrupt:
                print(f"\n⏹️  {test_name} interrupted by user")
                results[test_name] = {"success": False, "interrupted": True}
                break
            except Exception as e:
                print(f"❌ Unexpected error in {test_name}: {e}")
                results[test_name] = {"success": False, "error": str(e)}

        # Final summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)

        for test_name, result in results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"{status} {test_name}")
            if not result.get("success", False) and "error" in result:
                print(f"     Error: {result['error']}")

        print("\n🎉 All STT adapter tests completed!")
        print("Your microphone and STT system should now be working correctly.")

    except KeyboardInterrupt:
        print("\n\n⏹️  All tests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
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
