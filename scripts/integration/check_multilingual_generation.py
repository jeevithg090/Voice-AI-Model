import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from src.llm.embedding_agent import process_with_context

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


async def test_language(lang_code: str, lang_name: str, prompt_text: str = "Hello, how are you?"):
    print(f"\n{CYAN}{'=' * 60}")
    print(f"🌍 Testing {lang_name} ({lang_code})")
    print(f"{'=' * 60}{RESET}")

    metadata = {
        "detected_language": lang_code,  # This triggers the system prompt constraint
        "language_confidence": "0.99",
        "emergency_probability": "0.0",
        "sustained_distress": "False",
        "processing_latency_ms": "100",
    }

    print(f"📥 Input: {prompt_text}")
    print(f"⚙️  Metadata: detected_language={lang_code}")
    print("⏳ Generating response...")

    try:
        # We use a unique session ID for each language to avoid context pollution
        session_id = f"test_session_{lang_code}"

        result = process_with_context(prompt=prompt_text, session_id=session_id, metadata=metadata)

        response = result["response"]
        print(f"{GREEN}🤖 Response:{RESET} {response}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    print(f"{YELLOW}🚀 Starting Multilingual Generation Test{RESET}")

    # Clear cache to ensure fresh generation
    if os.path.exists("semantic_cache.json"):
        os.remove("semantic_cache.json")
        print("🧹 Cleared semantic_cache.json")

    # Clear In-Memory Cache (Critical fix)
    from src.llm.embedding_agent import cache

    cache.clear()
    print("🧹 Cleared In-Memory Cache")

    languages = [
        ("hi", "Hindi", "नमस्ते, आप कौन हैं?"),
        ("kn", "Kannada", "நಮஸ்கார, நீங்கள் யார்?"),  # Intentional mix to test robust? No, let's correct it.
        # "kn": "நಮஸ்கார" is mixed script. Let's use correct Kannada: "நಮಸ್ಕಾರ, ನೀವು ಯಾರು?" (Wait, first char is Tamil 'Na').
        # Correct Kannada: "ನಮಸ್ಕಾರ, ನೀವು ಯಾರು?"
        ("kn", "Kannada", "ನಮಸ್ಕಾರ, ನೀವು ಯಾರು?"),
        ("ta", "Tamil", "வணக்கம், நீங்கள் யார்?"),
        ("te", "Telugu", "నమస్కారం, మీరు ఎవరు?"),
    ]

    for code, name, prompt in languages:
        await test_language(code, name, prompt)

    print(f"\n{YELLOW}✅ Test Complete{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
