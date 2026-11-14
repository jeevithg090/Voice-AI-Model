import sys
import os
import asyncio

# Add current directory to path
sys.path.append(os.getcwd())

from src.llm.embedding_agent import process_with_context

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Prompt Variants
PROMPTS = {
    "Original": """You are a **real-time multilingual voice assistant** for an Indian voice support system.

You **must** operate in the user’s detected language with **native fluency**.

---

## CORE RULES (STRICT)

1. **Language**
    - Always respond in `detected_language` which is provided in the context below.
    - If `language_confidence < 0.5`, politely ask which language the user prefers.
    - Never mix languages unless the user explicitly code-switches.
2. **Grounding**
    - Use **only the provided context/data**.
    - Never fabricate, assume, or hallucinate information.
    - If the answer is not in the data, say you are unsure and offer human handover.
3. **Voice-first Responses**
    - Responses must be **spoken-word friendly**.
    - Keep answers **concise** (1–3 short sentences, under ~30 words when possible).
    - Avoid bullet points or technical formatting.

---

## CONTEXT FROM AUDIO LAYER
You have access to the following real-time parameters:

- **Detected Language**: {detected_language} (Confidence: {language_confidence})
- **Emergency Probability**: {emergency_probability}
- **Sustained Distress**: {sustained_distress}
- **Audio Processing Latency**: {processing_latency_ms} ms
""",

    "Simplified": """You are a helpful Indian voice assistant.
Your goal is to answer the user's question concisely and accurately.

CRITICAL INSTRUCTION:
You MUST respond IN THE SAME LANGUAGE as the user's input.
If the user speaks Hindi, reply in Hindi.
If the user speaks Tamil, reply in Tamil.
If the user speaks Kannada, reply in Kannada.
If the user speaks Telugu, reply in Telugu.

Context:
- Detected Language: {detected_language}
- User Distress: {sustained_distress}
""",

    "Minimal": """Respond in {detected_language} language only. Be concise.""",

    "LanguageSpecific": """You are a helpful assistant.
You strictly speak {language_name}.
User Input Language: {language_name} ({detected_language}).
Reply ONLY in {language_name} script.
"""
}


async def test_prompt(name: str, system_prompt: str, lang_code: str, user_input: str):
    print(f"\n{CYAN}🧪 Testing Prompt Variant: {name} [{lang_code}]{RESET}")
    print(f"📥 Input: {user_input}")
    
    # Simple map for test
    lang_map = {"kn": "Kannada", "te": "Telugu", "ta": "Tamil", "hi": "Hindi"}
    
    metadata = {
        "detected_language": lang_code,
        "language_name": lang_map.get(lang_code, "English"), # Added for new prompt
        "language_confidence": "0.99",
        "emergency_probability": "0.0",
        "sustained_distress": "False",
        "processing_latency_ms": "100"
    }

    try:
        # Clear previous session context for fair test
        session_id = f"test_prompt_{name}_{lang_code}"
        
        # Inject metadata into prompt manually here since we are passing custom system_prompt
        # But wait, process_with_context does the formatting if metadata is passed.
        # So we pass the UNFORMATTED string.
        
        result = process_with_context(
            prompt=user_input,
            session_id=session_id,
            system_prompt=system_prompt, # Passing the template
            metadata=metadata
        )
        
        print(f"{GREEN}🤖 Response:{RESET} {result['response']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    print(f"{YELLOW}🚀 Starting Prompt Engineering Test{RESET}")
    
    # We focus on the problematic languages: Kannada and Telugu
    test_cases = [
        ("kn", "ನಮಸ್ಕಾರ, ನೀವು ಯಾರು?"), # Kannada: Hello, who are you?
        ("te", "నమస్కారం, మీరు ఎవరు?"), # Telugu: Hello, who are you?
        ("ta", "வணக்கம், நீங்கள் யார்?") # Tamil
    ]
    
    # 1. Clear Semantic Cache File
    if os.path.exists("semantic_cache.json"):
        os.remove("semantic_cache.json")
        print("🧹 Deleted semantic_cache.json")

    # 2. Flush Redis
    import redis
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.flushdb()
        print("🧹 Flushed Redis Database")
    except Exception as e:
        print(f"⚠️ Could not flush Redis: {e}")
    
    # 3. Clear In-Memory Semantic Cache (CRITICAL: It loads at import time)
    from src.llm.embedding_agent import cache
    cache.clear() # Use proper API
    # cache.save_cache() # Not needed, clear() handles file removal
    print("🧹 Cleared In-Memory Semantic Cache")
    
    for lang_code, input_text in test_cases:
        # Map code to name
        lang_map = {"kn": "Kannada", "te": "Telugu", "ta": "Tamil", "hi": "Hindi"}
        lang_name = lang_map.get(lang_code, "English")
        
        # Manually format the prompt for the test (simulating dynamic injection)
        prompt_template = PROMPTS["LanguageSpecific"]
        # We need to pre-format the language name since process_with_context only handles the standard keys
        # Actually, process_with_context formatting is based on 'metadata'.
        # So we can pass 'language_name' in metadata!
        
        # Only test LanguageSpecific
        # We need to pass the template to process_with_context, which will format it.
        await test_prompt("LanguageSpecific", prompt_template, lang_code, input_text)

if __name__ == "__main__":
    if os.path.exists("semantic_cache.json"):
        os.remove("semantic_cache.json")
    asyncio.run(main())
