"""
Embedding Agent for Layer 2 - Voice AI Pipeline
Generates embeddings, performs semantic cache lookup, and manages conversation context.
Now with natural fillers and back-channeling support!
"""

from .ollama_client import embed_text, chat
from .semantic_cache import SemanticCache
from .redis_context import RedisContextManager
from .filler_manager import (
    FillerBackchannelManager, 
    get_enhanced_system_prompt,
    FillerType
)
from src.config import settings

# Language Code Map
LANGUAGE_MAP = {
    "hi": "Hindi",
    "kn": "Kannada", 
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "en": "English"
}

# Initialize the semantic cache, Redis context, and filler manager
cache = SemanticCache()
context_manager = RedisContextManager()
filler_manager = FillerBackchannelManager()

# Default system prompt for medical assistant
# Default system prompt for medical assistant
DEFAULT_SYSTEM_PROMPT = """You are a **real-time multilingual voice assistant** for an Indian voice support system.

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

---

## PRIORITY HANDLING (ALWAYS CHECK FIRST)

### Emergency & Distress

- If `sustained_distress == "True"` or `emergency_probability > 0.7`:
    - **IMMEDIATELY** acknowledge distress in the user’s language.
    - Use a calm, reassuring tone.
    - Encourage contacting emergency services:
        - Police: **100**
        - Ambulance: **102 / 108**
        - Women’s helpline: **181**
    - Example intent: reassure, ensure safety, guide next steps.

User safety **overrides all other goals**.

---

## RESPONSE FLOW (INTERNAL)

1. Check emergency signals from the context above.
2. Confirm language and confidence.
3. Ground response in provided data.
4. Respond clearly and naturally.
5. If uncertain, escalate gracefully.

---

## CONVERSATION STYLE

- Sound like a **helpful human**, not a chatbot.
- Acknowledge understanding when appropriate:
    - e.g., “You’re asking about…”
- Ask a **clarifying question** if the query is ambiguous.
- End responses with a natural turn-ending cue.

---

## FAILURE & HANDOVER

If:
- the data is insufficient, or
- the question is outside scope, or
- the user needs human assistance,

→ Say so clearly and politely, and offer to connect to a human.
"""


def embedding_agent(prompt: str, similarity_threshold: float = 0.85):
    """
    Generates embedding for the prompt and performs semantic cache lookup.
    
    Args:
        prompt: Input text from STT layer
        similarity_threshold: Minimum similarity score for cache hit (0.0-1.0)
        
    Returns:
        dict with either cached response or embedding for LLM processing
    """
    
    # 1. Generate embedding via local Ollama
    embedding = embed_text(prompt)
    
    # 2. Perform semantic lookup from cache
    cached = cache.find_similar(embedding, prompt=prompt, threshold=similarity_threshold)
    
    if cached:
        return {
            "cached": True,
            "cachedFrom": "semantic-cache",
            "response": cached["response"],
            "similarity": cached["similarity"],
            "originalPrompt": cached["prompt"]
        }
    
    # 3. No semantic match - return embedding for LLM
    return {
        "cached": False,
        "embedding": embedding,
        "prompt": prompt
    }


def process_with_context(
    prompt: str,
    session_id: str = None,
    system_prompt: str = None,
    similarity_threshold: float = 0.85,
    metadata: dict = None,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
):
    """
    Full pipeline with Redis context: Embedding → Cache → Context → LLM → Store
    
    Args:
        prompt: Input text from STT layer
        session_id: Redis session ID
        system_prompt: System instructions
        similarity_threshold: Cache threshold
        metadata: Context metadata
        stream: If True, returns a generator for LLM response
        temperature: LLM sampling temperature
        max_tokens: Max tokens to generate
        
    Returns:
        If stream=False: dict with full response
        If stream=True: dict with 'response_generator' and metadata
    """
    from .ollama_client import generate_response, chat, chat_stream
    
    is_custom_system_prompt = system_prompt is not None

    # Use default system prompt if not provided
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Inject metadata into default system prompt when caller didn't provide a custom prompt.
    if metadata and not is_custom_system_prompt:
        try:
            defaults = {
                "detected_language": "Unknown",
                "language_confidence": "N/A",
                "emergency_probability": "0.0",
                "sustained_distress": "False",
                "processing_latency_ms": "0"
            }
            context_data = {**defaults, **metadata}
            system_prompt = system_prompt.format(**context_data)
        except Exception as e:
            print(f"[WARN] Failed to format system prompt with metadata: {e}")
            system_prompt += f"\n\n[CONTEXT METADATA]: {metadata}"
            
        # --- DYNAMIC INJECTION: Force Language Instruction ---
        lang_code = metadata.get("detected_language", "en")
        lang_name = LANGUAGE_MAP.get(lang_code, "English")
        
        if lang_code in ["hi", "kn", "ta", "te", "ml", "bn", "gu", "mr"]:
            print(f"ℹ️  Switching to Minimal System Prompt for {lang_name}")
            system_prompt = (
                f"You are a helpful Indian voice assistant speaking {lang_name}.\n"
                f"User Input Language: {lang_name} ({lang_code}).\n"
                f"CRITICAL: Reply ONLY in {lang_name} script.\n"
                f"Keep answers short and helpful.\n"
                f"Do not translate to English.\n"
            )
            if metadata.get("sustained_distress") == "True":
                 system_prompt += "\nUSER IS IN DISTRESS. HELP THEM."
    
    # Fast-fail context path when Redis is unavailable.
    redis_available = False
    if session_id == "temp_session":
        redis_available = False
    else:
        try:
            redis_available = context_manager.ping()
        except Exception:
            redis_available = False

    # Create/Get session logic
    is_new_session = False
    if session_id is None:
        if redis_available:
            try:
                session_id = context_manager.create_session()
                is_new_session = True
            except Exception:
                session_id = "temp_session"
        else:
            session_id = "temp_session"
            
    if session_id != "temp_session" and redis_available:
        try:
            session = context_manager.get_session(session_id)
            if session is None:
                session_id = context_manager.create_session()
                is_new_session = True
        except Exception:
            session_id = "temp_session"
    elif session_id != "temp_session" and not redis_available:
        session_id = "temp_session"
    
    # Step 2: Check semantic cache
    try:
        result = embedding_agent(prompt, similarity_threshold)
    except Exception as e:
        print(f"[WARN] Embedding agent failed: {e}")
        result = {"cached": False, "embedding": [], "prompt": prompt}
    
    if result["cached"]:
        if redis_available and session_id != "temp_session":
            try:
                context_manager.add_turn(session_id, "user", prompt)
                context_manager.add_turn(session_id, "assistant", result["response"])
            except Exception:
                pass
        
        # If streaming is requested but we have cache, we simulate a stream
        if stream:
            def cached_stream_gen():
                yield result["response"]
            
            return {
                "response_generator": cached_stream_gen(),
                "session_id": session_id,
                "source": "semantic-cache",
                "similarity": result["similarity"],
                "new_session": is_new_session,
                "cached": True,
                "stream": True
            }

        return {
            "response": result["response"],
            "session_id": session_id,
            "source": "semantic-cache",
            "similarity": result["similarity"],
            "new_session": is_new_session
        }
    
    # Step 3/4: Conversation context (Redis-backed) or stateless fallback.
    if redis_available and session_id != "temp_session":
        try:
            context_manager.add_turn(session_id, "user", prompt)
            messages = context_manager.get_context_for_llm(session_id, system_prompt)
            if not messages:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        except Exception as e:
            print(f"[WARN] Redis context unavailable for this turn: {e}")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    # Step 5: Call LLM
    if stream:
        # Wrapper to handle side-effects (saving to cache/context) after stream ends
        def stream_wrapper():
            full_response = ""
            stream_gen = chat_stream(messages=messages, max_tokens=max_tokens, temperature=temperature)
            
            for token in stream_gen:
                full_response += token
                yield token
            
            # Post-stream processing: Save to Context & Cache
            try:
                if redis_available and session_id != "temp_session":
                    context_manager.add_turn(session_id, "assistant", full_response)
                if result.get("embedding"):
                    cache.add_entry(prompt, result["embedding"], full_response)
            except Exception as e:
                print(f"[ERROR] Failed to save post-stream context: {e}")
                
        # Handle Filler (Same logic, simple check)
        filler_manager.reset_turn()
        filler_audio, used_filler = filler_manager.get_response_prefix(prompt, 1500, None)
        
        return {
            "response_generator": stream_wrapper(),
            "session_id": session_id,
            "source": "llm",
            "model": settings.LLM_MODEL,
            "new_session": is_new_session,
            "filler_audio": filler_audio if used_filler else None,
            "language": filler_manager._session_language,
            "stream": True
        }
        
    else:
        # Non-streaming fallback
        filler_manager.reset_turn()
        filler_audio, used_filler = filler_manager.get_response_prefix(prompt, 1500, None)
        
        response = chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        
        if redis_available and session_id != "temp_session":
            try:
                context_manager.add_turn(session_id, "assistant", response)
            except Exception:
                pass
        
        if result.get("embedding"):
            cache.add_entry(prompt, result["embedding"], response)
        
        return {
            "response": response,
            "session_id": session_id,
            "source": "llm",
            "model": settings.LLM_MODEL,
            "new_session": is_new_session,
            "context_turns": len(messages),
            "filler_audio": filler_audio if used_filler else None,
            "language": filler_manager._session_language
        }


def process_query(prompt: str, system_prompt: str = None, similarity_threshold: float = 0.85):
    """
    Single-turn pipeline (no context): Embedding → Cache → LLM → Store
    
    Args:
        prompt: Input text from STT layer
        system_prompt: Optional system instructions
        similarity_threshold: Minimum similarity for cache hit
        
    Returns:
        dict with response and metadata
    """
    from .ollama_client import generate_response
    
    # Step 1: Check semantic cache
    result = embedding_agent(prompt, similarity_threshold)
    
    if result["cached"]:
        return {
            "response": result["response"],
            "source": "semantic-cache",
            "similarity": result["similarity"],
            "latency_saved": True
        }
    
    # Step 2: No cache hit - call LLM
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    response = generate_response(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=256,
        temperature=0.7
    )
    
    # Cache the response
    if result.get("embedding"):
        cache.add_entry(
            prompt=prompt,
            embedding=result["embedding"],
            response=response
        )
    
    return {
        "response": response,
        "source": "llm",
        "model": settings.LLM_MODEL,
        "cached_for_future": True
    }


def end_session(session_id: str) -> bool:
    """End a conversation session."""
    return context_manager.clear_session(session_id)


def get_session_history(session_id: str):
    """Get full conversation history for a session."""
    return context_manager.get_context(session_id)
