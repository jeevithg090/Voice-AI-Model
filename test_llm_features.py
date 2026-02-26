import sys
import os
import time
import json
import asyncio
from typing import Dict, Any, List

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def log_section(title):
    print(f"\n{CYAN}{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}{RESET}")

def log_success(msg):
    print(f"{GREEN}   ✅ {msg}{RESET}")

def log_fail(msg):
    print(f"{RED}   ❌ {msg}{RESET}")
    return False

def log_info(msg):
    print(f"   ℹ️  {msg}")

async def test_infrastructure():
    log_section("Checking Infrastructure")
    
    # Check 1: Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        if r.ping():
            log_success("Redis is running and accessible")
        else:
            log_fail("Redis ping failed")
            return False
    except ImportError:
        log_fail("redis-py not installed")
        return False
    except Exception as e:
        log_fail(f"Redis check failed: {e}")
        return False

    # Check 2: Ollama directly
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            log_success(f"Ollama is running. Models found: {len(models)}")
            
            required = ["llama3.2:1b", "nomic-embed-text"]
            missing = [req for req in required if not any(req in m for m in models)]
            
            if missing:
                log_fail(f"Missing required models: {missing}")
                log_info(f"Available: {models}")
                return False
            else:
                log_success("All required models (llama3.2:1b, nomic-embed-text) present")
        else:
            log_fail(f"Ollama returned status {resp.status_code}")
            return False
    except Exception as e:
        log_fail(f"Ollama check failed: {e}")
        return False
        
    return True

async def test_ollama_client():
    log_section("Testing Ollama Client Module")
    try:
        from src.llm import ollama_client
        
        # Embed
        text = "Hello world"
        emb = ollama_client.embed_text(text)
        if len(emb) > 0:
            log_success(f"Embedding generated (dim: {len(emb)})")
        else:
            log_fail("Embedding empty")
            return False
            
        # Chat
        resp = ollama_client.chat([{"role": "user", "content": "Say hi briefly"}])
        if resp:
            log_success(f"Chat response received: {resp}")
        else:
            log_fail("Chat response empty")
            return False
            
        return True
    except Exception as e:
        log_fail(f"Ollama client test error: {e}")
        return False

async def test_semantic_cache():
    log_section("Testing Semantic Cache")
    try:
        from src.llm.semantic_cache import SemanticCache
        cache = SemanticCache(cache_file="test_cache.json")
        cache.clear()
        
        # Add entry
        embedding = [0.1] * 768 # Dummy
        cache.add_entry("test prompt", embedding, "test response")
        log_success("Entry added to cache")
        
        # Retrieve
        result = cache.find_similar(embedding, threshold=0.99)
        if result and result['response'] == "test response":
            log_success(f"Cache hit successful (Match Type: {result.get('match_type', 'semantic')})")
        else:
            log_fail("Cache hit failed")
            return False
            
        # Exact Match Test
        result_exact = cache.find_similar(embedding, prompt="test prompt", threshold=0.99)
        if result_exact and result_exact.get('match_type') == 'exact':
            log_success("Exact match hash lookup successful ✅")
        else:
            log_fail(f"Exact match failed. Got: {result_exact.get('match_type') if result_exact else 'None'}")
            return False
            
        # Cleanup
        cache.clear()
        if os.path.exists("test_cache.json"):
            os.remove("test_cache.json")
            
        return True
    except Exception as e:
        log_fail(f"Semantic cache test error: {e}")
        return False

async def test_redis_context():
    log_section("Testing Redis Context")
    try:
        from src.llm.redis_context import RedisContextManager
        ctx = RedisContextManager(prefix="test_session")
        
        # Create session
        sid = ctx.create_session(language="en")
        log_success(f"Session created: {sid}")
        
        # Add turns
        ctx.add_turn(sid, "user", "Hello")
        ctx.add_turn(sid, "assistant", "Hi there")
        
        # Check history
        hist = ctx.get_context(sid)
        if len(hist) == 2:
            log_success("History preserved correctly")
        else:
            log_fail(f"History length mismatch. Expected 2, got {len(hist)}")
            return False
            
        # Clean up
        ctx.clear_session(sid)
        if not ctx.get_session(sid):
            log_success("Session cleared successfully")
        else:
            log_fail("Session clear failed")
            
        return True
    except Exception as e:
        log_fail(f"Redis context test error: {e}")
        return False

async def test_filler_manager():
    log_section("Testing Filler Manager")
    try:
        from src.llm.filler_manager import FillerBackchannelManager, FillerType
        mgr = FillerBackchannelManager()
        
        # English filler
        mgr.set_session_language("en")
        f_en = mgr.get_filler(filler_type=FillerType.THINKING)
        log_success(f"English filler: {f_en}")
        
        # Hindi filler
        mgr.set_session_language("hi")
        f_hi = mgr.get_filler(filler_type=FillerType.THINKING)
        log_success(f"Hindi filler: {f_hi}")
        
        # Reset turn to clear filler count limit
        mgr.reset_turn()
        
        # Response prefix logic
        prefix, used = mgr.get_response_prefix("Hard question?", processing_time_ms=2000)

        if used:
            log_success(f"Prefix generated: {prefix}")
        else:
            log_fail("Prefix generation failed for long process time")
            return False
            
        return True
    except Exception as e:
        log_fail(f"Filler manager test error: {e}")
        return False

async def test_full_pipeline():
    log_section("Testing Full Embedding Agent Pipeline")
    try:
        from src.llm.embedding_agent import process_with_context, end_session
        
        prompt = "I have a headache and I am 30 years old."
        log_info(f"Processing prompt: '{prompt}'")
        
        start = time.time()
        result = process_with_context(prompt)
        duration = time.time() - start
        
        sid = result['session_id']
        resp = result['response']
        
        log_success(f"Pipeline finished in {duration:.2f}s")
        log_info(f"Session ID: {sid}")
        log_info(f"Response: {resp}")
        
        if result.get('filler_audio'):
            log_success(f"Filler generated: {result['filler_audio']}")
        elif result.get('source') == 'semantic-cache':
            log_info("Response from cache (no filler generated)")
        else:
            log_info("No filler generated (might be fast response)")
            
        # Cleanup

        end_session(sid)
        return True
    except Exception as e:
        log_fail(f"Full pipeline test error: {e}")
        return False

async def main():
    print(f"\n{YELLOW}🚀 Starting Comprehensive LLM Feature Tests{RESET}")
    
    infra_ok = await test_infrastructure()
    if not infra_ok:
        print(f"\n{RED}🛑 Infrastructure checks failed. Stopping tests.{RESET}")
        return

    checks = [
        test_ollama_client,
        test_semantic_cache,
        test_redis_context,
        test_filler_manager,
        test_full_pipeline
    ]
    
    failed = 0
    for check in checks:
        if not await check():
            failed += 1
            
    print(f"\n{CYAN}{'='*60}")
    if failed == 0:
        print(f"{GREEN}🎉 ALL TESTS PASSED! The LLM stack is fully functional.{RESET}")
    else:
        print(f"{RED}⚠️  {failed} tests failed. Check logs above.{RESET}")
    print(f"{'='*60}{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
