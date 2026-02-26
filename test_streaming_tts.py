import sys
import os
import asyncio
import time

# Add current directory to path
sys.path.append(os.getcwd())

os.makedirs("test_output", exist_ok=True)

from src.llm.embedding_agent import process_with_context
from src.llm.stream_chunker import StreamChunker
from src.tts.tts_manager import TTSManager

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

async def test_streaming_flow(language_code: str, prompt: str):
    print(f"\n{CYAN}{'='*60}")
    print(f"🌊 Testing Stream Flow for {language_code}")
    print(f"{'='*60}{RESET}")
    
    tts_manager = TTSManager()
    metadata = {
        "detected_language": language_code,
        "language_confidence": "0.99"
    }

    print(f"📥 Input: {prompt}")
    print("⏳ Starting stream...")

    # Clear cache to force generation
    from src.llm.embedding_agent import cache
    cache.clear()

    # 1. Start Stream
    start_time = time.time()
    result = process_with_context(
        prompt=prompt,
        session_id=f"stream_test_{language_code}",
        metadata=metadata,
        stream=True
    )
    
    if not result.get("stream"):
        print("❌ Error: Result is not a stream!")
        return

    # 2. Process with Chunker & TTS
    chunker = StreamChunker()
    token_count = 0
    chunk_count = 0
    
    print("\n🎧 [TTS Simulation] Processing Chunks:")
    
    generator = result["response_generator"]
    
    for token in generator:
        token_count += 1
        if token_count == 1:
            ttft = (time.time() - start_time) * 1000
            print(f"{YELLOW}⚡ Time to First Token: {ttft:.2f}ms{RESET}")
            
        # Feed token to chunker
        for chunk in chunker.process(token):
            chunk_count += 1
            print(f"   🗣️  Chunk #{chunk_count}: {GREEN}{chunk}{RESET}")
            # Route to TTS
            audio_bytes = await tts_manager.speak(chunk, language_code)
            if audio_bytes:
                ext = "wav"
                filename = f"test_output/{language_code}_{chunk_count}.{ext}"
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                print(f"      💾 Saved: {filename} ({len(audio_bytes)} bytes)")
            
    # Flush remaining text
    last_chunk = chunker.flush()
    if last_chunk:
        chunk_count += 1
        print(f"   🗣️  Chunk #{chunk_count}: {GREEN}{last_chunk}{RESET}")
        audio_bytes = await tts_manager.speak(last_chunk, language_code)
        if audio_bytes:
            ext = "wav"
            filename = f"test_output/{language_code}_{chunk_count}.{ext}"
            with open(filename, "wb") as f:
                f.write(audio_bytes)
            print(f"      💾 Saved: {filename} ({len(audio_bytes)} bytes)")
        
    total_time = (time.time() - start_time) * 1000
    print(f"\n✅ Stream Complete. Total time: {total_time:.2f}ms. Generated {chunk_count} chunks.")

async def main():
    print(f"{YELLOW}🚀 Starting Streaming & Chunking Test{RESET}")
    
    # Test English (Piper)
    await test_streaming_flow("en", "Explain quantum computing in 3 sentences.")
    
    # Test Hindi (Piper + Indic Punctuation)
    await test_streaming_flow("hi", "What is the capital of India?")
    
    # Test Tamil (Edge TTS)
    await test_streaming_flow("ta", "Vanakkam! This is a Tamil TTS test.")

    # Test Kannada (Edge TTS)
    await test_streaming_flow("kn", "Namaskara! This is a Kannada TTS test.")

if __name__ == "__main__":

    asyncio.run(main())
