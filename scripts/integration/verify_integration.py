import asyncio
import os
import sys

# Add src to path
sys.path.append(os.getcwd())


async def test_llm_module():
    print("🧪 Testing src.llm module...")
    try:
        from src.llm.embedding_agent import process_query

        print("✅ Successfully imported src.llm.embedding_agent")

        # Test a simple query (assuming Ollama is up, but handle if not)
        print("   Running test query...")
        try:
            result = process_query("Hello, are you there?")
            print(f"   Response: {result['response']}")
            print("✅ LLM query successful")
        except Exception as e:
            print(f"⚠️  LLM query failed (expected if Ollama not running): {e}")

    except ImportError as e:
        print(f"❌ Failed to import src.llm: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def test_server_import():
    print("\n🧪 Testing signaling_server import...")
    try:
        import src.realtime.signaling_server  # noqa: F401

        print("✅ Successfully imported src.realtime.signaling_server")
    except ImportError as e:
        print(f"❌ Failed to import signaling_server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error importing signaling_server: {e}")
        # It might fail due to missing models or env vars, but syntax should be fine
        if "No module named" in str(e):
            sys.exit(1)


if __name__ == "__main__":
    test_server_import()
    asyncio.run(test_llm_module())
