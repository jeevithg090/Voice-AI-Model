"""LLM integration layer: Ollama client, semantic cache, Redis context, fillers.

Submodules are imported explicitly (e.g. ``from src.llm.embedding_agent import
process_with_context``) so that importing a lightweight helper such as
``src.llm.stream_chunker`` does not eagerly create Redis/cache singletons.
"""
