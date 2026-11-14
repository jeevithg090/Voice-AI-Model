import json

import pytest

from src.llm import semantic_cache as sc


@pytest.fixture(params=["faiss", "cosine"])
def cache(request, tmp_path):
    if request.param == "faiss" and not sc.FAISS_AVAILABLE:
        pytest.skip("faiss not installed")
    return sc.SemanticCache(
        dimension=3,
        cache_file=str(tmp_path / "cache.json"),
        use_cosine_only=request.param == "cosine",
    )


def test_empty_cache_misses(cache):
    assert cache.find_similar([1, 0, 0], prompt="hi") is None


def test_exact_match_counts_hits(cache):
    cache.add_entry("what are your hours", [1, 0, 0], "9 to 5")
    hit = cache.find_similar([0, 1, 0], prompt="what are your hours")
    assert hit["response"] == "9 to 5"
    assert hit["match_type"] == "exact"
    assert hit["similarity"] == 1.0
    cache.find_similar([0, 1, 0], prompt="what are your hours")
    assert cache.get_stats()["total_hits"] == 2


def test_semantic_match_respects_threshold(cache):
    cache.add_entry("opening hours?", [1, 0, 0], "9 to 5")
    hit = cache.find_similar([0.95, 0.05, 0], prompt="when do you open", threshold=0.9)
    assert hit is not None and hit["response"] == "9 to 5"
    assert cache.find_similar([0, 1, 0], prompt="something else", threshold=0.9) is None


def test_persistence_roundtrip(cache, tmp_path):
    cache.add_entry("hello", [1, 0, 0], "namaste")
    data = json.loads((tmp_path / "cache.json").read_text())
    assert data["entries"][0]["prompt"] == "hello"

    reloaded = sc.SemanticCache(dimension=3, cache_file=cache.cache_file, use_cosine_only=not cache.use_faiss)
    assert reloaded.find_similar([1, 0, 0], prompt="hello")["response"] == "namaste"
    assert reloaded.find_similar([1, 0, 0], prompt=None, threshold=0.9)["response"] == "namaste"


def test_clear_removes_exact_matches(cache, tmp_path):
    cache.add_entry("hello", [1, 0, 0], "namaste")
    cache.clear()
    assert cache.find_similar([1, 0, 0], prompt="hello") is None
    assert not (tmp_path / "cache.json").exists()
