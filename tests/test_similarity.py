import pytest

from src.llm.similarity import (
    batch_cosine_similarity,
    cosine_similarity,
    cosine_similarity_np,
    find_most_similar,
)


def test_cosine_basic_cases():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity_np([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine_similarity_np([0, 0], [1, 1]) == 0.0


def test_cosine_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_batch_and_find_most_similar():
    embeddings = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
    sims = batch_cosine_similarity([1, 0, 0], embeddings)
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert sims[1] == pytest.approx(0.0, abs=1e-6)

    idx, score = find_most_similar([1, 0, 0], embeddings, threshold=0.95)
    assert idx == 0 and score == pytest.approx(1.0, abs=1e-6)
    assert find_most_similar([0, 0, 1], embeddings, threshold=0.5) == (-1, 0.0)
    assert find_most_similar([1, 0, 0], [], threshold=0.5) == (-1, 0.0)
