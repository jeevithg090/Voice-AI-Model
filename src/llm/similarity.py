"""
Similarity Functions for Voice AI Pipeline
Provides cosine similarity and other similarity metrics.
"""

import math
from typing import List
import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two numeric vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Similarity score between -1.0 and 1.0
        1.0 = identical meaning
        0.0 = unrelated
        -1.0 = opposite meaning
    """
    v1 = [float(x) for x in vec1]
    v2 = [float(x) for x in vec2]

    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def cosine_similarity_np(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity using NumPy (faster for large vectors).
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Similarity score between -1.0 and 1.0
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return float(dot / (mag1 * mag2))


def batch_cosine_similarity(query: List[float], embeddings: List[List[float]]) -> List[float]:
    """
    Compute cosine similarity between a query and multiple embeddings.
    Vectorized for speed.
    
    Args:
        query: Query embedding vector
        embeddings: List of stored embedding vectors
        
    Returns:
        List of similarity scores
    """
    q = np.array(query, dtype=np.float32)
    emb = np.array(embeddings, dtype=np.float32)
    
    # Normalize
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    emb_norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb_normalized = emb / emb_norms
    
    # Compute all similarities at once
    similarities = np.dot(emb_normalized, q_norm)
    
    return similarities.tolist()


def find_most_similar(
    query: List[float], 
    embeddings: List[List[float]], 
    threshold: float = 0.85
) -> tuple:
    """
    Find the most similar embedding above threshold.
    
    Args:
        query: Query embedding vector
        embeddings: List of stored embedding vectors
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (best_index, best_similarity) or (-1, 0.0) if no match
    """
    if not embeddings:
        return (-1, 0.0)
    
    similarities = batch_cosine_similarity(query, embeddings)
    
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        return (best_idx, best_score)
    
    return (-1, 0.0)
