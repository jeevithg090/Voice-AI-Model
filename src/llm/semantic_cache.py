"""
Semantic Cache using FAISS for Voice AI Pipeline
Fast similarity search with persistent storage option.
"""

import numpy as np
import json
import os
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not installed. Using fallback cosine similarity.")


class SemanticCache:
    """
    Semantic cache with FAISS for fast similarity search.
    Falls back to brute-force cosine similarity if FAISS is unavailable.
    """
    
    def __init__(self, dimension: int = 768, cache_file: str = "semantic_cache.json", use_cosine_only: bool = False):
        """
        Initialize the semantic cache.
        
        Args:
            dimension: Embedding dimension (768 for nomic-embed-text)
            cache_file: Path to persist cache entries
            use_cosine_only: If True, use pure cosine similarity (no FAISS)
        """
        self.dimension = dimension
        self.cache_file = cache_file
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.hash_map: Dict[str, Dict[str, Any]] = {}  # Exact match cache: hash -> entry
        
        # Initialize FAISS index if available and not forcing cosine
        if FAISS_AVAILABLE and not use_cosine_only:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
            self.use_faiss = True
            self.mode = "faiss"
        else:
            self.index = None
            self.use_faiss = False
            self.mode = "cosine"
        
        # Load existing cache if available
        self._load_cache()
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of the text for exact match."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self):
        """Load cache from disk if exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
                    self.embeddings = data.get("embeddings", [])
                    
                    # Rebuild exact match hash map
                    for entry in self.entries:
                        if "prompt" in entry:
                            h = self._compute_hash(entry["prompt"])
                            self.hash_map[h] = entry
                    
                    # Rebuild FAISS index
                    if self.use_faiss and self.embeddings:
                        embeddings_np = self._normalize_embeddings(self.embeddings)
                        self.index.add(embeddings_np)
                        
                print(f"📂 Loaded {len(self.entries)} cache entries from {self.cache_file}")
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
    
    def _save_cache(self):
        """Persist cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    "entries": self.entries,
                    "embeddings": self.embeddings
                }, f)
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> np.ndarray:
        """Normalize embeddings for cosine similarity via inner product."""
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return arr / norms
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return float(dot / (mag1 * mag2))
    
    def add_entry(self, prompt: str, embedding: List[float], response: str):
        """
        Add a new entry to the cache.
        
        Args:
            prompt: Original user prompt
            embedding: Embedding vector
            response: LLM response to cache
        """
        entry = {
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "hit_count": 0
        }
        
        self.entries.append(entry)
        self.embeddings.append(embedding)
        
        # Update Exact Match Hash Map
        h = self._compute_hash(prompt)
        self.hash_map[h] = entry
        
        # Add to FAISS index
        if self.use_faiss:
            embedding_np = self._normalize_embeddings([embedding])
            self.index.add(embedding_np)
        
        # Persist to disk
        self._save_cache()
    
    def find_similar(
        self, 
        query_embedding: List[float], 
        prompt: Optional[str] = None,
        threshold: float = 0.85,
        top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Find the most similar cached entry.
        
        Args:
            query_embedding: Query embedding vector
            prompt: Original prompt text (required for exact match O(1) lookup)
            threshold: Minimum similarity score for a match
            top_k: Number of results to consider
            
        Returns:
            Best matching entry or None if no match above threshold
        """
        # 1. Exact Match via Hashing (O(1))
        if prompt:
            h = self._compute_hash(prompt)
            if h in self.hash_map:
                entry = self.hash_map[h].copy()
                entry["similarity"] = 1.0
                entry["hit_count"] += 1
                entry["match_type"] = "exact"
                self._save_cache()
                return entry

        if not self.entries:
            return None
        
        if self.use_faiss:
            # FAISS search
            query_np = self._normalize_embeddings([query_embedding])
            scores, indices = self.index.search(query_np, min(top_k, len(self.entries)))
            
            best_score = scores[0][0]
            best_idx = indices[0][0]
            
            if best_score >= threshold:
                entry = self.entries[best_idx].copy()
                entry["similarity"] = float(best_score)
                
                # Update hit count
                self.entries[best_idx]["hit_count"] += 1
                self._save_cache()
                
                return entry
        else:
            # Fallback: brute-force cosine similarity
            best_score = 0.0
            best_entry = None
            best_idx = -1
            
            for i, stored_embedding in enumerate(self.embeddings):
                score = self._cosine_similarity(query_embedding, stored_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = self.entries[i]
                    best_idx = i
            
            if best_score >= threshold and best_entry:
                result = best_entry.copy()
                result["similarity"] = best_score
                
                # Update hit count
                self.entries[best_idx]["hit_count"] += 1
                self._save_cache()
                
                return result
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.get("hit_count", 0) for e in self.entries)
        return {
            "total_entries": len(self.entries),
            "total_hits": total_hits,
            "using_faiss": self.use_faiss,
            "dimension": self.dimension
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.entries = []
        self.embeddings = []
        
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        
        print("🗑️  Cache cleared")
