#!/usr/bin/env python3
"""
Standalone FAISS lookup latency benchmark (semantic-cache proxy).
"""

from __future__ import annotations

import argparse
import json
import time

import faiss
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    embs = rng.normal(size=(args.entries, args.dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-12)

    index = faiss.IndexFlatIP(args.dim)
    index.add(embs)

    # warmup
    index.search(embs[0:1], 1)

    timings = []
    for i in range(args.runs):
        q = embs[(i % args.entries) : ((i % args.entries) + 1)]
        t0 = time.perf_counter()
        index.search(q, 1)
        timings.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings, dtype=np.float64)
    out = {
        "entries": args.entries,
        "dimension": args.dim,
        "runs": args.runs,
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
