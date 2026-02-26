#!/usr/bin/env python3
"""
Chunked LLM latency benchmark for Ollama streaming.

Measures per configured stream-chunk size:
- TTFT (time-to-first-token)
- First chunk time
- Total response time
- Inter-chunk cadence
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "count": float(len(values)),
        "avg_ms": float(sum(values) / len(values)),
        "min_ms": float(min(values)),
        "p50_ms": float(statistics.median(values)),
        "p95_ms": percentile(values, 95),
        "max_ms": float(max(values)),
    }


@dataclass
class Cfg:
    endpoint: str
    model: str
    prompt: str
    runs: int
    warmup: int
    num_predict: int
    chunk_sizes: List[int]
    out: str


def stream_once(cfg: Cfg) -> Dict[str, Any]:
    payload = {
        "model": cfg.model,
        "prompt": cfg.prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": cfg.num_predict},
    }
    req = urllib.request.Request(
        cfg.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft_ms = None
    token_events = 0
    token_event_times: List[float] = []
    total_chars = 0

    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            evt = json.loads(raw.decode("utf-8"))
            tok = evt.get("response", "")
            if tok:
                now_ms = (time.perf_counter() - start) * 1000.0
                token_events += 1
                token_event_times.append(now_ms)
                total_chars += len(tok)
                if ttft_ms is None:
                    ttft_ms = now_ms
            if evt.get("done"):
                break

    total_ms = (time.perf_counter() - start) * 1000.0
    return {
        "ttft_ms": float(ttft_ms if ttft_ms is not None else total_ms),
        "total_ms": float(total_ms),
        "token_events": token_events,
        "token_event_times_ms": token_event_times,
        "total_chars": total_chars,
    }


def chunk_metrics(token_times: List[float], chunk_size: int) -> Dict[str, Any]:
    if not token_times:
        return {
            "first_chunk_ms": 0.0,
            "chunk_count": 0,
            "avg_inter_chunk_ms": 0.0,
            "inter_chunk_ms": [],
        }
    chunk_times = []
    for i in range(0, len(token_times), chunk_size):
        chunk_times.append(token_times[min(i + chunk_size - 1, len(token_times) - 1)])
    inter = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
    return {
        "first_chunk_ms": float(chunk_times[0]),
        "chunk_count": len(chunk_times),
        "avg_inter_chunk_ms": float(sum(inter) / len(inter)) if inter else 0.0,
        "inter_chunk_ms": inter,
    }


def run(cfg: Cfg):
    for _ in range(cfg.warmup):
        stream_once(cfg)

    base_runs = [stream_once(cfg) for _ in range(cfg.runs)]
    out: Dict[str, Any] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": cfg.endpoint,
            "model": cfg.model,
            "runs": cfg.runs,
            "warmup": cfg.warmup,
            "num_predict": cfg.num_predict,
            "chunk_sizes": cfg.chunk_sizes,
            "prompt": cfg.prompt,
        },
        "base_stream": {},
        "chunking": {},
    }

    ttfts = [r["ttft_ms"] for r in base_runs]
    totals = [r["total_ms"] for r in base_runs]
    token_events = [float(r["token_events"]) for r in base_runs]
    out["base_stream"] = {
        "ttft_ms": summarize(ttfts),
        "total_ms": summarize(totals),
        "token_events": summarize(token_events),
    }

    for n in cfg.chunk_sizes:
        firsts = []
        counts = []
        inter_avg = []
        for r in base_runs:
            m = chunk_metrics(r["token_event_times_ms"], n)
            firsts.append(m["first_chunk_ms"])
            counts.append(float(m["chunk_count"]))
            inter_avg.append(float(m["avg_inter_chunk_ms"]))
        out["chunking"][str(n)] = {
            "first_chunk_ms": summarize(firsts),
            "chunk_count": summarize(counts),
            "avg_inter_chunk_ms": summarize(inter_avg),
        }

    p = Path(cfg.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved={p}")


def parse() -> Cfg:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--prompt", default="Caller says there is smoke in the building. Give a short safety-first response.")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--num-predict", type=int, default=64)
    ap.add_argument("--chunk-sizes", default="1,4,8,16")
    ap.add_argument(
        "--out",
        default=f"benchmark_outputs/ollama_chunk_latency_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    a = ap.parse_args()
    chunks = [int(x.strip()) for x in a.chunk_sizes.split(",") if x.strip()]
    return Cfg(
        endpoint=a.endpoint,
        model=a.model,
        prompt=a.prompt,
        runs=max(1, a.runs),
        warmup=max(0, a.warmup),
        num_predict=max(1, a.num_predict),
        chunk_sizes=chunks,
        out=a.out,
    )


if __name__ == "__main__":
    run(parse())
