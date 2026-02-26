#!/usr/bin/env python3
"""
Extract reproducible evidence metrics for the Solvathon Layer 1 research paper.

Outputs:
1) JSON snapshot with raw and aggregated metrics.
2) Markdown summary with publication-ready tables.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "evidence"

LID_NOTEBOOK = ROOT / "Indian_Language_ID_Colab.ipynb"
WEBRTC_LOG = ROOT / "gpu_wav_samples" / "test_logs" / "webrtc_live_test.log"
WEBSOCKET_LOG = ROOT / "gpu_wav_samples" / "test_logs" / "websocket_live_test_v2.log"
VOICE_TURN_EN = ROOT / "gpu_wav_samples" / "voice_turn_en_1.json"
VOICE_TURN_HI = ROOT / "gpu_wav_samples" / "voice_turn_hi_1.json"

PROMPTS = [
    "Translate 'Hello, how are you?' to Hindi.",
    "Translate 'Hello, how are you?' to Kannada.",
    "Translate 'Hello, how are you?' to Telugu.",
    "Translate 'Hello, how are you?' to Tamil.",
    "Write a short sentence in Kannada.",
]


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _nearest_rank_p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    rank = max(1, math.ceil(0.95 * len(vals)))
    return float(vals[rank - 1])


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p95": _nearest_rank_p95(values),
    }


def _read_wav_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            channels = wf.getnchannels()
            bits = wf.getsampwidth() * 8
            duration = frames / sr if sr else 0.0
        return {
            "path": str(path),
            "sample_rate_hz": sr,
            "channels": channels,
            "bits_per_sample": bits,
            "duration_s": duration,
        }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def extract_lid_times(notebook_path: Path) -> Dict[str, Any]:
    times: List[int] = []
    payload = _safe_json_load(notebook_path)
    if payload is None:
        return {"source": str(notebook_path), "times_ms": times, "stats_ms": _summary_stats([])}

    pattern = re.compile(r"Time:\s*(\d+)ms")
    for cell in payload.get("cells", []):
        for output in cell.get("outputs", []):
            for key in ("text",):
                text_lines = output.get(key, [])
                if isinstance(text_lines, str):
                    text_lines = [text_lines]
                for line in text_lines:
                    match = pattern.search(line)
                    if match:
                        times.append(int(match.group(1)))

    stats = _summary_stats([float(x) for x in times])
    return {"source": str(notebook_path), "times_ms": times, "stats_ms": stats}


def extract_reliability_logs() -> Dict[str, Any]:
    return {
        "webrtc_log": _safe_json_load(WEBRTC_LOG),
        "websocket_log": _safe_json_load(WEBSOCKET_LOG),
    }


def _decode_b64_len(value: str) -> int:
    if not value:
        return 0
    try:
        return len(base64.b64decode(value))
    except Exception:
        return 0


def extract_voice_turn(path: Path) -> Dict[str, Any]:
    data = _safe_json_load(path) or {}
    language = data.get("language")
    transcript = data.get("transcript", "")
    response = data.get("response", "")
    audio_b64 = data.get("audio_b64", "")

    base = {
        "source": str(path),
        "ok": data.get("ok"),
        "session_id": data.get("session_id"),
        "language": language,
        "transcript": transcript,
        "response": response,
        "transcript_chars": len(transcript or ""),
        "response_chars": len(response or ""),
        "audio_mime": data.get("audio_mime"),
        "audio_b64_len": len(audio_b64 or ""),
        "audio_decoded_bytes": _decode_b64_len(audio_b64 or ""),
    }

    if language == "en":
        base["input_wav"] = _read_wav_meta(ROOT / "gpu_wav_samples" / "en_1.wav")
        base["output_wav"] = _read_wav_meta(ROOT / "gpu_wav_samples" / "pipeline_reply_en_1.wav")
    elif language == "hi":
        base["input_wav"] = _read_wav_meta(ROOT / "gpu_wav_samples" / "hi_1.wav")
        base["output_wav"] = _read_wav_meta(ROOT / "gpu_wav_samples" / "pipeline_reply_hi_1.wav")

    return base


def run_ollama_checks(base_url: str, model: str) -> Dict[str, Any]:
    status: Dict[str, Any] = {"base_url": base_url, "model": model}
    latencies_ms: List[float] = []
    prompt_runs: List[Dict[str, Any]] = []

    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=5)
        tags_resp.raise_for_status()
        tags_data = tags_resp.json()
        status["server_running"] = True
        status["models"] = [m.get("name") for m in tags_data.get("models", [])]
    except Exception as exc:
        status["server_running"] = False
        status["error"] = str(exc)
        return {"status": status, "prompt_runs": prompt_runs, "stats_ms": _summary_stats([])}

    for prompt in PROMPTS:
        row: Dict[str, Any] = {"prompt": prompt}
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
                timeout=120,
            )
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            text = resp.json().get("response", "")
            row.update(
                {
                    "ok": True,
                    "latency_ms": latency,
                    "response_chars": len(text),
                    "response_preview": text[:240],
                }
            )
            latencies_ms.append(latency)
        except Exception as exc:
            row.update({"ok": False, "error": str(exc)})
        prompt_runs.append(row)

    return {
        "status": status,
        "prompt_runs": prompt_runs,
        "stats_ms": _summary_stats(latencies_ms),
    }


def run_pipeline_connectivity(server_url: str) -> Dict[str, Any]:
    cmd = ["python3", str(ROOT / "test_pipeline.py"), "--server", server_url]
    started_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180, check=False)
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        summary_match = re.search(r"(\d+)/(\d+) tests passed", combined)
        parsed = None
        if summary_match:
            parsed = {"passed": int(summary_match.group(1)), "total": int(summary_match.group(2))}
        return {
            "started_at_utc": started_at,
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "parsed_summary": parsed,
            "output": combined.strip(),
        }
    except Exception as exc:
        return {
            "started_at_utc": started_at,
            "command": " ".join(cmd),
            "exit_code": None,
            "parsed_summary": None,
            "output": "",
            "error": str(exc),
        }


def check_redis_status() -> Dict[str, Any]:
    cmd = ["redis-cli", "-h", "localhost", "-p", "6379", "ping"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        return {
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "misconf_detected": "MISCONF" in stdout or "MISCONF" in stderr,
        }
    except FileNotFoundError:
        return {
            "command": " ".join(cmd),
            "exit_code": None,
            "stdout": "",
            "stderr": "redis-cli not found",
            "misconf_detected": False,
        }
    except Exception as exc:
        return {
            "command": " ".join(cmd),
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
            "misconf_detected": False,
        }


def build_markdown(snapshot: Dict[str, Any]) -> str:
    lid = snapshot["lid_timings"]
    llm = snapshot["ollama_live_checks"]
    pipe = snapshot["pipeline_connectivity"]
    reliability = snapshot["artifact_reliability"]
    voice = snapshot["voice_turn_examples"]

    webrtc = reliability.get("webrtc_log") or {}
    websocket = reliability.get("websocket_log") or {}

    lines: List[str] = []
    lines.append("# Solvathon Layer 1 Evidence Snapshot")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{snapshot['generated_at_utc']}`")
    lines.append(f"- Measurement date: `{snapshot['measurement_date']}`")
    lines.append("")

    lines.append("## LID Timing (Notebook-Extracted)")
    lines.append("")
    lid_stats = lid["stats_ms"]
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}"
    lines.append("| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {lid_stats['n']} | {_fmt(lid_stats['mean'])} | {_fmt(lid_stats['median'])} | "
        f"{_fmt(lid_stats['min'])} | {_fmt(lid_stats['max'])} | {_fmt(lid_stats['p95'])} |"
    )
    lines.append("")
    lines.append(f"Raw timings (ms): `{lid['times_ms']}`")
    lines.append("")

    lines.append("## Live LLM Latency (Ollama)")
    lines.append("")
    llm_stats = llm["stats_ms"]
    lines.append("| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {llm_stats['n']} | {_fmt(llm_stats['mean'])} | {_fmt(llm_stats['median'])} | "
        f"{_fmt(llm_stats['min'])} | {_fmt(llm_stats['max'])} | {_fmt(llm_stats['p95'])} |"
    )
    lines.append("")
    lines.append("| prompt | ok | latency_ms | response_chars |")
    lines.append("|---|---|---:|---:|")
    for row in llm["prompt_runs"]:
        latency = row.get("latency_ms")
        latency_cell = f"{latency:.2f}" if isinstance(latency, (int, float)) else "-"
        lines.append(
            f"| {row['prompt']} | {row.get('ok')} | {latency_cell} | {row.get('response_chars', '-')} |"
        )
    lines.append("")

    lines.append("## Connectivity and Reliability")
    lines.append("")
    lines.append("| source | key findings |")
    lines.append("|---|---|")
    lines.append(
        "| webrtc_live_test.log | "
        f"offer_status={webrtc.get('offer_status')}, answer_type={webrtc.get('answer_type')}, "
        f"ice_final={webrtc.get('ice_final')}, connected={webrtc.get('connected')} |"
    )
    lines.append(
        "| websocket_live_test_v2.log | "
        f"connected={websocket.get('connected')}, recv_media_events={websocket.get('recv_media_events')}, "
        f"ws_closed_cleanly={websocket.get('ws_closed_cleanly')} |"
    )
    parsed = pipe.get("parsed_summary") or {}
    lines.append(
        "| test_pipeline.py live probe | "
        f"exit_code={pipe.get('exit_code')}, passed={parsed.get('passed', '-')}/{parsed.get('total', '-')} |"
    )
    lines.append("")

    lines.append("## Voice-Turn Examples")
    lines.append("")
    lines.append("| file | language | transcript_chars | response_chars | decoded_audio_bytes | input_duration_s | output_duration_s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in voice:
        in_d = ((row.get("input_wav") or {}).get("duration_s"))
        out_d = ((row.get("output_wav") or {}).get("duration_s"))
        lines.append(
            f"| {Path(row['source']).name} | {row.get('language')} | {row.get('transcript_chars')} | "
            f"{row.get('response_chars')} | {row.get('audio_decoded_bytes')} | "
            f"{in_d:.2f} | {out_d:.2f} |"
        )
    lines.append("")

    lines.append("## Environment Flags")
    lines.append("")
    redis_status = snapshot["redis_status"]
    lines.append(f"- Redis MISCONF detected: `{redis_status.get('misconf_detected')}`")
    lines.append(f"- `data/splits.json` exists: `{snapshot['data_splits_exists']}`")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metrics for research-paper evidence.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evidence outputs")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--llm-model", default="llama3.2:3b", help="Ollama model for latency checks")
    parser.add_argument("--pipeline-url", default="http://localhost:8080", help="Voice pipeline server URL")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    measurement_date = now_utc.date().isoformat()

    snapshot: Dict[str, Any] = {
        "generated_at_utc": now_utc.isoformat() + "Z",
        "measurement_date": measurement_date,
        "artifact_reliability": extract_reliability_logs(),
        "lid_timings": extract_lid_times(LID_NOTEBOOK),
        "voice_turn_examples": [extract_voice_turn(VOICE_TURN_EN), extract_voice_turn(VOICE_TURN_HI)],
        "ollama_live_checks": run_ollama_checks(args.ollama_url, args.llm_model),
        "pipeline_connectivity": run_pipeline_connectivity(args.pipeline_url),
        "redis_status": check_redis_status(),
        "data_splits_exists": (ROOT / "data" / "splits.json").exists(),
    }

    json_path = out_dir / "metrics_snapshot.json"
    md_path = out_dir / "evidence_summary.md"

    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(snapshot), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
