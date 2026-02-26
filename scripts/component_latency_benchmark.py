#!/usr/bin/env python3
"""
Component-by-component latency benchmark for Solvathon Layer 1.

Measures:
- Speech preprocessing
- WAD/VAD (energy-based endpointing)
- Speech encoder (Mimi)
- Language detection (MMS-LID)
- STT (Whisper)
- Emergency detection (Mimi + classifier)
- LLM (Ollama, stream TTFT + total)
- TTS (TTSManager)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    MimiModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForSequenceClassification,
)

from src.config import settings
from src.models.emergency_classifier import MimiEmergencyClassifier
from src.tts.tts_manager import TTSManager


SUPPORTED_LID_CODES = {
    "eng", "hin", "tam", "tel", "kan", "ben", "guj", "mal", "mar", "pan", "ori", "asm", "urd"
}
LANGUAGE_CODE_MAP = {
    "eng": "en",
    "hin": "hi",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "count": float(len(values)),
        "avg_ms": float(sum(values) / len(values)),
        "min_ms": float(min(values)),
        "p50_ms": float(statistics.median(values)),
        "p95_ms": percentile(values, 95),
        "max_ms": float(max(values)),
    }


def fmt_stats(stats: Dict[str, float]) -> str:
    return (
        f"count={int(stats['count'])} "
        f"avg={stats['avg_ms']:.2f}ms "
        f"min={stats['min_ms']:.2f}ms "
        f"p95={stats['p95_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms"
    )


@dataclass
class RunConfig:
    audio_path: str
    runs: int
    warmup: int
    language: str
    prompt: str
    tts_text: str
    output_json: str
    llm_fallback_model: str
    sample_rate: int = 48000


class ComponentBench:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = pick_device()

        self.audio_48k, _ = librosa.load(cfg.audio_path, sr=cfg.sample_rate, mono=True)
        self.audio_48k = self.audio_48k.astype(np.float32)
        if self.audio_48k.size == 0:
            raise ValueError("Input audio is empty after decode.")

        self.results: Dict[str, Any] = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "audio_path": cfg.audio_path,
                "audio_seconds": float(self.audio_48k.shape[0] / cfg.sample_rate),
                "runs": cfg.runs,
                "warmup": cfg.warmup,
                "device": self.device,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "settings": {
                    "whisper_model": settings.WHISPER_MODEL,
                    "lid_model": settings.LID_MODEL,
                    "ollama_base_url": settings.OLLAMA_BASE_URL,
                    "llm_model": settings.LLM_MODEL,
                    "tts_backend": settings.TTS_BACKEND,
                },
            },
            "components": {},
        }

    def _run_timed(
        self,
        name: str,
        fn,
        warmup: Optional[int] = None,
        runs: Optional[int] = None,
    ) -> Tuple[List[float], Any]:
        warm = self.cfg.warmup if warmup is None else warmup
        total_runs = self.cfg.runs if runs is None else runs
        last_out = None

        for _ in range(warm):
            last_out = fn()

        timings: List[float] = []
        for _ in range(total_runs):
            start = time.perf_counter()
            last_out = fn()
            elapsed = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed)
        return timings, last_out

    def _save_component(self, name: str, times: List[float], extra: Optional[Dict[str, Any]] = None):
        payload: Dict[str, Any] = {"latency_ms": summarize(times)}
        if extra:
            payload["details"] = extra
        self.results["components"][name] = payload
        print(f"[OK] {name}: {fmt_stats(payload['latency_ms'])}")
        if extra:
            print(f"     details={extra}")

    def _save_failure(self, name: str, err: Exception):
        self.results["components"][name] = {"error": str(err)}
        print(f"[FAIL] {name}: {err}")

    def bench_speech_preprocess(self):
        def _fn():
            x = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000)
            x = x.astype(np.float32)
            x *= 2.0
            x = np.clip(x, -1.0, 1.0)
            return x.shape[0]

        times, out = self._run_timed("speech_preprocess", _fn)
        self._save_component("speech_preprocess", times, {"output_samples_16k": int(out)})

    def bench_wad(self):
        threshold = settings.SILENCE_THRESHOLD
        frame_samples = int(0.02 * 48000)  # 20ms
        audio = self.audio_48k

        def _fn():
            speech_frames = 0
            for i in range(0, len(audio), frame_samples):
                frame = audio[i : i + frame_samples]
                if frame.size == 0:
                    continue
                energy = float(np.mean(np.abs(frame)))
                if energy > threshold:
                    speech_frames += 1
            return speech_frames

        times, speech_frames = self._run_timed("wad_energy", _fn)
        self._save_component(
            "wad_energy",
            times,
            {"threshold": threshold, "frame_ms": 20, "speech_frames": int(speech_frames)},
        )

    def bench_speech_encoder(self):
        fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        mimi = MimiModel.from_pretrained("kyutai/mimi").to(self.device).eval()
        audio_24k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=fe.sampling_rate)

        def _fn():
            inputs = fe(raw_audio=audio_24k, sampling_rate=fe.sampling_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                codes = mimi.encode(inputs["input_values"]).audio_codes
            return tuple(codes.shape)

        times, shape = self._run_timed("speech_encoder_mimi", _fn)
        self._save_component("speech_encoder_mimi", times, {"audio_codes_shape": list(shape)})

    def bench_language_detection(self):
        lid_processor = AutoFeatureExtractor.from_pretrained(settings.LID_MODEL)
        lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.LID_MODEL).to(self.device).eval()
        audio_16k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000)

        def _fn():
            inputs = lid_processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(lid_model(**inputs).logits, dim=-1)[0]
            top_idx = int(torch.argmax(probs).item())
            code = lid_model.config.id2label[top_idx]
            prob = float(probs[top_idx].item())
            return code, prob

        times, pred = self._run_timed("language_detection_mms_lid", _fn)
        code, prob = pred
        normalized = LANGUAGE_CODE_MAP.get(code, "en")
        self._save_component(
            "language_detection_mms_lid",
            times,
            {"top_code": code, "normalized_lang": normalized, "confidence": round(prob, 4)},
        )

    def bench_stt(self):
        whisper_lang = self.cfg.language
        processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL).to(self.device).eval()
        audio_16k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000).astype(np.float32)
        audio_16k = np.clip(audio_16k * 2.0, -1.0, 1.0)

        try:
            forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
        except Exception:
            forced_ids = None

        def _fn():
            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                kwargs = {"forced_decoder_ids": forced_ids} if forced_ids else {}
                out_ids = model.generate(inputs.input_features.to(self.device), **kwargs)
            transcript = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            return transcript

        times, transcript = self._run_timed("stt_whisper", _fn)
        self._save_component("stt_whisper", times, {"transcript_preview": transcript[:160]})

    def bench_emergency_detection(self):
        fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        mimi = MimiModel.from_pretrained("kyutai/mimi").to(self.device).eval()
        clf = MimiEmergencyClassifier().to(self.device)
        clf.load_state_dict(torch.load(settings.EMERGENCY_CLASSIFIER_PATH, map_location=self.device))
        clf.eval()

        audio_24k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=fe.sampling_rate)

        def _fn():
            inputs = fe(raw_audio=audio_24k, sampling_rate=fe.sampling_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                codes = mimi.encode(inputs["input_values"]).audio_codes
                prob = float(clf(codes).item())
            return prob

        times, prob = self._run_timed("emergency_detection", _fn)
        self._save_component("emergency_detection", times, {"emergency_probability": round(prob, 4)})

    def bench_llm(self):
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": self.cfg.prompt,
            "stream": True,
            "options": {"temperature": 0.2, "num_predict": 64},
        }

        def _one_stream_call() -> Tuple[float, float, int]:
            start = time.perf_counter()
            first_token_ms: Optional[float] = None
            chars = 0
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    event = json.loads(raw)
                    tok = event.get("response", "")
                    if tok and first_token_ms is None:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    chars += len(tok)
                    if event.get("done", False):
                        break
            total_ms = (time.perf_counter() - start) * 1000.0
            return float(first_token_ms or math.nan), total_ms, chars

        try:
            for _ in range(self.cfg.warmup):
                _one_stream_call()

            ttft_ms: List[float] = []
            total_ms: List[float] = []
            out_chars = 0
            for _ in range(self.cfg.runs):
                ttft, total, chars = _one_stream_call()
                if not math.isnan(ttft):
                    ttft_ms.append(ttft)
                total_ms.append(total)
                out_chars = chars

            if not ttft_ms:
                raise RuntimeError("LLM stream returned no token events.")

            self.results["components"]["llm_ollama"] = {
                "ttft_ms": summarize(ttft_ms),
                "total_ms": summarize(total_ms),
                "details": {"output_chars_last_run": out_chars},
            }
            print(f"[OK] llm_ollama TTFT: {fmt_stats(self.results['components']['llm_ollama']['ttft_ms'])}")
            print(f"[OK] llm_ollama TOTAL: {fmt_stats(self.results['components']['llm_ollama']['total_ms'])}")
        except Exception as ollama_err:
            self.results["components"]["llm_ollama"] = {"error": str(ollama_err)}
            print(f"[FAIL] llm_ollama: {ollama_err}")
            self.bench_llm_transformers_fallback()

    def bench_llm_transformers_fallback(self):
        model_id = self.cfg.llm_fallback_model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()
        prompt_ids = tokenizer(self.cfg.prompt, return_tensors="pt").input_ids.to(self.device)
        target_tokens = 64

        def _one() -> Tuple[float, float, int]:
            with torch.no_grad():
                start = time.perf_counter()
                out = model(input_ids=prompt_ids, use_cache=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                ttft_ms = (time.perf_counter() - start) * 1000.0
                past = out.past_key_values
                generated = [next_token]
                for _ in range(target_tokens - 1):
                    step = model(input_ids=next_token, past_key_values=past, use_cache=True)
                    next_token = torch.argmax(step.logits[:, -1, :], dim=-1, keepdim=True)
                    past = step.past_key_values
                    generated.append(next_token)
                total_ms = (time.perf_counter() - start) * 1000.0
            generated_ids = torch.cat(generated, dim=1)
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return ttft_ms, total_ms, len(text)

        for _ in range(self.cfg.warmup):
            _one()

        ttft_ms: List[float] = []
        total_ms: List[float] = []
        out_chars = 0
        for _ in range(self.cfg.runs):
            ttft, total, chars = _one()
            ttft_ms.append(ttft)
            total_ms.append(total)
            out_chars = chars

        self.results["components"]["llm_transformers_fallback"] = {
            "ttft_ms": summarize(ttft_ms),
            "total_ms": summarize(total_ms),
            "details": {"model": model_id, "output_chars_last_run": out_chars, "generated_tokens": target_tokens},
        }
        print(
            f"[OK] llm_transformers_fallback TTFT: "
            f"{fmt_stats(self.results['components']['llm_transformers_fallback']['ttft_ms'])}"
        )
        print(
            f"[OK] llm_transformers_fallback TOTAL: "
            f"{fmt_stats(self.results['components']['llm_transformers_fallback']['total_ms'])}"
        )

    def bench_tts(self):
        manager = TTSManager()
        text = self.cfg.tts_text
        language = self.cfg.language

        async def _call():
            return await manager.speak(text, language)

        for _ in range(self.cfg.warmup):
            asyncio.run(_call())

        times: List[float] = []
        audio_len = 0
        for _ in range(self.cfg.runs):
            start = time.perf_counter()
            out = asyncio.run(_call()) or b""
            elapsed = (time.perf_counter() - start) * 1000.0
            times.append(elapsed)
            audio_len = len(out)

        self._save_component(
            "tts",
            times,
            {"language": language, "audio_bytes_last_run": audio_len},
        )

    def run_all(self):
        order = [
            ("speech_preprocess", self.bench_speech_preprocess),
            ("wad_energy", self.bench_wad),
            ("speech_encoder_mimi", self.bench_speech_encoder),
            ("language_detection_mms_lid", self.bench_language_detection),
            ("stt_whisper", self.bench_stt),
            ("emergency_detection", self.bench_emergency_detection),
            ("llm_ollama", self.bench_llm),
            ("tts", self.bench_tts),
        ]
        for name, fn in order:
            print(f"\n=== Benchmark: {name} ===")
            try:
                fn()
            except Exception as e:
                self._save_failure(name, e)

        out_path = Path(self.cfg.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        print(f"\nSaved results JSON: {out_path}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Component latency benchmark.")
    parser.add_argument(
        "--audio",
        default="sample.wav",
        help="Input audio file used for speech/WAD/LID/STT/emergency tests.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per component.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per component.")
    parser.add_argument("--language", default="en", help="Language code for STT/TTS.")
    parser.add_argument(
        "--prompt",
        default="Caller says there is smoke in the building. Give a short safety-first response.",
        help="Prompt used for LLM benchmark.",
    )
    parser.add_argument(
        "--tts-text",
        default="Emergency services have been alerted. Please stay calm and move to a safe area.",
        help="Text used for TTS benchmark.",
    )
    parser.add_argument(
        "--out",
        default=f"benchmark_outputs/component_latency_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--llm-fallback-model",
        default="distilgpt2",
        help="Transformers model used when Ollama is unavailable.",
    )
    args = parser.parse_args()

    return RunConfig(
        audio_path=args.audio,
        runs=max(1, args.runs),
        warmup=max(0, args.warmup),
        language=args.language,
        prompt=args.prompt,
        tts_text=args.tts_text,
        output_json=args.out,
        llm_fallback_model=args.llm_fallback_model,
    )


def main():
    cfg = parse_args()
    print("Component Latency Benchmark")
    print(f"audio={cfg.audio_path} runs={cfg.runs} warmup={cfg.warmup} language={cfg.language}")
    bench = ComponentBench(cfg)
    bench.run_all()


if __name__ == "__main__":
    main()
