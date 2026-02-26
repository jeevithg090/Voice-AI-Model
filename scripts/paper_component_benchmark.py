#!/usr/bin/env python3
"""
Paper-aligned component benchmark (GPU-oriented).

Benchmarks:
- Speech preprocessing
- WAD (energy-based)
- Speech encoder (Mimi)
- Language detection (MMS-LID)
- STT (Whisper)
- Emergency detection (Mimi + classifier)
- TTS by language (en, hi, te, ta, kn) with Piper/Edge fallback visibility
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    MimiModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForSequenceClassification,
)

from src.config import settings
from src.models.emergency_classifier import MimiEmergencyClassifier
from src.tts.tts_manager import TTSManager


LANGUAGES = ["en", "hi", "te", "ta", "kn"]
TTS_TEXT = {
    "en": "Emergency services have been informed. Please remain calm.",
    "hi": "आपातकालीन सहायता को सूचित कर दिया गया है। कृपया शांत रहें।",
    "te": "అత్యవసర సేవలకు సమాచారం ఇచ్చాము. దయచేసి ప్రశాంతంగా ఉండండి.",
    "ta": "அவசர சேவைகளுக்கு தகவல் அளிக்கப்பட்டுள்ளது. தயவுசெய்து அமைதியாக இருங்கள்.",
    "kn": "ತುರ್ತು ಸೇವೆಗೆ ಮಾಹಿತಿ ನೀಡಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಶಾಂತವಾಗಿರಿ.",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    audio: str
    runs: int
    warmup: int
    whisper_lang: str
    out: str
    sr_in: int = 48000


class Runner:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.device = pick_device()
        audio, _ = librosa.load(cfg.audio, sr=cfg.sr_in, mono=True)
        self.audio_48k = audio.astype(np.float32)
        if self.audio_48k.size == 0:
            raise ValueError("decoded input audio is empty")

        self.results: Dict[str, Any] = {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "audio_path": cfg.audio,
                "audio_seconds": float(self.audio_48k.shape[0] / cfg.sr_in),
                "runs": cfg.runs,
                "warmup": cfg.warmup,
                "device": self.device,
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "models": {
                    "whisper": settings.WHISPER_MODEL,
                    "lid": settings.LID_MODEL,
                    "emergency_classifier": settings.EMERGENCY_CLASSIFIER_PATH,
                    "tts_backend": settings.TTS_BACKEND,
                    "piper_models_dir": settings.PIPER_MODELS_DIR,
                },
            },
            "components": {},
            "tts_by_language": {},
        }

    def _measure(self, fn, warmup: Optional[int] = None, runs: Optional[int] = None):
        w = self.cfg.warmup if warmup is None else warmup
        r = self.cfg.runs if runs is None else runs
        last = None
        for _ in range(w):
            last = fn()
        times = []
        for _ in range(r):
            t0 = time.perf_counter()
            last = fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times, last

    def bench_speech_preprocess(self):
        def _fn():
            x = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000).astype(np.float32)
            x = np.clip(x * 2.0, -1.0, 1.0)
            return x.shape[0]

        times, out = self._measure(_fn)
        self.results["components"]["speech_preprocess"] = {
            "latency_ms": summarize(times),
            "details": {"output_samples_16k": int(out)},
        }

    def bench_wad(self):
        threshold = settings.SILENCE_THRESHOLD
        frame = int(0.02 * 48000)

        def _fn():
            speech = 0
            for i in range(0, len(self.audio_48k), frame):
                seg = self.audio_48k[i : i + frame]
                if seg.size == 0:
                    continue
                if float(np.mean(np.abs(seg))) > threshold:
                    speech += 1
            return speech

        times, speech_frames = self._measure(_fn)
        self.results["components"]["wad_energy"] = {
            "latency_ms": summarize(times),
            "details": {"threshold": threshold, "frame_ms": 20, "speech_frames": int(speech_frames)},
        }

    def bench_mimi_and_emergency(self):
        fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        mimi = MimiModel.from_pretrained("kyutai/mimi").to(self.device).eval()
        audio_24k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=fe.sampling_rate)

        def _mimi():
            x = fe(raw_audio=audio_24k, sampling_rate=fe.sampling_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                codes = mimi.encode(x["input_values"]).audio_codes
            return tuple(codes.shape), codes

        mimi_times, out = self._measure(lambda: _mimi())
        shape, codes = out
        self.results["components"]["speech_encoder_mimi"] = {
            "latency_ms": summarize(mimi_times),
            "details": {"audio_codes_shape": list(shape)},
        }

        clf = MimiEmergencyClassifier().to(self.device)
        clf.load_state_dict(torch.load(settings.EMERGENCY_CLASSIFIER_PATH, map_location=self.device))
        clf.eval()

        def _emg():
            x = fe(raw_audio=audio_24k, sampling_rate=fe.sampling_rate, return_tensors="pt").to(self.device)
            with torch.no_grad():
                c = mimi.encode(x["input_values"]).audio_codes
                p = float(clf(c).item())
            return p

        emg_times, prob = self._measure(_emg)
        self.results["components"]["emergency_detection"] = {
            "latency_ms": summarize(emg_times),
            "details": {"emergency_probability": round(prob, 6)},
        }

    def bench_lid(self):
        proc = AutoFeatureExtractor.from_pretrained(settings.LID_MODEL)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.LID_MODEL).to(self.device).eval()
        audio_16k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000)

        def _fn():
            x = proc(audio_16k, sampling_rate=16000, return_tensors="pt")
            x = {k: v.to(self.device) for k, v in x.items()}
            with torch.no_grad():
                probs = torch.softmax(model(**x).logits, dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            return model.config.id2label[idx], float(probs[idx].item())

        times, pred = self._measure(_fn)
        code, conf = pred
        self.results["components"]["language_detection_mms_lid"] = {
            "latency_ms": summarize(times),
            "details": {"top_code": code, "confidence": round(conf, 6)},
        }

    def bench_stt(self):
        proc = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL).to(self.device).eval()
        audio_16k = librosa.resample(self.audio_48k, orig_sr=48000, target_sr=16000).astype(np.float32)
        audio_16k = np.clip(audio_16k * 2.0, -1.0, 1.0)
        try:
            forced = proc.get_decoder_prompt_ids(language=self.cfg.whisper_lang, task="transcribe")
        except Exception:
            forced = None

        def _fn():
            x = proc(audio_16k, sampling_rate=16000, return_tensors="pt")
            kwargs = {"forced_decoder_ids": forced} if forced else {}
            with torch.no_grad():
                y = model.generate(x.input_features.to(self.device), **kwargs)
            txt = proc.batch_decode(y, skip_special_tokens=True)[0].strip()
            return txt

        times, txt = self._measure(_fn)
        self.results["components"]["stt_whisper"] = {
            "latency_ms": summarize(times),
            "details": {"transcript_preview": txt[:200]},
        }

    def bench_tts_languages(self):
        mgr = TTSManager()
        loop_text = {k: TTS_TEXT[k] for k in LANGUAGES}

        async def _speak(text: str, lang: str):
            return await mgr.speak(text, lang)

        for lang in LANGUAGES:
            text = loop_text[lang]
            expected_engine = mgr.get_engine(lang)
            piper_model = mgr._pick_piper_model(lang) if expected_engine == "piper" else None
            routed_engine = "piper" if (expected_engine == "piper" and piper_model) else "edge"
            err = None

            try:
                for _ in range(self.cfg.warmup):
                    asyncio.run(_speak(text, lang))
                times = []
                size = 0
                for _ in range(self.cfg.runs):
                    t0 = time.perf_counter()
                    out = asyncio.run(_speak(text, lang)) or b""
                    times.append((time.perf_counter() - t0) * 1000.0)
                    size = len(out)
                self.results["tts_by_language"][lang] = {
                    "latency_ms": summarize(times),
                    "details": {
                        "expected_engine": expected_engine,
                        "routed_engine": routed_engine,
                        "piper_model_present": bool(piper_model),
                        "audio_bytes_last_run": size,
                    },
                }
            except Exception as e:
                err = str(e)
                self.results["tts_by_language"][lang] = {
                    "error": err,
                    "details": {
                        "expected_engine": expected_engine,
                        "routed_engine": routed_engine,
                        "piper_model_present": bool(piper_model),
                    },
                }

            if err:
                print(f"[FAIL] tts_{lang}: {err}")
            else:
                s = self.results["tts_by_language"][lang]["latency_ms"]
                print(f"[OK] tts_{lang}: avg={s['avg_ms']:.2f} min={s['min_ms']:.2f} p95={s['p95_ms']:.2f} max={s['max_ms']:.2f}")

    def run(self):
        self.bench_speech_preprocess()
        self.bench_wad()
        self.bench_mimi_and_emergency()
        self.bench_lid()
        self.bench_stt()
        self.bench_tts_languages()
        out = Path(self.cfg.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        print(f"saved={out}")


def parse() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="sample.wav")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--whisper-lang", default="en")
    p.add_argument(
        "--out",
        default=f"benchmark_outputs/paper_component_latency_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    a = p.parse_args()
    return Cfg(audio=a.audio, runs=max(1, a.runs), warmup=max(0, a.warmup), whisper_lang=a.whisper_lang, out=a.out)


if __name__ == "__main__":
    cfg = parse()
    print(f"audio={cfg.audio} runs={cfg.runs} warmup={cfg.warmup} device={pick_device()}")
    Runner(cfg).run()
