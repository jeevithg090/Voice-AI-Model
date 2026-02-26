# Solvathon Layer 1
## GPU Component Latency Validation Report

- Report date (UTC): `2026-02-22`
- Validation window (UTC): `04:58` to `05:06`
- Target host: `129.212.185.44` (DigitalOcean ROCm droplet)
- GPU: `AMD Instinct MI300X VF (gfx942)`

## 1. Objective

This report validates per-component latency on the GPU deployment for:

1. LLM
2. Speech preprocessing
3. WAD (energy-based voice activity detection)
4. Language detection
5. TTS
6. STT
7. Emergency detection

## 2. Test Environment

- OS: Ubuntu `24.04.3 LTS`
- ROCm stack: visible via `rocm-smi`
- Python runtime in test container: `3.12.3`
- Torch runtime: `2.10.0+rocm7.1`
- Whisper model: `openai/whisper-small`
- LID model: `facebook/mms-lid-256`
- LLM runtime: Ollama `0.16.3`, model `llama3.2:3b`

## 3. Methodology

- Speech pipeline benchmark executed inside `rocm` container with:
  - runs: `3`
  - warmup: `1`
  - input: `sample.wav` (`3.390125s`)
- Primary artifact: `component_latency_gpu.json`
- LLM latency was measured on host Ollama endpoint (`127.0.0.1:11434`) because the container network could not route to host-local Ollama.
- LLM measurement config:
  - endpoint: `/api/generate` (streaming)
  - model: `llama3.2:3b`
  - prompt: emergency safety response prompt
  - runs: `3` with warmup `1`

## 4. Latency Results

All values are in milliseconds.

| Component | Avg | Min | P95 | Max | Source |
|---|---:|---:|---:|---:|---|
| Speech preprocessing | 0.73 | 0.70 | 0.76 | 0.76 | `component_latency_gpu.json` |
| WAD (energy VAD) | 0.91 | 0.90 | 0.91 | 0.91 | `component_latency_gpu.json` |
| Speech encoder (Mimi) | 14.85 | 14.23 | 15.72 | 15.86 | `component_latency_gpu.json` |
| Language detection (MMS-LID) | 21.05 | 20.35 | 21.76 | 21.85 | `component_latency_gpu.json` |
| STT (Whisper) | 96.53 | 95.83 | 97.52 | 97.68 | `component_latency_gpu.json` |
| Emergency detection (Mimi + classifier) | 15.64 | 15.21 | 16.16 | 16.24 | `component_latency_gpu.json` |
| TTS | 277.99 | 256.02 | 309.26 | 314.33 | `component_latency_gpu.json` |
| LLM TTFT (Ollama stream) | 165.19 | 154.21 | 182.96 | 182.96 | `llm_ollama_latency_host.json` |
| LLM total (Ollama stream) | 333.26 | 302.14 | 390.88 | 390.88 | `llm_ollama_latency_host.json` |

## 5. Component-Specific Notes

- WAD:
  - threshold: `0.01`
  - frame size: `20ms`
  - speech frames detected: `110`
- Language detection:
  - top code: `tam`
  - normalized language: `ta`
  - confidence: `0.999`
- STT transcript preview:
  - `"This is Tamil Piper Sodhanai Kural."`
- Emergency detection:
  - probability on sample: `0.0` (non-emergency sample)
- TTS:
  - audio bytes generated (last run): `634732`

## 6. Proof of GPU Usage

### 6.1 ROCm Device Evidence

From `rocm_smi.txt`:

- `Card Series: AMD Instinct MI300X VF`
- `GFX Version: gfx942`
- `GPU Memory Allocated (VRAM%): 9`

### 6.2 Ollama GPU Offload Evidence

From `ollama_gpu_offload_log.txt`:

- `using device ROCm0 (AMD Instinct MI300X VF)`
- `offloaded 29/29 layers to GPU`
- `ROCm0 model buffer size = 1918.35 MiB`
- `ROCm0 KV buffer size = 14336.00 MiB`

### 6.3 Ollama Service and Model Evidence

From `ollama_status.txt`:

- service state: `active`
- version: `0.16.3`
- model available: `llama3.2:3b`

## 7. Artifact Integrity (SHA-256)

- `component_latency_gpu.json`
  - `5d24bb1d67323a9cfb877e057a2f69961760f5c422eb4fff1de4d246caa46ef1`
- `llm_ollama_latency_host.json`
  - `99fe31ec76c1357be09f1534857b939afa7a29fb0bfeaeff59c1512e476614be`
- `ollama_gpu_offload_log.txt`
  - `64f5c82ae13d0c924d7ce0cb29c15ebdba7b6ad958bc8eff80298d365db6742c`
- `ollama_status.txt`
  - `b94ee26d604576254d9378740cb8d188461c915c0f59dc50d7052a51989ac19f`
- `rocm_smi.txt`
  - `eabf4617fb2bad2fc71647860078e5377ffa45a3479615e6d040fa56fb9ed4bb`

## 8. Limitations and Interpretation

- Sample size is small (`n=3` measured runs per component), so P95 is directional, not statistically stable.
- LLM was measured from host process due container-to-host loopback isolation; this does not invalidate latency but should be noted for strict same-process benchmarking.
- Results are valid as a deployment validation snapshot for the tested hardware and model versions at the above UTC timestamps.

## 9. Conclusion

The GPU deployment successfully produced component-level latency measurements for all requested components. Evidence confirms that the LLM workload was executed with ROCm GPU layer offload, and the measured latencies are documented with reproducible artifacts and integrity hashes.
