# Paper Tables: Measured GPU Validation Report

- Run folder: `/Users/jeevithg/Downloads/solvathon_layer1-main 3/paper/evidence_remote/run_2026-02-22T0530Z`
- Component benchmark timestamp: `2026-02-22T05:16:55.452613+00:00`
- LLM chunk benchmark timestamp: `2026-02-22T05:18:21.249646+00:00`
- GPU mode: `cuda` (`torch 2.10.0+rocm7.1`)

## 1) Paper Approximation vs Measured Latency

| Stage (from paper tables) | Paper Approx (ms) | Measured Avg (ms) | Status | Measurement Note |
|---|---:|---:|---|---|
| Voice Activity Detection (WAD) | 8-15 | 0.92 | Measured | Measured via energy endpointing |
| Language Identification | 9-16 | 21.18 | Measured | MMS-LID GPU inference |
| ASR (Whisper-small) | 40 | 99.63 | Measured | Whisper-small on GPU |
| Semantic Cache Lookup | 3-8 | 1.04 | Measured | FAISS IndexFlatIP lookup (10k entries) |
| LLM First-Token Generation | 12-18 | 176.83 | Measured | Ollama stream TTFT |
| LLM Full Response (~25 tokens) | 30 | 368.90 | Measured | Ollama stream total |
| TTS First Chunk | 20-30 | 188.25 | Measured | Proxy: LLM chunk-4 first chunk availability |
| Full TTS Synthesis (English) | 50 | 251.04 | Measured | TTSManager routed to Edge (no English Piper model) |
| Audio Transmission (WebRTC) | 5-15 | - | Not isolated | Not isolated in this offline benchmark run |

| End-to-End | Paper Approx (ms) | Measured Avg (ms) | Formula |
|---|---:|---:|---|
| Speech-to-speech proxy | 140 | 742.39 | ASR + LLM(total) + TTS(en) + preprocess+WAD+LID |

## 2) Component-by-Component GPU Results

| Component | Avg (ms) | Min (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| speech_preprocess | 0.74 | 0.71 | 0.75 | 0.75 |
| wad_energy | 0.92 | 0.91 | 0.94 | 0.94 |
| speech_encoder_mimi | 14.74 | 14.06 | 15.74 | 15.91 |
| language_detection_mms_lid | 21.18 | 20.30 | 22.30 | 22.47 |
| stt_whisper | 99.63 | 98.08 | 102.10 | 102.53 |
| emergency_detection | 15.16 | 15.08 | 15.22 | 15.22 |

## 3) Multilingual TTS (Piper/Edge Fallback Verification)

| Language | Routed Engine | Piper Model Present | Avg (ms) | Min (ms) | P95 (ms) | Max (ms) |
|---|---|---|---:|---:|---:|---:|
| en | edge | False | 251.04 | 241.01 | 257.46 | 257.81 |
| hi | edge | False | 509.51 | 318.41 | 609.45 | 610.54 |
| te | edge | False | 374.22 | 248.81 | 546.08 | 573.37 |
| ta | piper | True | 1623.41 | 1487.58 | 1791.26 | 1816.24 |
| kn | edge | False | 317.85 | 213.80 | 448.98 | 468.76 |

## 4) LLM Chunking Latency (Ollama on GPU)

- Base TTFT avg: `176.83 ms`
- Base total avg: `368.90 ms`

| Chunk Size (token events) | First Chunk Avg (ms) | Avg Inter-Chunk (ms) | Avg Chunk Count |
|---:|---:|---:|---:|
| 1 | 176.83 | 3.98 | 48.20 |
| 4 | 188.25 | 15.20 | 12.60 |
| 8 | 203.81 | 28.83 | 6.60 |
| 16 | 235.41 | 51.12 | 3.60 |

## 5) Proof Artifacts (with SHA-256)

- `paper_component_latency_gpu.json`
  - sha256: `11a2689e9d321aeb34eae82f240ba741f7b30d3b032019537167288a0ac330b7`
- `ollama_chunk_latency_host.json`
  - sha256: `4909c5463bc0888eae90425f7cd49fa345c90d7df159b1b1106a40011d1fdf5e`
- `semantic_cache_faiss_lookup_gpu.json`
  - sha256: `28bc8c733e77d56cfda751d32cd41b83de91acb602389c64eb1efd8e457344d2`
- `ollama_gpu_offload_log.txt`
  - sha256: `9a660bf0a27d442bfe40f41fd676aa9bb6d23e802d7e54b6ce46c792a14ec8e3`
- `rocm_smi.txt`
  - sha256: `dc29e44c774af5bbf823c79dee64fb52d8e724836412047cd3d3a93e137f5b80`

## 6) Validation Notes

- LLM metrics are measured on host Ollama (`127.0.0.1:11434`) with GPU offload verified via systemd logs.
- TTS fallback is validated: English/Hindi/Telugu/Kannada routed to Edge because Piper models were absent; Tamil routed to Piper because model existed.
- Audio transmission latency (WebRTC network leg) was not isolated in this run; it requires a live browser/telephony end-to-end timing harness.