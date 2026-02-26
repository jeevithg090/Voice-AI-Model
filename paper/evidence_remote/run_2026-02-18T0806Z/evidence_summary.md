# Solvathon Layer 1 Evidence Snapshot

- Generated at (UTC): `2026-02-18T08:06:38.076787Z`
- Measurement date: `2026-02-18`

## LID Timing (Notebook-Extracted)

| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|
| 7 | 274.29 | 209.00 | 64.00 | 587.00 | 587.00 |

Raw timings (ms): `[193, 209, 587, 64, 482, 165, 220]`

## Live LLM Latency (Ollama)

| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|
| 0 | - | - | - | - | - |

| prompt | ok | latency_ms | response_chars |
|---|---|---:|---:|

## Connectivity and Reliability

| source | key findings |
|---|---|
| webrtc_live_test.log | offer_status=200, answer_type=answer, ice_final=completed, connected=True |
| websocket_live_test_v2.log | connected=True, recv_media_events=627, ws_closed_cleanly=True |
| test_pipeline.py live probe | exit_code=1, passed=4/6 |

## Voice-Turn Examples

| file | language | transcript_chars | response_chars | decoded_audio_bytes | input_duration_s | output_duration_s |
|---|---|---:|---:|---:|---:|---:|
| voice_turn_en_1.json | en | 60 | 320 | 2310380 | 4.69 | 24.07 |
| voice_turn_hi_1.json | hi | 9 | 47 | 447468 | 2.05 | 4.66 |

## Environment Flags

- Redis MISCONF detected: `False`
- `data/splits.json` exists: `False`

