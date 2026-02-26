# Solvathon Layer 1 Evidence Snapshot

- Generated at (UTC): `2026-02-18T07:30:09+00:00Z`
- Measurement date: `2026-02-18`

## LID Timing (Notebook-Extracted)

| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|
| 7 | 274.29 | 209.00 | 64.00 | 587.00 | 587.00 |

Raw timings (ms): `[193, 209, 587, 64, 482, 165, 220]`

## Live LLM Latency (Ollama)

| n | mean_ms | median_ms | min_ms | max_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|
| 5 | 16467.03 | 16880.32 | 8913.05 | 23440.92 | 23440.92 |

| prompt | ok | latency_ms | response_chars |
|---|---|---:|---:|
| Translate 'Hello, how are you?' to Hindi. | True | 18964.91 | 479 |
| Translate 'Hello, how are you?' to Kannada. | True | 8913.05 | 236 |
| Translate 'Hello, how are you?' to Telugu. | True | 14135.97 | 394 |
| Translate 'Hello, how are you?' to Tamil. | True | 16880.32 | 474 |
| Write a short sentence in Kannada. | True | 23440.92 | 94 |

## Connectivity and Reliability

| source | key findings |
|---|---|
| webrtc_live_test.log | offer_status=200, answer_type=answer, ice_final=completed, connected=True |
| websocket_live_test_v2.log | connected=True, recv_media_events=627, ws_closed_cleanly=True |
| test_pipeline.py live probe | exit_code=1, passed=0/6 |

## Voice-Turn Examples

| file | language | transcript_chars | response_chars | decoded_audio_bytes | input_duration_s | output_duration_s |
|---|---|---:|---:|---:|---:|---:|
| voice_turn_en_1.json | en | 60 | 320 | 2310380 | 4.69 | 24.07 |
| voice_turn_hi_1.json | hi | 9 | 47 | 447468 | 2.05 | 4.66 |

## Environment Flags

- Redis MISCONF detected: `True`
- `data/splits.json` exists: `False`

