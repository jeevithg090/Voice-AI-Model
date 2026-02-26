# Piper Tamil Project Documentation (Technical)

## 1. Scope

This repository is an automation wrapper for training a Tamil single-speaker Piper model on ROCm 7.1 with a reproducible shell pipeline. The core logic lives in:

- `/Users/jeevithg/Downloads/Piper Tamil/piper_ta_rocm71_train.sh` (dispatcher)
- `/Users/jeevithg/Downloads/Piper Tamil/FInal-Voice/piper_ta_rocm71_train.sh` (canonical pipeline)

The pipeline covers:

1. host bootstrapping (apt packages, locks, storage mount checks)
2. dataset acquisition + integrity validation
3. Piper source checkout and Python environment build
4. transcript/audio materialization and metadata generation
5. training, checkpoint resolution, ONNX export, and audio sanity sample

## 2. Repository Topology

```text
/Users/jeevithg/Downloads/Piper Tamil/
├── piper_ta_rocm71_train.sh                    # Thin launcher
├── PROJECT_DOCUMENTATION.md                    # This file
├── FInal-Voice/
│   ├── README.md
│   ├── piper_ta_rocm71_train.sh                # Canonical training pipeline
│   └── .gitignore
└── tamil_female_mono/
    ├── IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip
    └── IndicTTS_Phase3_Tamil_fem_Speaker2_mono.zip
```

Notes:

- Git metadata is present under `/Users/jeevithg/Downloads/Piper Tamil/FInal-Voice/.git`.
- The workspace root itself is not a git repository.

## 3. Execution Entry Points

### 3.1 Root dispatcher

`/Users/jeevithg/Downloads/Piper Tamil/piper_ta_rocm71_train.sh`

Behavior:

- resolves `SCRIPT_DIR`
- resolves `TARGET_SCRIPT="${SCRIPT_DIR}/FInal-Voice/piper_ta_rocm71_train.sh"`
- hard-fails if target script is absent
- delegates using `exec bash "${TARGET_SCRIPT}" "$@"`

This process replacement ensures PID/exit code represent the canonical script directly.

### 3.2 Canonical trainer

`/Users/jeevithg/Downloads/Piper Tamil/FInal-Voice/piper_ta_rocm71_train.sh`

Shell runtime guarantees:

- `set -euo pipefail`
- non-zero on any uncaught failing command
- undefined variable usage is fatal
- pipeline failures are propagated

## 4. Runtime Control Flow

High-level sequence:

1. resolve defaults and env overrides
2. detect root/sudo mode
3. optionally configure `/scratch` via `LABEL=DOSCRATCH`
4. finalize derived paths (`WORK_ROOT`, `DATA_ZIP_DIR`, `DATASET_ARCHIVE`, `PIPER_REPO_DIR`)
5. install host dependencies with apt lock retry
6. validate scalar parameters (`MAX_STEPS`) and `ESPEAK_VOICE` availability
7. ensure dataset archive exists and passes `unzip -tqq`
8. extract required phase zips from parent archive
9. clone/fetch/checkout Piper repo
10. create venv and install build/train dependencies
11. compile monotonic align and C extensions
12. execute ROCm CUDA visibility probe via Python
13. build training corpus (wav extraction + `metadata.csv`)
14. resolve base checkpoint and max-step policy
15. run `python -m piper.train fit`
16. detect new checkpoint generated during this run
17. export ONNX and synthesize Tamil sample utterance

## 5. Configuration Contract

All variables are sourced from environment with script defaults:

| Variable | Type | Default | Validation | Internal Consumer |
|---|---|---|---|---|
| `WORK_ROOT` | path | `/scratch/piper_tamil` or `$HOME/piper_tamil` | fallback if `/scratch` not mounted | all outputs |
| `DATA_ZIP_DIR` | path | `$WORK_ROOT/datasets/tamil_female_mono` | none | dataset ingress |
| `DATASET_ARCHIVE` | path | `$DATA_ZIP_DIR/tamil_female_mono.zip` | must be valid zip | download/extract |
| `PIPER_REPO_DIR` | path | `$WORK_ROOT/piper1-gpl` | git dir created if absent | source build |
| `PIPER_REF` | git ref | `v1.4.1` | checkout must resolve | version pin |
| `TORCH_VERSION` | version str | `2.10.0+rocm7.1` | pip resolver | training runtime |
| `VOICE_NAME` | string | `ta_IN_iitm_female_s1` | none | `--data.voice_name` |
| `MODEL_BASENAME` | string | `ta_IN-iitm-female-s1-medium` | none | run artifact naming |
| `ESPEAK_VOICE` | code | `ta` | must exist in `espeak-ng --voices` | phonemization |
| `INCLUDE_ADDITIONAL` | `0/1` | `0` | parsed by Python bool compare | optional corpus branch |
| `BATCH_SIZE` | int | `24` | implicit trainer validation | `--data.batch_size` |
| `MAX_STEPS` | int | `120000` | strictly positive integer | `--trainer.max_steps` policy |
| `VAL_SPLIT` | float | `0.05` | implicit trainer validation | validation split |
| `NUM_TEST_EXAMPLES` | int | `16` | implicit trainer validation | eval examples |
| `PRECISION` | enum str | `16-mixed` | trainer accepts value | precision mode |
| `VAL_CHECK_INTERVAL` | int | `50` | implicit trainer validation | validation cadence |
| `BASE_CKPT_URL` | URL | default HF checkpoint | downloadable and readable | resume source |
| `DATASET_BASE_URL` | URL | hard-coded Google Drive endpoint | HTML token parse must succeed | archive download |

## 6. Dependency Resolution

### 6.1 System packages

Installed via apt:

- `git`
- `build-essential`
- `cmake`
- `ninja-build`
- `python3-venv`
- `ffmpeg`
- `libsndfile1`
- `unzip`
- `rsync`
- `curl`
- `tmux`
- `espeak-ng`

Lock strategy:

- lock files polled every 5s using `fuser`
- max wait: 1800s
- apt commands retried up to 5 attempts with 10s backoff

### 6.2 Python packages

Installed in `PIPER_REPO_DIR/.venv`:

1. `pip/setuptools/wheel` upgrade
2. `torch==${TORCH_VERSION}` from ROCm 7.1 index
3. `pip install -e '.[train]'` in Piper repo
4. `onnxscript`

Build steps:

- `./build_monotonic_align.sh`
- `python setup.py build_ext --inplace`

## 7. Dataset Ingestion Internals

### 7.1 Archive acquisition

Function `download_gdrive_archive`:

- obtains HTML via `curl -L -sS "${DATASET_BASE_URL}"`
- extracts `confirm` + `uuid` with `sed` capture groups
- builds final URL: `${base_url}&confirm=${confirm}&uuid=${uuid}`
- resumes downloads with `curl -C -`
- validates zip via `unzip -tqq`

If existing archive is invalid, it is deleted and re-downloaded.

### 7.2 Required embedded archives

From parent archive:

- `IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip`
- `IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip`

Extraction mode:

- `unzip -j -o` to `DATA_ZIP_DIR`
- ignores folder hierarchy (junk paths)

### 7.3 Transcript parsing + alignment

Inline Python parser:

- regex: `^\(\s*([^\s]+)\s+"(.*)"\s*\)\s*$`
- expected format: `(utt_id "transcript")`
- UTF-8 BOM tolerant decode (`utf-8-sig`)
- transcript normalization:
  - collapse whitespace
  - trim
  - replace `|` with space

Data jobs:

1. `phase2`: `mono/Read_speech/txt.done.data` + wavs in `mono/Read_speech/wav/`
2. `phase3_s1`: `speaker1/txt.done.data` + wavs in `speaker1/wav/`
3. optional `phase2_additional` when `INCLUDE_ADDITIONAL=1`

Output materialization:

- wavs extracted into `$WORK_ROOT/data_ta_single_speaker/audio/<job>/`
- metadata rows written as `relative_wav_path|text` with `|` delimiter
- metadata sorted lexicographically by path

Hard quality gates:

- each enabled job must produce at least one aligned row
- total rows must be `>= 3000`

## 8. Training and Checkpoint Semantics

Training invocation:

```bash
python -m piper.train fit \
  --data.voice_name "${VOICE_NAME}" \
  --data.csv_path "${META_PATH}" \
  --data.audio_dir "${AUDIO_DIR}" \
  --data.cache_dir "${RUN_ROOT}/cache" \
  --data.config_path "${RUN_ROOT}/${MODEL_BASENAME}.onnx.json" \
  --data.espeak_voice "${ESPEAK_VOICE}" \
  --model.sample_rate 22050 \
  --data.batch_size "${BATCH_SIZE}" \
  --data.validation_split "${VAL_SPLIT}" \
  --data.num_test_examples "${NUM_TEST_EXAMPLES}" \
  --ckpt_path "${BASE_CKPT}" \
  --trainer.default_root_dir "${RUN_ROOT}" \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision "${PRECISION}" \
  --trainer.max_steps "${EFFECTIVE_MAX_STEPS}" \
  --trainer.val_check_interval "${VAL_CHECK_INTERVAL}" \
  --trainer.log_every_n_steps 20
```

### 8.1 Effective max steps

The script treats `MAX_STEPS` as "additional steps" when the base checkpoint already has a larger global step.

Algorithm:

1. parse base step from `BASE_CKPT_URL` (`step%3DNNN` or `step=NNN`)
2. if `MAX_STEPS <= BASE_CKPT_STEP`:
   - `EFFECTIVE_MAX_STEPS = BASE_CKPT_STEP + MAX_STEPS`
3. else:
   - `EFFECTIVE_MAX_STEPS = MAX_STEPS`

This avoids immediate trainer termination when resuming from high-step checkpoints.

### 8.2 New checkpoint detection

To ensure the current run produced output:

- create marker: `${RUN_ROOT}/.train_started_<timestamp>`
- after training, list `*.ckpt` newer than marker (excluding `base.ckpt`)
- fail if none found
- choose newest checkpoint by mtime for export

## 9. Artifact Layout

Primary output root:

`$WORK_ROOT/runs/${MODEL_BASENAME}`

Expected files:

- `base.ckpt` (downloaded resume checkpoint)
- `train_<YYYYMMDD_HHMMSS>.log`
- `*.ckpt` (trainer-produced checkpoints)
- `${MODEL_BASENAME}.onnx`
- `${MODEL_BASENAME}.onnx.json`
- `sample.wav`

Dataset workspace:

`$WORK_ROOT/data_ta_single_speaker`

- `raw/` (copied phase zip files)
- `audio/` (extracted wav corpus)
- `metadata.csv`

## 10. Idempotency and Re-run Behavior

Safe to re-run with same parameters:

- apt installs are idempotent
- dataset archive reused if valid
- phase zip extraction skipped if already present
- piper repo clone skipped if git dir exists
- base checkpoint download skipped if file exists and non-empty

Not fully deterministic across reruns:

- training stochasticity (no explicit seed control in wrapper)
- newest checkpoint selection depends on runtime timing and produced files

## 11. Failure Modes

Non-exhaustive fatal conditions:

- apt lock timeout
- invalid/corrupt dataset archive after download
- inability to parse Google Drive confirm/uuid tokens
- missing required extracted phase zips
- invalid `MAX_STEPS`
- unavailable `ESPEAK_VOICE`
- ROCm GPU not visible to PyTorch
- insufficient aligned rows (`<3000`)
- no checkpoint generated by trainer in current run

## 12. Performance and Tuning Notes

Primary knobs:

- `BATCH_SIZE`: increase until near memory limit for throughput
- `PRECISION`: `16-mixed` is default; `bf16-mixed` may improve stability on compatible hardware
- `VAL_CHECK_INTERVAL`: larger values reduce validation overhead
- `MAX_STEPS`: controls training duration and total compute cost

I/O considerations:

- prefer fast local NVMe-backed `WORK_ROOT` (e.g., `/scratch`) for unzip and checkpoint writes
- dataset extraction currently copies full wav bytes from zip into filesystem before training

## 13. Extension Points

Potential technical extensions:

1. add Phase3 speaker2 path wiring in dataset job definitions
2. expose random seed and deterministic trainer flags
3. add explicit checkpoint selection strategy (best val metric vs newest mtime)
4. migrate Google Drive download logic to stable API or pre-signed object storage
5. add health checks for free disk, GPU VRAM, and expected dataset hours before training

## 14. Minimal Invocation Examples

Default:

```bash
bash /Users/jeevithg/Downloads/Piper\ Tamil/piper_ta_rocm71_train.sh
```

Custom run:

```bash
WORK_ROOT=/scratch/piper_tamil \
BATCH_SIZE=40 \
MAX_STEPS=60000 \
VAL_CHECK_INTERVAL=100 \
PRECISION=bf16-mixed \
bash /Users/jeevithg/Downloads/Piper\ Tamil/piper_ta_rocm71_train.sh
```

Post-run verification:

```bash
ls -lh "$WORK_ROOT/runs/ta_IN-iitm-female-s1-medium/"*.onnx*
```
