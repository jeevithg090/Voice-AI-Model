# Piper Tamil ROCm 7.1 Training (IITM Female Mono)

This repo contains a single script to train Piper Tamil on ROCm 7.1 (Ubuntu 24.04), using:
- `IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip`
- `IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip`

The script auto-downloads `tamil_female_mono.zip` from Google Drive using `curl` + confirm token (`confirm` + `uuid`), then extracts only required phase zips.
If an existing `tamil_female_mono.zip` is corrupted/partial, the script automatically deletes it and re-downloads.
The script also auto-waits/retries if `apt/dpkg` is temporarily locked (for example by `unattended-upgrades`).

## One Command (on your server console)

```bash
bash piper_ta_rocm71_train.sh
```

## Fresh Server Setup

```bash
git clone https://github.com/jeevithg090/FInal-Voice.git
cd FInal-Voice
bash piper_ta_rocm71_train.sh
```

## Important Env Vars

- `WORK_ROOT` (default: `/scratch/piper_tamil` if available, otherwise `$HOME/piper_tamil`)
- `DATA_ZIP_DIR` (default: `$WORK_ROOT/datasets/tamil_female_mono`)
- `MAX_STEPS` (default: `120000`, treated as additional steps when base checkpoint step is larger)
- `BATCH_SIZE` (default: `24`)
- `PRECISION` (default: `16-mixed`, alternatives: `bf16-mixed`, `32-true`)
- `PIPER_REF` (default: `v1.4.1`)
- `TORCH_VERSION` (default: `2.10.0+rocm7.1`)
- `VAL_CHECK_INTERVAL` (default: `50`)
- `INCLUDE_ADDITIONAL` (`0` default, set `1` to include Phase2 Additional_data)

Google Drive URL is hard-coded in the script:

`https://drive.usercontent.google.com/download?id=1lTi5J7iX2dlo8GB16OhnDdqXlJyfuHjI&export=download&authuser=0`

Example:

```bash
MAX_STEPS=60000 BATCH_SIZE=40 VAL_CHECK_INTERVAL=50 bash piper_ta_rocm71_train.sh
```

## Outputs

Final artifacts are written to:

- `$WORK_ROOT/runs/ta_IN-iitm-female-s1-medium/`

Expected final files:

- `ta_IN-iitm-female-s1-medium.onnx`
- `ta_IN-iitm-female-s1-medium.onnx.json`
- `sample.wav`
