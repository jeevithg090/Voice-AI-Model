#!/usr/bin/env bash
set -euo pipefail

# ---------------- Configuration ----------------
DEFAULT_WORK_ROOT="/scratch/piper_tamil"
if [[ ! -d /scratch ]]; then
  DEFAULT_WORK_ROOT="${HOME}/piper_tamil"
fi
WORK_ROOT="${WORK_ROOT:-${DEFAULT_WORK_ROOT}}"
DATA_ZIP_DIR="${DATA_ZIP_DIR:-}"
DATASET_ARCHIVE="${DATASET_ARCHIVE:-}"
DATASET_BASE_URL="https://drive.usercontent.google.com/download?id=1lTi5J7iX2dlo8GB16OhnDdqXlJyfuHjI&export=download&authuser=0"
PIPER_REPO_DIR="${PIPER_REPO_DIR:-}"
PIPER_REF="${PIPER_REF:-v1.4.1}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0+rocm7.1}"

VOICE_NAME="${VOICE_NAME:-ta_IN_iitm_female_s1}"
MODEL_BASENAME="${MODEL_BASENAME:-ta_IN-iitm-female-s1-medium}"
ESPEAK_VOICE="${ESPEAK_VOICE:-ta}"

INCLUDE_ADDITIONAL="${INCLUDE_ADDITIONAL:-0}" # 0=clean Tamil only, 1=include Phase2 Additional_data
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_STEPS="${MAX_STEPS:-120000}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
NUM_TEST_EXAMPLES="${NUM_TEST_EXAMPLES:-16}"
PRECISION="${PRECISION:-16-mixed}" # alternatives: bf16-mixed, 32-true
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"

BASE_CKPT_URL="${BASE_CKPT_URL:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/te/te_IN/maya/medium/epoch%3D3277-step%3D839202.ckpt}"

PHASE2_ZIP="IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip"
PHASE3_ZIP="IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

wait_for_apt_locks() {
  local waited=0
  local max_wait=1800
  while fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock >/dev/null 2>&1; do
    if (( waited == 0 )); then
      echo "Waiting for apt/dpkg lock (likely unattended-upgrades) ..."
    fi
    sleep 5
    waited=$((waited + 5))
    if (( waited >= max_wait )); then
      echo "ERROR: Timed out waiting for apt/dpkg locks."
      return 1
    fi
  done
}

run_apt_cmd() {
  local attempt
  for attempt in 1 2 3 4 5; do
    wait_for_apt_locks || return 1
    if "$@"; then
      return 0
    fi
    echo "APT command failed (attempt ${attempt}/5), retrying in 10s ..."
    sleep 10
  done
  echo "ERROR: APT command failed after retries: $*"
  return 1
}

is_valid_zip_archive() {
  local archive_path="$1"
  [[ -s "${archive_path}" ]] || return 1
  unzip -tqq "${archive_path}" >/dev/null 2>&1 || return 1
  return 0
}

download_gdrive_archive() {
  mkdir -p "${DATA_ZIP_DIR}"

  if [[ -s "${DATASET_ARCHIVE}" ]]; then
    if is_valid_zip_archive "${DATASET_ARCHIVE}"; then
      echo "Using existing dataset archive: ${DATASET_ARCHIVE}"
      return 0
    fi
    echo "Existing dataset archive is invalid/corrupted, re-downloading: ${DATASET_ARCHIVE}"
    rm -f "${DATASET_ARCHIVE}"
  fi

  local base_url
  local html
  local confirm
  local uuid
  local download_url

  base_url="${DATASET_BASE_URL}"
  html="$(curl -L -sS "${base_url}")"

  confirm="$(printf "%s" "${html}" | sed -n 's/.*name="confirm" value="\([^"]*\)".*/\1/p' | head -n1)"
  uuid="$(printf "%s" "${html}" | sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' | head -n1)"

  if [[ -z "${confirm}" || -z "${uuid}" ]]; then
    echo "ERROR: Could not extract Google Drive confirm token/uuid."
    echo "Set DATASET_ARCHIVE manually or verify DATASET_BASE_URL."
    return 1
  fi

  download_url="${base_url}&confirm=${confirm}&uuid=${uuid}"
  echo "Downloading dataset archive from hard-coded Google Drive URL ..."
  rm -f "${DATASET_ARCHIVE}"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "${DATASET_ARCHIVE}" "${download_url}"

  if ! is_valid_zip_archive "${DATASET_ARCHIVE}"; then
    echo "ERROR: Downloaded file is not a valid ZIP archive: ${DATASET_ARCHIVE}"
    return 1
  fi
}

extract_required_phase_archives() {
  local need_extract=0
  if [[ ! -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" ]]; then
    need_extract=1
  fi
  if [[ ! -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    need_extract=1
  fi

  if [[ "${need_extract}" -eq 0 ]]; then
    echo "Required phase zips already exist in ${DATA_ZIP_DIR}"
    return 0
  fi

  echo "Extracting required phase archives from ${DATASET_ARCHIVE} ..."
  unzip -j -o "${DATASET_ARCHIVE}" "${PHASE2_ZIP}" "${PHASE3_ZIP}" -d "${DATA_ZIP_DIR}" >/dev/null

  if [[ ! -s "${DATA_ZIP_DIR}/${PHASE2_ZIP}" || ! -s "${DATA_ZIP_DIR}/${PHASE3_ZIP}" ]]; then
    echo "ERROR: Failed to extract required phase archives."
    return 1
  fi
}

# ---------------- Scratch disk setup ----------------
if lsblk -o LABEL | grep -q 'DOSCRATCH'; then
  ${SUDO} mkdir -p /scratch
  if ! grep -q 'LABEL=DOSCRATCH' /etc/fstab; then
    echo 'LABEL=DOSCRATCH /scratch ext4 discard,errors=remount-ro 0 2' | ${SUDO} tee -a /etc/fstab >/dev/null
  fi
  ${SUDO} mount /scratch || true
fi

if [[ "${WORK_ROOT}" == /scratch/* ]] && ! mountpoint -q /scratch; then
  WORK_ROOT="${HOME}/piper_tamil"
fi

# Resolve derived paths after WORK_ROOT is finalized.
if [[ -z "${DATA_ZIP_DIR}" ]]; then
  DATA_ZIP_DIR="${WORK_ROOT}/datasets/tamil_female_mono"
fi
if [[ -z "${DATASET_ARCHIVE}" ]]; then
  DATASET_ARCHIVE="${DATA_ZIP_DIR}/tamil_female_mono.zip"
fi
if [[ -z "${PIPER_REPO_DIR}" ]]; then
  PIPER_REPO_DIR="${WORK_ROOT}/piper1-gpl"
fi

# ---------------- System packages ----------------
if [[ -n "${SUDO}" ]]; then
  run_apt_cmd ${SUDO} apt-get update
  run_apt_cmd ${SUDO} env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git build-essential cmake ninja-build python3-venv ffmpeg libsndfile1 unzip rsync curl tmux espeak-ng
else
  run_apt_cmd apt-get update
  run_apt_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git build-essential cmake ninja-build python3-venv ffmpeg libsndfile1 unzip rsync curl tmux espeak-ng
fi

if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]] || (( MAX_STEPS <= 0 )); then
  echo "ERROR: MAX_STEPS must be a positive integer, got: ${MAX_STEPS}"
  exit 1
fi

if ! espeak-ng --voices | awk 'NR>1 {print $2}' | grep -Fxq "${ESPEAK_VOICE}"; then
  echo "ERROR: espeak voice '${ESPEAK_VOICE}' not found."
  echo "Run 'espeak-ng --voices' and set ESPEAK_VOICE to a valid language code."
  exit 1
fi

mkdir -p "${WORK_ROOT}"

# ---------------- Dataset download + extraction ----------------
download_gdrive_archive
extract_required_phase_archives

# ---------------- Piper source + Python env ----------------
if [[ ! -d "${PIPER_REPO_DIR}/.git" ]]; then
  git clone https://github.com/OHF-Voice/piper1-gpl.git "${PIPER_REPO_DIR}"
fi

cd "${PIPER_REPO_DIR}"
git fetch --tags --force
git checkout "${PIPER_REF}"
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.1 "torch==${TORCH_VERSION}"
python -m pip install -e '.[train]'
python -m pip install onnxscript

./build_monotonic_align.sh
python setup.py build_ext --inplace

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("gpu visible:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: ROCm GPU is not visible to PyTorch.")
print("device:", torch.cuda.get_device_name(0))
PY

# ---------------- Dataset preparation ----------------
DATA_ROOT="${WORK_ROOT}/data_ta_single_speaker"
RAW_DIR="${DATA_ROOT}/raw"
AUDIO_DIR="${DATA_ROOT}/audio"
META_PATH="${DATA_ROOT}/metadata.csv"

mkdir -p "${RAW_DIR}" "${AUDIO_DIR}"

for f in "${PHASE2_ZIP}" "${PHASE3_ZIP}"; do
  if [[ ! -f "${DATA_ZIP_DIR}/${f}" ]]; then
    echo "ERROR: Missing ${DATA_ZIP_DIR}/${f}"
    exit 1
  fi
  rsync -a "${DATA_ZIP_DIR}/${f}" "${RAW_DIR}/${f}"
done

export RAW_DIR AUDIO_DIR META_PATH INCLUDE_ADDITIONAL PHASE2_ZIP PHASE3_ZIP
python - <<'PY'
import csv
import os
import re
import wave
import zipfile
from pathlib import Path

raw_dir = Path(os.environ["RAW_DIR"])
audio_dir = Path(os.environ["AUDIO_DIR"])
meta_path = Path(os.environ["META_PATH"])
include_additional = os.environ.get("INCLUDE_ADDITIONAL", "0") == "1"
phase2_zip = raw_dir / os.environ["PHASE2_ZIP"]
phase3_zip = raw_dir / os.environ["PHASE3_ZIP"]

pat = re.compile(r'^\(\s*([^\s]+)\s+"(.*)"\s*\)\s*$')


def parse_txt_done(zf, member):
    out = {}
    text = zf.read(member).decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        utt, txt = m.group(1), m.group(2)
        txt = re.sub(r"\s+", " ", txt).strip().replace("|", " ")
        if txt:
            out[utt] = txt
    return out


jobs = [
    (phase2_zip, "mono/Read_speech/txt.done.data", "mono/Read_speech/wav/", "phase2"),
    (phase3_zip, "speaker1/txt.done.data", "speaker1/wav/", "phase3_s1"),
]
if include_additional:
    jobs.append(
        (phase2_zip, "mono/Additional_data/all_bi_mono_sus_text.txt", "mono/Additional_data/wav/", "phase2_additional")
    )

rows = []
total_seconds = 0.0
job_counts = {}
job_seconds = {}

for zpath, txt_member, wav_prefix, out_subdir in jobs:
    with zipfile.ZipFile(zpath) as zf:
        txt_map = parse_txt_done(zf, txt_member)
        wav_names = [n for n in zf.namelist() if n.startswith(wav_prefix) and n.endswith(".wav")]
        out_dir = audio_dir / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        kept_rows = 0
        kept_seconds = 0.0

        for wav_name in wav_names:
            stem = Path(wav_name).stem
            text = txt_map.get(stem)
            if not text:
                continue

            dst = out_dir / f"{stem}.wav"
            with zf.open(wav_name) as src, open(dst, "wb") as out_f:
                out_f.write(src.read())

            with wave.open(str(dst), "rb") as wf:
                utt_seconds = wf.getnframes() / float(wf.getframerate())
                total_seconds += utt_seconds
                kept_seconds += utt_seconds

            rel = dst.relative_to(audio_dir).as_posix()
            rows.append((rel, text))
            kept_rows += 1

        job_counts[out_subdir] = kept_rows
        job_seconds[out_subdir] = kept_seconds
        if kept_rows == 0:
            raise SystemExit(f"ERROR: No aligned rows found for {out_subdir}. Check {zpath} and {txt_member}.")

rows.sort(key=lambda x: x[0])

if len(rows) < 3000:
    raise SystemExit(f"ERROR: Too few rows: {len(rows)}")

meta_path.parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)

print(f"metadata: {meta_path}")
print(f"rows: {len(rows)}")
print(f"hours: {total_seconds / 3600:.2f}")
for name in sorted(job_counts):
    print(f"{name}: rows={job_counts[name]}, hours={job_seconds[name] / 3600:.2f}")
PY

# ---------------- Training ----------------
RUN_ROOT="${WORK_ROOT}/runs/${MODEL_BASENAME}"
mkdir -p "${RUN_ROOT}"

BASE_CKPT="${RUN_ROOT}/base.ckpt"
if [[ ! -s "${BASE_CKPT}" ]]; then
  curl -L --retry 5 --retry-delay 5 --fail -o "${BASE_CKPT}" "${BASE_CKPT_URL}"
fi

TRAIN_LOG="${RUN_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
EFFECTIVE_MAX_STEPS="${MAX_STEPS}"
TRAIN_START_MARKER="${RUN_ROOT}/.train_started_$(date +%Y%m%d_%H%M%S)"
touch "${TRAIN_START_MARKER}"

# When resuming from a pretrained checkpoint with a large global step,
# make MAX_STEPS behave like "additional steps" if needed.
if [[ "${BASE_CKPT_URL}" =~ step%3D([0-9]+) ]]; then
  BASE_CKPT_STEP="${BASH_REMATCH[1]}"
elif [[ "${BASE_CKPT_URL}" =~ step=([0-9]+) ]]; then
  BASE_CKPT_STEP="${BASH_REMATCH[1]}"
else
  BASE_CKPT_STEP=""
fi

if [[ -n "${BASE_CKPT_STEP}" ]] && (( MAX_STEPS <= BASE_CKPT_STEP )); then
  EFFECTIVE_MAX_STEPS="$((BASE_CKPT_STEP + MAX_STEPS))"
  echo "MAX_STEPS=${MAX_STEPS} is <= base checkpoint step ${BASE_CKPT_STEP}."
  echo "Using EFFECTIVE_MAX_STEPS=${EFFECTIVE_MAX_STEPS} (base step + additional steps)."
fi

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
  --trainer.log_every_n_steps 20 \
  2>&1 | tee "${TRAIN_LOG}"

mapfile -t NEW_CKPTS < <(find "${RUN_ROOT}" -type f -name '*.ckpt' ! -name 'base.ckpt' -newer "${TRAIN_START_MARKER}" -print)
if (( ${#NEW_CKPTS[@]} == 0 )); then
  echo "ERROR: Training produced no new checkpoint in this run."
  echo "Check log: ${TRAIN_LOG}"
  exit 1
fi
LATEST_CKPT="$(ls -1t "${NEW_CKPTS[@]}" | head -n1)"
echo "Using checkpoint: ${LATEST_CKPT}"

rm -f "${TRAIN_START_MARKER}"

python -m piper.train.export_onnx \
  --checkpoint "${LATEST_CKPT}" \
  --output-file "${RUN_ROOT}/${MODEL_BASENAME}.onnx"

echo "வணக்கம். இது தமிழ் பைப்பர் சோதனை குரல்." | \
  python -m piper --model "${RUN_ROOT}/${MODEL_BASENAME}.onnx" --output_file "${RUN_ROOT}/sample.wav"

echo "Done."
echo "Model:  ${RUN_ROOT}/${MODEL_BASENAME}.onnx"
echo "Config: ${RUN_ROOT}/${MODEL_BASENAME}.onnx.json"
echo "Sample: ${RUN_ROOT}/sample.wav"
