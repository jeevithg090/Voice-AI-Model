#!/usr/bin/env bash
set -euo pipefail

# Canonical training script lives in FInal-Voice.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/FInal-Voice/piper_ta_rocm71_train.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "ERROR: Missing canonical script: ${TARGET_SCRIPT}"
  exit 1
fi

exec bash "${TARGET_SCRIPT}" "$@"
