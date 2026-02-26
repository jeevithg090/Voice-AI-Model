#!/usr/bin/env bash
set -euo pipefail

SCRATCH_LABEL="DOSCRATCH"
SCRATCH_MNT="/mnt/scratch"
TARGET_USER="${SUDO_USER:-$(whoami)}"

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y docker.io docker-compose-plugin
  usermod -aG docker "$TARGET_USER" || true
fi

device=$(blkid -L "$SCRATCH_LABEL" 2>/dev/null || true)
if [ -n "$device" ]; then
  fs_type=$(blkid -o value -s TYPE "$device" 2>/dev/null || true)
  if [ -z "$fs_type" ]; then
    fs_type=$(lsblk -no FSTYPE "$device" 2>/dev/null | head -n 1 || true)
  fi
  if [ -z "$fs_type" ]; then
    echo "Unable to determine filesystem type for $device. Skipping scratch mount."
    exit 1
  fi

  mkdir -p "$SCRATCH_MNT"
  entry_source="LABEL=$SCRATCH_LABEL"
  if [ ! -e "/dev/disk/by-label/$SCRATCH_LABEL" ]; then
    entry_source="$device"
  fi

  if grep -q "$SCRATCH_LABEL" /etc/fstab || grep -q "$SCRATCH_MNT" /etc/fstab; then
    sed -i.bak "\\#${SCRATCH_MNT}#d;/${SCRATCH_LABEL}/d" /etc/fstab
  fi
  echo "$entry_source $SCRATCH_MNT $fs_type discard,errors=remount-ro,nofail 0 2" >> /etc/fstab
  mount -a

  mkdir -p "$SCRATCH_MNT/hf" "$SCRATCH_MNT/torch"
  chown -R "$TARGET_USER":"$TARGET_USER" "$SCRATCH_MNT"
else
  echo "Scratch disk with label $SCRATCH_LABEL not found. Skipping scratch mount."
fi

echo "Bootstrap complete. Log out/in to apply docker group changes."
