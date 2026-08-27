#!/bin/bash
# Download an openpi policy checkpoint. Defaults to pi05_libero (pi0.5
# fine-tuned on vanilla LIBERO -- the checkpoint used for zero-shot LIBERO-X
# eval in this pipeline's README results). ~12 GB.
#
# Usage: ./download_checkpoint.sh /path/to/workspace [checkpoint_name]
set -euo pipefail

WORKSPACE="${1:?Usage: download_checkpoint.sh /path/to/workspace [checkpoint_name]}"
CHECKPOINT="${2:-pi05_libero}"

if ! command -v gsutil &> /dev/null; then
  echo "gsutil not found. Install the Google Cloud SDK, or download manually:" >&2
  echo "  https://storage.googleapis.com/openpi-assets/checkpoints/$CHECKPOINT/" >&2
  exit 1
fi

mkdir -p "$WORKSPACE/checkpoints"
gsutil -m cp -r "gs://openpi-assets/checkpoints/$CHECKPOINT" "$WORKSPACE/checkpoints/"
echo "Downloaded to $WORKSPACE/checkpoints/$CHECKPOINT"
