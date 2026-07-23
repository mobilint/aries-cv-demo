#!/bin/bash
set -euo pipefail

WEAPON_ASSET_BUCKET_ID="${HF_WEAPON_ASSET_BUCKET_ID:-mobilint/aries-weapon-detection-demo-assets}"
FIRE_ASSET_BUCKET_ID="${HF_FIRE_ASSET_BUCKET_ID:-mobilint/aries-fire-detection-demo-assets}"
ULTRALYTICS_ASSET_BUCKET_ID="${HF_ULTRALYTICS_ASSET_BUCKET_ID:-mobilint/aries-ultralytics-demo-assets}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HF_UPLOAD_VENV_DIR:-${HF_DOWNLOAD_VENV_DIR:-$SCRIPT_DIR/.hf_venv}}"
WEAPON_LOCAL_DIR="$SCRIPT_DIR/assets/weapon"
FIRE_LOCAL_DIR="$SCRIPT_DIR/assets/fire"
ULTRALYTICS_LOCAL_DIR="$SCRIPT_DIR/assets/ultralytics"

if [ -d "$HOME/.local/bin" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Please install uv before running this script."
  echo "See: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "Preparing Hugging Face upload environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi

if [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo "Cannot find venv python at $VENV_DIR/bin/python"
  exit 1
fi

uv pip install --python "$VENV_PYTHON" huggingface-hub

HF_CLI="$VENV_DIR/bin/hf"
if [ ! -x "$HF_CLI" ]; then
  echo "Cannot find hf CLI at $HF_CLI"
  exit 1
fi

upload_bucket() {
  local label="$1"
  local bucket_id="$2"
  local local_dir="$3"
  local bucket_uri="hf://buckets/${bucket_id}"

  if [ ! -d "$local_dir" ]; then
    echo "Local directory not found for ${label}: $local_dir"
    exit 1
  fi

  echo "Uploading ${label} assets to Hugging Face bucket: $bucket_uri"
  echo "Remote files matching included asset patterns but missing locally will be deleted."
  "$HF_CLI" buckets sync "$local_dir" "$bucket_uri" \
    --delete \
    --include "mxq/*" \
    --include "video/*.mp4" \
    --include "video/**/*.mp4" \
    --exclude "mxq/.gitkeep" \
    --exclude "video/.gitkeep"

  echo "${label} assets uploaded from $local_dir"
}

upload_bucket "weapon" "$WEAPON_ASSET_BUCKET_ID" "$WEAPON_LOCAL_DIR"
upload_bucket "fire" "$FIRE_ASSET_BUCKET_ID" "$FIRE_LOCAL_DIR"
upload_bucket "ultralytics" "$ULTRALYTICS_ASSET_BUCKET_ID" "$ULTRALYTICS_LOCAL_DIR"

echo "All CV assets uploaded and synchronized successfully."