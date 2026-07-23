#!/bin/bash
set -euo pipefail

WEAPON_ASSET_BUCKET_ID="${HF_WEAPON_ASSET_BUCKET_ID:-mobilint/aries-weapon-detection-demo-assets}"
FIRE_ASSET_BUCKET_ID="${HF_FIRE_ASSET_BUCKET_ID:-mobilint/aries-fire-detection-demo-assets}"
ULTRALYTICS_ASSET_BUCKET_ID="${HF_ULTRALYTICS_ASSET_BUCKET_ID:-mobilint/aries-ultralytics-demo-assets}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HF_DOWNLOAD_VENV_DIR:-$SCRIPT_DIR/.hf_venv}"
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

echo "Preparing Hugging Face download environment: $VENV_DIR"
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

download_bucket() {
  local label="$1"
  local bucket_id="$2"
  local local_dir="$3"
  local bucket_uri="hf://buckets/${bucket_id}"

  mkdir -p "$local_dir"

  echo "Downloading ${label} assets from Hugging Face bucket: $bucket_uri"
  "$HF_CLI" buckets sync "$bucket_uri" "$local_dir" \
    --include "mxq/*" \
    --include "video/*.mp4" \
    --include "video/**/*.mp4"

  echo "${label} assets downloaded to $local_dir"
}

download_bucket "weapon" "$WEAPON_ASSET_BUCKET_ID" "$WEAPON_LOCAL_DIR"
download_bucket "fire" "$FIRE_ASSET_BUCKET_ID" "$FIRE_LOCAL_DIR"
download_bucket "ultralytics" "$ULTRALYTICS_ASSET_BUCKET_ID" "$ULTRALYTICS_LOCAL_DIR"

echo "All CV assets downloaded successfully."