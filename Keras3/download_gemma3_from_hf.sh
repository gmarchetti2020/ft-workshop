#!/bin/bash

# A script to download the Gemma 3 1B model from Hugging Face.
# It sources configuration from a file.
#
# Usage: ./download_gemma3_from_hf.sh <download_dir> [path/to/config_file.conf]
#   - download_dir: The local directory where the model will be saved.
#   - config_file.conf: (Optional) Path to a configuration file containing HF_TOKEN.
#                       Defaults to 'config.conf' in the current directory.

set -euo pipefail

# --- Argument Validation ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <download_dir> [config_file]" >&2
    echo "Example: $0 /mnt/content/gemma3 config.conf" >&2
    exit 1
fi

DOWNLOAD_DIR="$1"
CONFIG_FILE="${2:-config.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at '$CONFIG_FILE'" >&2
    exit 1
fi

# Source configuration for HF_TOKEN
source "$CONFIG_FILE"

# --- Execution ---
echo "Starting model download using config from '${CONFIG_FILE}'..."

~/.local/bin/huggingface-cli download \
  --repo-type model google/gemma-3-1b-it \
  --local-dir "$DOWNLOAD_DIR" \
  --token $HF_TOKEN

echo "Model downloaded. Output is in '${DOWNLOAD_DIR}'."
