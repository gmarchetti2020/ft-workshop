#!/bin/bash

# This script converts a Keras model to Hugging Face format (creating a .safetensors file)
# and then copies the .safetensors file to a final destination directory.
#
# Usage: ./4.convert_to_hf.sh <keras_model_path> <hf_output_dir> <final_dest_dir>
#   - keras_model_path: Path to the input Keras model file (.keras).
#   - hf_output_dir: Directory where the converted Hugging Face model files will be saved.
#                    This script assumes the safetensors file will be named 'model.safetensors' inside this directory.
#   - final_dest_dir: The final directory to which the 'model.safetensors' file will be copied.

set -euo pipefail

# --- Argument Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <keras_model_path> <hf_output_dir> <final_dest_dir>" >&2
    echo "Example: $0 /mnt/content/finetuned3/model.keras /mnt/content/finetuned3_hf /mnt/content/frankenmodel_hf" >&2
    exit 1
fi

KERAS_MODEL_PATH="$1"
HF_OUTPUT_DIR="$2"
FINAL_DEST_DIR="$3"

# The python script that performs the conversion.
# The original script had a typo '33.gemma3keras__hfconverter.py'.
# Based on the provided files, 'gemma3keras_hf_converter.py' is the correct script in this context.
PYTHON_CONVERTER_SCRIPT="gemma3keras_hf_converter.py"
SAFETENSORS_FILENAME="model.safetensors"

# --- Pre-flight checks ---
if [ ! -f "$KERAS_MODEL_PATH" ]; then
    echo "Error: Keras model file not found at '$KERAS_MODEL_PATH'" >&2
    exit 1
fi

if [ ! -f "$PYTHON_CONVERTER_SCRIPT" ]; then
    echo "Error: Python converter script not found: '$PYTHON_CONVERTER_SCRIPT'" >&2
    exit 1
fi

# --- Execution ---
echo "Starting Keras to Hugging Face conversion for '$KERAS_MODEL_PATH'..."
python "$PYTHON_CONVERTER_SCRIPT" "$KERAS_MODEL_PATH" "$HF_OUTPUT_DIR" --model_size 1b

echo "Conversion complete. Safetensors created in '$HF_OUTPUT_DIR'."

SAFETENSORS_SRC_PATH="${HF_OUTPUT_DIR}/${SAFETENSORS_FILENAME}"

echo "Copying '$SAFETENSORS_SRC_PATH' to '$FINAL_DEST_DIR'..."
mkdir -p "$FINAL_DEST_DIR"
cp "$SAFETENSORS_SRC_PATH" "$FINAL_DEST_DIR"

echo "--- ✅ Done ---"
