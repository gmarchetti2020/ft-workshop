#!/bin/bash
#
# This script sets up the environment by installing necessary packages,
# configuring system settings, and mounting a GCS bucket.
#
# It also cleans up redundant repository setup commands and adds robust
# error handling.
#
# Usage: ./setup.sh <your-gcs-bucket-name>

set -euo pipefail


# --- System Configuration ---
echo "⚙️  Configuring transparent huge pages..."
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"


# --- Mount GCS Bucket ---
echo "⚙️  Mounting hyperdisk ..."
sudo mkdir -p /mnt/content
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme0n2
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/content
echo "✔️ Hyperdisk mounted to '/mnt/content'."

# --- Python Dependencies ---
echo "⚙️  Installing Python packages..."
pip install jax[tpu] tensorflow-cpu keras-hub sentencepiece google-cloud-aiplatform
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install -U huggingface_hub[cli]

echo "--- ✅ Setup complete ---"
