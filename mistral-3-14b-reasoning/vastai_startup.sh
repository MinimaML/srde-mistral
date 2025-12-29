#!/bin/bash
#===============================================================================
# SRDE Training Startup Script
# 
# PASTE THIS DIRECTLY INTO VAST.AI STARTUP COMMAND
# 
# Set these environment variables in Vast.ai:
#   HF_TOKEN=hf_your_token  (required)
#   WEBHOOK_URL=https://...  (optional, for Discord/Slack notifications)
#   EXAMPLES_PER_DOMAIN=50000  (optional, default 50000)
#   MAX_STEPS=10000  (optional, default 10000)
#===============================================================================

set -e

# Config with defaults
GITHUB_REPO="https://github.com/MinimaML/srde-mistral.git"
WORK_DIR="/workspace"
EXAMPLES_PER_DOMAIN="${EXAMPLES_PER_DOMAIN:-50000}"
MAX_STEPS="${MAX_STEPS:-10000}"

# Notification helper
notify() {
    echo "[$(date +%H:%M:%S)] $1"
    [ -n "${WEBHOOK_URL}" ] && curl -s -X POST "${WEBHOOK_URL}" -H "Content-Type: application/json" -d "{\"content\":\"🤖 **SRDE**: $1\"}" --max-time 5 || true
}

notify "Starting SRDE training pipeline"

# Setup
cd ${WORK_DIR}
pip install -q torch transformers accelerate datasets huggingface_hub tqdm wandb safetensors
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "[INFO] Flash Attention not available"

# Clone repo
echo "[1/4] Cloning SRDE code..."
rm -rf srde 2>/dev/null || true
git clone ${GITHUB_REPO} srde
cd srde

# HuggingFace login
echo "[2/4] Logging into HuggingFace..."
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

# Build dataset
echo "[3/4] Building dataset (${EXAMPLES_PER_DOMAIN} per domain)..."
notify "Downloading and tokenizing dataset"
mkdir -p data
python build_and_upload_dataset.py \
    --repo_name MinimaML/srde-dataset \
    --examples_per_domain ${EXAMPLES_PER_DOMAIN} \
    --work_dir ./data \
    --hf_token ${HF_TOKEN} 2>&1 || echo "[WARN] Upload failed, using local data"

# Train
echo "[4/4] Training (${MAX_STEPS} steps)..."
notify "Training started"
mkdir -p checkpoints

python train.py \
    --pretokenized_dir ./data/tokenized \
    --output_dir ./checkpoints \
    --max_steps ${MAX_STEPS} \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --flash_attention \
    --use_muon \
    --compile \
    --gradient_checkpointing \
    --bf16 \
    --save_steps 1000

notify "TRAINING COMPLETE!"
echo "=============================================="
echo "Done! Checkpoints: $(pwd)/checkpoints"
echo "=============================================="
sleep infinity
