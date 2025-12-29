#!/bin/bash
#===============================================================================
# SRDE Training Startup Script for Vast.ai
# 
# Clones code from GitHub, builds dataset, tokenizes, and trains.
# 
# USAGE:
#   1. Create GitHub repo with all SRDE code
#   2. Set environment variables below
#   3. Use this script as Vast.ai startup command
#===============================================================================

set -e

#===============================================================================
# CONFIGURATION - EDIT THESE
#===============================================================================

# REQUIRED: Your GitHub repo URL
GITHUB_REPO="${GITHUB_REPO:-https://github.com/YOUR_USERNAME/srde-mistral}"

# REQUIRED: HuggingFace token for model access
export HF_TOKEN="${HF_TOKEN:-your_hf_token}"

# Dataset config
EXAMPLES_PER_DOMAIN="${EXAMPLES_PER_DOMAIN:-50000}"  # 50K per domain = 300K total
MAX_STEPS="${MAX_STEPS:-10000}"

# Optional: Webhook for notifications
WEBHOOK_URL="${WEBHOOK_URL:-}"

# Directories
WORK_DIR="/workspace"
CODE_DIR="${WORK_DIR}/srde"
DATA_DIR="${WORK_DIR}/data"
CHECKPOINTS_DIR="${WORK_DIR}/checkpoints"

#===============================================================================
# NOTIFICATION HELPER
#===============================================================================

notify() {
    local message="$1"
    echo "[NOTIFY] $message"
    if [ -n "${WEBHOOK_URL}" ]; then
        curl -s -X POST "${WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"🤖 **SRDE**: $message\"}" \
            --max-time 10 || true
    fi
}

#===============================================================================
# STEP 1: SETUP
#===============================================================================

echo "=============================================="
echo "SRDE Training Pipeline"
echo "=============================================="
echo "GitHub Repo: ${GITHUB_REPO}"
echo "Examples/Domain: ${EXAMPLES_PER_DOMAIN}"
echo "Max Steps: ${MAX_STEPS}"
echo "=============================================="

notify "Starting training pipeline"

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch transformers accelerate datasets huggingface_hub tqdm wandb safetensors

# Install Flash Attention (optional, may fail on some systems)
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention not available"

#===============================================================================
# STEP 2: CLONE REPO
#===============================================================================

echo "[2/5] Cloning code from GitHub..."
cd ${WORK_DIR}

if [ -d "${CODE_DIR}" ]; then
    cd ${CODE_DIR}
    git pull
else
    git clone ${GITHUB_REPO} ${CODE_DIR}
    cd ${CODE_DIR}
fi

# HuggingFace login
echo "[2/5] Logging into HuggingFace..."
huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential

#===============================================================================
# STEP 3: BUILD DATASET
#===============================================================================

echo "[3/5] Building dataset (downloading + tokenizing)..."
notify "Downloading and tokenizing dataset (${EXAMPLES_PER_DOMAIN} per domain)"

cd ${CODE_DIR}

python build_and_upload_dataset.py \
    --repo_name $(whoami)/srde-dataset \
    --examples_per_domain ${EXAMPLES_PER_DOMAIN} \
    --work_dir ${DATA_DIR} \
    --hf_token ${HF_TOKEN} \
    --skip_download 2>/dev/null || true

# If upload failed, that's fine - we have local data
echo "Dataset ready at: ${DATA_DIR}/tokenized"

#===============================================================================
# STEP 4: TRAIN
#===============================================================================

echo "[4/5] Starting training..."
notify "Training started (${MAX_STEPS} steps)"

mkdir -p ${CHECKPOINTS_DIR}

python train.py \
    --pretokenized_dir ${DATA_DIR}/tokenized \
    --output_dir ${CHECKPOINTS_DIR} \
    --max_steps ${MAX_STEPS} \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --flash_attention \
    --use_muon \
    --compile \
    --gradient_checkpointing \
    --bf16 \
    --save_steps 1000 \
    --log_steps 50 \
    --heartbeat_steps 100

#===============================================================================
# STEP 5: COMPLETE
#===============================================================================

echo "[5/5] Training complete!"
notify "Training COMPLETE! Checkpoints at ${CHECKPOINTS_DIR}"

echo "=============================================="
echo "TRAINING FINISHED"
echo "=============================================="
echo "Checkpoints: ${CHECKPOINTS_DIR}"
echo "Latest: $(ls -1t ${CHECKPOINTS_DIR}/checkpoint-* | head -1)"
echo "=============================================="

# Keep container alive for download
echo "Container staying alive for file access. Press Ctrl+C to stop."
sleep infinity
