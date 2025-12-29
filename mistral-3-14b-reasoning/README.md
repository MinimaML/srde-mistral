# SRDE for Mistral-3-14B-Reasoning

**Sparse Routed Delta Experts** - A parameter-efficient alternative to Mixture of Experts.

## 🚀 Quick Start (Vast.ai)

### Fire-and-Forget Training

1. **Upload your training data** to cloud storage (HuggingFace, S3, etc.)

2. **Edit `vastai_startup.sh`**:
   ```bash
   HF_TOKEN="your_huggingface_token"
   DATA_URL="https://your-data-url/train.jsonl"
   MAX_STEPS=5000
   ```

3. **Create Vast.ai instance** with the startup script:
   ```bash
   # In Vast.ai "On-start Script" field:
   curl -sL https://raw.githubusercontent.com/your-repo/srde/main/vastai_startup.sh | bash
   ```

4. **That's it!** The script handles everything:
   - ✅ Installs dependencies
   - ✅ Downloads model weights
   - ✅ Downloads your data
   - ✅ Auto-resumes from crashes
   - ✅ Sends Discord/Slack notifications
   - ✅ Saves checkpoints every 500 steps

## 📁 File Structure

```
mistral-3-14b-reasoning/
├── config.py           # SRDE hyperparameters
├── srde.py             # Core architecture
├── losses.py           # Loss functions
├── scheduler.py        # Training schedulers
├── train.py            # Robust training script
├── inference.py        # Generation + analysis
├── monitor.py          # Training monitor
├── vastai_startup.sh   # Vast.ai startup script
└── requirements.txt    # Dependencies
```

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py \
    --train_file data.jsonl \
    --output_dir ./checkpoints \
    --max_steps 5000

# Resume from checkpoint (automatic if crashed)
python train.py \
    --train_file data.jsonl \
    --output_dir ./checkpoints

# Monitor training (separate terminal)
python monitor.py --output_dir ./checkpoints

# Inference with trained weights
python inference.py \
    --srde_weights ./checkpoints/checkpoint-best/srde_weights.pt \
    --prompt "Solve this problem:"
```

## 💾 Crash Recovery

The training script is **crash-resistant**:

- **Auto-resume**: Automatically finds latest checkpoint on restart
- **Graceful shutdown**: Saves checkpoint on `Ctrl+C` or SIGTERM
- **OOM recovery**: Clears CUDA cache and continues
- **Heartbeat**: Writes status every 50 steps for monitoring

## 📊 Training Phases

| Phase             | Steps    | What's Trained               |
| ----------------- | -------- | ---------------------------- |
| 1. Warmup         | 0-100    | Nothing (establish baseline) |
| 2. Mask Learning  | 100-600  | Which weights to modify      |
| 3. Delta Training | 600-1600 | Modification values          |
| 4. Joint          | 1600+    | Everything together          |

## ⚙️ Configuration

Key parameters in `config.py`:

| Parameter          | Default | Description                 |
| ------------------ | ------- | --------------------------- |
| `num_experts`      | 8       | Number of sparse experts    |
| `top_k`            | 2       | Experts activated per token |
| `initial_sparsity` | 0.05    | Starting sparsity (5%)      |
| `target_sparsity`  | 0.01    | Final sparsity (1%)         |

## 🔔 Notifications

Set `WEBHOOK_URL` in `vastai_startup.sh` to receive:
- 🚀 Training started
- ✅ Training completed
- ❌ Training failed (with retry info)
- ♻️ Resumed from checkpoint

## Hardware Requirements

| Task      | VRAM  | Recommended  |
| --------- | ----- | ------------ |
| Inference | ~28GB | 1× H100/H200 |
| Training  | ~40GB | 2× H100/H200 |

## Citation

```bibtex
@misc{srde2024,
  title={Sparse Routed Delta Experts: A Parameter-Efficient Alternative to MoE},
  author={Collins, William},
  year={2024}
}
```
