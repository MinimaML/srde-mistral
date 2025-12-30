# SRDE: Sparse Routed Delta Experts

Train domain-specialized reasoning capabilities on top of a frozen base model using sparse expert routing.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build dataset (downloads ~400k examples across 6 domains)
python build_and_upload_dataset.py \
    --examples_per_domain 100000 \
    --model_name mistralai/Ministral-3-14B-Reasoning-2512 \
    --work_dir ./data

# Train with all optimizations
python train.py \
    --model_name mistralai/Ministral-3-14B-Reasoning-2512 \
    --pretokenized_dir ./data/tokenized \
    --jsonl_data ./data/data.jsonl \
    --output_dir ./checkpoints \
    --max_steps 10000 \
    --use_muon \
    --bf16 \
    --supervised_experts \
    --expert_pretrain_steps 500 \
    --warm_start \
    --progressive_unlock linear
```

## 📊 Architecture

SRDE adds trainable sparse experts on top of a frozen base model:

```
Input → Embed (frozen) → Transformer Layers with SRDE:
                         ├── Attention (frozen)
                         └── MLP → SRDE Layer:
                                  ├── Router → picks top-k experts
                                  ├── Experts → domain-specific deltas
                                  └── Base MLP (frozen)
                         → LM Head → Output
```

**Key Properties:**
- Base model stays frozen (14B frozen + ~1.6B trainable SRDE params)
- Experts specialize in different domains (math, code, logic, etc.)
- Router learns to select relevant experts per token

## 🎯 Training Optimizations

### Supervised Expert Pre-Training
Pre-train each expert on its domain data before joint training:
```bash
--supervised_experts \
--expert_pretrain_steps 500
```

### Router Warm-Start
Initialize router weights from K-means clustering of hidden states:
```bash
--warm_start
```

### Progressive Expert Unlocking
Train experts one at a time to prevent interference:
```bash
--progressive_unlock linear    # Gradual unlock
--progressive_unlock warmup    # 1 expert, then all
```

### Muon Optimizer
Use Muon+AdamW for ~35% faster training:
```bash
--use_muon
```

## 📁 Domains

The training data covers 6 reasoning domains:

| Domain   | Expert | Example Datasets             |
| -------- | ------ | ---------------------------- |
| Math     | 0      | GSM8K, MetaMathQA, Orca-Math |
| Logic    | 1      | LogiQA, ReClor, HellaSwag    |
| Code     | 2      | CodeAlpaca, Code-Feedback    |
| Science  | 3      | SciQ, ARC, SciBench          |
| Planning | 4      | AQuA-RAT, BoardgameQA        |
| Abstract | 5      | PIQA, WinoGrande             |

## 🔧 Files

| File                          | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `train.py`                    | Main training script with all optimizations |
| `srde.py`                     | SRDE architecture (Router, Experts, Layers) |
| `config.py`                   | Configuration classes                       |
| `build_and_upload_dataset.py` | Dataset builder                             |
| `evaluate.py`                 | Evaluation script                           |
| `inference.py`                | Interactive inference                       |
| `vastai_startup.sh`           | Vast.ai deployment script                   |

## 💻 Inference

```bash
# Generate text
python inference.py \
    --srde_weights ./checkpoints/checkpoint-10000/srde_weights.pt \
    --prompt "Explain quicksort step by step:"

# Interactive mode
python inference.py --interactive \
    --srde_weights ./checkpoints/checkpoint-10000/srde_weights.pt
```

## 📈 Evaluation

```bash
python evaluate.py \
    --checkpoint ./checkpoints/checkpoint-10000 \
    --eval_file ./data/data.jsonl \
    --compare_baseline
```

## 🖥️ Vast.ai Deployment

```bash
# Template: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
# On-start script:
curl -sSL https://raw.githubusercontent.com/MinimaML/srde-mistral/main/mistral-3-14b-reasoning/vastai_startup.sh | bash
```

## 📄 License

MIT License

## 🔗 Citation

```bibtex
@misc{srde2024,
  title={SRDE: Sparse Routed Delta Experts},
  author={MinimaML},
  year={2024},
  url={https://github.com/MinimaML/srde-mistral}
}
```
