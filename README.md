# SRDE - Sparse Routed Delta Experts

**Parameter-efficient enhancement for Mistral-3-14B-Reasoning**

Adds 6 domain experts (~12M params) that dynamically route to improve:
- Advanced Math
- Formal Logic  
- Algorithm Design
- Scientific Reasoning
- Multi-step Planning
- Abstract/Symbolic reasoning

## Quick Start (Vast.ai)

### 1. Set Environment Variables

```bash
export GITHUB_REPO="https://github.com/YOUR_USERNAME/srde-mistral"
export HF_TOKEN="hf_your_token"
export EXAMPLES_PER_DOMAIN=50000  # 300K total
export MAX_STEPS=10000
```

### 2. Run on Vast.ai

Use `vastai_startup.sh` as your startup script. It will:
1. Clone this repo
2. Download datasets (GSM8K, MATH, CodeContests, etc.)
3. Pre-tokenize for Mistral
4. Train with Flash Attention + Muon optimizer

### 3. Estimated Cost

| Metric | Value              |
| ------ | ------------------ |
| Data   | 300K examples      |
| Time   | ~12 hours          |
| Cost   | **~£28** (1× H200) |

## Local Usage

```bash
# Install
pip install -r requirements.txt

# Build dataset
python build_and_upload_dataset.py \
    --repo_name YOUR_USER/srde-dataset \
    --examples_per_domain 50000

# Train
python train.py \
    --pretokenized_dir ./data \
    --flash_attention \
    --use_muon \
    --compile
```

## Configuration

| Parameter         | Default | Description          |
| ----------------- | ------- | -------------------- |
| `num_experts`     | 6       | Domain expert count  |
| `top_k`           | 2       | Experts per token    |
| `target_sparsity` | 1%      | Final delta sparsity |
| `max_steps`       | 10000   | Training steps       |

## Files

| File                          | Purpose              |
| ----------------------------- | -------------------- |
| `train.py`                    | Main training script |
| `srde.py`                     | Core architecture    |
| `config.py`                   | Configuration        |
| `build_and_upload_dataset.py` | Dataset pipeline     |
| `vastai_startup.sh`           | Cloud startup script |
| `muon.py`                     | Muon optimizer       |

## Expert Domains

| ID  | Domain   | Datasets                 |
| --- | -------- | ------------------------ |
| 0   | Math     | GSM8K, MATH, MetaMathQA  |
| 1   | Logic    | LogiQA, ReClor           |
| 2   | Code     | CodeContests, APPS       |
| 3   | Science  | SciQ, ARC                |
| 4   | Planning | StrategyQA, HotpotQA     |
| 5   | Abstract | BIG-Bench Hard, AQuA-RAT |

## License

Apache License 2.0
