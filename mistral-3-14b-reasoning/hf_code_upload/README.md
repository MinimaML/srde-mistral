---
license: apache-2.0
language:
- en
tags:
- srde
- sparse-experts
- moe
- reasoning
- mistral
base_model: mistralai/Ministral-3-14B-Reasoning-2512
library_name: transformers
pipeline_tag: text-generation
datasets:
- openai/gsm8k
- meta-math/MetaMathQA
- google-research-datasets/mbpp
- allenai/sciq
- allenai/ai2_arc
---

# SaRDinE-14B8x4P

**S**parse **R**outed **D**elta **E**xperts on Mistral-14B-Reasoning.

> 14B base params | 8 experts per layer | ~4% sparsity (alpha)

## What is SaRDinE?

SaRDinE is a novel MoE-alternative architecture. Unlike traditional MoE which fragments model capacity across experts, SaRDinE:

- Keeps **100% of the base model** frozen and active
- Adds **sparse trainable delta experts** on top
- Routes tokens to domain-specialized experts (math, code, science, etc.)

**Result:** Full base model capability PLUS domain specialization.

## Quick Start

### Option 1: BF16 (Requires ~30GB VRAM)

```python
import torch
from transformers import AutoTokenizer

# Clone the repo for the model code
# The model uses trust_remote_code=True

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-14B-Reasoning-2512",
    trust_remote_code=True
)

# Load SRDE model
from huggingface_hub import hf_hub_download
import sys
sys.path.insert(0, hf_hub_download("MinimaML/SaRDinE-14B8x4P", "modeling_sardine.py", local_dir="."))

from modeling_sardine import SaRDinEForCausalLM

model = SaRDinEForCausalLM.from_pretrained(
    "MinimaML/SaRDinE-14B8x4P",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
prompt = "Solve step by step: What is 15% of 80?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: Quantized Base Model (Requires ~16GB VRAM)

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Quantization config for base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load with quantization
from modeling_sardine import SaRDinEForCausalLM

model = SaRDinEForCausalLM.from_pretrained(
    "MinimaML/SaRDinE-14B8x4P",
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Architecture

| Component            | Value                          |
| -------------------- | ------------------------------ |
| Base Model           | Mistral-14B-Reasoning (frozen) |
| Trainable Parameters | ~2.4B (sparse deltas)          |
| Experts per Layer    | 8                              |
| Top-K Routing        | 2                              |
| Current Sparsity     | ~4%                            |
| Augmented Layers     | 40                             |

## Training

- **Phase 1**: Supervised expert pre-training on domain-specific data
- **Phase 2**: Joint fine-tuning with progressive expert unlocking

### Domains
| Domain   | Data Sources                       |
| -------- | ---------------------------------- |
| Math     | GSM8K, MetaMathQA, Orca-Math       |
| Logic    | BigBench, CommonsenseQA, HellaSwag |
| Code     | MBPP, HumanEval, CodeFeedback      |
| Science  | SciQ, ARC, MMLU                    |
| Planning | HotpotQA, SCROLLS                  |
| Abstract | BigBench, AQuA-RAT, Winogrande     |

## Files

- `srde_weights.pt` - Trained SRDE delta weights (~5GB)
- `modeling_sardine.py` - Model architecture
- `configuration_sardine.py` - Configuration class
- `config.json` - Model config

## Citation

```bibtex
@misc{sardine2025,
  title={SaRDinE: Sparse Routed Delta Experts},
  author={MinimaML},
  year={2025},
  url={https://github.com/MinimaML/srde-mistral}
}
```

## License

Apache 2.0

## Links

- **GitHub**: [MinimaML/srde-mistral](https://github.com/MinimaML/srde-mistral)
- **Base Model**: [Mistral-14B-Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512)
