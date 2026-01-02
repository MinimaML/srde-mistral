# SaRDinE Demo - ZeroGPU v2
import gradio as gr
import torch
import spaces
from transformers import AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
import os

# Download model files
print("Downloading model files...")
os.makedirs("./model", exist_ok=True)
for filename in ["modeling_sardine.py", "configuration_sardine.py", "srde_weights.pt", "config.json"]:
    try:
        hf_hub_download("MinimaML/SaRDinE-14B8x4P", filename, local_dir="./model")
        print(f"  Downloaded {filename}")
    except Exception as e:
        print(f"  Could not download {filename}: {e}")

# Add model to path
import sys
sys.path.insert(0, "./model")

# Global model variable (loaded on first GPU call)
model = None
tokenizer = None


def load_model():
    """Load model on first call."""
    global model, tokenizer
    
    if model is not None:
        return
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Ministral-3-14B-Reasoning-2512",
        trust_remote_code=True
    )
    
    print("Loading model with 4-bit quantization...")
    from modeling_sardine import create_srde_model
    
    model = create_srde_model(
        model_name="mistralai/Ministral-3-14B-Reasoning-2512",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load SRDE weights
    weights_path = "./model/srde_weights.pt"
    if os.path.exists(weights_path):
        model.load_srde_weights(weights_path)
        print("Loaded SRDE weights")
    print("Model loaded!")


@spaces.GPU(duration=120)
def generate(prompt, max_tokens, temperature, top_p):
    """Generate text from prompt using ZeroGPU."""
    load_model()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature) if float(temperature) > 0 else 0.01,
            top_p=float(top_p),
            do_sample=float(temperature) > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Example prompts
examples = [
    ["Solve step by step: What is 15% of 80?", 256, 0.7, 0.9],
    ["Write a Python function to check if a number is prime.", 256, 0.7, 0.9],
    ["Explain why the sky is blue in simple terms.", 256, 0.7, 0.9],
    ["What is the capital of France? Answer in one word.", 32, 0.1, 0.9],
]

# Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3),
        gr.Slider(minimum=32, maximum=512, value=256, step=32, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Response", lines=10),
    title="🐟 SaRDinE-14B8x4P Demo",
    description="""
    **S**parse **R**outed **D**elta **E**xperts on Mistral-14B-Reasoning.
    
    SaRDinE is a novel MoE-alternative architecture that adds trainable sparse expert deltas 
    to a frozen base model for domain specialization.
    
    - **Base Model**: Mistral-14B-Reasoning (frozen)
    - **Experts**: 8 per layer, ~4% sparsity
    - **Domains**: Math, Code, Science, Logic, Planning, Abstract
    
    ⚡ Powered by ZeroGPU (first request may take ~60s to load)
    
    [Model Page](https://huggingface.co/MinimaML/SaRDinE-14B8x4P) | 
    [GitHub](https://github.com/MinimaML/srde-mistral)
    """,
    examples=examples,
    cache_examples=False,
    theme=gr.themes.Default()
)

if __name__ == "__main__":
    demo.launch()
