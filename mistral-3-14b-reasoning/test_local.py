"""
Quick local test for SaRDinE with 4-bit quantization.
Run this on your local machine with 12GB VRAM.
"""
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-14B-Reasoning-2512",
    trust_remote_code=True
)

print("Loading model with 4-bit quantization...")
from srde import create_srde_model

model = create_srde_model(
    model_name="mistralai/Ministral-3-14B-Reasoning-2512",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    # Note: We need to modify create_srde_model to accept quantization_config
)

# Download and load SRDE weights from HuggingFace
from huggingface_hub import hf_hub_download
weights_path = hf_hub_download("MinimaML/SaRDinE-14B8x4P", "srde_weights.pt")
model.load_srde_weights(weights_path)

print(f"\nVRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Test generation
print("\n--- Testing generation ---")
prompt = "What is 15% of 80? Think step by step."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nPrompt: {prompt}")
print(f"Response: {response}")
