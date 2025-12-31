"""
SRDE Inference Script

Load a trained SRDE model and generate text.
"""
import argparse
import torch
from transformers import AutoTokenizer, TextStreamer

from config import SRDEConfig
from srde import create_srde_model


def parse_args():
    parser = argparse.ArgumentParser(description="SRDE Inference")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Ministral-3-14B-Reasoning-2512",
        help="Base model name"
    )
    parser.add_argument(
        "--srde_weights",
        type=str,
        default=None,
        help="Path to SRDE weights (if not provided, uses untrained SRDE)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the concept of sparse expert models in simple terms:",
        help="Input prompt"
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true", default=True)
    
    # Extended reasoning options
    parser.add_argument(
        "--extended_thinking",
        action="store_true",
        help="Enable extended reasoning mode (longer generation, self-check prompt)"
    )
    parser.add_argument(
        "--best_of_n",
        type=int,
        default=1,
        help="Generate N answers and return the best/most common (majority vote)"
    )
    
    # SRDE config (should match training)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--target_sparsity", type=float, default=0.01)
    
    return parser.parse_args()


def load_model(args):
    """Load SRDE model with optional trained weights."""
    
    config = SRDEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        target_sparsity=args.target_sparsity
    )
    
    print(f"Loading model: {args.model_name}")
    model = create_srde_model(
        model_name=args.model_name,
        config=config,
        torch_dtype=torch.bfloat16
    )
    
    if args.srde_weights:
        print(f"Loading SRDE weights from: {args.srde_weights}")
        model.load_srde_weights(args.srde_weights)
    
    model.eval()
    return model


def generate(model, tokenizer, prompt: str, args):
    """Generate text from prompt with optional extended reasoning."""
    
    # Extended thinking mode: add self-check prompt and increase tokens
    if getattr(args, 'extended_thinking', False):
        prompt = f"""Think step by step. After reaching an answer, double-check your work.
If you find an error, correct it before giving your final answer.

{prompt}"""
        max_tokens = max(args.max_tokens, 2048)  # Allow longer reasoning
    else:
        max_tokens = args.max_tokens
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Setup streamer
    streamer = TextStreamer(tokenizer, skip_special_tokens=True) if args.stream else None
    
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    if getattr(args, 'extended_thinking', False):
        print(f"[Extended Thinking Mode: max_tokens={max_tokens}]")
    print(f"{'='*50}\n")
    
    # Best-of-N sampling
    n_samples = getattr(args, 'best_of_n', 1)
    if n_samples > 1:
        print(f"[Generating {n_samples} samples for majority vote...]")
        all_outputs = []
        for i in range(n_samples):
            with torch.no_grad():
                outputs = model.base_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_outputs.append(text)
            print(f"  Sample {i+1}/{n_samples} generated")
        
        # Simple majority vote (return most common or first if all different)
        from collections import Counter
        # Extract just the answer part (last sentence or line)
        answers = [o.strip().split('\n')[-1] for o in all_outputs]
        most_common = Counter(answers).most_common(1)[0][0]
        # Return the full output that ends with this answer
        for o in all_outputs:
            if o.strip().endswith(most_common) or most_common in o:
                print(f"\n[Majority Vote Result]\n{o}")
                return outputs
        print(f"\n{all_outputs[0]}")
        return outputs
    
    # Standard single generation
    with torch.no_grad():
        outputs = model.base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer
        )
    
    if not args.stream:
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated)
    
    return outputs


def analyze_routing(model, tokenizer, prompt: str):
    """Analyze which experts are selected for a prompt."""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    print(f"\n{'='*50}")
    print("Expert Routing Analysis")
    print(f"{'='*50}")
    
    # Get hidden states
    with torch.no_grad():
        outputs = model.base_model(
            **inputs,
            output_hidden_states=True
        )
    
    # Analyze routing for each SRDE layer
    for layer_idx, srde_layer in model.srde_layers.items():
        # Get hidden state for this layer
        layer_num = int(layer_idx)
        if outputs.hidden_states and layer_num < len(outputs.hidden_states):
            hidden = outputs.hidden_states[layer_num]
            
            # Get router decisions
            router_logits, router_weights, selected_experts = srde_layer.router(hidden)
            
            # Average expert usage across sequence
            avg_weights = router_logits.softmax(dim=-1).mean(dim=[0, 1])
            
            print(f"\nLayer {layer_idx}:")
            for expert_idx, weight in enumerate(avg_weights):
                bar = "█" * int(weight.item() * 40)
                print(f"  Expert {expert_idx}: {weight.item():.3f} {bar}")
    
    return


def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = load_model(args)
    
    # Generate
    generate(model, tokenizer, args.prompt, args)
    
    # Analyze routing
    print("\n")
    analyze_routing(model, tokenizer, args.prompt)


def interactive_mode():
    """Interactive chat mode."""
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = load_model(args)
    
    print("\n" + "="*60)
    print("SRDE Interactive Mode")
    print("Type 'quit' to exit, 'analyze' to see routing for last prompt")
    print("="*60 + "\n")
    
    last_prompt = None
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'analyze' and last_prompt:
                analyze_routing(model, tokenizer, last_prompt)
                continue
            elif not prompt:
                continue
            
            last_prompt = prompt
            print("\nSRDE: ", end="")
            generate(model, tokenizer, prompt, args)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        interactive_mode()
    else:
        main()
