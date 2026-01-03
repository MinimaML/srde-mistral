#!/usr/bin/env python3
"""
SaRDinE Training Script
Run with: accelerate launch train.py

Or single GPU: python train.py
"""

import os
# Disable tokenizer parallelism to avoid 'Already borrowed' error in multi-GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import json
import random
import shutil
import threading
import numpy as np
from collections import deque
from pathlib import Path

import torch
import wandb
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from huggingface_hub import login
from accelerate import Accelerator

from config import SRDEConfig
from srde import create_srde_model
from muon import Muon

# === CONFIG ===
os.environ['WANDB_API_KEY'] = 'abe32d5463fb2265eaea4563a571c07e5a39b7b6'
if os.environ.get('HF_TOKEN'):
    login(token=os.environ['HF_TOKEN'])

MODEL_NAME = 'mistralai/Ministral-3-14B-Reasoning-2512'
SRDE_CFG = SRDEConfig()
SEQ_LEN = 2048
TARGET_TOKENS = 20_000_000_000
BUFFER_TOKENS_PER_EXPERT = 1_000_000
BATCH_SIZE = 4
GRAD_ACCUM = 4

CONFIG = {
    'wandb_project': 'sardine-collab',
    'model_name': MODEL_NAME,
    'max_length': SEQ_LEN,
    'target_tokens': TARGET_TOKENS,
    'buffer_tokens': BUFFER_TOKENS_PER_EXPERT,
    'max_steps': 1000000,
    'lr': 1e-4,
    'muon_lr': 0.02,
    'warmup_steps': 1000,
    'save_steps': 5000,
    'num_experts': SRDE_CFG.num_experts,
    'batch_size': BATCH_SIZE,
    'grad_accum': GRAD_ACCUM,
    'grad_accum': GRAD_ACCUM,
    # --- PATHS (Change these for your setup!) ---
    'data_dir': '/mnt/sardine',       # Where to cache training data
    'checkpoint_dir': '/mnt/sardine/checkpoints',   # Where to save model checkpoints
}

# Parse args to override config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
parser.add_argument('--data_dir', type=str, default=None, help='Directory to cache training data')
args, unknown = parser.parse_known_args()

if args.checkpoint_dir:
    CONFIG['checkpoint_dir'] = args.checkpoint_dir
if args.data_dir:
    CONFIG['data_dir'] = args.data_dir

# Set Hugging Face cache directory
if CONFIG['data_dir']:
    os.environ['HF_HOME'] = CONFIG['data_dir']
    print(f"Dataset cache set to: {CONFIG['data_dir']}")

DOMAINS = {
    'math': {'expert': 0, 'weight': 0.30},
    'code': {'expert': 1, 'weight': 0.30},
    'science': {'expert': 2, 'weight': 0.15},
    'logic': {'expert': 3, 'weight': 0.10},
    'planning': {'expert': 4, 'weight': 0.08},
    'abstract': {'expert': 5, 'weight': 0.07},
}

DATA_SOURCES = {
    'math': [('nvidia/OpenMathInstruct-2', None), ('meta-math/MetaMathQA', None)],
    'code': [('ise-uiuc/Magicoder-OSS-Instruct-75K', None), ('bigcode/starcoderdata', None)],
    'science': [('HuggingFaceFW/fineweb-edu', 'sample-10BT')],
    'logic': [('Rowan/hellaswag', None)],
    'planning': [('hotpot_qa', 'fullwiki')],
    'abstract': [('deepmind/aqua_rat', 'raw'), ('winogrande', 'winogrande_xl')],
}


def format_sample(sample, source_name):
    try:
        if 'OpenMath' in source_name:
            return f"Problem: {sample.get('problem', '')}\nSolution: {sample.get('generated_solution', '')}"
        if 'MetaMath' in source_name:
            return f"Q: {sample.get('query', '')}\nA: {sample.get('response', '')}"
        if 'Magicoder' in source_name:
            return f"### Instruction:\n{sample.get('problem', '')}\n### Solution:\n{sample.get('solution', '')}"
        if 'starcoder' in source_name:
            return sample.get('content', '')
        if 'hellaswag' in source_name:
            return f"{sample.get('ctx', '')} {sample.get('endings', [''])[int(sample.get('label', 0))]}"
        if 'hotpot' in source_name:
            return f"Q: {sample.get('question', '')}\nA: {sample.get('answer', '')}"
        if 'aqua' in source_name:
            return f"Q: {sample.get('question', '')}\nA: {sample.get('rationale', '')}"
        if 'winogrande' in source_name:
            return sample.get('sentence', '')
        return sample.get('text', str(sample)[:1000])
    except:
        return ''


class BufferedDomainStream:
    def __init__(self, domain, sources, tokenizer, buffer_tokens=1_000_000):
        self.domain = domain
        self.sources = sources
        self.tokenizer = tokenizer
        self.buffer_tokens = buffer_tokens
        self.buffer = deque()
        self.buffer_token_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._fill_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)

    def _fill_loop(self):
        import time
        from requests.exceptions import RequestException, ConnectionError, ReadTimeout
        
        # Exponential backoff config
        MAX_RETRIES = 5
        BASE_DELAY = 5
        
        while not self.stop_event.is_set():
            with self.lock:
                if self.buffer_token_count >= self.buffer_tokens:
                    time.sleep(0.1)
                    continue

            any_success_this_cycle = False
            
            for source_name, subset in self.sources:
                if self.stop_event.is_set():
                    break
                
                # Retry loop for initial connection
                for attempt in range(MAX_RETRIES):
                    try:
                        # Attempt to connect to streaming dataset
                        ds = load_dataset(source_name, subset, split='train', streaming=True, token=True)
                        
                        # Test connection immediately
                        iterator = iter(ds)
                        
                        # Iterate with error handling inside the stream
                        while True:
                            if self.stop_event.is_set(): break
                            try:
                                sample = next(iterator)
                                
                                # Process sample
                                text = format_sample(sample, source_name)
                                if len(text) < 50:
                                    continue
                                    
                                tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                                with self.lock:
                                    self.buffer.append({'text': text, 'domain': self.domain, 'tokens': tokens})
                                    self.buffer_token_count += tokens
                                    
                                    # Break inner loop if buffer full
                                    if self.buffer_token_count >= self.buffer_tokens * 1.2:
                                        break
                                
                                any_success_this_cycle = True
                                
                            except StopIteration:
                                break # End of dataset
                            except (RequestException, ConnectionError, ReadTimeout) as e:
                                print(f"⚠️ Stream interrupted for {source_name}: {e}. Resuming next source...")
                                time.sleep(BASE_DELAY)
                                break # Move to next source
                            except Exception as e:
                                print(f"⚠️ Unexpected stream error {source_name}: {e}")
                                break
                        
                        # If we successfully iterated (even if we broke early), consider this source 'done' for now
                        break 
                        
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            delay = BASE_DELAY * (2 ** attempt)
                            print(f"⚠️ Connection failed for {source_name} (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            print(f"❌ Failed to load {source_name} after {MAX_RETRIES} attempts: {e}")
            
            # If we went through all sources and got ZERO data (total outage?), sleep longer
            if not any_success_this_cycle:
                print(f"⚠️ No data received from domain '{self.domain}'. Retrying in 30s...")
                time.sleep(30)

    def get_sample(self):
        with self.lock:
            if self.buffer:
                sample = self.buffer.popleft()
                self.buffer_token_count -= sample['tokens']
                return sample
        return None

    def buffer_status(self):
        with self.lock:
            return self.buffer_token_count


class BufferedStreamDataset(IterableDataset):
    def __init__(self, domains, tokenizer, buffer_tokens=1_000_000):
        self.domains = list(domains.keys())
        self.weights = [domains[d]['weight'] for d in self.domains]
        self.tokenizer = tokenizer
        self.buffer_tokens = buffer_tokens
        self.streams = {}
        self.started = False

    def start_streams(self):
        if self.started:
            return
        print("Starting background streams...")
        for domain in self.domains:
            sources = DATA_SOURCES.get(domain, [])
            self.streams[domain] = BufferedDomainStream(domain, sources, self.tokenizer, self.buffer_tokens)
            self.streams[domain].start()
        self.started = True

        print("Filling buffers...")
        import time
        while True:
            statuses = {d: s.buffer_status() for d, s in self.streams.items()}
            min_fill = min(statuses.values())
            print(f"  Buffer: {min_fill/1e6:.2f}M / {self.buffer_tokens/1e6:.0f}M", end='\r')
            if min_fill >= self.buffer_tokens * 0.5:
                print("\nBuffers ready!")
                break
            time.sleep(1)

    def stop_streams(self):
        for stream in self.streams.values():
            stream.stop()

    def __iter__(self):
        self.start_streams()
        rng = random.Random(42)
        while True:
            domain = rng.choices(self.domains, weights=self.weights, k=1)[0]
            sample = self.streams[domain].get_sample()
            if sample:
                yield sample
            else:
                for d in self.domains:
                    sample = self.streams[d].get_sample()
                    if sample:
                        yield sample
                        break


def get_loss(out):
    if isinstance(out, dict):
        return out.get('loss')
    return getattr(out, 'loss', None)


def get_params_for_muon(params):
    return [p for p in params if p.ndim >= 2], [p for p in params if p.ndim < 2]


def get_gpu_stats():
    stats = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            stats[f'gpu{i}_gb'] = allocated
            stats[f'gpu{i}_pct'] = (allocated / total) * 100
        stats['gpu_total_gb'] = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
    return stats


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=CONFIG['grad_accum'])
    is_main = accelerator.is_main_process
    device = accelerator.device
    world_size = accelerator.num_processes

    if is_main:
        wandb.init(project=CONFIG['wandb_project'], name=f'sardine-{world_size}gpu', config={**CONFIG, 'world_size': world_size})

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = BufferedStreamDataset(DOMAINS, tokenizer, CONFIG['buffer_tokens'])

    def collate_fn(batch):
        texts = [b['text'][:8192] for b in batch]
        enc = tokenizer(texts, truncation=True, max_length=CONFIG['max_length'], padding='max_length', return_tensors='pt')
        return enc['input_ids'], enc['attention_mask']

    train_dl = DataLoader(dataset, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)

    accelerator.print(f"Loading model...")
    device_map = {'': str(device)}
    model = create_srde_model(CONFIG['model_name'], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map)

    trainable_count = 0
    for name, param in model.named_parameters():
        if 'srde_layers' in name or 'expert' in name or 'router' in name or 'vocabulary' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    accelerator.print(f'Trainable: {trainable_count/1e6:.1f}M')

    if hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    accelerator.print(f'GPU stats: {get_gpu_stats()}')

    accelerator.print(f'\n🌊 SaRDinE-14B: Sparse Routed Delta Experts Training ({world_size} GPUs) 🌊')
    all_params = [p for p in model.parameters() if p.requires_grad]
    muon_p, adam_p = get_params_for_muon(all_params)
    accelerator.print(f'Muon: {sum(p.numel() for p in muon_p)/1e6:.1f}M | Adam: {sum(p.numel() for p in adam_p)/1e6:.1f}M')

    opts, scheds = [], []
    if muon_p:
        o = Muon(muon_p, lr=CONFIG['muon_lr'], momentum=0.95)
        opts.append(o)
        scheds.append(get_cosine_schedule_with_warmup(o, CONFIG['warmup_steps'], CONFIG['max_steps']))
    if adam_p:
        o = bnb.optim.AdamW8bit(adam_p, lr=CONFIG['lr'])
        opts.append(o)
        scheds.append(get_cosine_schedule_with_warmup(o, CONFIG['warmup_steps'], CONFIG['max_steps']))

    prepared = accelerator.prepare(model, train_dl, *opts, *scheds)
    model = prepared[0]
    train_dl = prepared[1]
    n = len(opts)
    opts = list(prepared[2:2+n])
    scheds = list(prepared[2+n:])

    output_dir = Path(CONFIG['checkpoint_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    step = 0
    total_tokens = 0
    tokens_per_batch = CONFIG['batch_size'] * CONFIG['max_length'] * world_size

    pbar = tqdm(total=CONFIG['target_tokens'], unit='tok', unit_scale=True, disable=not is_main)

    # --- CHECKPOINT RESUMPTION ---
    resume_ckpt = output_dir / 'latest'
    if resume_ckpt.exists() and (resume_ckpt / 'training_state.pt').exists():
        accelerator.print(f'Resuming from {resume_ckpt}...')
        try:
            state = torch.load(resume_ckpt / 'training_state.pt', map_location='cpu')
            step = state.get('step', 0)
            total_tokens = state.get('total_tokens', 0)
            
            # Load optimizer states
            for i, o in enumerate(opts):
                if f'optimizer_{i}' in state:
                    o.load_state_dict(state[f'optimizer_{i}'])
            
            # Load scheduler states  
            for i, s in enumerate(scheds):
                if f'scheduler_{i}' in state:
                    s.load_state_dict(state[f'scheduler_{i}'])
            
            # Load SRDE weights
            if (resume_ckpt / 'weights.pt').exists():
                accelerator.unwrap_model(model).load_srde_weights(str(resume_ckpt / 'weights.pt'))
            
            # Restore RNG states for reproducibility
            if 'rng_state' in state:
                torch.set_rng_state(state['rng_state'])
            if 'cuda_rng_state' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state(state['cuda_rng_state'])
            
            pbar.update(total_tokens)
            accelerator.print(f'Resumed at step {step}, {total_tokens/1e9:.2f}B tokens')
        except Exception as e:
            accelerator.print(f'Failed to resume: {e}, starting fresh')
            step = 0
            total_tokens = 0
    # -----------------------------

    model.train()
    try:
        for ids, mask in train_dl:
            if total_tokens >= CONFIG['target_tokens']:
                accelerator.print(f'\nReached {total_tokens/1e9:.1f}B tokens!')
                break
            if os.path.exists('STOP_TRAINING'):
                break

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out = model(ids, attention_mask=mask, labels=ids)
                    loss = get_loss(out)
                if loss is not None:
                    accelerator.backward(loss)
                    for o in opts:
                        o.step()
                        o.zero_grad()
                    for s in scheds:
                        s.step()

            step += 1
            total_tokens += tokens_per_batch

            if step % 10 == 0 and is_main:
                lr = scheds[0].get_last_lr()[0] if scheds else 0
                lv = loss.item() if loss else 0
                buf = sum(s.buffer_status() for s in dataset.streams.values()) / 1e6 if dataset.streams else 0
                gpu_stats = get_gpu_stats()
                
                # --- SRDE STATS ---
                srde_metrics = {}
                try:
                    entropies, confidences, usages = [], [], []
                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, 'srde_layers'):
                        for n, m in unwrapped.srde_layers.items():
                             if hasattr(m, 'stats_queue') and m.stats_queue:
                                s = m.stats_queue[-1]
                                entropies.append(s['entropy'])
                                confidences.append(s['confidence'])
                                usages.append(s['usage'])
                                
                                if step % 100 == 0:
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(6, 2))
                                    ax.bar(range(len(s['usage'])), s['usage'])
                                    ax.set_title(f'Layer {n} Usage')
                                    srde_metrics[f'experts/layer_{n}'] = wandb.Image(fig)
                                    plt.close(fig)

                        if entropies:
                            srde_metrics['router/entropy'] = np.mean(entropies)
                            srde_metrics['router/confidence'] = np.mean(confidences)
                            avg_usage = np.mean(usages, axis=0) # [num_experts]
                            srde_metrics['experts/global_std'] = np.std(avg_usage)
                except Exception as e:
                    print(f"Stats error: {e}")
                # ------------------

                wandb.log({'loss': lv, 'step': step, 'lr': lr, 'tokens_B': total_tokens/1e9, 'buffer_M': buf, **gpu_stats, **srde_metrics})
                pbar.set_postfix({'loss': f'{lv:.4f}', 'gpu': f"{gpu_stats.get('gpu_total_gb', 0):.1f}GB"})
                pbar.update(tokens_per_batch * 10)

            # --- CHECKPOINT SAVING (all ranks sync first!) ---
            if step % CONFIG['save_steps'] == 0:
                accelerator.wait_for_everyone()  # ALL ranks must sync before save
                if is_main:
                    ckpt = output_dir / f'ckpt-{step}'
                    ckpt.mkdir(exist_ok=True, parents=True)
                    
                    # Save SRDE weights
                    accelerator.unwrap_model(model).save_srde_weights(str(ckpt / 'weights.pt'))
                    
                    # Save full training state for resumption
                    training_state = {
                        'step': step,
                        'total_tokens': total_tokens,
                        'config': CONFIG,
                        'rng_state': torch.get_rng_state(),
                    }
                    
                    # Save optimizer states
                    for i, o in enumerate(opts):
                        training_state[f'optimizer_{i}'] = o.state_dict()
                    
                    # Save scheduler states
                    for i, s in enumerate(scheds):
                        training_state[f'scheduler_{i}'] = s.state_dict()
                    
                    # Save CUDA RNG state if available
                    if torch.cuda.is_available():
                        training_state['cuda_rng_state'] = torch.cuda.get_rng_state()
                    
                    # Atomic save: write to temp then rename
                    temp_path = ckpt / 'training_state.pt.tmp'
                    torch.save(training_state, temp_path)
                    temp_path.rename(ckpt / 'training_state.pt')
                    
                    # Update 'latest' symlink/copy for easy resumption
                    latest = output_dir / 'latest'
                    if latest.exists():
                        import shutil
                        shutil.rmtree(latest)
                    shutil.copytree(ckpt, latest)
                    
                    accelerator.print(f'\nSaved: {ckpt}')

    except KeyboardInterrupt:
        accelerator.print('\nInterrupted')
    finally:
        dataset.stop_streams()
        
        # --- FINAL CHECKPOINT (all ranks sync!) ---
        if step > 0:
            accelerator.wait_for_everyone()  # ALL ranks must sync
            if is_main:
                final = output_dir / f'final-{int(total_tokens/1e9)}B'
                final.mkdir(exist_ok=True, parents=True)
                
                # Save SRDE weights
                accelerator.unwrap_model(model).save_srde_weights(str(final / 'weights.pt'))
                
                # Save full training state
                training_state = {
                    'step': step,
                    'total_tokens': total_tokens,
                    'config': CONFIG,
                    'rng_state': torch.get_rng_state(),
                }
                for i, o in enumerate(opts):
                    training_state[f'optimizer_{i}'] = o.state_dict()
                for i, s in enumerate(scheds):
                    training_state[f'scheduler_{i}'] = s.state_dict()
                if torch.cuda.is_available():
                    training_state['cuda_rng_state'] = torch.cuda.get_rng_state()
                
                torch.save(training_state, final / 'training_state.pt')
                
                # Update 'latest' for resumption
                latest = output_dir / 'latest'
                if latest.exists():
                    shutil.rmtree(latest)
                shutil.copytree(final, latest)
                
                accelerator.print(f'\nFinal saved: {final}')
                wandb.finish()
        
        cleanup()


if __name__ == '__main__':
    main()
