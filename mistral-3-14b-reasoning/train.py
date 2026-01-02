#!/usr/bin/env python3
"""
SaRDinE Training Script
Run with: accelerate launch train.py

Or single GPU: python train.py
"""

import os
import gc
import json
import random
import threading
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
login(token='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')

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
    'checkpoint_dir': './checkpoints',
}

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
    'code': [('ise-uiuc/Magicoder-OSS-Instruct-75K', None), ('bigcode/starcoderdata', 'python')],
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
        while not self.stop_event.is_set():
            with self.lock:
                if self.buffer_token_count >= self.buffer_tokens:
                    time.sleep(0.1)
                    continue

            for source_name, subset in self.sources:
                if self.stop_event.is_set():
                    break
                try:
                    ds = load_dataset(source_name, subset, split='train', streaming=True, token=True)
                    for sample in ds:
                        if self.stop_event.is_set():
                            break
                        text = format_sample(sample, source_name)
                        if len(text) < 50:
                            continue
                        tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                        with self.lock:
                            self.buffer.append({'text': text, 'domain': self.domain, 'tokens': tokens})
                            self.buffer_token_count += tokens
                            if self.buffer_token_count >= self.buffer_tokens * 1.2:
                                break
                except Exception as e:
                    print(f"Stream error {source_name}: {e}")

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

    accelerator.print(f'\n=== Training ({world_size} GPUs) ===')
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

                wandb.log({'loss': lv, 'step': step, 'lr': lr, 'tokens_B': total_tokens/1e9, 'buffer_M': buf, **gpu_stats})
                pbar.set_postfix({'loss': f'{lv:.4f}', 'gpu': f"{gpu_stats.get('gpu_total_gb', 0):.1f}GB"})
                pbar.update(tokens_per_batch * 10)

            if step % CONFIG['save_steps'] == 0 and is_main:
                accelerator.wait_for_everyone()
                ckpt = output_dir / f'ckpt-{step}'
                ckpt.mkdir(exist_ok=True, parents=True)
                accelerator.unwrap_model(model).save_srde_weights(str(ckpt / 'weights.pt'))
                accelerator.print(f'\nSaved: {ckpt}')

    except KeyboardInterrupt:
        accelerator.print('\nInterrupted')
    finally:
        dataset.stop_streams()
        if is_main and step > 0:
            accelerator.wait_for_everyone()
            final = output_dir / f'final-{total_tokens//1e9:.0f}B'
            final.mkdir(exist_ok=True, parents=True)
            accelerator.unwrap_model(model).save_srde_weights(str(final / 'weights.pt'))
            wandb.finish()
        cleanup()


if __name__ == '__main__':
    main()
