#!/usr/bin/env python3
"""
MedGemma AntiHallu Demo — FastAPI Backend (Dockerized)

Loads original MedGemma-1.5-4B and DAPO-trained (iter_004) checkpoint side by
side.  Both run simple greedy generation + neuron-probe scoring — no multi-layer
defense pipeline.  This gives an honest 1-vs-1 comparison of the base model
against the DAPO/GRPO fine-tuned model.

Docker usage:
    Launched via docker-compose with GPU access.
    Models mounted at /model_dapo and HF cache at /hf_cache.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths (Docker) ──
APP_DIR = Path("/app")
sys.path.insert(0, str(APP_DIR))

from medgemma_detector import (
    HallucinationDetector,
    get_model_layers,
    extract_activations_single,
    format_chat_prompt,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'

# Models are pre-cached in the mounted volume — run offline to avoid
# empty-token errors (Bearer '' is illegal in newer httpx).
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Force model/tokenizer cache under /hf_cache (Docker volume).
HF_CACHE_ROOT = os.environ.get('HF_HOME', '/hf_cache')
HF_HUB_CACHE = os.environ.get('HUGGINGFACE_HUB_CACHE', f'{HF_CACHE_ROOT}/hub')
TRANSFORMERS_CACHE = os.environ.get('TRANSFORMERS_CACHE', HF_HUB_CACHE)
os.environ['HF_HOME'] = HF_CACHE_ROOT
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_HUB_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

# ── FastAPI ──
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ── Config ──
MODEL_ID = "google/medgemma-1.5-4b-it"
GRPO_CHECKPOINT = Path(os.environ.get('GRPO_MODEL_PATH', '/model_dapo'))
CALIBRATION_DIR = APP_DIR / "calibration" / "medgemma-1.5-4b-it"
EXAMPLES_FILE = APP_DIR / "examples.json"
CACHE_FILE = APP_DIR / "cache.json"
HF_TOKEN = os.environ.get('HF_TOKEN')

# Risk thresholds (from calibration config)
THRESHOLDS = {'low': 0.3, 'medium': 0.5, 'high': 0.7}

# Default system prompt for the DAPO model
DAPO_SYSTEM_PROMPT = os.environ.get('DAPO_SYSTEM_PROMPT', '')


app = FastAPI(title="MedGemma AntiHallu Demo")

# ── Global state ──
state = {
    'models_loaded': False,
    'original_model': None,
    'grpo_model': None,
    'tokenizer': None,
    'neuron_indices': None,
    'classifier': None,
    'probe_layer_idx': None,
    'original_probe_layer': None,
    'grpo_probe_layer': None,
    'cache': {},
}


# ============================================================
# Model Loading
# ============================================================

def load_tokenizer() -> AutoTokenizer:
    """Load tokenizer with MedGemma EOS fix."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir=TRANSFORMERS_CACHE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # MedGemma terminates with <end_of_turn> (token 106), not <eos> (token 1)
    eot_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    if eot_id != tokenizer.unk_token_id:
        tokenizer.eos_token = '<end_of_turn>'
    return tokenizer


def load_model_on_device(model_path: str, device: str) -> AutoModelForCausalLM:
    """Load a model in bf16 mode on a specific device."""
    print(f"Loading model from {model_path} on {device}...")
    print(f"  Cache dir: {TRANSFORMERS_CACHE}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={'': device},
        trust_remote_code=True,
        attn_implementation='sdpa',
        token=HF_TOKEN,
        cache_dir=TRANSFORMERS_CACHE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


def load_cache():
    """Load pre-cached example responses."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


# ============================================================
# Inference helpers
# ============================================================

def generate_response(model, tokenizer, question, device, max_new_tokens=256, system_prompt=None):
    """Generate a greedy response with MedGemma EOS fix."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"{system_prompt}\n\n{question}\n\nAnswer:"
    else:
        prompt = format_chat_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            eos_token_id=[1, 106],
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def score_response(model, probe_layer, tokenizer, question, answer, device):
    """Score a Q/A pair using the neuron probe."""
    text = f"Question: {question}\nAnswer: {answer}"
    hidden = extract_activations_single(model, probe_layer, text, tokenizer, device)
    features = hidden[state['neuron_indices']].reshape(1, -1)
    prob = float(state['classifier'].predict_proba(features)[0, 1])

    if prob < THRESHOLDS['low']:
        risk = 'SAFE'
    elif prob < THRESHOLDS['medium']:
        risk = 'UNCERTAIN'
    elif prob < THRESHOLDS['high']:
        risk = 'ELEVATED'
    else:
        risk = 'HIGH'

    return {'hallucination_prob': round(prob, 4), 'risk_level': risk}


def run_original(question: str) -> Dict:
    """Generate and score with original model (baseline, no defense)."""
    model = state['original_model']
    probe_layer = state['original_probe_layer']
    tokenizer = state['tokenizer']
    device = str(next(model.parameters()).device)

    t0 = time.time()
    response = generate_response(model, tokenizer, question, device)
    score = score_response(model, probe_layer, tokenizer, question, response, device)

    return {
        'response': response,
        'hallucination_prob': score['hallucination_prob'],
        'risk_level': score['risk_level'],
        'latency_ms': round((time.time() - t0) * 1000),
    }


def run_grpo(question: str, system_prompt: str = None) -> Dict:
    """Generate and score with DAPO/GRPO-trained model (direct, no pipeline)."""
    model = state['grpo_model']
    probe_layer = state['grpo_probe_layer']
    tokenizer = state['tokenizer']
    device = str(next(model.parameters()).device)

    t0 = time.time()
    response = generate_response(model, tokenizer, question, device, system_prompt=system_prompt)
    score = score_response(model, probe_layer, tokenizer, question, response, device)

    return {
        'response': response,
        'hallucination_prob': score['hallucination_prob'],
        'risk_level': score['risk_level'],
        'latency_ms': round((time.time() - t0) * 1000),
    }


# ============================================================
# API
# ============================================================

class GenerateRequest(BaseModel):
    question: str
    system_prompt: str = None


@app.on_event("startup")
async def startup_event():
    """Load models and calibration on server start."""
    print("=" * 60)
    print("MedGemma AntiHallu Demo — Starting (Docker)")
    print("=" * 60)

    # Determine devices
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        original_device, grpo_device = 'cuda:0', 'cuda:1'
    elif n_gpus == 1:
        original_device = grpo_device = 'cuda:0'
        print("WARNING: Single GPU — both models on cuda:0 (~16GB)")
    else:
        raise RuntimeError("No GPU available")

    # Load tokenizer (shared)
    state['tokenizer'] = load_tokenizer()
    print(f"Tokenizer loaded (eos_token={state['tokenizer'].eos_token!r})")

    # Load calibration
    import pickle
    with open(CALIBRATION_DIR / 'config.json') as f:
        config = json.load(f)
    state['neuron_indices'] = np.load(CALIBRATION_DIR / 'neuron_indices.npy')
    with open(CALIBRATION_DIR / 'classifier.pkl', 'rb') as f:
        state['classifier'] = pickle.load(f)
    state['probe_layer_idx'] = config['layer_idx']
    print(f"Calibration loaded: layer {config['layer_idx']}, "
          f"{len(state['neuron_indices'])} neurons, CV AUROC={config['cv_auroc']:.3f}")

    # Load original model
    state['original_model'] = load_model_on_device(MODEL_ID, original_device)
    original_layers = get_model_layers(state['original_model'])
    state['original_probe_layer'] = original_layers[state['probe_layer_idx']]

    # Load GRPO checkpoint
    if GRPO_CHECKPOINT.exists():
        state['grpo_model'] = load_model_on_device(str(GRPO_CHECKPOINT), grpo_device)
    else:
        print(f"WARNING: GRPO checkpoint not found at {GRPO_CHECKPOINT}")
        print("  Falling back to original model as GRPO placeholder")
        state['grpo_model'] = load_model_on_device(MODEL_ID, grpo_device)
    grpo_layers = get_model_layers(state['grpo_model'])
    state['grpo_probe_layer'] = grpo_layers[state['probe_layer_idx']]

    print("Direct L1 comparison mode (no defense pipeline)")

    # Load cache
    state['cache'] = load_cache()
    if state['cache']:
        print(f"Loaded {len(state['cache'])} cached responses")

    state['models_loaded'] = True
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 60)
    print("Server ready!")
    print("=" * 60)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not state['models_loaded']:
        raise HTTPException(503, "Models still loading")

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # Always run live inference — cache is only used as Django-side fallback
    # Determine system prompt: per-request > global default > none
    sys_prompt = req.system_prompt or DAPO_SYSTEM_PROMPT or None

    # Run both models — direct greedy generation + probe scoring
    original_result = run_original(question)
    grpo_result = run_grpo(question, system_prompt=sys_prompt)

    result = {
        'question': question,
        'original': original_result,
        'defended': grpo_result,
        'cached': False,
    }
    return result


@app.get("/api/examples")
async def get_examples():
    if EXAMPLES_FILE.exists():
        with open(EXAMPLES_FILE) as f:
            return json.load(f)
    return {"categories": []}


@app.get("/api/health")
async def health():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.mem_get_info(i)
            gpu_info.append({
                'device': i,
                'name': torch.cuda.get_device_name(i),
                'free_gb': round(mem[0] / 1e9, 1),
                'total_gb': round(mem[1] / 1e9, 1),
            })

    return {
        'models_loaded': state['models_loaded'],
        'gpu_count': torch.cuda.device_count(),
        'gpu_memory': gpu_info,
        'cached_responses': len(state['cache']),
    }


class ConfigRequest(BaseModel):
    system_prompt: str = None


@app.post("/api/config")
async def set_config(req: ConfigRequest):
    """Update the default DAPO system prompt at runtime (no restart needed)."""
    global DAPO_SYSTEM_PROMPT
    if req.system_prompt is not None:
        DAPO_SYSTEM_PROMPT = req.system_prompt
    return {'system_prompt': DAPO_SYSTEM_PROMPT}


@app.get("/api/config")
async def get_config():
    return {'system_prompt': DAPO_SYSTEM_PROMPT}


if __name__ == '__main__':
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )
