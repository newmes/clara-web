#!/usr/bin/env python3
"""
Production Hallucination Detector for Medical LLMs

Two modes:
  --calibrate   Select discriminative neurons on MedHallu, save artifacts
  --interactive  Load calibration, generate + score medical Q&A interactively

Example:
  # One-time calibration (~15 min)
  CUDA_VISIBLE_DEVICES=0 python production/medgemma_detector.py --calibrate

  # Interactive use
  CUDA_VISIBLE_DEVICES=0 python production/medgemma_detector.py --interactive
"""

import os
import sys
import json
import pickle
import argparse
import gc
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ANSI colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'
CYAN = '\033[96m'

MODEL_CONFIGS = {
    'google/medgemma-1.5-4b-it': {
        'name': 'MedGemma-1.5-4B',
        'n_layers': 34,
        'hidden_size': 2560,
    },
    'google/medgemma-4b-it': {
        'name': 'MedGemma-4B',
        'n_layers': 34,
        'hidden_size': 2560,
    },
    'google/medgemma-27b-text-it': {
        'name': 'MedGemma-27B',
        'n_layers': 62,
        'hidden_size': 4608,
    },
    'Qwen/Qwen3-8B': {
        'name': 'Qwen3-8B',
        'n_layers': 36,
        'hidden_size': 4096,
    },
}


# ============================================================================
# Model utilities
# ============================================================================

def get_model_layers(model):
    """Get transformer layers regardless of architecture."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        if hasattr(lm, 'layers'):
            return lm.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in model: {type(model)}")


def load_model(model_id: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={'': device},
        trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def format_chat_prompt(question: str, tokenizer) -> str:
    """Format a medical question using the model's chat template."""
    messages = [
        {"role": "user", "content": f"Answer the following medical question concisely and accurately:\n\n{question}"}
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Answer the following medical question concisely and accurately:\n\n{question}\n\nAnswer:"


# ============================================================================
# Activation extraction
# ============================================================================

def extract_activations_batch(
    model, tokenizer, texts: List[str],
    layer_idx: int, device: str,
    max_length: int = 512,
) -> np.ndarray:
    """Extract last-token activations from a specific layer for a batch of texts."""
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    all_activations = []

    for text in tqdm(texts, desc=f"Extracting layer {layer_idx}", leave=False):
        inputs = tokenizer(
            text, return_tensors='pt',
            truncation=True, max_length=max_length,
        ).to(device)

        captured = []
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured.append(output[0].detach())
            else:
                captured.append(output.detach())

        hook = layer.register_forward_hook(capture_hook)
        with torch.no_grad():
            model(**inputs)
        hook.remove()

        hidden = captured[0][0, -1, :].float().cpu().numpy()
        all_activations.append(hidden)

    return np.array(all_activations)


def extract_activations_single(
    model, layer, text: str, tokenizer, device: str,
) -> np.ndarray:
    """Extract last-token activation for a single text (fast path)."""
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=512,
    ).to(device)

    captured = []
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured.append(output[0].detach())
        else:
            captured.append(output.detach())

    hook = layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    return captured[0][0, -1, :].float().cpu().numpy()


# ============================================================================
# Neuron selection (nested CV)
# ============================================================================

def compute_neuron_aurocs(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-neuron AUROC (max of auroc, 1-auroc)."""
    n_neurons = activations.shape[1]
    aurocs = np.zeros(n_neurons)
    for i in range(n_neurons):
        try:
            auc = roc_auc_score(labels, activations[:, i])
            aurocs[i] = max(auc, 1 - auc)
        except Exception:
            aurocs[i] = 0.5
    return aurocs


def select_neurons_nested_cv(
    activations: np.ndarray, labels: np.ndarray,
    n_inner_folds: int = 3, top_k: int = 100,
) -> np.ndarray:
    """Select top-k neurons via inner CV with median-rank aggregation."""
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    all_ranks = []
    for train_idx, _ in inner_cv.split(activations, labels):
        fold_aurocs = compute_neuron_aurocs(activations[train_idx], labels[train_idx])
        ranks = np.argsort(np.argsort(-fold_aurocs))
        all_ranks.append(ranks)
    median_ranks = np.median(all_ranks, axis=0)
    return np.argsort(median_ranks)[:top_k]


# ============================================================================
# Self-calibration utilities
# ============================================================================

def generate_self_calibration_data(
    model, tokenizer, device: str,
    n_generate: int = 200,
    max_new_tokens: int = 256,
) -> Tuple[List[Dict], np.ndarray]:
    """Generate balanced long-form calibration data with reliable labels.

    Strategy: For each MedHallu sample, prompt the model to expand/paraphrase
    BOTH the ground truth AND the hallucinated answer. This produces:
    - Long-form factual text (model expanding verified GT) → label 0
    - Long-form hallucinated text (model expanding false info) → label 1

    Both are in the model's generation style and similarly long, breaking
    the length-hallucination confound that causes false positives on
    model-generated text.
    """
    from datasets import load_dataset

    token = os.environ.get('HF_TOKEN')
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial",
                           split="train", token=token)

    qa_pairs = []
    for item in dataset:
        q = item.get('Question', '')
        gt = item.get('Ground Truth', '')
        halluc = item.get('Hallucinated Answer', '')
        if q and gt and halluc:
            qa_pairs.append((q, gt, halluc))

    rng = np.random.RandomState(42)
    n_pairs = min(n_generate // 2, len(qa_pairs))
    indices = rng.choice(len(qa_pairs), n_pairs, replace=False)

    generated_samples = []
    generated_labels = []

    print(f"\nSelf-calibration: expanding {n_pairs} Q/A pairs (factual + hallucinated each)...")
    for idx in tqdm(indices, desc="Self-calibrating"):
        q, gt, halluc_ans = qa_pairs[idx]

        for answer_text, label in [(gt, 0), (halluc_ans, 1)]:
            # Prompt the model to expand the answer in its own voice
            expand_prompt = (
                f"A student asked: \"{q}\"\n\n"
                f"A textbook says: \"{answer_text}\"\n\n"
                f"Please provide a comprehensive, detailed explanation "
                f"of this answer for the student."
            )
            messages = [{"role": "user", "content": expand_prompt}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = expand_prompt + "\n\nDetailed explanation:"

            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            expanded = tokenizer.decode(out[0][input_length:], skip_special_tokens=True)

            if len(expanded.strip()) < 10:
                continue

            # Score text uses same format as score() at inference time
            generated_samples.append({
                'question': q, 'answer': expanded,
                'text': f"Question: {q}\nAnswer: {expanded}",
            })
            generated_labels.append(label)

    labels = np.array(generated_labels, dtype=int) if generated_labels else np.array([], dtype=int)
    n_f = int((labels == 0).sum()) if len(labels) > 0 else 0
    n_h = int((labels == 1).sum()) if len(labels) > 0 else 0
    print(f"Self-calibration: {len(generated_samples)} generated "
          f"({n_f} factual, {n_h} halluc)")
    return generated_samples, labels


# ============================================================================
# Calibration pipeline
# ============================================================================

def load_medhallu(max_samples: int) -> Tuple[List[Dict], np.ndarray]:
    """Load MedHallu dataset for calibration."""
    from datasets import load_dataset
    print(f"Loading MedHallu (max={max_samples})...")
    token = os.environ.get('HF_TOKEN')
    dataset = load_dataset(
        "UTAustin-AIHealth/MedHallu", "pqa_artificial",
        split="train", token=token,
    )
    print(f"Raw MedHallu: {len(dataset)} items")

    samples, labels = [], []
    for item in dataset:
        if len(samples) >= max_samples:
            break
        question = item.get('Question', '')
        ground_truth = item.get('Ground Truth', '')
        hallucinated = item.get('Hallucinated Answer', '')

        if hallucinated:
            samples.append({
                'question': question,
                'answer': hallucinated,
                'text': f"Question: {question}\nAnswer: {hallucinated}",
            })
            labels.append(1)

        if ground_truth and len(samples) < max_samples:
            samples.append({
                'question': question,
                'answer': ground_truth,
                'text': f"Question: {question}\nAnswer: {ground_truth}",
            })
            labels.append(0)

    print(f"Loaded {len(samples)} samples ({sum(labels)} halluc, {len(labels)-sum(labels)} factual)")
    return samples, np.array(labels)


def calibrate(model, tokenizer, device: str, model_id: str, output_dir: Path,
              n_samples: int = 2000, top_k: int = 100, self_calibrate: int = 0):
    """Run calibration: probe layers, select neurons, train classifier, save artifacts.

    Args:
        self_calibrate: Number of model-generated responses to include in
            calibration data. Reduces false positives on generated text by
            exposing neuron selection to the model's own generation style.
    """
    config = MODEL_CONFIGS[model_id]
    n_layers = len(get_model_layers(model))

    samples, labels = load_medhallu(n_samples)

    # Generate self-calibration data (long-form model-generated text)
    gen_samples, gen_labels = None, None
    if self_calibrate > 0:
        gen_samples, gen_labels = generate_self_calibration_data(
            model, tokenizer, device, n_generate=self_calibrate,
        )

    # Decide which data to use for each phase
    if gen_samples is not None and len(gen_samples) >= 100:
        # Full self-calibration: use ONLY generated data for everything
        # (neuron selection + classifier training) to match inference distribution
        cal_samples = gen_samples
        cal_labels = gen_labels
        print(f"\nUsing {len(cal_samples)} generated samples for calibration "
              f"({int((cal_labels==0).sum())} factual, {int((cal_labels==1).sum())} halluc)")
    else:
        # Default: use MedHallu (for backward compatibility / no self-cal)
        cal_samples = samples
        cal_labels = labels

    cal_texts = [s['text'] for s in cal_samples]

    # Probe 3 candidate layers
    probe_layers = sorted(set(int(n_layers * p) for p in [0.4, 0.6, 0.8]))
    best_layer, best_auroc = probe_layers[1], 0.0
    layer_results = {}

    print(f"\nProbing {len(probe_layers)} layers...")
    for layer_idx in probe_layers:
        pct = layer_idx / n_layers * 100
        print(f"  Layer {layer_idx} ({pct:.0f}% depth)...", end=" ", flush=True)

        acts = extract_activations_batch(model, tokenizer, cal_texts, layer_idx, device)
        aurocs = compute_neuron_aurocs(acts, cal_labels)
        top_idx = np.argsort(aurocs)[-top_k:]
        clf = LogisticRegression(max_iter=5000, C=0.1, random_state=42)
        clf.fit(acts[:, top_idx], cal_labels)
        auroc = roc_auc_score(cal_labels, clf.predict_proba(acts[:, top_idx])[:, 1])

        layer_results[layer_idx] = {'auroc': float(auroc), 'depth_pct': pct}
        print(f"AUROC={auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer_idx

        del acts; gc.collect(); torch.cuda.empty_cache()

    print(f"\nBest layer: {best_layer} (AUROC={best_auroc:.4f})")

    # Full extraction at best layer
    print(f"Full extraction at layer {best_layer}...")
    activations = extract_activations_batch(model, tokenizer, cal_texts, best_layer, device)

    # Nested CV neuron selection
    print("Selecting neurons via nested CV (3-fold inner)...")
    neuron_indices = select_neurons_nested_cv(activations, cal_labels, top_k=top_k)

    # Train final classifier
    print("Training classifier...")
    X = activations[:, neuron_indices]
    clf = LogisticRegression(max_iter=5000, C=0.1, random_state=42)
    clf.fit(X, cal_labels)

    # 5-fold CV for validation AUROC
    print("Validating via 5-fold CV...")
    from scipy import stats
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs = []
    for train_idx, test_idx in cv.split(X, cal_labels):
        sel = select_neurons_nested_cv(activations[train_idx], cal_labels[train_idx], top_k=top_k)
        c = LogisticRegression(max_iter=5000, C=0.1, random_state=42)
        c.fit(activations[train_idx][:, sel], cal_labels[train_idx])
        cv_aurocs.append(roc_auc_score(
            cal_labels[test_idx], c.predict_proba(activations[test_idx][:, sel])[:, 1]
        ))

    cv_mean = np.mean(cv_aurocs)
    cv_ci = stats.t.ppf(0.975, 4) * np.std(cv_aurocs) / np.sqrt(5)
    print(f"  CV AUROC: {cv_mean:.4f} [{cv_mean-cv_ci:.4f}, {cv_mean+cv_ci:.4f}]")

    # Compute factual centroids (for potential firewall use)
    factual_centroids = activations[cal_labels == 0][:, neuron_indices].mean(axis=0)

    # Save artifacts
    model_short = model_id.split('/')[-1]
    save_dir = output_dir / model_short
    save_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        'model_id': model_id,
        'model_name': config['name'],
        'layer_idx': int(best_layer),
        'top_k': top_k,
        'n_calibration_samples': len(samples),
        'n_self_calibration': self_calibrate,
        'cv_auroc': float(cv_mean),
        'cv_auroc_ci': float(cv_ci),
        'cv_fold_aurocs': [float(a) for a in cv_aurocs],
        'layer_results': {str(k): v for k, v in layer_results.items()},
        'thresholds': {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
        },
        'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    np.save(save_dir / 'neuron_indices.npy', neuron_indices)
    with open(save_dir / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    np.save(save_dir / 'factual_centroids.npy', factual_centroids)

    print(f"\nCalibration saved to {save_dir}/")
    print(f"  config.json         - Layer, thresholds, AUROC")
    print(f"  neuron_indices.npy  - {len(neuron_indices)} neuron indices")
    print(f"  classifier.pkl      - LogisticRegression classifier")
    print(f"  factual_centroids.npy - Factual activation centroids")

    del activations; gc.collect()
    return config_data


# ============================================================================
# Hallucination Detector
# ============================================================================

class HallucinationDetector:
    """Score medical Q&A pairs for hallucination risk using saved calibration."""

    def __init__(self, model, tokenizer, config_dir: Path, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load calibration artifacts
        with open(config_dir / 'config.json') as f:
            self.config = json.load(f)

        self.neuron_indices = np.load(config_dir / 'neuron_indices.npy')
        with open(config_dir / 'classifier.pkl', 'rb') as f:
            self.clf = pickle.load(f)

        self.layer_idx = self.config['layer_idx']
        self.layer = get_model_layers(model)[self.layer_idx]
        self.thresholds = self.config['thresholds']
        self.ensemble_thresholds = self.config.get('ensemble_thresholds', {
            'low': 0.20, 'medium': 0.35, 'high': 0.50,
        })

        # Load factual centroids for distance-based scoring
        centroids_path = config_dir / 'factual_centroids.npy'
        self.factual_centroids = np.load(centroids_path) if centroids_path.exists() else None

        print(f"Detector loaded: layer {self.layer_idx}, "
              f"{len(self.neuron_indices)} neurons, "
              f"CV AUROC={self.config['cv_auroc']:.3f}"
              f"{', centroids loaded' if self.factual_centroids is not None else ''}")

    def score(self, question: str, answer: str) -> Dict:
        """Score a Q/A pair for hallucination risk."""
        text = f"Question: {question}\nAnswer: {answer}"
        hidden = extract_activations_single(
            self.model, self.layer, text, self.tokenizer, self.device,
        )
        features = hidden[self.neuron_indices].reshape(1, -1)
        prob = float(self.clf.predict_proba(features)[0, 1])

        if prob < self.thresholds['low']:
            risk = 'LOW'
        elif prob < self.thresholds['medium']:
            risk = 'UNCERTAIN'
        elif prob < self.thresholds['high']:
            risk = 'ELEVATED'
        else:
            risk = 'HIGH'

        return {
            'hallucination_prob': prob,
            'risk_level': risk,
            'layer': self.layer_idx,
            'n_neurons': len(self.neuron_indices),
        }

    def centroid_distance_score(self, neuron_acts: np.ndarray) -> float:
        """Compute hallucination risk from L2 distance to factual centroid.

        More robust to distribution shift than the classifier boundary since
        it measures relative displacement rather than absolute position.
        Returns sigmoid-transformed score in [0, 1].
        """
        if self.factual_centroids is None:
            return 0.5
        dist = float(np.linalg.norm(neuron_acts - self.factual_centroids))
        centroid_norm = float(np.linalg.norm(self.factual_centroids))
        if centroid_norm < 1e-8:
            return 0.5
        relative_dist = dist / centroid_norm
        return float(1.0 / (1.0 + np.exp(-4.0 * (relative_dist - 0.5))))

    def score_raw(self, question: str, answer: str) -> Tuple[float, np.ndarray]:
        """Score a Q/A pair and return both probability and raw neuron activations."""
        text = f"Question: {question}\nAnswer: {answer}"
        hidden = extract_activations_single(
            self.model, self.layer, text, self.tokenizer, self.device,
        )
        neuron_acts = hidden[self.neuron_indices]
        prob = float(self.clf.predict_proba(neuron_acts.reshape(1, -1))[0, 1])
        return prob, neuron_acts

    def generate(self, question: str, max_new_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Generate a response to a medical question."""
        prompt = format_chat_prompt(question, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    @staticmethod
    def _extract_key_terms(text: str) -> set:
        """Extract content words from text for Jaccard similarity."""
        stops = {'the','a','an','is','are','was','were','be','been','being',
                 'and','or','of','to','in','for','it','its','that','this',
                 'with','on','by','as','at','from','has','have','had','not',
                 'but','can','may','will','would','could','should','do','does',
                 'did','if','so','than','then','they','them','their','we','our',
                 'you','your','he','she','his','her','what','which','who','when',
                 'where','how','all','each','both','few','more','most','some',
                 'such','no','only','very','just','also','about','into','over',
                 'after','before','between','through','during','while','because',
                 'here','there','these','those','other','like','well','much',
                 'many','often','usually','typically','generally','commonly'}
        return set(text.lower().split()) - stops

    def generate_and_score(self, question: str, n_consistency: int = 3,
                           **gen_kwargs) -> Dict:
        """Generate response(s) and score with ensemble (neuron + consistency).

        When n_consistency > 1, combines two orthogonal signals via geometric
        mean (soft AND): only flags HIGH when BOTH neuron classifier AND
        consistency scoring agree the response is hallucinated.

        Args:
            n_consistency: Number of generations (1 = neuron-only, >1 = ensemble)
        """
        # Generate multiple responses
        responses = []
        for _ in range(max(1, n_consistency)):
            r = self.generate(question, **gen_kwargs)
            if r.strip():
                responses.append(r)

        if not responses:
            return {'hallucination_prob': 1.0, 'risk_level': 'HIGH',
                    'answer': '', 'question': question, 'method': 'none'}

        if n_consistency <= 1 or len(responses) == 1:
            # Single generation or insufficient data — neuron classifier only
            answer = responses[0]
            result = self.score(question, answer)
            result['answer'] = answer
            result['question'] = question
            result['method'] = 'neuron'
            return result

        # === Score each response with neuron classifier ===
        neuron_probs = []
        neuron_acts_list = []
        for r in responses:
            prob, acts = self.score_raw(question, r)
            neuron_probs.append(prob)
            neuron_acts_list.append(acts)
        neuron_probs = np.array(neuron_probs)

        # === Compute consistency (Jaccard key-term overlap) ===
        term_sets = [self._extract_key_terms(r) for r in responses]
        pairwise_sims = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                union = term_sets[i] | term_sets[j]
                if union:
                    pairwise_sims.append(len(term_sets[i] & term_sets[j]) / len(union))
        consistency = float(np.mean(pairwise_sims)) if pairwise_sims else 0.5
        # Transform to risk: high consistency → low risk
        consistency_risk = max(0.0, min(1.0, 1.0 - (consistency - 0.05) / 0.25))

        # === Compute relative neuron signal ===
        neuron_spread = float(neuron_probs.max() - neuron_probs.min())
        if neuron_spread > 0.05:
            # Relative ranking: rescale within the N-response pool to cancel
            # distribution shift (absolute scores are ~0.85-0.99 for all
            # generated text, but BoN selection confirms relative ordering works)
            neuron_risk = float((neuron_probs.mean() - neuron_probs.min()) / neuron_spread)
        else:
            # Spread too small for reliable ranking — use centroid distance
            mean_acts = np.mean(neuron_acts_list, axis=0)
            neuron_risk = self.centroid_distance_score(mean_acts)

        # === Select best response ===
        # Highest avg Jaccard similarity to others, neuron score as tiebreaker
        best_idx = 0
        best_score = -1.0
        for i in range(len(term_sets)):
            sims = []
            for j in range(len(term_sets)):
                if i != j:
                    union = term_sets[i] | term_sets[j]
                    sims.append(len(term_sets[i] & term_sets[j]) / len(union) if union else 0)
            avg_sim = float(np.mean(sims)) if sims else 0.0
            # Tiebreaker: prefer lower neuron prob (less likely hallucination)
            score = avg_sim - 0.01 * neuron_probs[i]
            if score > best_score:
                best_score = score
                best_idx = i
        answer = responses[best_idx]

        # === Ensemble: geometric mean (soft AND) ===
        # sqrt(a * b) requires BOTH signals high for output to be high.
        # If consistency_risk=0.90 but neuron_risk=0.10 → ensemble=0.30 (LOW)
        ensemble_prob = float(np.sqrt(consistency_risk * neuron_risk))

        thresholds = self.ensemble_thresholds
        if ensemble_prob < thresholds['low']:
            risk = 'LOW'
        elif ensemble_prob < thresholds['medium']:
            risk = 'UNCERTAIN'
        elif ensemble_prob < thresholds['high']:
            risk = 'ELEVATED'
        else:
            risk = 'HIGH'

        return {
            'hallucination_prob': ensemble_prob,
            'risk_level': risk,
            'answer': answer,
            'question': question,
            'consistency': consistency,
            'consistency_risk': consistency_risk,
            'neuron_risk': neuron_risk,
            'neuron_spread': neuron_spread,
            'n_generations': len(responses),
            'method': 'ensemble',
        }


# ============================================================================
# Interactive mode
# ============================================================================

def risk_color(risk: str) -> str:
    """Return ANSI color code for risk level."""
    return {
        'LOW': GREEN,
        'UNCERTAIN': YELLOW,
        'ELEVATED': YELLOW,
        'HIGH': RED,
    }.get(risk, RESET)


def print_result(result: Dict):
    """Pretty-print a scored result."""
    risk = result['risk_level']
    prob = result['hallucination_prob']
    color = risk_color(risk)

    print(f"\n{DIM}{'─' * 70}{RESET}")
    print(f"{BOLD}Response:{RESET}")
    print(result['answer'])
    print(f"\n{DIM}{'─' * 70}{RESET}")

    # Risk bar
    bar_len = 40
    filled = int(prob * bar_len)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"  Hallucination risk: {color}{BOLD}[{bar}] {prob:.1%} {risk}{RESET}")
    method = result.get('method', 'neuron')
    if method == 'ensemble':
        print(f"  {DIM}(ensemble: consistency={result.get('consistency_risk', 0):.3f}, "
              f"neuron_risk={result.get('neuron_risk', 0):.3f}, "
              f"spread={result.get('neuron_spread', 0):.3f}, "
              f"{result.get('n_generations', 1)} generations){RESET}")
    elif method == 'consistency':
        print(f"  {DIM}(consistency={result.get('consistency', 0):.3f}, "
              f"{result.get('n_generations', 1)} generations){RESET}")
    else:
        print(f"  {DIM}(layer {result.get('layer', '?')}, "
              f"{result.get('n_neurons', '?')} neurons){RESET}")


def interactive(detector: HallucinationDetector, n_consistency: int = 5):
    """Interactive Q&A loop with hallucination scoring."""
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  Medical Hallucination Detector — Interactive Mode{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"  Model: {detector.config['model_name']}")
    print(f"  Scoring: {'ensemble (n=' + str(n_consistency) + ')' if n_consistency > 1 else 'neuron classifier'}")
    print(f"  Calibrated: {detector.config['calibrated_at']}")
    print(f"\n  Type a medical question and press Enter.")
    print(f"  Commands: {DIM}/retry{RESET} (regenerate), {DIM}/score <answer>{RESET} (score custom), {DIM}/quit{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")

    last_question = None

    while True:
        try:
            user_input = input(f"\n{BOLD}[?]{RESET} Medical question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ('/quit', '/exit', 'quit', 'exit'):
            print(f"{DIM}Goodbye.{RESET}")
            break

        if user_input.lower() == '/retry' and last_question:
            print(f"{DIM}Regenerating...{RESET}")
            result = detector.generate_and_score(last_question, n_consistency=n_consistency)
            print_result(result)
            continue

        if user_input.lower().startswith('/score '):
            if last_question is None:
                print(f"{RED}Ask a question first.{RESET}")
                continue
            custom_answer = user_input[7:].strip()
            result = detector.score(last_question, custom_answer)
            result['answer'] = custom_answer
            result['question'] = last_question
            print_result(result)
            continue

        # Normal question
        last_question = user_input
        print(f"{DIM}Generating{'  (' + str(n_consistency) + ' samples)' if n_consistency > 1 else ''}...{RESET}")
        t0 = time.time()
        result = detector.generate_and_score(user_input, n_consistency=n_consistency)
        elapsed = time.time() - t0
        print_result(result)
        print(f"  {DIM}({elapsed:.1f}s){RESET}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Production Hallucination Detector for Medical LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-time calibration
  CUDA_VISIBLE_DEVICES=0 python production/medgemma_detector.py --calibrate

  # Interactive mode
  CUDA_VISIBLE_DEVICES=0 python production/medgemma_detector.py --interactive

  # Calibrate with custom settings
  python production/medgemma_detector.py --calibrate --model google/medgemma-4b-it --n-calibration 3000
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--calibrate', action='store_true',
                      help='Run calibration (one-time, ~15 min)')
    mode.add_argument('--interactive', action='store_true',
                      help='Interactive Q&A with hallucination scoring')

    parser.add_argument('--model', type=str, default='google/medgemma-1.5-4b-it',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to use (default: medgemma-1.5-4b-it)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--n-calibration', type=int, default=2000,
                        help='Number of calibration samples (default: 2000)')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of discriminative neurons (default: 100)')
    parser.add_argument('--self-calibrate', type=int, default=0, metavar='N',
                        help='Generate N model responses for self-calibration to reduce '
                             'false positives on generated text (recommended: 200)')
    parser.add_argument('--n-consistency', type=int, default=5, metavar='N',
                        help='Number of generations for consistency scoring in interactive '
                             'mode (1=neuron-only, 3-5=consistency, default: 5)')
    parser.add_argument('--calibration-dir', type=str, default=None,
                        help='Calibration directory (default: production/calibration/)')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    script_dir = Path(__file__).parent
    cal_dir = Path(args.calibration_dir) if args.calibration_dir else script_dir / 'calibration'
    model_short = args.model.split('/')[-1]
    model_cal_dir = cal_dir / model_short

    model, tokenizer = load_model(args.model, device)

    # Verify layer count
    actual_layers = len(get_model_layers(model))
    expected = MODEL_CONFIGS[args.model]['n_layers']
    if actual_layers != expected:
        print(f"Note: model has {actual_layers} layers (expected {expected})")
        MODEL_CONFIGS[args.model]['n_layers'] = actual_layers

    if args.calibrate:
        calibrate(
            model, tokenizer, device, args.model,
            output_dir=cal_dir,
            n_samples=args.n_calibration,
            top_k=args.top_k,
            self_calibrate=args.self_calibrate,
        )

    elif args.interactive:
        if not (model_cal_dir / 'config.json').exists():
            print(f"{RED}No calibration found at {model_cal_dir}{RESET}")
            print(f"Run with --calibrate first.")
            sys.exit(1)

        detector = HallucinationDetector(model, tokenizer, model_cal_dir, device)
        interactive(detector, n_consistency=args.n_consistency)


if __name__ == '__main__':
    main()
