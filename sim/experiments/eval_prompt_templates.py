"""Evaluate different prompt templates on baseline MedGemma — multi-turn support.

5 prompt templates (A~E) × multi-turn conversation (T2→T3→T4→T5→...).
Single model load, all prompts evaluated sequentially.

Usage:
    python -m src.experiments.eval_prompt_templates \
        --test-data data/training_v3_test_ood/sft_data.jsonl \
        --gpu 6 --max-turns 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sim.agents.llm_client import generate_json as gemini_generate_json, set_caller
from sim.engine.mood import MoodState, compute_interaction_quality, compute_grade_distortion
from sim.config.defaults import normalize_ae_term

PATIENT_MODEL = "gemini-2.0-flash"
MEDGEMMA_BASE = "google/medgemma-4b-it"


# ═══════════════════════════════════════════════════════════
# 1. Prompt Templates A~E
# ═══════════════════════════════════════════════════════════

def template_a_current(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template A — Current baseline prompt (control)."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    profile = scenario["drug_ae_profile"]

    vis_text = json.dumps(vis.get("findings", []), ensure_ascii=False, indent=2) if vis.get("findings") else "No significant visual findings."
    gen_obs = "; ".join(vis.get("general_observations", []))
    profile_lines = [f"  - {ae['ae_term']} ({ae['incidence_pct']}): {ae['common_symptoms']}" for ae in profile]
    profile_text = "\n".join(profile_lines)

    sys = f"""You are an AI nurse conducting Turn {turn} of a daily video call with a cancer patient.
You've just heard the patient's initial report and received visual analysis from a separate system.

CLINICAL CONTEXT:
- Drug: {drug}
- Indication: {indication}

VISUAL ASSESSMENT (from MedGemma-Vision front-end — already analyzed patient video):
{vis_text}
General: {gen_obs}

NON-VISUAL AE PROFILE FOR THIS DRUG (these require conversation to detect):
{profile_text}

PATIENT INTERACTION PROFILE:
- Under-report likelihood: {quality['under_report_prob']:.2f} — {'HIGH: patient likely hiding symptoms' if quality['under_report_prob'] > 0.4 else 'moderate' if quality['under_report_prob'] > 0.2 else 'low: patient is forthcoming'}
- Engagement: {quality['engagement']:.2f}

YOUR OBJECTIVES (dual):
  (a) DETECT non-visual AEs through conversation — ask about specific symptoms from the drug profile
  (b) MAINTAIN patient comfort — be warm, empathetic, build trust

STRATEGY:
1. Acknowledge what the patient shared (empathy first)
2. If visual findings exist, acknowledge them naturally ("I noticed..." or "The camera picked up...")
3. Ask about TOP non-visual AEs for this drug — use open-ended, non-threatening language
4. Maximum 3 targeted questions (don't overwhelm)
5. Use OARS: Open questions, Affirmations, Reflective listening, Summarizing

Output JSON only."""

    usr = _build_user_prompt_with_history(scenario, turn, history)
    return sys, usr


def template_b_concise(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template B — Fixed format + concise output."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    profile = scenario["drug_ae_profile"]

    vis_text = json.dumps(vis.get("findings", []), ensure_ascii=False, indent=2) if vis.get("findings") else "No visual concerns."
    gen_obs = "; ".join(vis.get("general_observations", []))
    top_aes = ", ".join(f"{ae['ae_term']}({ae['incidence_pct']})" for ae in profile[:5])

    sys = f"""You are a warm, experienced oncology nurse on a video call (Turn {turn}).

Drug: {drug} for {indication}
Visual scan: {gen_obs if gen_obs else 'unremarkable'}
Top non-visual AEs to screen: {top_aes}

RULES:
- Keep responses SHORT and warm (1-2 sentences per field)
- Ask exactly 2 focused questions about symptoms the patient hasn't mentioned
- No medical jargon — use everyday language
- Output JSON only."""

    usr = _build_user_prompt_with_history(scenario, turn, history, concise=True)
    return sys, usr


def template_c_rich_persona(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template C — Rich context + patient persona + speaking style."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    profile = scenario["drug_ae_profile"]
    demo = scenario.get("patient_demographics", {})
    persona = scenario.get("patient_persona_type", "unknown")
    day = scenario["treatment_day"]

    vis_text = json.dumps(vis.get("findings", []), ensure_ascii=False, indent=2) if vis.get("findings") else "No significant visual findings."
    gen_obs = "; ".join(vis.get("general_observations", []))
    profile_lines = [f"  - {ae['ae_term']} ({ae['incidence_pct']}): {ae['common_symptoms']}" for ae in profile]
    profile_text = "\n".join(profile_lines)

    persona_desc = {
        "stoic_minimizer": "stoic and tends to downplay symptoms. May say 'I'm fine' when not. Needs gentle, persistent probing.",
        "anxious_reporter": "anxious and may over-report or catastrophize. Needs reassurance and calm tone.",
        "cooperative_honest": "generally cooperative and honest. Direct questions work well.",
        "depressed_withdrawn": "withdrawn and low energy. May not volunteer information. Needs warmth and patience.",
        "angry_frustrated": "frustrated with treatment. May be short-tempered. Needs validation and respect.",
    }.get(persona, "personality unknown — adapt to their tone.")

    under_report = quality['under_report_prob']
    engagement = quality['engagement']
    if under_report > 0.4:
        interaction_tip = "HIGH under-report risk — ask indirectly, normalize symptoms, don't accept 'fine' at face value."
    elif under_report > 0.2:
        interaction_tip = "Moderate under-report risk — gentle follow-ups needed."
    else:
        interaction_tip = "Low under-report risk — patient is likely forthcoming."

    sys = f"""You are a caring, experienced oncology nurse conducting a daily video check-in (Turn {turn}).
Speak naturally — like a warm friend who happens to be medically trained.
Use simple language. Never use medical jargon with the patient.

PATIENT:
- {demo.get('age', '?')}yo {demo.get('sex', '?')}, Day {day} of treatment
- Personality: {persona_desc}
- {interaction_tip}
- Engagement level: {'low — keep it brief and light' if engagement < 0.4 else 'moderate' if engagement < 0.7 else 'high — patient is receptive'}

CLINICAL CONTEXT:
- Drug: {drug} for {indication}

VISUAL ASSESSMENT:
{vis_text}
General: {gen_obs}

NON-VISUAL AEs TO SCREEN (ask about these through conversation):
{profile_text}

YOUR APPROACH:
1. Warmly acknowledge what the patient shared — reflect their words back
2. Mention any visual concerns naturally ("I noticed you look a bit tired today")
3. Ask 2-3 open-ended questions targeting TOP unmentioned AEs
4. Match the patient's energy — if they're low-key, be gentle; if chatty, be conversational
5. End with something encouraging

Output JSON only."""

    usr = _build_user_prompt_with_history(scenario, turn, history)
    return sys, usr


def template_d_minimal(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template D — Minimal context, rely on MedGemma's medical knowledge."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    gen_obs = "; ".join(vis.get("general_observations", []))

    sys = f"""You are a kind oncology nurse on a video call (Turn {turn}) with a patient taking {drug} for {indication}.

Visual observation: {gen_obs if gen_obs else 'nothing notable'}

Your job: be warm, listen carefully, and ask 2-3 gentle questions about how they're feeling.
Based on your medical knowledge of {drug}, ask about common side effects they haven't mentioned yet.
Keep it conversational — no checklists, no jargon.

Output JSON only."""

    usr = _build_user_prompt_with_history(scenario, turn, history, concise=True)
    return sys, usr


def template_e_fewshot(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template E — Maximum context + few-shot example."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    profile = scenario["drug_ae_profile"]
    demo = scenario.get("patient_demographics", {})
    persona = scenario.get("patient_persona_type", "unknown")
    day = scenario["treatment_day"]

    vis_text = json.dumps(vis.get("findings", []), ensure_ascii=False, indent=2) if vis.get("findings") else "No significant visual findings."
    gen_obs = "; ".join(vis.get("general_observations", []))

    profile_with_probes = []
    probe_examples = {
        "nausea": "Have certain food smells been bothering you lately?",
        "constipation": "How have things been going in the bathroom — any changes?",
        "diarrhoea": "Has your stomach been giving you any trouble — loose stools or urgency?",
        "peripheral_neuropathy": "Have you noticed any tingling or odd sensations in your hands or feet?",
        "neuropathy_peripheral": "Any numbness or tingling in your fingers or toes?",
        "anaemia": "Have you been feeling more winded than usual, even with small activities?",
        "fatigue": "How's your energy been — are you needing more rest than before?",
        "arthralgia": "Any new aches or stiffness in your joints?",
        "decreased_appetite": "How's your appetite been? Are you eating about the same as before?",
        "insomnia": "How have you been sleeping — any trouble getting to sleep or staying asleep?",
        "headache": "Any headaches or pressure in your head lately?",
        "dyspnoea": "Have you noticed being short of breath at all?",
    }
    for ae in profile:
        term = ae["ae_term"]
        probe = probe_examples.get(term, f"Have you experienced any {term.replace('_', ' ')}?")
        profile_with_probes.append(f"  - {term} ({ae['incidence_pct']}): {ae['common_symptoms']}\n    → Example probe: \"{probe}\"")
    profile_text = "\n".join(profile_with_probes)

    persona_desc = {
        "stoic_minimizer": "stoic, downplays symptoms",
        "anxious_reporter": "anxious, may over-report",
        "cooperative_honest": "cooperative and honest",
        "depressed_withdrawn": "withdrawn, low energy",
        "angry_frustrated": "frustrated, short-tempered",
    }.get(persona, "unknown")

    sys = f"""You are a caring oncology nurse on a daily video call (Turn {turn}).

PATIENT: {demo.get('age', '?')}yo {demo.get('sex', '?')}, Day {day}, personality: {persona_desc}

DRUG: {drug} for {indication}

VISUAL: {gen_obs if gen_obs else 'unremarkable'}
{vis_text}

NON-VISUAL AEs TO PROBE (with example questions):
{profile_text}

STYLE: Empathetic, warm, everyday language. Reflect the patient's words. Maximum 3 questions.

Output JSON only."""

    fewshot_example = """{
  "approach_style": "empathetic",
  "acknowledgment": "Thank you for sharing that — I can tell it's been a tough few days. I want to make sure we're staying on top of everything together.",
  "questions": [
    {
      "question": "You mentioned feeling tired — has it been the kind where you just can't catch up on rest, or more like your body feels heavy?",
      "target_ae": "anaemia"
    },
    {
      "question": "How have things been going with your stomach — any nausea or changes in appetite?",
      "target_ae": "nausea"
    }
  ],
  "visual_followup": null,
  "concerns": ["fatigue pattern warrants anaemia screening", "GI symptoms not yet assessed"]
}"""

    usr = _build_user_prompt_with_history(scenario, turn, history, fewshot=fewshot_example)
    return sys, usr


# ═══════════════════════════════════════════════════════════
# 2. User Prompt Builder (shared, multi-turn aware)
# ═══════════════════════════════════════════════════════════

def _build_user_prompt_with_history(
    scenario: dict, turn: int, history: list[dict],
    concise: bool = False, fewshot: str | None = None,
) -> str:
    """Build user prompt with conversation history for multi-turn."""
    parts = []

    if turn == 2:
        t1 = history[0] if history else {}
        parts.append(f"PATIENT'S INITIAL REPORT:\n{json.dumps(t1, indent=2, ensure_ascii=False)}")
    else:
        parts.append("CONVERSATION SO FAR:")
        for h in history:
            role = h.get("_role", "unknown")
            turn_num = h.get("_turn", "?")
            if role == "patient":
                if turn_num == 1:
                    parts.append(f"\n[T1 — Patient initial report]")
                    display = {k: v for k, v in h.items() if not k.startswith("_")}
                    parts.append(json.dumps(display, indent=2, ensure_ascii=False))
                else:
                    parts.append(f"\n[T{turn_num} — Patient response]")
                    responses = h.get("responses", [])
                    for r in responses:
                        parts.append(f"  Patient: {r.get('answer', '(no answer)')}")
                    emotional = h.get("emotional_reaction", "")
                    if emotional:
                        parts.append(f"  (Emotional tone: {emotional})")
            elif role == "nurse":
                parts.append(f"\n[T{turn_num} — Your previous response]")
                ack = h.get("acknowledgment", "")
                if ack:
                    parts.append(f"  You said: {ack}")
                for q in h.get("questions", []):
                    parts.append(f"  You asked: {q.get('question', '')}")

        parts.append(f"\nNow generate Turn {turn} response.")

    if fewshot:
        parts.append(f"\nEXAMPLE OUTPUT:\n{fewshot}")
        parts.append(f"\nYOUR OUTPUT (same format):")
    else:
        if concise:
            parts.append(_output_schema_concise())
        else:
            parts.append(_output_schema_standard())

    return "\n\n".join(parts)


def _output_schema_standard() -> str:
    return """OUTPUT:
{
    "approach_style": "<choose one: empathetic, neutral, concerned, urgent>",
    "acknowledgment": "string — brief empathetic response (1-2 sentences)",
    "questions": [
        {
            "question": "string — what you ask the patient",
            "target_ae": "string or null — which AE you're probing for"
        }
    ],
    "visual_followup": "string or null",
    "concerns": ["string — your clinical suspicions"]
}"""


def _output_schema_concise() -> str:
    return """OUTPUT:
{
    "approach_style": "<choose one: empathetic, neutral, concerned, urgent>",
    "acknowledgment": "1-2 sentences max",
    "questions": [
        {"question": "string", "target_ae": "string or null"}
    ],
    "visual_followup": "string or null",
    "concerns": ["string"]
}"""


# ═══════════════════════════════════════════════════════════
# 3. T3/T5 Patient Response (Gemini) — multi-turn aware
# ═══════════════════════════════════════════════════════════

def generate_patient_response(
    scenario: dict, nurse_response: dict, mood: MoodState,
    quality: dict, grade_distortion: int, turn: int, history: list[dict],
) -> dict:
    """Generate patient response to nurse (works for T3, T5, T7, ...)."""
    set_caller("patient_eval")
    demo = scenario.get("patient_demographics", {})
    gt_nv = scenario["gt_non_visual_aes"]
    gt_vis = scenario.get("gt_visual_aes", [])
    day = scenario["treatment_day"]

    all_gt = []
    for ae in gt_nv:
        all_gt.append({"ae": ae["ae_term"], "grade": ae["grade"], "symptoms": ae["symptom_description"]})
    for ae in gt_vis:
        all_gt.append({"ae": ae["ae_term"], "grade": ae["grade"], "type": "visual"})

    already_revealed = set()
    for h in history:
        if h.get("_role") == "patient" and h.get("_turn", 0) > 1:
            for r in h.get("responses", []):
                sym = r.get("revealed_symptom")
                if sym:
                    already_revealed.add(sym)

    prev_context = ""
    if turn > 3:
        prev_context = f"\nYou have already revealed these symptoms in previous turns: {list(already_revealed)}" if already_revealed else "\nYou haven't revealed much yet."
        prev_context += f"\nThe nurse has been {'caring and warm' if quality['engagement'] > 0.5 else 'professional'}."
        prev_context += f"\nYour guard may have lowered slightly because of the ongoing conversation."

    system_prompt = f"""You are a clinical trial patient responding to your nurse's follow-up questions (Turn {turn}).

PATIENT PROFILE:
- Age: {demo.get('age', 60)}, Sex: {demo.get('sex', 'M')}
- Personality: {scenario.get('patient_persona_type', 'stoic_minimizer')}

MOOD:
- Under-report probability: {quality['under_report_prob']:.2f}
- Grade distortion: {grade_distortion:+d}
- Engagement: {quality['engagement']:.2f}
{prev_context}

RULES:
- Answer each question based on your ACTUAL symptoms below
- If the nurse was caring, your defensiveness has lowered (reveal more)
- If asked about a symptom you HAVE: reveal based on mood (full/partial/evasive)
- If asked about something you DON'T have: say no naturally
- Each additional turn of warm conversation makes you slightly more open

Output JSON only."""

    questions = nurse_response.get("questions", [])
    user_prompt = f"""Day {day} — TURN {turn}: Responding to nurse

NURSE'S QUESTIONS:
{json.dumps(questions, indent=2, ensure_ascii=False)}

YOUR ACTUAL STATE:
- Active AEs: {json.dumps(all_gt, ensure_ascii=False)}

OUTPUT:
{{
    "responses": [
        {{
            "to_question": "string (nurse's question)",
            "answer": "string (your natural response)",
            "revealed_symptom": "string|null (AE term if newly revealed)",
            "honesty_level": "full|partial|evasive|denied"
        }}
    ],
    "new_info_revealed": true/false,
    "emotional_reaction": "string (how you feel now)"
}}"""

    try:
        result = gemini_generate_json(system_prompt, user_prompt, model=PATIENT_MODEL)
    except Exception as e:
        result = {"responses": [], "new_info_revealed": False, "emotional_reaction": "neutral", "_error": str(e)}
    result["_turn"] = turn
    result["_role"] = "patient"
    return result


# ═══════════════════════════════════════════════════════════
# 4. Scoring (reused logic)
# ═══════════════════════════════════════════════════════════

def measure_behavior(t_response: dict) -> dict:
    responses = t_response.get("responses", [])
    n_responses = len(responses)
    n_revealed = sum(1 for r in responses if r.get("revealed_symptom"))
    n_full_honest = sum(1 for r in responses if r.get("honesty_level") == "full")
    emotional = t_response.get("emotional_reaction", "neutral")

    emotion_scores = {
        "relaxed": 1.0, "open": 0.9, "comfortable": 0.85,
        "neutral": 0.5, "calm": 0.5,
        "slightly defensive": 0.3, "guarded": 0.25,
        "defensive": 0.15, "hostile": 0.05, "withdrawn": 0.1,
        "anxious": 0.3, "tearful": 0.35, "frustrated": 0.2,
    }
    emotional_openness = emotion_scores.get(emotional.lower().strip(), 0.4)
    honesty_rate = n_full_honest / max(n_responses, 1)
    reveal_rate = n_revealed / max(n_responses, 1)
    new_info = t_response.get("new_info_revealed", False)

    mood_proxy = honesty_rate * 0.35 + reveal_rate * 0.30 + emotional_openness * 0.25 + (1.0 if new_info else 0.0) * 0.10

    return {
        "n_responses": n_responses, "n_revealed": n_revealed,
        "honesty_rate": round(honesty_rate, 3), "reveal_rate": round(reveal_rate, 3),
        "new_info_revealed": new_info, "emotional_reaction": emotional,
        "emotional_openness": round(emotional_openness, 3), "mood_proxy": round(mood_proxy, 3),
    }


def score_cumulative(all_patient_turns: list[dict], gt_nv_aes: list[dict], all_nurse_turns: list[dict]) -> dict:
    """Score across all conversation turns cumulatively."""
    gt_names = {normalize_ae_term(ae.get("ae_term", "")) for ae in gt_nv_aes}
    detected_all = set()
    mood_scores = []

    for pt in all_patient_turns:
        behavior = measure_behavior(pt)
        mood_scores.append(behavior["mood_proxy"])
        for r in pt.get("responses", []):
            sym = r.get("revealed_symptom", "")
            if sym:
                detected_all.add(normalize_ae_term(sym))

    for nt in all_nurse_turns:
        for q in nt.get("questions", []):
            target = q.get("target_ae", "")
            if target:
                for t in target.split("|"):
                    detected_all.add(normalize_ae_term(t.strip()))

    if gt_names:
        ae_recall = sum(1 for g in gt_names if any(g in d or d in g for d in detected_all)) / len(gt_names)
    else:
        ae_recall = 1.0

    avg_mood = sum(mood_scores) / len(mood_scores) if mood_scores else 0.5
    reveal_bonus = sum(1 for pt in all_patient_turns for r in pt.get("responses", []) if r.get("revealed_symptom")) / max(len(gt_names), 1) * 0.1
    ae_score = min(ae_recall + reveal_bonus, 1.0)
    pareto = ae_score * avg_mood

    return {
        "ae_score": round(ae_score, 3), "mood_score": round(avg_mood, 3),
        "pareto_score": round(pareto, 3), "ae_recall": round(ae_recall, 3),
        "detected_aes": sorted(detected_all), "n_detected": len(detected_all),
        "mood_per_turn": [round(m, 3) for m in mood_scores],
    }


# ═══════════════════════════════════════════════════════════
# 5. Multi-Turn Evaluation Loop
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# T4 Final Assessment — AE Detection + Reasoning + Action
# ═══════════════════════════════════════════════════════════

def generate_final_assessment(scenario: dict, history: list[dict], nurse_fn) -> dict:
    """Generate T4-style final assessment after multi-turn conversation."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario.get("visual_assessment", {})
    profile = scenario.get("drug_ae_profile", [])
    audio = scenario.get("audio_assessment", "")

    vis_findings = vis.get("findings", [])
    vis_text_parts = []
    for f in vis_findings:
        vis_text_parts.append(f"{f['ae_term']} Grade {f.get('grade','?')} (confidence {round(f.get('confidence',0)*100)}%): {f.get('reasoning','')}")
    gen_obs = "; ".join(vis.get("general_observations", []))
    vis_text = "\n".join(vis_text_parts) if vis_text_parts else gen_obs or "unremarkable"

    profile_lines = [f"  - {ae['ae_term']} ({ae.get('incidence_pct','?')})" for ae in profile[:10]]
    profile_text = "\n".join(profile_lines)

    transcript_parts = []
    for h in history:
        role = h.get("_role", "?")
        turn_n = h.get("_turn", "?")
        if role == "patient" and turn_n == 1:
            greeting = h.get("greeting", "")
            symptoms = h.get("reported_symptoms", [])
            sym_str = ", ".join(s.get("symptom", s) if isinstance(s, dict) else str(s) for s in symptoms) if symptoms else "none reported"
            transcript_parts.append(f"[T1 Patient]: {greeting} Symptoms: {sym_str}")
        elif role == "nurse":
            ack = h.get("acknowledgment", "")[:120]
            qs = [q.get("question", "")[:80] for q in h.get("questions", []) if isinstance(q, dict)]
            transcript_parts.append(f"[T{turn_n} Nurse]: {ack} Questions: {'; '.join(qs)}")
        elif role == "patient" and turn_n > 1:
            answers = [r.get("answer", "")[:150] for r in h.get("responses", []) if isinstance(r, dict)]
            emotional = h.get("emotional_reaction", "")
            transcript_parts.append(f"[T{turn_n} Patient]: {' | '.join(answers)} (mood: {emotional})")
    transcript = "\n".join(transcript_parts)

    sys_prompt = f"""You are an AI nurse completing a FINAL ASSESSMENT after a video call with a cancer patient.

CLINICAL CONTEXT:
- Drug: {drug} for {indication}
- Known AEs for this drug:
{profile_text}

MULTIMODAL INPUT:
- Visual analysis (MedGemma): {vis_text}
- Audio analysis (HeAR): {audio if audio else 'no abnormalities detected'}

CRITICAL: You MUST report ALL suspected AEs based on:
1. Visual findings from camera analysis
2. Audio findings (cough detection)
3. Patient-reported symptoms in conversation
Even low-confidence suspicions should be reported. If the patient mentions ANY symptom, classify it.

Output JSON with integer grades (1-4), not strings."""

    usr_prompt = f"""CONVERSATION TRANSCRIPT:
{transcript}

Based on ALL evidence (visual + audio + conversation), provide your assessment.
OUTPUT:
{{
    "detected_aes": [
        {{
            "ae_term": "rash_acneiform",
            "estimated_grade": 2,
            "confidence": "high",
            "evidence": "Visual detection + patient reported itching"
        }}
    ],
    "action": "no_action|monitor_closely|recommend_conmed|recommend_early_visit|recommend_hospital_visit",
    "action_reason": "string",
    "overall_concern_level": "low|moderate|high",
    "missed_screening": []
}}"""

    result = nurse_fn(sys_prompt, usr_prompt)
    detected = result.get("detected_aes", [])
    if isinstance(detected, list):
        for ae in detected:
            if isinstance(ae.get("estimated_grade"), str):
                try:
                    ae["estimated_grade"] = int(ae["estimated_grade"])
                except (ValueError, TypeError):
                    ae["estimated_grade"] = 1
            ae.setdefault("grade", ae.get("estimated_grade", 1))
    result["_turn"] = "T4_assessment"
    return result


def score_t4_assessment(t4: dict, gt_nv_aes: list[dict]) -> dict:
    """Score T4 assessment against ground truth."""
    gt_terms = {normalize_ae_term(ae["ae_term"]): ae.get("grade", 1) for ae in gt_nv_aes}
    detected = t4.get("detected_aes", [])
    if isinstance(detected, str):
        detected = []

    detected_terms = {}
    for d in detected:
        if isinstance(d, dict):
            term = normalize_ae_term(d.get("ae_term", ""))
            grade = d.get("estimated_grade", 0)
            try:
                grade = int(grade)
            except (ValueError, TypeError):
                grade = 0
            detected_terms[term] = grade

    tp = 0
    grade_errors = []
    matched_gt = set()
    for gt_term, gt_grade in gt_terms.items():
        for det_term, det_grade in detected_terms.items():
            if gt_term in det_term or det_term in gt_term:
                tp += 1
                grade_errors.append(abs(gt_grade - det_grade))
                matched_gt.add(gt_term)
                break

    n_gt = len(gt_terms)
    n_det = len(detected_terms)
    recall = tp / n_gt if n_gt > 0 else 1.0
    precision = tp / n_det if n_det > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    grade_mae = sum(grade_errors) / len(grade_errors) if grade_errors else 0.0

    action = t4.get("action", "unknown")
    concern = t4.get("overall_concern_level", "unknown")

    return {
        "ae_recall": round(recall, 3),
        "ae_precision": round(precision, 3),
        "ae_f1": round(f1, 3),
        "grade_mae": round(grade_mae, 2),
        "n_gt": n_gt,
        "n_detected": n_det,
        "action": action,
        "concern_level": concern,
        "detected_terms": list(detected_terms.keys()),
    }


def template_f_realistic(scenario: dict, quality: dict, turn: int, history: list[dict]) -> tuple[str, str]:
    """Template F — Realistic context with full clinical data available in deployment."""
    drug = scenario["drug_name"]
    indication = scenario["indication"]
    vis = scenario["visual_assessment"]
    profile = scenario["drug_ae_profile"]
    demo = scenario.get("patient_demographics", {})
    day = scenario["treatment_day"]

    vis_text = json.dumps(vis.get("findings", []), ensure_ascii=False, indent=2) if vis.get("findings") else "No significant visual findings."
    gen_obs = "; ".join(vis.get("general_observations", []))
    profile_lines = [f"  - {ae['ae_term']} ({ae['incidence_pct']}): {ae['common_symptoms']}" for ae in profile]
    profile_text = "\n".join(profile_lines)

    audio = scenario.get("audio_assessment", "")
    labs_text = scenario.get("current_labs_text", "")
    vitals_text = scenario.get("current_vitals_text", "")
    meds_text = scenario.get("current_medications_text", "")
    hx_text = scenario.get("medical_history_text", "")
    ecog = scenario.get("ecog", "?")

    clinical_block = ""
    if labs_text or vitals_text or meds_text:
        clinical_block = f"""
LATEST LABS (vs baseline):
{labs_text or '  Not available'}

VITALS: {vitals_text or 'Not available'}

ECOG: {ecog}

CURRENT MEDICATIONS:
{meds_text or '  Not available'}

MEDICAL HISTORY:
{hx_text or '  Not available'}
"""

    audio_block = ""
    if audio:
        audio_block = f"\nAUDIO ANALYSIS (HeAR model): {audio}\n"

    sys = f"""You are an oncology nurse on a brief daily video check-in (Turn {turn}).
Be warm but CONCISE — real patients hang up on long-winded nurses.
Each message: 1 short acknowledgment + 1-2 focused questions. Max 3 sentences total.

PATIENT: {demo.get('age', '?')}yo {demo.get('sex', '?')}, {demo.get('race', '?')}, Day {day} of {drug} for {indication}

VISUAL ASSESSMENT (MedGemma-4B finetuned):
{vis_text}
General: {gen_obs}
{audio_block}{clinical_block}
AEs TO SCREEN (pick 1-2 most relevant given the above data):
{profile_text}

CLINICAL REASONING:
- Cross-reference visual findings with lab trends (e.g. rising ALT with skin findings = possible hepatic involvement)
- Consider medical history (e.g. pre-existing DM → watch hyperglycemia, CKD → watch creatinine)
- Audio + visual combined: cough + rash may indicate different etiology than rash alone
- Prioritize questions by clinical urgency, not by AE list order

RULES:
- If visual findings exist, ask about those FIRST
- Ask only 1-2 questions per turn, each under 15 words
- Reflect what the patient said briefly, then ask
- Never list-dump multiple questions at once

Output JSON only."""

    usr = _build_user_prompt_with_history(scenario, turn, history)
    return sys, usr


TEMPLATES = {
    "A_current": template_a_current,
    "B_concise": template_b_concise,
    "C_rich": template_c_rich_persona,
    "D_minimal": template_d_minimal,
    "E_fewshot": template_e_fewshot,
    "F_realistic": template_f_realistic,
}


def run_multiturn(
    scenario: dict, t1_visible: dict, nurse_fn, template_fn,
    quality: dict, grade_distortion: int, mood: MoodState,
    max_turns: int = 3,
) -> dict:
    """Run multi-turn conversation: T2→T3→T4→T5→... and score."""
    history = []

    t1_entry = {**t1_visible, "_role": "patient", "_turn": 1}
    history.append(t1_entry)

    nurse_turns = []
    patient_turns = []
    turn_timings = []

    for round_idx in range(max_turns):
        nurse_turn_num = 2 + round_idx * 2
        patient_turn_num = nurse_turn_num + 1

        sys_prompt, usr_prompt = template_fn(scenario, quality, nurse_turn_num, history)

        t0 = time.time()
        t2 = nurse_fn(sys_prompt, usr_prompt)
        nurse_time = time.time() - t0

        t2["_turn"] = nurse_turn_num
        t2["_role"] = "nurse"
        history.append(t2)
        nurse_turns.append(t2)

        t3 = generate_patient_response(
            scenario, t2, mood, quality, grade_distortion,
            patient_turn_num, history,
        )
        history.append(t3)
        patient_turns.append(t3)
        turn_timings.append(round(nurse_time, 1))

    # T4: Final Assessment
    t4_assessment = generate_final_assessment(
        scenario, history, nurse_fn,
    )

    cumulative = score_cumulative(patient_turns, scenario["gt_non_visual_aes"], nurse_turns)

    t4_score = score_t4_assessment(t4_assessment, scenario["gt_non_visual_aes"])
    cumulative["t4_assessment"] = t4_score

    per_turn_scores = []
    detected_so_far = set()
    gt_names = {normalize_ae_term(ae.get("ae_term", "")) for ae in scenario["gt_non_visual_aes"]}
    for i, (nt, pt) in enumerate(zip(nurse_turns, patient_turns)):
        for r in pt.get("responses", []):
            sym = r.get("revealed_symptom", "")
            if sym:
                detected_so_far.add(normalize_ae_term(sym))
        for q in nt.get("questions", []):
            target = q.get("target_ae", "")
            if target:
                for t in target.split("|"):
                    detected_so_far.add(normalize_ae_term(t.strip()))
        recall_i = sum(1 for g in gt_names if any(g in d or d in g for d in detected_so_far)) / len(gt_names) if gt_names else 1.0
        beh = measure_behavior(pt)
        per_turn_scores.append({
            "turn": 2 + i * 2,
            "ae_recall": round(recall_i, 3),
            "mood": round(beh["mood_proxy"], 3),
            "emotional": beh["emotional_reaction"],
            "n_revealed": beh["n_revealed"],
            "nurse_time_s": turn_timings[i],
        })

    return {
        **cumulative,
        "per_turn": per_turn_scores,
        "total_nurse_time_s": sum(turn_timings),
        "n_turns": max_turns,
    }


def reconstruct_scenario(sft_example: dict) -> dict:
    ctx = sft_example["context"]
    gt_nv = sft_example.get("gt_non_visual_aes", [])
    mood_raw = ctx.get("patient_mood", {})
    return {
        "drug_name": ctx["drug_name"],
        "indication": ctx["indication"],
        "visual_assessment": ctx["visual_assessment"],
        "drug_ae_profile": ctx["drug_ae_profile"],
        "treatment_day": ctx["treatment_day"],
        "gt_non_visual_aes": gt_nv,
        "gt_visual_aes": [],
        "patient_demographics": {"age": 60, "sex": "M"},
        "patient_persona_type": "stoic_minimizer",
        "patient_mood": mood_raw,
    }


# ═══════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=3, help="Number of nurse turns per conversation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--templates", nargs="+", default=list(TEMPLATES.keys()),
                        help="Which templates to evaluate (default: all)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Custom model path (default: use MEDGEMMA_BASE)")
    parser.add_argument("--model-label", type=str, default="baseline",
                        help="Label for this model run (used in output filename)")
    args = parser.parse_args()

    test_data = []
    with open(args.test_data) as f:
        for line in f:
            test_data.append(json.loads(line))
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    model_path = args.model_path or MEDGEMMA_BASE
    print(f"Test samples: {len(test_data)} | Max turns: {args.max_turns} | Templates: {args.templates}")
    print(f"Model: {args.model_label} ({model_path})")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\nLoading model on GPU {args.gpu}...")
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(MEDGEMMA_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    def nurse_fn(system_prompt: str, user_prompt: str) -> dict:
        chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(match.group()) if match else {"_raw": raw[:500]}
        except (json.JSONDecodeError, AttributeError):
            return {"_raw": raw[:500]}

    all_results = {}
    for tmpl_name in args.templates:
        if tmpl_name not in TEMPLATES:
            print(f"  Unknown template: {tmpl_name}, skipping")
            continue

        template_fn = TEMPLATES[tmpl_name]
        print(f"\n{'='*70}")
        print(f"  TEMPLATE: {tmpl_name} ({args.max_turns} turns × {len(test_data)} samples)")
        print(f"{'='*70}")

        results = []
        for i, sft_ex in enumerate(test_data):
            scenario = reconstruct_scenario(sft_ex)
            t1_visible = sft_ex["context"]["patient_said"]
            mood_raw = scenario["patient_mood"]

            mood = MoodState(persona_type=scenario["patient_persona_type"], seed=args.seed + i)
            for dim, val in mood_raw.items():
                if dim in mood.state:
                    mood.state[dim] = val
            quality = compute_interaction_quality(mood)
            grade_distortion = compute_grade_distortion(mood)

            r = run_multiturn(
                scenario, t1_visible, nurse_fn, template_fn,
                quality, grade_distortion, mood,
                max_turns=args.max_turns,
            )
            results.append(r)

            gt_n = len(scenario["gt_non_visual_aes"])
            turns_str = " → ".join(
                f"T{t['turn']}(AE={t['ae_recall']:.1f},M={t['mood']:.2f})"
                for t in r["per_turn"]
            )
            t4 = r.get("t4_assessment", {})
            t4_str = f"T4(R={t4.get('ae_recall',0):.1f},P={t4.get('ae_precision',0):.1f},G={t4.get('grade_mae',0):.1f},{t4.get('action','?')})"
            print(f"  [{i+1}/{len(test_data)}] {turns_str} → {t4_str}  |  GT:{gt_n} Det:{r['n_detected']}  {r['total_nurse_time_s']:.1f}s")

        avg_ae = sum(r["ae_score"] for r in results) / len(results)
        avg_mood = sum(r["mood_score"] for r in results) / len(results)
        avg_pareto = sum(r["pareto_score"] for r in results) / len(results)
        avg_time = sum(r["total_nurse_time_s"] for r in results) / len(results)

        n_turns = args.max_turns
        turn_ae_avgs = []
        turn_mood_avgs = []
        for t_idx in range(n_turns):
            t_ae = [r["per_turn"][t_idx]["ae_recall"] for r in results if t_idx < len(r["per_turn"])]
            t_mood = [r["per_turn"][t_idx]["mood"] for r in results if t_idx < len(r["per_turn"])]
            turn_ae_avgs.append(sum(t_ae) / len(t_ae) if t_ae else 0)
            turn_mood_avgs.append(sum(t_mood) / len(t_mood) if t_mood else 0)

        # T4 assessment averages
        t4_recalls = [r.get("t4_assessment", {}).get("ae_recall", 0) for r in results]
        t4_precisions = [r.get("t4_assessment", {}).get("ae_precision", 0) for r in results]
        t4_f1s = [r.get("t4_assessment", {}).get("ae_f1", 0) for r in results]
        t4_grade_maes = [r.get("t4_assessment", {}).get("grade_mae", 0) for r in results]
        from collections import Counter
        t4_actions = Counter(r.get("t4_assessment", {}).get("action", "unknown") for r in results)

        avg_t4_recall = sum(t4_recalls) / len(t4_recalls) if t4_recalls else 0
        avg_t4_precision = sum(t4_precisions) / len(t4_precisions) if t4_precisions else 0
        avg_t4_f1 = sum(t4_f1s) / len(t4_f1s) if t4_f1s else 0
        avg_t4_grade_mae = sum(t4_grade_maes) / len(t4_grade_maes) if t4_grade_maes else 0

        all_results[tmpl_name] = {
            "avg_ae": round(avg_ae, 3), "avg_mood": round(avg_mood, 3),
            "avg_pareto": round(avg_pareto, 3), "avg_time": round(avg_time, 1),
            "turn_ae_progression": [round(x, 3) for x in turn_ae_avgs],
            "turn_mood_progression": [round(x, 3) for x in turn_mood_avgs],
            "t4_recall": round(avg_t4_recall, 3),
            "t4_precision": round(avg_t4_precision, 3),
            "t4_f1": round(avg_t4_f1, 3),
            "t4_grade_mae": round(avg_t4_grade_mae, 2),
            "t4_actions": dict(t4_actions),
            "n_samples": len(results),
        }

        print(f"\n  >>> {tmpl_name}: AE={avg_ae:.3f}  Mood={avg_mood:.3f}  Pareto={avg_pareto:.3f}  ({avg_time:.1f}s)")
        print(f"      Turn progression — AE: {' → '.join(f'{x:.3f}' for x in turn_ae_avgs)}")
        print(f"      Turn progression — Mood: {' → '.join(f'{x:.3f}' for x in turn_mood_avgs)}")
        print(f"      T4 Assessment — Recall={avg_t4_recall:.3f} Precision={avg_t4_precision:.3f} F1={avg_t4_f1:.3f} GradeMAE={avg_t4_grade_mae:.2f}")
        print(f"      T4 Actions: {dict(t4_actions)}")

    # Final comparison
    print(f"\n\n{'='*80}")
    print("  FINAL COMPARISON — Conversation Quality")
    print(f"{'='*80}")
    print(f"{'Template':<15} {'AE':>7} {'Mood':>7} {'Pareto':>8} {'Time':>7}  AE by turn")
    print("-" * 80)
    for name, r in all_results.items():
        ae_prog = " → ".join(f"{x:.2f}" for x in r["turn_ae_progression"])
        print(f"{name:<15} {r['avg_ae']:>7.3f} {r['avg_mood']:>7.3f} {r['avg_pareto']:>8.3f} {r['avg_time']:>6.1f}s  [{ae_prog}]")

    print(f"\n{'Template':<15} Mood by turn")
    print("-" * 50)
    for name, r in all_results.items():
        mood_prog = " → ".join(f"{x:.2f}" for x in r["turn_mood_progression"])
        print(f"{name:<15} [{mood_prog}]")

    print(f"\n{'='*80}")
    print("  FINAL COMPARISON — T4 Assessment (AE Detection + Reasoning)")
    print(f"{'='*80}")
    print(f"{'Template':<15} {'Recall':>8} {'Precis':>8} {'F1':>8} {'GradeMAE':>10}  Actions")
    print("-" * 80)
    for name, r in all_results.items():
        actions_str = ", ".join(f"{k}:{v}" for k, v in r.get("t4_actions", {}).items())
        print(f"{name:<15} {r.get('t4_recall',0):>8.3f} {r.get('t4_precision',0):>8.3f} {r.get('t4_f1',0):>8.3f} {r.get('t4_grade_mae',0):>10.2f}  {actions_str}")

    out_path = Path(args.test_data).parent / f"eval_prompt_templates_{args.model_label}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
