"""MedGemma Nurse Engine — generates nurse responses via vLLM OpenAI API.

Calls an external vLLM server (OpenAI-compatible) for MedGemma inference
instead of loading the model locally.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests as _requests

log = logging.getLogger("care-ai-api.nurse")

from dca_server.ae_channels import AE_DETECTION_CHANNELS

AE_SYMPTOM_DESCRIPTIONS: dict[str, str] = {
    "nausea": "feeling sick to stomach, queasiness, aversion to food smells",
    "vomiting": "throwing up, unable to keep food or liquids down",
    "fatigue": "extreme tiredness, no energy, needing to rest frequently",
    "diarrhea": "loose/watery stools, frequent urgent bowel movements",
    "diarrhoea": "loose/watery stools, frequent urgent bowel movements",
    "constipation": "difficulty having bowel movements, bloating, abdominal discomfort",
    "peripheral_neuropathy": "tingling, numbness, burning sensation in hands and feet",
    "neuropathy": "numbness, tingling, weakness in extremities",
    "headache": "head pain, pressure, throbbing",
    "cough": "persistent dry or productive cough",
    "dyspnoea": "shortness of breath, difficulty breathing, getting winded easily",
    "dyspnea": "shortness of breath, difficulty breathing",
    "anorexia": "loss of appetite, no desire to eat, food seems unappealing",
    "decreased_appetite": "eating much less than usual, food doesn't appeal",
    "stomatitis": "mouth sores, pain when eating or drinking, difficulty swallowing",
    "arthralgia": "joint pain, stiffness, difficulty moving joints",
    "myalgia": "muscle aches, soreness, body pain",
    "abdominal_pain": "stomach or belly pain, cramping",
    "pyrexia": "fever, chills, feeling hot and cold alternately",
    "pruritus": "intense itching of the skin, scratching urge",
    "insomnia": "difficulty falling asleep, waking during the night",
    "back_pain": "pain in the lower or upper back",
    "dysgeusia": "metallic or altered taste, food tastes strange",
    "pneumonitis": "dry cough, shortness of breath, chest discomfort",
    "hypertension": "headache, dizziness, visual changes from high blood pressure",
    "bleeding": "unusual bleeding, bruising easily",
}


def _extract_drug_ae_profile(rule_set: dict) -> list[dict]:
    """Extract non-visual AE profile from rule_set for nurse context."""
    ae_profile = rule_set.get("ae_profile", [])
    non_visual = []
    for ae in ae_profile:
        term = ae["ae_term"]
        channels = AE_DETECTION_CHANNELS.get(term, {"channels": ["patient_reported"]})
        if "video_detectable" not in channels.get("channels", []):
            pct = f"{ae['incidence_all_grade']*100:.0f}%"
            symptoms = AE_SYMPTOM_DESCRIPTIONS.get(term, f"symptoms of {term.replace('_', ' ')}")
            non_visual.append({
                "ae_term": term,
                "incidence_pct": pct,
                "common_symptoms": symptoms,
            })
    return non_visual[:8]


class NurseEngine:
    """MedGemma nurse response generator via vLLM OpenAI API."""

    def __init__(
        self,
        vllm_base_url: str = "http://clara-medgemma4b-base:8000/v1",
        vllm_model_id: str = "medgemma-1.5-4b-it",
        vllm_api_key: str = "EMPTY",
        **kwargs,
    ):
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.vllm_model_id = vllm_model_id
        self.vllm_api_key = vllm_api_key
        log.info("NurseEngine using vLLM: %s (model=%s)", self.vllm_base_url, self.vllm_model_id)

        self._session = _requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.vllm_api_key}",
            "Content-Type": "application/json",
        })

        self._drug_profiles: dict[str, list[dict]] = {}

    def load_drug_context(self, rule_set_path: str | Path) -> dict:
        """Load and cache drug context from a rule_set file."""
        rule_set = json.loads(Path(rule_set_path).read_text())
        drug_name = rule_set.get("drug_name", "Unknown")
        indication = rule_set.get("indication", "")
        profile = _extract_drug_ae_profile(rule_set)
        self._drug_profiles[drug_name] = {
            "drug_name": drug_name,
            "indication": indication,
            "ae_profile": profile,
        }
        return self._drug_profiles[drug_name]

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove MedGemma thinking tokens (<unused94>thought ... <unused94>)."""
        original_len = len(text)
        # Case 1: closed thinking block — <unused94>thought ... <unused94>
        text = re.sub(r'<unused\d+>thought.*?<unused\d+>\s*', '', text, flags=re.DOTALL)
        # Case 2: unclosed thinking (truncated) — <unused94>thought ... EOF
        text = re.sub(r'<unused\d+>thought.*', '', text, flags=re.DOTALL)
        # Strip markdown code fences: ```json ... ```
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.strip()
        if len(text) != original_len:
            log.info("Stripped thinking/fences (%d → %d chars)", original_len, len(text))
        return text

    def _call_vllm(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Call vLLM OpenAI-compatible chat completions API."""
        payload = {
            "model": self.vllm_model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "response_format": {"type": "json_object"},
        }
        resp = self._session.post(
            f"{self.vllm_base_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        log.info("vLLM tokens: prompt=%s, completion=%s",
                 usage.get("prompt_tokens"), usage.get("completion_tokens"))
        content = data["choices"][0]["message"]["content"]
        return self._strip_thinking(content)

    def build_system_prompt(
        self,
        drug_name: str,
        indication: str,
        visual_assessment: dict,
        drug_ae_profile: list[dict],
        audio_assessment: dict | None = None,
        medical_transcript: str | None = None,
    ) -> str:
        vis_findings = visual_assessment.get("findings", [])
        vis_text = json.dumps(vis_findings, ensure_ascii=False, indent=2) if vis_findings else "No significant visual findings."
        gen_obs = "; ".join(visual_assessment.get("general_observations", []))

        profile_lines = []
        for ae in drug_ae_profile[:6]:
            profile_lines.append(
                f"  - {ae.get('ae_term','')} ({ae.get('incidence_pct','')}): {ae.get('common_symptoms','')}"
            )
        profile_text = "\n".join(profile_lines) if profile_lines else "Not available."

        # Audio assessment section
        audio_section = ""
        if audio_assessment and audio_assessment.get("cough_detected"):
            n = audio_assessment.get("num_cough_segments", 0)
            ctype = audio_assessment.get("majority_type", "unknown")
            votes = audio_assessment.get("vote_counts", {})
            seg_details = []
            for seg in audio_assessment.get("segments", [])[:5]:
                seg_details.append(
                    f"  - {seg.get('start_sec', 0):.1f}–{seg.get('end_sec', 0):.1f}s: "
                    f"{seg.get('label', '?')} (conf {seg.get('confidence', 0):.0%})"
                )
            seg_text = "\n".join(seg_details) if seg_details else "  (no details)"
            audio_section = (
                f"\nAUDIO ASSESSMENT (from HeAR cough detection):\n"
                f"Cough detected: YES — {n} segment(s), majority type: {ctype}\n"
                f"Vote distribution: {votes}\n"
                f"Segment details:\n{seg_text}\n"
            )
        elif audio_assessment is not None:
            audio_section = (
                f"\nAUDIO ASSESSMENT (from HeAR cough detection):\n"
                f"Cough detected: NO — no cough segments found in audio.\n"
            )

        # MedASR transcript section
        medasr_section = ""
        if medical_transcript:
            medasr_section = (
                f"\nMEDICAL SPEECH TRANSCRIPT (from MedASR):\n"
                f'"{medical_transcript}"\n'
                f"Note: This is a medical-domain ASR transcription of the patient's audio. "
                f"It may capture medical terms, symptoms, or drug names more accurately than general STT.\n"
            )

        return (
            f"You are an AI nurse conducting a daily video call with a cancer patient.\n"
            f"You've just heard the patient's report and received visual analysis from a separate system.\n\n"
            f"CLINICAL CONTEXT:\n- Drug: {drug_name}\n- Indication: {indication}\n\n"
            f"VISUAL ASSESSMENT (from MedGemma-Vision front-end):\n{vis_text}\nGeneral: {gen_obs}\n\n"
            f"{audio_section}"
            f"{medasr_section}"
            f"NON-VISUAL AE PROFILE FOR THIS DRUG (these require conversation to detect):\n{profile_text}\n\n"
            f"YOUR OBJECTIVES (dual):\n"
            f"  (a) DETECT non-visual AEs through conversation — ask about specific symptoms from the drug profile\n"
            f"  (b) MAINTAIN patient comfort — be warm, empathetic, build trust\n\n"
            f"STRATEGY:\n"
            f"1. Acknowledge what the patient shared (empathy first)\n"
            f"2. If visual findings exist, ALWAYS state the detected AE name AND its CTCAE grade (e.g. 'Grade 2 maculopapular rash')\n"
            f"3. If cough was detected in audio, mention the cough type (dry/wet) and ask about duration\n"
            f"4. Ask about TOP non-visual AEs for this drug — use open-ended, non-threatening language\n"
            f"5. Maximum 2 targeted questions (keep it brief)\n"
            f"6. Keep the acknowledgment under 2 sentences. Keep each question under 1 sentence.\n\n"
            f"Output JSON only."
        )

    def build_user_prompt(self, patient_text: str, has_audio: bool = False, medical_transcript: str | None = None) -> str:
        audio_field = ""
        if has_audio:
            audio_field = (
                f'    "audio_followup": "string|null (comment on detected cough — type, advice, concern level)",\n'
            )
        transcript_section = ""
        if medical_transcript:
            transcript_section = f"\nMEDICAL ASR TRANSCRIPT:\n{medical_transcript}\n"
        return (
            f"PATIENT'S REPORT:\n{patient_text}\n{transcript_section}\n"
            f"OUTPUT:\n"
            f'{{\n'
            f'    "approach_style": "empathetic|neutral|concerned|urgent",\n'
            f'    "acknowledgment": "string (brief empathetic response to patient report)",\n'
            f'    "questions": [\n'
            f'        {{\n'
            f'            "question": "string (what you ask the patient)",\n'
            f'            "target_ae": "string|null (which non-visual AE you\'re probing for)",\n'
            f'            "rationale": "string (why you\'re asking this)"\n'
            f'        }}\n'
            f'    ],\n'
            f'    "visual_followup": "string|null (MUST include AE name + CTCAE grade, e.g. \'I noticed signs of Grade 2 maculopapular rash\')",\n'
            f'{audio_field}'
            f'    "preliminary_concerns": ["string (initial suspicions)"]\n'
            f'}}'
        )

    def generate_response(
        self,
        patient_text: str,
        visual_assessment: dict,
        drug_name: str | None = None,
        indication: str | None = None,
        drug_ae_profile: list[dict] | None = None,
        audio_assessment: dict | None = None,
        medical_transcript: str | None = None,
    ) -> dict:
        """Generate a nurse response given patient input and visual assessment.

        If drug_name is not provided, falls back to the default loaded drug context.
        """
        if drug_name and drug_name in self._drug_profiles:
            ctx = self._drug_profiles[drug_name]
            drug_name = ctx["drug_name"]
            indication = indication or ctx["indication"]
            drug_ae_profile = drug_ae_profile or ctx["ae_profile"]
        elif not drug_name and self._drug_profiles:
            ctx = next(iter(self._drug_profiles.values()))
            drug_name = ctx["drug_name"]
            indication = indication or ctx["indication"]
            drug_ae_profile = drug_ae_profile or ctx["ae_profile"]
        else:
            drug_name = drug_name or "Unknown"
            indication = indication or ""
            drug_ae_profile = drug_ae_profile or []

        sys_prompt = self.build_system_prompt(
            drug_name, indication, visual_assessment, drug_ae_profile,
            audio_assessment=audio_assessment,
            medical_transcript=medical_transcript,
        )
        usr_prompt = self.build_user_prompt(
            patient_text,
            has_audio=audio_assessment is not None,
            medical_transcript=medical_transcript,
        )

        messages = [
            {"role": "user", "content": sys_prompt + "\n\n" + usr_prompt},
        ]

        raw = self._call_vllm(messages)

        log.info("=== NurseEngine generate_response ===")
        log.info("Raw output:\n%s", raw[:1000])

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {"raw_response": raw}
        except (json.JSONDecodeError, AttributeError):
            log.warning("JSON parse failed, returning raw response")
            parsed = {"raw_response": raw}

        log.info("Parsed keys: %s", list(parsed.keys()))
        return parsed

    def generate_chat_response(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> dict:
        """Generate a follow-up chat response given full conversation history.

        Args:
            system_prompt: The system prompt built during the initial consult.
            messages: Conversation history (role: user/assistant dicts, system excluded).

        Returns:
            Parsed dict response from MedGemma.
        """
        chat_messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Understood. I will follow these instructions as the oncology nurse."},
        ] + messages

        raw = self._call_vllm(chat_messages)

        log.info("=== NurseEngine generate_chat_response ===")
        log.info("Raw output:\n%s", raw[:1000])

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {"raw_response": raw}
        except (json.JSONDecodeError, AttributeError):
            log.warning("JSON parse failed, returning raw response")
            parsed = {"raw_response": raw}

        log.info("Parsed keys: %s", list(parsed.keys()))
        return parsed

    def chat_response_to_speech_text(self, response: dict) -> str:
        """Convert a chat response dict to natural speech text for TTS."""
        if "acknowledgment" in response or "questions" in response:
            return self.response_to_speech_text(response)

        raw = response.get("raw_response", "")
        if raw:
            return raw

        for key in ("response", "answer", "text", "message"):
            if key in response:
                return str(response[key])

        return " ".join(str(v) for v in response.values() if v)

    def response_to_speech_text(self, response: dict) -> str:
        """Convert structured nurse JSON response into concise speech text for TTS.

        Keeps acknowledgment + first question only to stay under ~200 chars.
        """
        parts = []
        ack = response.get("acknowledgment", "")
        if ack:
            parts.append(ack)

        visual = response.get("visual_followup")
        if visual:
            parts.append(visual)

        # Only the first question — keep it short for TTS
        questions = response.get("questions", [])
        if questions:
            q = questions[0]
            q_text = q.get("question", "") if isinstance(q, dict) else str(q)
            if q_text:
                parts.append(q_text)

        if not parts:
            raw = response.get("raw_response", "")
            if raw:
                parts.append(raw)

        speech = " ".join(parts)
        log.info("Speech text (%d chars): %s", len(speech), speech[:200])
        return speech
