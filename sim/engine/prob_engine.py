"""Prob Engine — LLM → rand → LLM 핵심 패턴

모든 확률적 의사결정의 공통 엔진.

패턴:
  1. LLM이 현재 컨텍스트를 보고 확률 분포를 출력
  2. Sampler가 그 분포에서 값을 추출 (rand)
  3. 추출된 결과를 다음 LLM 호출의 입력으로 전달

이 패턴이 환자 생성, 운명표 생성, 일별 시뮬레이션 모두에 반복 적용된다.
"""

import json
from typing import Any

from sim.agents.llm_client import generate_json, DEFAULT_MODEL
from sim.engine.sampler import Sampler


# ── 확률 추정 (LLM 호출 1: 확률을 뱉어라) ─────────────

PROB_SYSTEM_PROMPT = """You are a medical probability estimator.
Given a clinical context (drug, disease, patient characteristics), output realistic probability distributions.

Rules:
- Base estimates on published clinical trial data, FDA labels, medical literature.
- All probabilities must be between 0 and 1.
- For categorical options, probabilities must sum to approximately 1.0.
- For numeric distributions, provide appropriate parameters.
- If conditional on previously sampled values, adjust probabilities accordingly.
- Include brief 'reasoning' for key estimates when it aids traceability.

Output ONLY valid JSON matching the requested schema. No explanations outside JSON."""


def estimate_probabilities(
    context: str,
    question: str,
    output_schema: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """LLM에게 확률 분포를 추정하도록 요청한다.

    Args:
        context: 현재까지의 환자/약물/상태 정보
        question: 무엇의 확률을 추정할지
        output_schema: LLM이 출력할 JSON 구조 (확률 필드 포함)
        model: LLM 모델 ID

    Returns:
        확률이 채워진 JSON dict
    """
    user_prompt = f"""{context}

--- QUESTION ---
{question}

--- OUTPUT FORMAT ---
{json.dumps(output_schema, indent=2, ensure_ascii=False)}"""

    return generate_json(PROB_SYSTEM_PROMPT, user_prompt, model=model)


# ── 상세 생성 (LLM 호출 2: 결정된 조건으로 데이터를 채워라) ──

DETAIL_SYSTEM_PROMPT = """You are a medical data generator for clinical trial simulation.
Given a set of predetermined conditions (already decided by probabilistic sampling),
generate medically realistic detailed data that is internally consistent.

Rules:
- The predetermined conditions are FIXED. Do not change them.
- Fill in all remaining fields with medically plausible values.
- Ensure internal consistency (e.g., CKD → elevated creatinine, diabetes → elevated glucose).
- Use realistic ranges from clinical practice.
- Output ONLY valid JSON matching the requested schema. No explanations outside JSON."""


def generate_details(
    context: str,
    predetermined: dict,
    output_schema: dict,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
) -> dict:
    """샘플링된 조건을 바탕으로 상세 데이터를 생성한다.

    Args:
        context: 약물/질환 컨텍스트
        predetermined: 코드에서 이미 결정된 값들 (LLM이 변경 불가)
        output_schema: LLM이 출력할 JSON 구조
        system_prompt: 커스텀 시스템 프롬프트 (None이면 기본값 사용)
        model: LLM 모델 ID
        max_tokens: 최대 출력 토큰

    Returns:
        상세 데이터가 채워진 JSON dict
    """
    user_prompt = f"""{context}

--- PREDETERMINED CONDITIONS (DO NOT CHANGE) ---
{json.dumps(predetermined, indent=2, ensure_ascii=False)}

--- OUTPUT FORMAT ---
Fill in the following JSON structure. The predetermined conditions above are already decided.
Your job is to fill in the remaining details to make a medically consistent record.

{json.dumps(output_schema, indent=2, ensure_ascii=False)}"""

    return generate_json(
        system_prompt or DETAIL_SYSTEM_PROMPT,
        user_prompt,
        model=model,
        max_tokens=max_tokens,
    )


# ── LLM → rand → LLM 통합 함수 ──────────────────────

def prob_decide_and_generate(
    context: str,
    prob_question: str,
    prob_schema: dict,
    detail_schema: dict,
    sampler: Sampler,
    model: str = DEFAULT_MODEL,
    detail_system_prompt: str | None = None,
    max_tokens: int = 8192,
) -> tuple[dict, dict, dict]:
    """LLM → rand → LLM 전체 패턴을 한 번에 실행.

    Args:
        context: 현재 컨텍스트
        prob_question: 확률 추정 질문
        prob_schema: 확률 출력 스키마
        detail_schema: 상세 데이터 출력 스키마
        sampler: 난수 생성기
        model: LLM 모델 ID
        detail_system_prompt: 상세 생성용 커스텀 시스템 프롬프트
        max_tokens: 상세 생성 최대 토큰

    Returns:
        (probabilities, sampled_values, detailed_output)
    """
    # Step 1: LLM estimates probabilities
    probabilities = estimate_probabilities(context, prob_question, prob_schema, model)

    # Step 2: Code samples from probabilities
    sampled = _auto_sample(probabilities, sampler)

    # Step 3: LLM generates details given sampled values
    detailed = generate_details(
        context=context,
        predetermined=sampled,
        output_schema=detail_schema,
        system_prompt=detail_system_prompt,
        model=model,
        max_tokens=max_tokens,
    )

    return probabilities, sampled, detailed


def _auto_sample(prob_output: dict, sampler: Sampler) -> dict:
    """LLM의 확률 출력에서 자동으로 샘플링.

    LLM 출력이 다양한 형태일 수 있으므로, 재귀적으로 탐색하여
    'probability', 'options', 'distribution' 키가 있는 항목을 샘플링한다.
    """
    result = {}

    for key, value in prob_output.items():
        if key in ("_reasoning", "reasoning", "_note"):
            continue

        if isinstance(value, dict):
            # 확률 스펙인지 확인
            if "type" in value and value["type"] in (
                "categorical", "boolean", "numeric", "integer",
                "multi_boolean", "fixed",
            ):
                result[key] = sampler.sample_from_spec(value)
            elif "options" in value and isinstance(value["options"], dict):
                # {"options": {"A": 0.7, "B": 0.3}} 형태
                result[key] = sampler.categorical(value["options"])
            elif "probability" in value and isinstance(value["probability"], (int, float)):
                # {"probability": 0.3} 형태
                result[key] = sampler.boolean(value["probability"])
            elif "distribution" in value:
                # {"distribution": "normal", "mean": 68, ...} 형태
                dist = value["distribution"]
                params = {k: v for k, v in value.items() if k != "distribution" and k != "reasoning"}
                result[key] = sampler.numeric(dist, params)
            else:
                # 중첩된 dict → 재귀
                result[key] = _auto_sample(value, sampler)
        elif isinstance(value, (int, float, str, bool)):
            # 확률이 아닌 고정값 → 그대로 전달
            result[key] = value

    return result
