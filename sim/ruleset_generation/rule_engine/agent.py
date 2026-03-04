import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import AsyncOpenAI

from rule_engine.config import RuleEngineConfig
from rule_engine.prompts import SYSTEM_PROMPT, format_evidence_prompt
from rule_engine.rate_limiter import RateLimiter
from rule_engine.schema import EvidenceBundle, RuleSet

log = logging.getLogger(__name__)

# JSON schema for the RuleSet, used in system prompt to guide output format
_RULESET_JSON_SCHEMA = json.dumps(RuleSet.model_json_schema(), indent=2)


@dataclass
class AgentLog:
    """Captures the full reasoning trace for a synthesis run."""

    timestamp: str = ""
    model: str = ""
    evidence_prompt: str = ""
    reasoning_trace: str = ""
    raw_response: str = ""
    rule_set: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stage_logs: list[dict] = field(default_factory=list)


def _repair_json(text: str) -> str:
    """Attempt to fix common JSON issues from LLM output."""
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix missing commas between JSON elements (e.g., "}\n{" or "]\n[" or value\n"key")
    # Pattern: closing bracket/brace followed by whitespace then opening bracket/brace
    text = re.sub(r'(\})\s*(\{)', r'\1,\2', text)
    text = re.sub(r'(\])\s*(\[)', r'\1,\2', text)
    # Pattern: closing bracket/brace followed by whitespace then quoted string (next key)
    text = re.sub(r'(\})\s*(")', r'\1,\2', text)
    text = re.sub(r'(\])\s*(")', r'\1,\2', text)
    # Pattern: quoted string or number followed by newline then quoted string (missing comma between values/keys)
    text = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', text)
    text = re.sub(r'(\d)\s*\n(\s*")', r'\1,\n\2', text)
    text = re.sub(r'(true|false|null)\s*\n(\s*")', r'\1,\n\2', text)
    # Remove duplicate commas from over-correction
    text = re.sub(r',\s*,', ',', text)
    return text


def _extract_and_parse_ruleset(raw_text: str) -> RuleSet:
    """Try to extract a valid RuleSet JSON from raw LLM text."""
    # Try direct parse
    try:
        return RuleSet(**json.loads(raw_text))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try after repair
    repaired = _repair_json(raw_text)
    try:
        return RuleSet(**json.loads(repaired))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON object in the text
    brace_start = raw_text.find("{")
    brace_end = raw_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = raw_text[brace_start : brace_end + 1]
        repaired = _repair_json(candidate)
        return RuleSet(**json.loads(repaired))

    raise ValueError("Could not extract valid JSON RuleSet from response")


_OUTPUT_INSTRUCTION = (
    "\n\nRespond with a single JSON object matching this schema — no markdown fences, "
    "no commentary before or after the JSON:\n"
    f"{_RULESET_JSON_SCHEMA}"
)


async def synthesize_rules(
    drugs: list[str],
    indication: str,
    evidence: EvidenceBundle,
    config: RuleEngineConfig | None = None,
) -> tuple[RuleSet, AgentLog]:
    """Run the synthesis agent to produce a RuleSet from evidence.

    Uses the OpenAI client directly (bypassing agno) for full control
    over reasoning trace capture and token budget.

    Returns:
        Tuple of (parsed RuleSet, AgentLog with full reasoning trace).
    """
    if config is None:
        config = RuleEngineConfig()

    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )
    limiter = RateLimiter(config.rate_limit_rpm)
    prompt = format_evidence_prompt(evidence)

    drug_label = " + ".join(drugs)

    agent_log = AgentLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=config.llm_model,
        evidence_prompt=prompt,
    )

    log.info("Synthesizing rules for %s / %s", drug_label, indication)
    await limiter.acquire()
    response = await client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + _OUTPUT_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8192,
        temperature=0.6,
        response_format={"type": "json_object"},
    )

    choice = response.choices[0]
    msg = choice.message

    # Capture reasoning trace (gpt-oss-120b emits reasoning_content)
    reasoning = getattr(msg, "reasoning_content", None) or ""
    agent_log.reasoning_trace = reasoning
    if reasoning:
        log.info("Captured reasoning trace (%d chars) for %s / %s",
                 len(reasoning), drug_label, indication)

    # Capture raw content
    raw_text = msg.content or ""
    agent_log.raw_response = raw_text

    if choice.finish_reason == "length":
        log.warning("Response truncated (finish_reason=length) for %s / %s", drug_label, indication)

    # Parse the RuleSet with robust fallback
    rule_set = _extract_and_parse_ruleset(raw_text)
    agent_log.rule_set = rule_set.model_dump()
    return rule_set, agent_log
