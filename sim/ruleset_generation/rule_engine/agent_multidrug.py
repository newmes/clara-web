"""Multi-indication merge agent — combines individual rule sets into one unified rule set.

Takes 2-3 individually generated rule sets (each for one drug-regimen/indication pair),
cross-regimen DDI evidence, and produces a single merged rule set via one LLM call.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from openai import AsyncOpenAI

from rule_engine.agent import AgentLog, _extract_and_parse_ruleset
from rule_engine.agent_multistage import _aggressive_repair_ruleset
from rule_engine.config import RuleEngineConfig
from rule_engine.evidence.ddi import DDIEvidence
from rule_engine.rate_limiter import RateLimiter
from rule_engine.prompts_multidrug import MULTIDRUG_MERGE_PROMPT
from rule_engine.schema import RuleSet

log = logging.getLogger(__name__)

_RULESET_JSON_SCHEMA = json.dumps(RuleSet.model_json_schema(), indent=2)


def _format_individual_rule_sets(
    individual_results: list[tuple[RuleSet, AgentLog | None, list[str]]],
) -> str:
    """Format individual rule sets into compact text for the merge prompt.

    Limits AEs to top 15 by frequency to fit in the context window.
    """
    sections: list[str] = []
    for i, (rs, _log, _warnings) in enumerate(individual_results, 1):
        sections.append(f"=== R{i}: {' + '.join(rs.drugs)} / {rs.indication} ===")
        sections.append(f"Phase:{rs.phase} Dur:{rs.treatment_duration_days}d")

        # Regimen (compact)
        for reg in rs.regimen:
            sections.append(f"  {reg.drug}: {reg.dose} {reg.route} q{reg.cycle_days}d {reg.schedule}")

        # Demographics (one line)
        d = rs.demographics
        sections.append(f"Demo: age {d.age.mean:.0f}±{d.age.std:.0f} [{d.age.min}-{d.age.max}], "
                        f"{d.sex.pct_male:.0f}%M/{d.sex.pct_female:.0f}%F")

        # AEs — top 15 by frequency
        sorted_aes = sorted(rs.adverse_events, key=lambda a: a.frequency_pct, reverse=True)
        top_aes = sorted_aes[:15]
        sections.append(f"AEs ({len(rs.adverse_events)} total, top {len(top_aes)}):")
        for ae in top_aes:
            g34 = sum(v for k, v in ae.severity_distribution.items() if k in ("grade_3", "grade_4", "grade_5"))
            sections.append(f"  {ae.event}:{ae.frequency_pct}% g34={g34:.1f}% d{ae.median_onset_days}")

        # Efficacy (one line)
        eff = rs.efficacy
        sections.append(f"Eff: ORR={eff.overall_response_rate_pct}% CR={eff.complete_response_rate_pct}% "
                        f"PFS={eff.median_pfs_months}mo OS={eff.median_os_months}mo")
        sections.append("")

    return "\n".join(sections)


def _format_ddi_evidence(ddi_evidence: DDIEvidence) -> str:
    """Format DDI evidence for the merge prompt."""
    if not ddi_evidence.pairs:
        return "No cross-regimen drug-drug interactions found."

    sections: list[str] = []
    sections.append(f"Cross-regimen DDI pairs ({len(ddi_evidence.pairs)}):")
    for pair in ddi_evidence.pairs:
        line = f"  {pair.drug_a} <-> {pair.drug_b}"
        if pair.drugbank_relation:
            line += f" — DrugBank: {pair.drugbank_relation}"
        if pair.shared_targets:
            line += f" — Shared targets: {', '.join(pair.shared_targets[:5])}"
            if len(pair.shared_targets) > 5:
                line += f" (+{len(pair.shared_targets) - 5} more)"
        sections.append(line)
    return "\n".join(sections)


def _compute_overlapping_aes(
    individual_results: list[tuple[RuleSet, AgentLog | None, list[str]]],
) -> str:
    """Pre-compute overlapping AEs across regimens with naive sums."""
    # Build: ae_name_lower -> [(regimen_idx, frequency_pct, source_drug)]
    ae_map: dict[str, list[tuple[int, float, str]]] = {}
    for i, (rs, _log, _warnings) in enumerate(individual_results):
        for ae in rs.adverse_events:
            key = ae.event.lower().strip()
            ae_map.setdefault(key, []).append((i + 1, ae.frequency_pct, ae.source_drug or rs.drugs[0]))

    overlapping = {k: v for k, v in ae_map.items() if len(v) > 1}
    if not overlapping:
        return "No overlapping AEs found across regimens."

    sections: list[str] = []
    sections.append(f"Overlapping AEs ({len(overlapping)}) — appear in multiple regimens:")
    for ae_name, entries in sorted(overlapping.items()):
        freqs = [f for _, f, _ in entries]
        naive_sum = sum(freqs)
        # Probabilistic independence model
        prob_combined = 1.0
        for f in freqs:
            prob_combined *= (1.0 - f / 100.0)
        prob_combined = min(95.0, (1.0 - prob_combined) * 100.0)

        regimen_details = ", ".join(f"R{idx}:{f:.1f}%({drug})" for idx, f, drug in entries)
        sections.append(f"  {ae_name}: {regimen_details}")
        sections.append(f"    naive_sum={naive_sum:.1f}%, probabilistic={prob_combined:.1f}%")

    return "\n".join(sections)


async def merge_rule_sets(
    individual_results: list[tuple[RuleSet, AgentLog | None, list[str]]],
    ddi_evidence: DDIEvidence,
    config: RuleEngineConfig,
) -> tuple[RuleSet, AgentLog]:
    """Merge individual rule sets into one unified multi-indication rule set.

    Args:
        individual_results: List of (RuleSet, AgentLog, warnings) from individual pipelines.
        ddi_evidence: Cross-regimen DDI evidence.
        config: Pipeline configuration.

    Returns:
        Tuple of (merged RuleSet, AgentLog).
    """
    client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
    )
    limiter = RateLimiter(config.rate_limit_rpm)

    # Collect all drugs and indications
    all_drugs: list[str] = []
    all_indications: list[str] = []
    for rs, _log, _warnings in individual_results:
        for d in rs.drugs:
            if d not in all_drugs:
                all_drugs.append(d)
        if rs.indication not in all_indications:
            all_indications.append(rs.indication)

    all_drugs_str = ", ".join(all_drugs)
    all_indications_str = ", ".join(all_indications)

    log.info("Merging %d rule sets: %s for %s", len(individual_results), all_drugs_str, all_indications_str)

    # Format inputs
    individual_text = _format_individual_rule_sets(individual_results)
    ddi_text = _format_ddi_evidence(ddi_evidence)
    overlap_text = _compute_overlapping_aes(individual_results)

    # Build merge prompt — aggressively cap sizes to fit context window
    # Model has 16384 tokens (~50K chars). Need room for reasoning + 8K output.
    # Target: prompt ≤ 15K chars (~5K tokens), leaving ~11K tokens for reasoning+output.
    n_results = len(individual_results)
    per_ruleset_budget = min(3000, 12000 // max(n_results, 1))
    individual_text_capped = individual_text[:per_ruleset_budget * n_results]

    # Compact schema — just key field names, not full JSON schema
    compact_schema = (
        "Output a JSON object with fields: drugs (list[str]), indication (str, comma-separated), "
        "phase (int), treatment_duration_days (int), regimen (list of {drug,dose,route,cycle_days,schedule}), "
        "demographics ({age:{min,max,mean,std}, sex:{pct_male,pct_female}, race_ethnicity:[{group,pct}]}), "
        "comorbidities ([{condition,prevalence_pct,impacts_dosing,ae_risk_modifiers}]), "
        "adverse_events ([{event,frequency_pct,severity_distribution:{grade_1,grade_2,grade_3,grade_4},median_onset_days,reversible,source_drug,triggers}]), "
        "efficacy ({overall_response_rate_pct,complete_response_rate_pct,median_pfs_months,median_os_months}), "
        "is_multi_indication (true), indications (list[str]), "
        "per_indication_efficacy ([{indication,regimen_drugs,efficacy,phase,treatment_duration_days}]), "
        "drug_interactions ([{drug_a,drug_b,interaction_type,description,severity,ae_impact,frequency_modifier,monitoring_recommendation}]), "
        "overlapping_ae_notes ([{event,contributing_drugs,unadjusted_frequency_sum,adjusted_frequency_pct,rationale}])"
    )

    merge_prompt = MULTIDRUG_MERGE_PROMPT.format(
        individual_rule_sets=individual_text_capped,
        ddi_evidence=ddi_text[:2000],
        overlapping_aes=overlap_text[:2000],
        all_drugs=all_drugs_str,
        all_indications=all_indications_str,
        schema=compact_schema,
    )

    system = "You are a clinical pharmacologist. Respond ONLY with valid JSON — no markdown fences, no commentary."

    log.info("Merge prompt: %d chars (budget %d/ruleset for %d rulesets)",
             len(merge_prompt), per_ruleset_budget, n_results)

    # Single LLM call
    await limiter.acquire()
    response = await client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": merge_prompt},
        ],
        max_tokens=8192,
        temperature=0.6,
        response_format={"type": "json_object"},
    )

    choice = response.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    content = msg.content or ""

    log.info("Merge LLM response: %d chars content, %d chars reasoning", len(content), len(reasoning))

    # Build agent log
    agent_log = AgentLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=config.llm_model,
        evidence_prompt=f"[multi-indication merge: {len(individual_results)} rule sets + DDI]",
        reasoning_trace=reasoning,
        raw_response=content,
    )

    # Parse the merged rule set
    rule_set = None
    try:
        rule_set = _extract_and_parse_ruleset(content)
    except Exception as first_err:
        log.warning("Primary merge parse failed: %s — attempting aggressive repair", first_err)
        try:
            rule_set = _aggressive_repair_ruleset(content, all_drugs, all_indications_str)
            log.info("Aggressive repair succeeded for merge output")
        except Exception as repair_err:
            log.error("Aggressive repair also failed for merge: %s", repair_err)
            raise repair_err from first_err

    if rule_set is None:
        raise ValueError("Failed to parse merge output into a valid RuleSet")

    # Ensure multi-indication fields are set
    rule_set.is_multi_indication = True
    rule_set.indications = all_indications
    if not rule_set.drugs or set(rule_set.drugs) != set(all_drugs):
        rule_set.drugs = all_drugs

    agent_log.rule_set = rule_set.model_dump()
    agent_log.stage_logs = [{
        "stage": "multi_indication_merge",
        "individual_count": len(individual_results),
        "ddi_pairs": len(ddi_evidence.pairs),
        "prompt_len": len(merge_prompt),
        "response_len": len(content),
        "reasoning_len": len(reasoning),
    }]

    log.info("Multi-indication merge complete: %d drugs, %d indications, %d AEs",
             len(rule_set.drugs), len(all_indications), len(rule_set.adverse_events))

    return rule_set, agent_log
