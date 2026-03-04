import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rule_engine.agent import AgentLog, synthesize_rules
from rule_engine.config import RuleEngineConfig
from rule_engine.converter import convert_ruleset, split_base_overlay
from rule_engine.evidence.collector import collect_evidence
from rule_engine.schema import RuleSet
from rule_engine.validator import validate_rule_set, validate_multi_rule_set

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    successful: list[tuple[list[str], str, Path]] = field(default_factory=list)  # (drugs, indication, output_path)
    failed: list[tuple[list[str], str, str]] = field(default_factory=list)       # (drugs, indication, error)
    warnings: dict[str, list[str]] = field(default_factory=dict)                 # drugs_key -> [warnings]


def _drugs_key(drugs: list[str]) -> str:
    """Create a stable key for a drug list."""
    return "+".join(drugs)


def _save_rule_set(rule_set: RuleSet, agent_log: AgentLog | None, output_dir: Path) -> Path:
    """Save a RuleSet as split base.json + type overlay, plus agent log with internal format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_slug = "+".join(d.lower().replace(" ", "_").replace("/", "-") for d in rule_set.drugs)
    indication_slug = rule_set.indication.lower().replace(" ", "_").replace("/", "-")
    subdir_name = f"{drug_slug}_{indication_slug}"

    # Convert to target schema format
    internal_data = json.loads(rule_set.model_dump_json())
    converted, schema_type = convert_ruleset(internal_data)

    # Split into base + overlay
    base_dict, overlay_dict = split_base_overlay(converted, schema_type)

    # Save into subdirectory: output/{drug}_{indication}/base.json + {schema_type}.json
    subdir = output_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)

    base_path = subdir / "base.json"
    base_path.write_text(json.dumps(base_dict, indent=2, ensure_ascii=False))

    overlay_path = subdir / f"{schema_type}.json"
    overlay_path.write_text(json.dumps(overlay_dict, indent=2, ensure_ascii=False))

    log.info(f"Saved rule set [{schema_type}]: {subdir}/ (base.json + {schema_type}.json)")

    # Save agent log flat in output_dir (not in subdir)
    if agent_log is not None:
        agent_log_data = asdict(agent_log)
        agent_log_data["internal_rule_set"] = internal_data
        log_filename = f"{subdir_name}_agent_log.json"
        log_path = output_dir / log_filename
        log_path.write_text(json.dumps(agent_log_data, indent=2, default=str))
        log.info(f"Saved agent log: {log_path}")

    return subdir


async def _process_single(
    drugs: list[str],
    indication: str,
    config: RuleEngineConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[RuleSet | None, AgentLog | None, list[str], str | None]:
    """Process a single drug-indication entry. Returns (rule_set, agent_log, warnings, error)."""
    drug_label = " + ".join(drugs)
    async with semaphore:
        try:
            # Stage 1: Evidence prefetch
            log.info(f"Collecting evidence for {drug_label} / {indication}")
            evidence = await collect_evidence(drugs, indication, config)

            # Stage 2: Synthesis
            log.info(f"Synthesizing rules for {drug_label} / {indication}")
            if config.multi_stage:
                from rule_engine.agent_multistage import synthesize_rules_multistage
                rule_set, agent_log = await synthesize_rules_multistage(drugs, indication, evidence, config)
            else:
                rule_set, agent_log = await synthesize_rules(drugs, indication, evidence, config)

            # Stage 3: Validate
            warnings = validate_rule_set(rule_set, evidence)
            agent_log.warnings = warnings
            return rule_set, agent_log, warnings, None

        except Exception as e:
            log.error(f"Pipeline failed for {drug_label} / {indication}: {e}")
            return None, None, [], str(e)


async def run_pipeline(
    entries: list[tuple[list[str], str]],
    config: RuleEngineConfig | None = None,
) -> PipelineResult:
    """Run the full pipeline for a list of (drugs, indication) entries."""
    if config is None:
        config = RuleEngineConfig()

    result = PipelineResult()
    semaphore = asyncio.Semaphore(config.max_concurrent)

    tasks = [_process_single(drugs, indication, config, semaphore) for drugs, indication in entries]
    outcomes = await asyncio.gather(*tasks)

    for (drugs, indication), (rule_set, agent_log, warnings, error) in zip(entries, outcomes):
        dk = _drugs_key(drugs)
        if error:
            result.failed.append((drugs, indication, error))
            continue

        if rule_set:
            output_path = _save_rule_set(rule_set, agent_log, config.output_dir)
            result.successful.append((drugs, indication, output_path))
            if warnings:
                result.warnings[dk] = warnings

    log.info(f"Pipeline complete: {len(result.successful)} succeeded, {len(result.failed)} failed")
    return result


@dataclass
class MultiPipelineResult:
    """Result of a multi-indication pipeline run."""
    merged_rule_set: RuleSet | None = None
    merged_path: Path | None = None
    individual_successful: list[tuple[list[str], str, Path]] = field(default_factory=list)
    individual_failed: list[tuple[list[str], str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _save_multi_rule_set(rule_set: RuleSet, agent_log, output_dir: Path) -> Path:
    """Save a multi-indication merged RuleSet with naming convention multi_{ind1}_{ind2}_{ind3}_rules.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    indication_slugs = []
    for ind in rule_set.indications:
        slug = ind.lower().replace(" ", "_").replace("/", "-")
        indication_slugs.append(slug)

    if not indication_slugs:
        # Fallback: use the comma-separated indication field
        slug = rule_set.indication.lower().replace(" ", "_").replace("/", "-").replace(",", "")
        indication_slugs = [slug]

    filename = "multi_" + "_".join(indication_slugs) + "_rules.json"
    output_path = output_dir / filename
    output_path.write_text(rule_set.model_dump_json(indent=2))
    log.info(f"Saved multi-indication rule set: {output_path}")

    # Save agent log
    if agent_log is not None:
        log_filename = "multi_" + "_".join(indication_slugs) + "_agent_log.json"
        log_path = output_dir / log_filename
        log_path.write_text(json.dumps(asdict(agent_log), indent=2, default=str))
        log.info(f"Saved multi-indication agent log: {log_path}")

    return output_path


async def run_multi_indication_pipeline(
    regimens: list[tuple[list[str], str]],
    config: RuleEngineConfig | None = None,
) -> MultiPipelineResult:
    """Run the multi-indication pipeline: generate individual rule sets, then merge.

    Args:
        regimens: List of (drugs, indication) tuples — 2-3 regimens.
        config: Pipeline configuration (uses defaults if None).

    Returns:
        MultiPipelineResult with merged rule set and individual results.
    """
    if config is None:
        config = RuleEngineConfig()

    result = MultiPipelineResult()

    # Validate input
    if len(regimens) < 2:
        raise ValueError(f"Multi-indication requires at least 2 regimens, got {len(regimens)}")

    # Force multi-stage mode
    config.multi_stage = True

    # Step 1: Generate individual rule sets with concurrency limit
    log.info("Step 1: Generating %d individual rule sets (max %d concurrent)",
             len(regimens), config.max_concurrent_multi)

    semaphore = asyncio.Semaphore(config.max_concurrent_multi)
    tasks = [
        _process_single(drugs, indication, config, semaphore)
        for drugs, indication in regimens
    ]
    outcomes = await asyncio.gather(*tasks)

    individual_results: list[tuple[RuleSet, object, list[str]]] = []
    for (drugs, indication), (rule_set, agent_log, warnings, error) in zip(regimens, outcomes):
        dk = _drugs_key(drugs)
        if error:
            log.error("Individual pipeline failed for %s / %s: %s", " + ".join(drugs), indication, error)
            result.individual_failed.append((drugs, indication, error))
            continue

        if rule_set:
            output_path = _save_rule_set(rule_set, agent_log, config.output_dir)
            result.individual_successful.append((drugs, indication, output_path))
            individual_results.append((rule_set, agent_log, warnings))
            if warnings:
                result.warnings.extend([f"[{dk}] {w}" for w in warnings])

    # Require at least 2 successful results
    if len(individual_results) < 2:
        log.error("Only %d individual rule sets succeeded (need >= 2)", len(individual_results))
        result.warnings.append(
            f"Multi-indication merge skipped: only {len(individual_results)}/{ len(regimens)} "
            "individual pipelines succeeded (minimum 2 required)"
        )
        return result

    log.info("Step 1 complete: %d/%d individual rule sets succeeded",
             len(individual_results), len(regimens))

    # Step 2: Collect DDI evidence
    log.info("Step 2: Collecting cross-regimen DDI evidence")
    from rule_engine.evidence.ddi import collect_ddi_evidence
    ddi_evidence = await collect_ddi_evidence(regimens, config)
    log.info("DDI evidence: %d pairs", len(ddi_evidence.pairs))

    # Step 3: LLM merge
    log.info("Step 3: Merging rule sets via LLM")
    from rule_engine.agent_multidrug import merge_rule_sets
    try:
        merged_rule_set, merge_agent_log = await merge_rule_sets(
            individual_results, ddi_evidence, config
        )
    except Exception as e:
        log.error("Merge failed: %s", e)
        result.warnings.append(f"Multi-indication merge LLM call failed: {e}")
        return result

    # Step 4: Validate
    log.info("Step 4: Validating merged rule set")
    individual_for_validation = [(rs, w) for rs, _log, w in individual_results]
    merge_warnings = validate_multi_rule_set(merged_rule_set, individual_for_validation, ddi_evidence)
    merge_agent_log.warnings = merge_warnings
    result.warnings.extend([f"[merged] {w}" for w in merge_warnings])

    # Step 5: Save
    merged_path = _save_multi_rule_set(merged_rule_set, merge_agent_log, config.output_dir)
    result.merged_rule_set = merged_rule_set
    result.merged_path = merged_path

    log.info("Multi-indication pipeline complete: merged %d rule sets -> %s",
             len(individual_results), merged_path)

    return result
