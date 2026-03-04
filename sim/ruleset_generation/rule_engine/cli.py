"""Rule Engine CLI — generate clinical trial simulation rules from evidence."""

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rule_engine.config import RuleEngineConfig
from rule_engine.pipeline import run_pipeline, run_multi_indication_pipeline

app = typer.Typer(name="rule_engine", help="Rule Discovery Pipeline for Clinical Trial Simulation")
console = Console()


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="JSON file with list of {drugs, indication} entries"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory for rule_sets"),
    max_concurrent: int = typer.Option(16, "--concurrency", "-c", help="Max concurrent pairs"),
    multi_stage: bool = typer.Option(False, "--multi-stage", help="Use multi-stage LLM pipeline for grounded synthesis"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate rule sets from a JSON input file of drug-indication entries.

    JSON format: [{"drugs": ["Drug1", "Drug2"], "indication": "disease"}, ...]
    Legacy format: [{"drug_name": "Drug1", "indication": "disease"}, ...] is also supported.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    data = json.loads(input_file.read_text())
    entries: list[tuple[list[str], str]] = []
    for item in data:
        if "drugs" in item:
            entries.append((item["drugs"], item["indication"]))
        elif "drug_name" in item:
            # Legacy single-drug format
            entries.append(([item["drug_name"]], item["indication"]))
        else:
            console.print(f"[red]Invalid entry (missing 'drugs' or 'drug_name'): {item}[/red]")
            raise typer.Exit(1)

    console.print(f"Loaded {len(entries)} drug-indication entries")

    config = RuleEngineConfig()
    if output_dir:
        config.output_dir = output_dir
    config.max_concurrent = max_concurrent
    config.multi_stage = multi_stage

    result = asyncio.run(run_pipeline(entries, config))

    # Summary table
    table = Table(title="Pipeline Results")
    table.add_column("Drug(s)")
    table.add_column("Indication")
    table.add_column("Status")
    table.add_column("Warnings")

    for drugs, indication, path in result.successful:
        dk = "+".join(drugs)
        n_warnings = len(result.warnings.get(dk, []))
        status = f"[green]OK[/green] → {path.name}"
        table.add_row(" + ".join(drugs), indication, status, str(n_warnings))

    for drugs, indication, error in result.failed:
        table.add_row(" + ".join(drugs), indication, "[red]FAILED[/red]", error[:60])

    console.print(table)


@app.command("generate-single")
def generate_single(
    drug_names: str = typer.Argument(..., help="Drug name(s), '+' separated for combo therapy"),
    indication: str = typer.Argument(..., help="Indication / disease"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o"),
    multi_stage: bool = typer.Option(False, "--multi-stage", help="Use multi-stage LLM pipeline for grounded synthesis"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a rule set for a single drug-indication pair (or combo).

    For combination therapy, separate drug names with '+':
        rule-engine generate-single "Enfortumab vedotin + Pembrolizumab" "urothelial carcinoma"
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    drugs = [d.strip() for d in drug_names.split("+")]
    drug_label = " + ".join(drugs)

    config = RuleEngineConfig()
    if output_dir:
        config.output_dir = output_dir
    config.multi_stage = multi_stage

    result = asyncio.run(run_pipeline([(drugs, indication)], config))

    if result.successful:
        drugs_out, ind, path = result.successful[0]
        dk = "+".join(drugs_out)
        console.print(f"[green]Success![/green] Rule set saved to: {path}")
        if result.warnings.get(dk):
            for w in result.warnings[dk]:
                console.print(f"  [yellow]Warning:[/yellow] {w}")
    else:
        _, _, error = result.failed[0]
        console.print(f"[red]Failed:[/red] {error}")
        raise typer.Exit(1)


@app.command("generate-multi")
def generate_multi(
    input_file: Path = typer.Argument(..., help="JSON file with {regimens: [{drugs: [...], indication: '...'}]}"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory for rule_sets"),
    concurrency: int = typer.Option(2, "--concurrency", "-c", help="Max concurrent regimen pipelines (default 2)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a unified multi-indication rule set from 2-3 drug-indication regimens.

    Input JSON format:
    {
        "regimens": [
            {"drugs": ["Drug1", "Drug2"], "indication": "Disease1"},
            {"drugs": ["Drug3"], "indication": "Disease2"},
            {"drugs": ["Drug4", "Drug5"], "indication": "Disease3"}
        ]
    }

    Generates individual rule sets for each regimen, then merges them into one
    unified rule set with combined AE profile, per-indication efficacy, and
    drug-drug interactions.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    data = json.loads(input_file.read_text())
    if "regimens" not in data:
        console.print("[red]Input JSON must have a 'regimens' key[/red]")
        raise typer.Exit(1)

    regimens: list[tuple[list[str], str]] = []
    for item in data["regimens"]:
        if "drugs" not in item or "indication" not in item:
            console.print(f"[red]Each regimen must have 'drugs' and 'indication': {item}[/red]")
            raise typer.Exit(1)
        regimens.append((item["drugs"], item["indication"]))

    if len(regimens) < 2:
        console.print(f"[red]Multi-indication requires at least 2 regimens, got {len(regimens)}[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(regimens)} regimens for multi-indication pipeline:")
    for drugs, indication in regimens:
        console.print(f"  {' + '.join(drugs)} | {indication}")

    config = RuleEngineConfig()
    if output_dir:
        config.output_dir = output_dir
    config.max_concurrent_multi = concurrency

    result = asyncio.run(run_multi_indication_pipeline(regimens, config))

    # Summary table
    table = Table(title="Multi-Indication Pipeline Results")
    table.add_column("Regimen")
    table.add_column("Indication")
    table.add_column("Status")

    for drugs, indication, path in result.individual_successful:
        table.add_row(" + ".join(drugs), indication, f"[green]OK[/green] -> {path.name}")
    for drugs, indication, error in result.individual_failed:
        table.add_row(" + ".join(drugs), indication, f"[red]FAILED[/red] {error[:50]}")

    # Merged result row
    if result.merged_rule_set and result.merged_path:
        n_aes = len(result.merged_rule_set.adverse_events)
        n_drugs = len(result.merged_rule_set.drugs)
        n_ddis = len(result.merged_rule_set.drug_interactions)
        table.add_row(
            f"[bold]MERGED ({n_drugs} drugs)[/bold]",
            ", ".join(result.merged_rule_set.indications),
            f"[green bold]OK[/green bold] -> {result.merged_path.name} ({n_aes} AEs, {n_ddis} DDIs)",
        )
    else:
        table.add_row("[bold]MERGED[/bold]", "-", "[red bold]FAILED[/red bold]")

    console.print(table)

    # Warnings summary
    if result.warnings:
        console.print(f"\n[yellow]{len(result.warnings)} warnings:[/yellow]")
        for w in result.warnings[:20]:
            console.print(f"  [yellow]Warning:[/yellow] {w}")
        if len(result.warnings) > 20:
            console.print(f"  ... and {len(result.warnings) - 20} more")

    if not result.merged_rule_set:
        raise typer.Exit(1)


@app.command()
def healthcheck(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Check LLM backend and data file availability."""
    import httpx

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    config = RuleEngineConfig()
    all_ok = True

    # Check LLM backend
    console.print(f"Checking LLM backend at {config.llm_base_url}...")
    try:
        url = config.llm_base_url.rstrip("/") + "/models"
        headers = {}
        if config.llm_api_key:
            headers["Authorization"] = f"Bearer {config.llm_api_key}"
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json()
        console.print(f"  [green]LLM OK[/green] — models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        console.print(f"  [red]LLM FAILED[/red] — {e}")
        all_ok = False

    # Check data files
    for label, path in [
        ("PrimeKG nodes", config.primekg_nodes),
        ("PrimeKG edges", config.primekg_edges),
        ("DrugBank dir", config.drugbank_dir),
    ]:
        exists = path.exists()
        status = "[green]OK[/green]" if exists else "[yellow]MISSING[/yellow]"
        console.print(f"  {label}: {status} ({path})")
        if not exists and "DrugBank" not in label:
            all_ok = False

    if all_ok:
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print("\n[yellow]Some checks failed — see above[/yellow]")
