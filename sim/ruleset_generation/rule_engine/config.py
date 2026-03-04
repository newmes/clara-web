"""Configuration for the Rule Discovery Pipeline.

Uses pydantic-settings for env-var overrides (prefix: RULE_ENGINE_).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


_DATA_ROOT = Path(__file__).resolve().parent.parent  # rule_discovery/


def _resolve_primekg_path(filename: str) -> Path:
    """Find PrimeKG CSV — check augmented paths first, then base filenames as fallback."""
    primary = _DATA_ROOT / "data" / "primekg" / filename
    if primary.exists():
        return primary
    fallback = _DATA_ROOT / "data" / "primekg_augmented" / filename
    if fallback.exists():
        return fallback
    # Try base (non-augmented) filename as last resort
    base_name = filename.replace("_exhaustion_augmented", "")
    if base_name != filename:
        base_primary = _DATA_ROOT / "data" / "primekg" / base_name
        if base_primary.exists():
            return base_primary
        base_fallback = _DATA_ROOT / "data" / "primekg_augmented" / base_name
        if base_fallback.exists():
            return base_fallback
    return primary  # return primary path even if missing (config shows expected location)


class RuleEngineConfig(BaseSettings):
    model_config = {"env_prefix": "RULE_ENGINE_"}

    # --- LLM backend (Gemini 2.0 Flash via OpenAI-compatible endpoint) ---
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.0-flash"
    llm_api_key: str = ""
    rate_limit_rpm: int = 150

    # --- Paths (data lives at rule_discovery/) ---
    project_root: Path = _DATA_ROOT
    output_dir: Path = _DATA_ROOT / "rule_sets"
    primekg_nodes: Path = _resolve_primekg_path("nodes_exhaustion_augmented.csv")
    primekg_edges: Path = _resolve_primekg_path("edges_exhaustion_augmented.csv")
    drugbank_dir: Path = _DATA_ROOT / "data" / "drugbank"
    onsides_db: Path = _DATA_ROOT / "data" / "onsides" / "onsides.db"

    # --- Concurrency ---
    max_concurrent: int = 16
    max_concurrent_multi: int = 2   # max concurrent regimen pipelines for multi-indication mode
    evidence_timeout: int = 10      # seconds per evidence source
    agent_timeout: int = 120        # seconds for LLM synthesis
    multi_stage: bool = False       # use multi-stage LLM pipeline for grounded synthesis

    # --- Project Data Sphere (SAS Viya CAS) ---
    pds_cas_url: str = "https://mpmprodvdmml.ondemand.sas.com/cas-shared-default-http/"
    pds_username: str = ""   # via RULE_ENGINE_PDS_USERNAME
    pds_password: str = ""   # via RULE_ENGINE_PDS_PASSWORD
    pds_data_dir: Path = _DATA_ROOT / "data" / "pds"

    # --- MCP servers (backup tool-calling) ---
    mcp_servers: list[str] = ["clinicaltrials", "openfda", "chembl", "pubmed"]
