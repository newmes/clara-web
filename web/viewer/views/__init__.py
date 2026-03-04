"""
views package — split from the monolithic views.py.

Re-exports every public view function so that
``from viewer import views; views.landing`` and
``from viewer.views import landing`` both work unchanged.
"""

# ── core ─────────────────────────────────────────────────
from .core import (
    landing,
    simulation_list,
)

# ── trial ────────────────────────────────────────────────
from .trial import (
    trial_viewer,
    patient_state,
    api_day_data,
    api_run_meta,
    api_patient_timeline,
    sse_stream,
)

# ── demo ─────────────────────────────────────────────────
from .demo import (
    demo_anti_hallucination,
    demo_patient_init,
    demo_daily_sim,
    demo_validate_sim,
    api_antihallu_examples,
    api_antihallu_generate,
    api_demo_saes,
    api_demo_generate,
    api_demo_reports,
)

# ── compare ──────────────────────────────────────────────
from .compare import (
    compare_dashboard,
    api_compare_data,
    api_compare_regenerate,
)

# ── sim_api ──────────────────────────────────────────────
from .sim_api import (
    api_sim_start,
    api_sim_status,
    api_sim_list,
    api_sim_stop,
    api_sim_log,
)

# ── doc ──────────────────────────────────────────────────
from .doc import (
    api_doc_generate,
    api_doc_list_saes,
    api_doc_download,
    api_doc_list,
    api_doc_save,
    api_doc_get_status,
    api_doc_update_status,
    sae_report_editor,
    doc_hub,
    crf_tables,
    api_crf_domain_data,
    api_crf_excel_download,
    api_doc_chat,
)

# ── care_api ─────────────────────────────────────────────
from .care_api import (
    demo_care_agent,
    api_care_agent_run,
    api_care_agent_patients,
    api_care_agent_media,
    api_care_agent_chat,
)

# ── medgemma ─────────────────────────────────────────────
from .medgemma import (
    api_medgemma_analyze,
    api_medgemma_analyze_base,
    api_multimodal_enhance,
)

# ── stats ────────────────────────────────────────────────
from .stats import (
    statistical_analysis,
    api_stats_data,
    api_stats_chat,
    api_stats_chat_demo,
)

# ── ruleset ──────────────────────────────────────────────
from .ruleset import (
    api_ruleset_drugs,
    api_ruleset_compare,
    api_ruleset_compare_all,
    demo_ruleset_generation,
    api_ruleset_generate,
    api_ruleset_generate_status,
)

# ── map ──────────────────────────────────────────────────
from .map import (
    api_map_meta,
    api_map_tilemap,
)
