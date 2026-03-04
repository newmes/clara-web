"""URL Configuration for Clinical Trial Viewer."""
from django.urls import path
from viewer import views

urlpatterns = [
    # Landing (tech showcase)
    path('', views.landing, name='landing'),

    # Simulation list
    path('simulations/', views.simulation_list, name='simulation_list'),

    # Demo pages
    path('demo/data-analysis-agent/',
         views.demo_anti_hallucination, name='demo_data_analysis_agent'),
    path('demo/care-agent/',
         views.demo_care_agent, name='demo_care_agent'),
    path('demo/data-collection-agent/',
         views.demo_care_agent, name='demo_data_collection_agent'),
    path('demo/patient-init/',
         views.demo_patient_init, name='demo_patient_init'),
    path('demo/daily-sim/',
         views.demo_daily_sim, name='demo_daily_sim'),
    path('demo/validate-sim/',
         views.demo_validate_sim, name='demo_validate_sim'),

    # Trial viewer
    path('trial/<str:run_id>/', views.trial_viewer, name='trial_viewer'),
    path('trial/<str:run_id>/<int:day>/', views.trial_viewer, name='trial_viewer_day'),

    # Patient state
    path('patient/<str:run_id>/<str:patient_id>/',
         views.patient_state, name='patient_state'),
    path('patient/<str:run_id>/<str:patient_id>/<int:day>/',
         views.patient_state, name='patient_state_day'),

    # API — day data
    path('api/day/<str:run_id>/<int:day>/',
         views.api_day_data, name='api_day_data'),

    # API — patient timeline
    path('api/patient/<str:run_id>/<str:patient_id>/',
         views.api_patient_timeline, name='api_patient_timeline'),

    # API — run meta
    path('api/run/<str:run_id>/',
         views.api_run_meta, name='api_run_meta'),

    # SSE stream
    path('api/sse/<str:run_id>/',
         views.sse_stream, name='sse_stream'),

    # Map metadata & tilemap
    path('api/map/<str:run_id>/',
         views.api_map_meta, name='api_map_meta'),
    path('api/map/<str:run_id>/tilemap/',
         views.api_map_tilemap, name='api_map_tilemap'),

    # Care Agent API
    path('api/care-agent/run/',
         views.api_care_agent_run, name='api_care_agent_run'),
    path('api/care-agent/patients/',
         views.api_care_agent_patients, name='api_care_agent_patients'),
    path('api/care-agent/media/<str:media_type>/<path:filename>',
         views.api_care_agent_media, name='api_care_agent_media'),
    path('api/care-agent/chat/',
         views.api_care_agent_chat, name='api_care_agent_chat'),

    # Compare dashboard
    path('compare/<str:run_id>/',
         views.compare_dashboard, name='compare_dashboard'),
    path('api/compare/<str:run_id>/',
         views.api_compare_data, name='api_compare_data'),
    path('api/compare/<str:run_id>/regenerate/',
         views.api_compare_regenerate, name='api_compare_regenerate'),

    # Live simulation API
    path('api/sim/start', views.api_sim_start, name='api_sim_start'),
    path('api/sim/status/<str:run_id>/',
         views.api_sim_status, name='api_sim_status'),
    path('api/sim/list', views.api_sim_list, name='api_sim_list'),
    path('api/sim/stop/<str:run_id>/',
         views.api_sim_stop, name='api_sim_stop'),
    path('api/sim/log/<str:run_id>/',
         views.api_sim_log, name='api_sim_log'),

    # Doc Agent — SAE document generation
    path('api/doc/generate', views.api_doc_generate, name='api_doc_generate'),
    path('api/doc/saes/<str:run_id>/<str:patient_id>/',
         views.api_doc_list_saes, name='api_doc_list_saes'),
    path('api/doc/download/<str:run_id>/<str:patient_id>/<str:filename>',
         views.api_doc_download, name='api_doc_download'),
    path('api/doc/list/<str:run_id>/',
         views.api_doc_list, name='api_doc_list'),
    path('api/doc/save', views.api_doc_save, name='api_doc_save'),
    path('api/doc/status', views.api_doc_update_status, name='api_doc_update_status'),
    path('api/doc/status/<str:run_id>/<str:patient_id>/<str:ae_slug>/',
         views.api_doc_get_status, name='api_doc_get_status'),

    # Doc Agent — Documents Hub & SAE Report Editor
    path('doc/<str:run_id>/',
         views.doc_hub, name='doc_hub'),
    path('doc/<str:run_id>/<str:patient_id>/<str:ae_slug>/',
         views.sae_report_editor, name='sae_report_editor'),

    # AntiHallu API
    path('api/antihallu/examples/',
         views.api_antihallu_examples, name='api_antihallu_examples'),
    path('api/antihallu/generate/',
         views.api_antihallu_generate, name='api_antihallu_generate'),

    # MedGemma API
    path('api/medgemma/analyze/',
         views.api_medgemma_analyze, name='api_medgemma_analyze'),
    path('api/medgemma/analyze-base/',
         views.api_medgemma_analyze_base, name='api_medgemma_analyze_base'),

    # CRF Tables
    path('doc/<str:run_id>/crf/',
         views.crf_tables, name='crf_tables'),
    path('api/crf/<str:run_id>/excel/',
         views.api_crf_excel_download, name='api_crf_excel_download'),
    path('api/crf/<str:run_id>/<str:domain>/',
         views.api_crf_domain_data, name='api_crf_domain_data'),

    # Statistical Analysis (CSR)
    path("doc/<str:run_id>/stats/",
         views.statistical_analysis, name="statistical_analysis"),
    path("api/stats/chat/",
         views.api_stats_chat_demo, name="api_stats_chat_demo"),
    path("api/stats/<str:run_id>/",
         views.api_stats_data, name="api_stats_data"),
    path("api/stats/<str:run_id>/chat/",
         views.api_stats_chat, name="api_stats_chat"),

    # Unified Doc Chat API (stats / crf / sae)
    path("api/doc/<str:run_id>/chat/",
         views.api_doc_chat, name="api_doc_chat"),

    # Multimodal Enhance API
    path('api/multimodal/enhance',
         views.api_multimodal_enhance, name='api_multimodal_enhance'),

    # Rule Set Generation
    path('demo/ruleset/',
         views.demo_ruleset_generation, name='demo_ruleset_generation'),
    path('api/ruleset/drugs/',
         views.api_ruleset_drugs, name='api_ruleset_drugs'),
    path('api/ruleset/compare-all/',
         views.api_ruleset_compare_all, name='api_ruleset_compare_all'),
    path('api/ruleset/compare/<str:drug_id>/',
         views.api_ruleset_compare, name='api_ruleset_compare'),
    path('api/ruleset/generate/',
         views.api_ruleset_generate, name='api_ruleset_generate'),
    path('api/ruleset/generate/status/<str:job_id>/',
         views.api_ruleset_generate_status, name='api_ruleset_generate_status'),

    # Demo API — auto-select latest run (no run_id needed)
    path('api/demo/saes/',
         views.api_demo_saes, name='api_demo_saes'),
    path('api/demo/generate/',
         views.api_demo_generate, name='api_demo_generate'),
    path('api/demo/reports/',
         views.api_demo_reports, name='api_demo_reports'),
]