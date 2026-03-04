"""
Statistical Analysis views: statistical_analysis, api_stats_data,
api_stats_chat, api_stats_chat_demo.

Also exports shared chat helpers used by doc.py:
    _sanitize_messages, _call_chat_llm, _match_sections, _retrieve_context,
    _STATS_CHAT_MODEL, _STATS_SYSTEM_PROMPT, _STATS_CHAT_URL
"""
import json
import logging
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from ._helpers import (
    DATA_DIR, PINNED_RUN_ID, _get_run_path,
    _load_rule_set, _load_run_meta, _list_patients, logger,
)


def statistical_analysis(request, run_id: str):
    """Statistical Analysis page — CSR-style tables and charts."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    rule_set = _load_rule_set(run_path)
    meta = _load_run_meta(run_path)
    drug_name = rule_set.get("drug_name") or meta.get("drug_name", "Unknown")
    indication = rule_set.get("indication") or meta.get("indication", "")

    sim_dir = run_path / "simulations"
    available_modes = []
    if sim_dir.exists():
        if list(sim_dir.glob("*_care_ai.jsonl")):
            available_modes.append("care_ai")
        if list(sim_dir.glob("*_natural.jsonl")):
            available_modes.append("natural")

    n_patients = len(_list_patients(run_path))

    from django.shortcuts import render
    return render(request, "doc/statistical_analysis.html", {
        "run_id": run_id,
        "drug_name": drug_name,
        "indication": indication,
        "n_patients": n_patients,
        "available_modes": available_modes,
        "model_name": meta.get("model", ""),
    })


@require_GET
def api_stats_data(request, run_id: str):
    """JSON API: compute and return all CSR statistics."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    mode = request.GET.get("mode", "care_ai")

    cache_path = run_path / "validation" / f"csr_stats_{mode}.json"
    sim_dir = run_path / "simulations"
    needs_compute = True
    if cache_path.exists():
        sim_files = list(sim_dir.glob("*.jsonl")) if sim_dir.exists() else []
        if sim_files:
            newest_sim = max(f.stat().st_mtime for f in sim_files)
            if cache_path.stat().st_mtime >= newest_sim:
                needs_compute = False

    if needs_compute:
        try:
            import sys as _sys
            _proj_root = str(Path(settings.BASE_DIR).parent)
            if _proj_root not in _sys.path:
                _sys.path.insert(0, _proj_root)
            from validation.csr_stats import compute_csr_stats
            stats = compute_csr_stats(str(run_path), mode)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(stats, f, ensure_ascii=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
    else:
        with open(cache_path) as f:
            stats = json.load(f)

    return JsonResponse(stats, json_dumps_params={"ensure_ascii": False})


# ─── Stats Chatbot ──────────────────────────────────────────

_STATS_CHAT_URL = os.environ.get("CTE_VLLM_BASE_URL", "").rstrip("/")
_STATS_CHAT_MODEL = os.environ.get("CTE_VLLM_MODEL_ID", "medgemma-4b-antihallu")


def _sanitize_messages(messages):
    """Ensure roles alternate user/assistant after system. Drop consecutive same-role messages."""
    if not messages:
        return messages
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            out.append(msg)
            continue
        if out and out[-1].get("role") == role:
            # merge into previous to avoid consecutive same-role
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n" + msg.get("content", "")}
        else:
            out.append(msg)
    # vLLM requires last message to be user
    if out and out[-1].get("role") != "user" and out[-1].get("role") != "system":
        out.append({"role": "user", "content": "(continue)"})
    return out


def _call_chat_llm(messages, query_meta):
    """Call vLLM for chat completions. Returns JsonResponse."""
    import re as _re

    if not _STATS_CHAT_URL:
        return JsonResponse({"error": "vLLM not configured (CTE_VLLM_BASE_URL)"}, status=500)

    messages = _sanitize_messages(messages)

    payload = {
        "model": _STATS_CHAT_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
        "repetition_penalty": 1.15,
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{_STATS_CHAT_URL}/chat/completions",
        data=body_bytes,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        t0 = time.time()
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency_ms = round((time.time() - t0) * 1000)
        answer = data["choices"][0]["message"]["content"]
        answer = _re.sub(r'<unused\d+>.*?<unused\d+>', '', answer, flags=_re.DOTALL).strip()
        query_meta["backend"] = "vllm"
        query_meta["model"] = _STATS_CHAT_MODEL
        return JsonResponse({
            "response": answer,
            "latency_ms": latency_ms,
            "query": query_meta,
        })
    except Exception as exc:
        detail = str(exc)
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
        logging.warning("Chat LLM call failed: %s — %s", exc, detail)
        return JsonResponse({"error": f"LLM call failed: {detail}"}, status=500)

_STATS_SYSTEM_PROMPT = (
    "You are CLARA's Statistical Analysis Assistant, an expert clinical trial biostatistician.\n"
    "You help researchers interpret CSR (Clinical Study Report) statistical results.\n\n"
    "STRICT RULES — violating any rule is a critical failure:\n"
    "1. Answer based ONLY on the provided data. NEVER fabricate or infer numbers.\n"
    "2. When quoting a number, copy it EXACTLY from the data. Do NOT round, combine, or paraphrase.\n"
    "3. DISTINGUISH between 'all-grade' and 'Grade 3+' (G3+) columns carefully.\n"
    "   - 'all=56.0%' means all-grade incidence is 56%.\n"
    "   - 'G3+=2.0%' means Grade 3+ incidence is 2%.\n"
    "   - These are DIFFERENT numbers. NEVER use the all-grade number when asked about G3+.\n"
    "4. Answer ONLY the question asked. Do NOT dump unrelated sections.\n"
    "5. Be concise: 2-5 sentences unless the user asks for detail.\n"
    "6. If the data does not contain the answer, say so. Do NOT guess.\n"
    "7. Use the same language as the user.\n"
    "8. When user references data with @[...] tags, focus on that specific data point.\n"
)


def _compact_stats(stats: dict, tab: str) -> str:
    """Legacy: tab-based extraction. Use _retrieve_context() instead."""
    return _retrieve_context(stats, "", tab)


# ─── Keyword → section mapping for lightweight RAG ───
_SECTION_KEYWORDS = {
    "demographics": [
        "age", "sex", "gender", "race", "ethnicity", "weight", "height", "bmi",
        "demographic", "population", "patient characteristics", "baseline",
        "나이", "성별", "인종", "체중", "환자 특성",
    ],
    "efficacy": [
        "orr", "dcr", "response", "tumor", "waterfall", "survival", "pfs", "os",
        "progression", "recist", "cr", "pr", "sd", "pd", "best response",
        "time to response", "ttr", "duration of response", "dor",
        "반응률", "종양", "생존", "효능",
    ],
    "safety": [
        "ae", "adverse", "toxicity", "side effect", "safety", "grade",
        "sae", "serious", "fatal",
        "이상반응", "부작용", "독성", "안전",
    ],
    "safety_detail": [
        "fatigue", "nausea", "diarrhea", "rash", "alopecia", "neuropathy",
        "stomatitis", "pruritus", "hyperglycemia", "anemia", "pneumonitis",
        "appetite", "infusion", "vomiting", "constipation", "pain",
    ],
    "treatment": [
        "dose", "rdi", "interruption", "reduction", "modification", "discontinu",
        "cycle", "duration", "administration", "drug",
        "투여", "용량", "중단",
    ],
    "labs": [
        "lab", "glucose", "hemoglobin", "platelet", "creatinine", "alt", "ast",
        "bilirubin", "albumin", "sodium", "anc", "neutrophil", "blood",
        "검사", "혈액",
    ],
    "ecog": [
        "ecog", "performance status", "functional", "ps",
        "수행능력",
    ],
    "conmeds": [
        "concomitant", "medication", "conmed", "supportive", "steroid",
        "병용약",
    ],
    "disposition": [
        "disposition", "enrolled", "completed", "discontinued", "death", "dropout",
        "withdrawal",
        "등록", "완료", "중단", "사망",
    ],
}


def _match_sections(message: str, tab: str) -> list:
    """Return list of matched section names based on message keywords + active tab."""
    msg_lower = message.lower()
    scores = {}
    for section, keywords in _SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[section] = score

    # Always include the active tab
    tab_to_section = {
        "safety": "safety", "demographics": "demographics", "efficacy": "efficacy",
        "labs": "labs", "ecog": "ecog", "conmeds": "conmeds",
        "treatment": "treatment", "disposition": "disposition",
    }
    if tab in tab_to_section:
        sec = tab_to_section[tab]
        scores[sec] = scores.get(sec, 0) + 2  # boost active tab

    # If safety_detail matched, ensure safety is also included
    if "safety_detail" in scores:
        scores["safety"] = scores.get("safety", 0) + scores["safety_detail"]

    # If nothing matched, default to safety (most common)
    if not scores:
        scores["safety"] = 1

    # Sort by score descending, return top sections
    return [s for s, _ in sorted(scores.items(), key=lambda x: -x[1]) if s != "safety_detail"]


def _extract_section(stats: dict, section: str, message: str) -> list:
    """Extract formatted lines for a given section."""
    lines = []
    msg_lower = message.lower()

    if section == "disposition":
        disp = stats.get("disposition", {})
        lines.append(f"[Disposition] Enrolled: {disp.get('enrolled',0)}, "
                     f"Completed: {disp.get('completed',{}).get('n',0)} ({disp.get('completed',{}).get('pct',0)}%), "
                     f"Discontinued: {disp.get('discontinued',{}).get('n',0)}, "
                     f"Deaths: {disp.get('deaths',{}).get('n',0)}")
        reasons = disp.get("reasons", {})
        if reasons:
            parts = [f"{k}: {v}" for k, v in reasons.items() if v]
            if parts:
                lines.append(f"  Reasons: {', '.join(parts)}")

    elif section == "demographics":
        demo = stats.get("demographics", {})
        age = demo.get("age", {})
        if age:
            lines.append(f"[Demographics] Age: mean={age.get('mean','?')} SD={age.get('std','?')}, "
                         f"median={age.get('median','?')}, range={age.get('min','?')}-{age.get('max','?')}")
        for cat_key in ("sex", "race", "ecog"):
            cat = demo.get(cat_key, {})
            if cat:
                parts = [f"{k}={v.get('n',0)}({v.get('pct',0)}%)" for k, v in cat.items()]
                lines.append(f"  {cat_key.title()}: {', '.join(parts)}")
        # BMI if asked
        bmi = demo.get("bmi", {})
        if bmi and any(kw in msg_lower for kw in ("bmi", "weight", "체중")):
            lines.append(f"  BMI: mean={bmi.get('mean','?')} SD={bmi.get('std','?')}, "
                         f"range={bmi.get('min','?')}-{bmi.get('max','?')}")

    elif section == "efficacy":
        eff = stats.get("efficacy", {})
        orr = eff.get("orr", {})
        dcr = eff.get("dcr", {})
        km_os = eff.get("km_os", {})
        km_pfs = eff.get("km_pfs", {})
        lines.append(f"[Efficacy] ORR: {orr.get('pct',0)}% (n={orr.get('n',0)}, "
                     f"CI: {orr.get('ci','?')}), DCR: {dcr.get('pct',0)}%")
        lines.append(f"  Median OS: {km_os.get('median','NR')} days, "
                     f"Median PFS: {km_pfs.get('median','NR')} days")
        br = eff.get("best_response", {})
        if br:
            parts = [f"{k}={v.get('n',0)}({v.get('pct',0)}%)" for k, v in br.items()]
            lines.append(f"  Best response: {', '.join(parts)}")
        ttr = eff.get("time_to_response", {})
        if ttr.get("n"):
            lines.append(f"  TTR: median={ttr.get('median','?')} days (n={ttr['n']})")
        dor = eff.get("dor", {})
        if dor.get("n"):
            lines.append(f"  DoR: median={dor.get('median','NR')} days, events={dor.get('events',0)}")
        # Waterfall individual data if asked
        wf = eff.get("waterfall", [])
        if wf and any(kw in msg_lower for kw in ("waterfall", "tumor change", "pd ", "pr ", "cr ", "종양")):
            lines.append("  Waterfall (per patient):")
            for w in wf:
                lines.append(f"    {w.get('pid','?')}: {w.get('change',0)}% ({w.get('response','?')})")

    elif section == "safety":
        safe = stats.get("safety", {})
        sm = safe.get("summary", {})
        lines.append(f"[Safety Summary]")
        lines.append(f"  Any AE: {sm.get('any_ae',{}).get('pct',0)}% (n={sm.get('any_ae',{}).get('n',0)})")
        lines.append(f"  Grade>=3 AE: {sm.get('grade_gte3',{}).get('pct',0)}% (n={sm.get('grade_gte3',{}).get('n',0)})")
        lines.append(f"  SAE (Serious): {sm.get('sae',{}).get('pct',0)}% (n={sm.get('sae',{}).get('n',0)})")
        lines.append(f"  Fatal AE: {sm.get('fatal',{}).get('pct',0)}% (n={sm.get('fatal',{}).get('n',0)})")
        lines.append(f"  Led to discontinuation: {sm.get('led_to_discont',{}).get('pct',0)}% (n={sm.get('led_to_discont',{}).get('n',0)})")
        lines.append(f"  Led to interruption: {sm.get('led_to_interrupt',{}).get('pct',0)}% (n={sm.get('led_to_interrupt',{}).get('n',0)})")
        by_term = safe.get("by_term", [])
        # Check if user asks about a specific AE
        specific_aes = [ae for ae in by_term
                        if ae["term"].replace("_", " ") in msg_lower
                        or ae["term"].replace("_", "") in msg_lower.replace(" ", "")]
        if specific_aes:
            lines.append("  AE table (all-grade% ≠ G3+%, do NOT confuse):")
            for ae in specific_aes:
                gd = ae.get("grade_dist", {})
                gd_str = ", ".join(f"G{g}={n}" for g, n in sorted(gd.items()) if int(n) > 0)
                lines.append(f"    {ae['term']}: ALL-GRADE={ae['all_grade']['pct']}% (n={ae['all_grade']['n']}), "
                             f"GRADE3+={ae['grade_gte3']['pct']}% (n={ae['grade_gte3']['n']}), "
                             f"onset=Day {ae.get('onset_median','?')} "
                             f"(IQR: {ae.get('onset_iqr','?')}), grades: {gd_str}")
        else:
            # Top 10 AEs summary
            lines.append("  AE table (all-grade% ≠ G3+%, do NOT confuse):")
            lines.append("    TERM | ALL-GRADE% | GRADE3+% | ONSET")
            for ae in by_term[:10]:
                lines.append(f"    {ae['term']} | {ae['all_grade']['pct']}% | "
                             f"{ae['grade_gte3']['pct']}% | Day {ae.get('onset_median','?')}")

    elif section == "treatment":
        tx = stats.get("treatment", {})
        n_total = stats.get("n_patients", stats.get("n_simulated", "?"))
        dr = tx.get('dose_reduction_all', {})
        di = tx.get('dose_interruption_all', {})
        dc = tx.get('discontinuation_all', {})
        lines.append(f"[Treatment] N={n_total}")
        lines.append(f"  Duration: median={tx.get('duration',{}).get('median','?')} days")
        lines.append(f"  Cycles: median={tx.get('cycles',{}).get('median','?')}")
        lines.append(f"  Dose REDUCTION: {dr.get('n',0)} patients = {dr.get('pct',0)}%")
        lines.append(f"  Dose INTERRUPTION: {di.get('n',0)} patients = {di.get('pct',0)}%")
        lines.append(f"  Discontinuation: {dc.get('n',0)} patients = {dc.get('pct',0)}%")
        # Per-drug detail
        per_drug = tx.get("per_drug", {})
        for drug_name, drug_data in per_drug.items():
            admins = drug_data.get("n_admins", {})
            rdi = drug_data.get("rdi_median", "?")
            lines.append(f"  {drug_name}: admins median={admins.get('median','?')} "
                         f"(range {admins.get('min','?')}-{admins.get('max','?')}), RDI={rdi}%")

    elif section == "labs":
        abn = stats.get("labs", {}).get("abnormalities", {})
        if abn:
            lines.append("[Labs] Abnormalities:")
            for test, v in abn.items():
                lines.append(f"  {test}: any={v.get('any_pct',0)}%, G3+={v.get('g3_pct',0)}%")

    elif section == "ecog":
        ec = stats.get("ecog_shift", {})
        sm_ec = ec.get("summary", {})
        if sm_ec:
            lines.append(f"[ECOG Shift] Improved: {sm_ec.get('improved',{}).get('pct',0)}%, "
                         f"Stable: {sm_ec.get('stable',{}).get('pct',0)}%, "
                         f"Worsened: {sm_ec.get('worsened',{}).get('pct',0)}%")
        # Individual shifts
        patients = ec.get("patients", [])
        if patients and any(kw in msg_lower for kw in ("shift", "individual", "patient", "detail")):
            for p in patients[:10]:
                lines.append(f"  {p.get('pid','?')}: {p.get('baseline',0)} → {p.get('worst',0)} → {p.get('last',0)}")

    elif section == "conmeds":
        cm = stats.get("concomitant_meds", {})
        tbl = cm.get("table", [])
        if tbl:
            lines.append(f"[Concomitant Meds] {cm.get('total_unique_meds',0)} unique medications:")
            for m in tbl[:10]:
                lines.append(f"  {m['medication']}: {m['n']} ({m['pct']}%)")

    return lines


def _retrieve_context(stats: dict, message: str, tab: str) -> str:
    """Keyword-based retrieval: select relevant sections from stats based on message content.
    Target: <2500 chars to fit within 4096 token model context.
    """
    sections = _match_sections(message, tab)

    lines = []
    total_chars = 0
    char_budget = 2400

    for section in sections:
        section_lines = _extract_section(stats, section, message)
        section_text = "\n".join(section_lines)
        if total_chars + len(section_text) > char_budget and lines:
            break  # budget exceeded, stop adding sections
        lines.extend(section_lines)
        total_chars += len(section_text) + 1

    return "\n".join(lines)


@csrf_exempt
def api_stats_chat(request, run_id: str):
    """Chat API for statistical analysis — answers questions using vLLM."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "Empty message"}, status=400)

    mode = body.get("mode", "natural")
    tab = body.get("tab", "")
    history = body.get("history", [])

    # Load cached stats
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    cache_path = run_path / "validation" / f"csr_stats_{mode}.json"
    if not cache_path.exists():
        # Fallback: try the other mode
        alt_mode = "care_ai" if mode == "natural" else "natural"
        alt_path = run_path / "validation" / f"csr_stats_{alt_mode}.json"
        if alt_path.exists():
            cache_path = alt_path
            mode = alt_mode
        else:
            return JsonResponse({"error": "Stats not computed yet. Load the stats page first."}, status=400)

    with open(cache_path) as f:
        stats = json.load(f)

    # Load run metadata
    rule_set = _load_rule_set(run_path)
    meta = _load_run_meta(run_path)
    drug_name = rule_set.get("drug_name") or meta.get("drug_name", "Unknown")
    indication = rule_set.get("indication") or meta.get("indication", "")
    n_patients = len(_list_patients(run_path))

    # Keyword-based retrieval — select relevant sections from stats
    matched_sections = _match_sections(message, tab)
    compact = _retrieve_context(stats, message, tab)
    context_block = (
        f"Drug: {drug_name} | Indication: {indication} | "
        f"Mode: {mode} | N={n_patients}\n\n{compact}"
    )

    # Put context in system message
    context_msg = _STATS_SYSTEM_PROMPT + "\n---\nData:\n" + context_block

    # Build messages for vLLM — keep history minimal
    messages = [{"role": "system", "content": context_msg}]
    for msg in history[-4:]:  # last 2 turns
        role = msg.get("role", "user")
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    # Build query metadata for debug view
    query_meta = {
        "source": f"csr_stats_{mode}.json",
        "model": _STATS_CHAT_MODEL,
        "matched_sections": matched_sections,
        "tab": tab or "(none)",
        "context_data": compact,
        "history_turns": len(history) // 2,
        "message": message,
    }

    # Call LLM (vLLM or Gemini fallback)
    return _call_chat_llm(messages, query_meta)


@csrf_exempt
def api_stats_chat_demo(request):
    """Demo chat API — uses pinned run, falling back to latest with stats."""
    pinned_dir = DATA_DIR / "runs" / PINNED_RUN_ID
    if pinned_dir.is_dir():
        for mode in ("natural", "care_ai"):
            if (pinned_dir / "validation" / f"csr_stats_{mode}.json").exists():
                return api_stats_chat(request, PINNED_RUN_ID)

    runs_dir = DATA_DIR / "runs"
    if not runs_dir.exists():
        return JsonResponse({"error": "No simulation runs found. Please run a simulation first."}, status=404)

    # Fallback: find latest run that has validation stats
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir():
            for mode in ("natural", "care_ai"):
                cache = d / "validation" / f"csr_stats_{mode}.json"
                if cache.exists():
                    try:
                        return api_stats_chat(request, d.name)
                    except Exception as exc:
                        logging.warning("Stats chat demo failed for run %s: %s", d.name, exc)
                        return JsonResponse(
                            {"error": f"Stats chat failed for run '{d.name}': {exc}. "
                             "The statistics data may be incomplete or corrupted. "
                             "Try re-running the simulation or computing stats again."},
                            status=500,
                        )

    return JsonResponse(
        {"error": "No runs with computed stats found. "
         "Run a simulation first, then navigate to the Statistical Analysis page "
         "to compute the stats before using the chat."},
        status=404,
    )
