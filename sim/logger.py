"""Logger — 시뮬레이션 전체의 중앙 로깅 유틸리티

logs/ 디렉토리에 타임스탬프된 로그 파일을 생성한다.

로그 레벨:
  - Console: INFO (진행 상황 요약)
  - File: DEBUG (모든 상세 — hazard 값, LLM 입출력, 이벤트 등)

사용법:
  from sim.logger import get_logger, log_llm_call, log_event

  logger = get_logger(__name__)
  logger.info("Phase 0 시작")
  logger.debug("상세 정보")
  log_llm_call(agent="rule_agent", input_tokens=..., output_tokens=..., ...)
  log_event(day=15, event_type="ae_onset", detail={...})
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ── 싱글톤 설정 ──────────────────────────────────────

_LOG_DIR = Path(__file__).parent.parent / "logs"
_initialized = False
_log_file_path: Path | None = None
_sim_start_time: float | None = None

# LLM 호출 통계 추적
_llm_stats = {
    "total_calls": 0,
    "total_input_chars": 0,
    "total_output_chars": 0,
    "total_latency_ms": 0,
    "calls": [],
}


def init_logging(run_name: str | None = None) -> Path:
    """로깅 시스템을 초기화한다. 시뮬레이션 시작 시 1회 호출.

    Args:
        run_name: 로그 파일명에 포함할 이름 (없으면 타임스탬프만)

    Returns:
        로그 파일 경로
    """
    global _initialized, _log_file_path, _sim_start_time

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_part = f"_{run_name}" if run_name else ""
    _log_file_path = _LOG_DIR / f"sim_{timestamp}{name_part}.log"
    _sim_start_time = time.time()

    # 루트 로거 설정
    root_logger = logging.getLogger("sim")
    root_logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()

    # File handler (DEBUG — 모든 상세)
    fh = logging.FileHandler(_log_file_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(fh)

    # Console handler (INFO — 요약만)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(ch)

    _initialized = True

    root_logger.info(f"Logging initialized → {_log_file_path}")
    root_logger.debug(f"Log file: {_log_file_path.resolve()}")

    return _log_file_path


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환한다.

    Args:
        name: 보통 __name__ 전달

    Returns:
        logging.Logger
    """
    if not _initialized:
        init_logging()
    return logging.getLogger(f"sim.{name}")


# ── LLM 호출 로깅 ────────────────────────────────────

def log_llm_call(
    agent: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    model: str,
    latency_ms: float,
    max_tokens: int = 0,
) -> None:
    """LLM 호출 1회를 상세 로깅한다.

    Args:
        agent: 호출한 에이전트 이름
        system_prompt: 시스템 프롬프트 (전문)
        user_prompt: 사용자 프롬프트 (전문)
        response_text: LLM 응답 텍스트 (전문)
        model: 사용된 모델 ID
        latency_ms: 응답 시간 (ms)
        max_tokens: 요청한 max_tokens
    """
    logger = get_logger("llm")

    input_chars = len(system_prompt) + len(user_prompt)
    output_chars = len(response_text)

    # 통계 업데이트
    _llm_stats["total_calls"] += 1
    _llm_stats["total_input_chars"] += input_chars
    _llm_stats["total_output_chars"] += output_chars
    _llm_stats["total_latency_ms"] += latency_ms
    _llm_stats["calls"].append({
        "agent": agent,
        "model": model,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "latency_ms": round(latency_ms),
    })

    call_num = _llm_stats["total_calls"]

    # INFO: 요약만
    logger.info(
        f"  [LLM #{call_num}] {agent} | {model} | "
        f"in={input_chars:,}ch out={output_chars:,}ch | {latency_ms:.0f}ms"
    )

    # DEBUG: 전문 기록
    logger.debug(f"{'─' * 60}")
    logger.debug(f"LLM CALL #{call_num} — {agent}")
    logger.debug(f"Model: {model}, max_tokens: {max_tokens}")
    logger.debug(f"Latency: {latency_ms:.0f}ms")
    logger.debug(f"─── SYSTEM PROMPT ({len(system_prompt)} chars) ───")
    logger.debug(system_prompt[:2000] + ("..." if len(system_prompt) > 2000 else ""))
    logger.debug(f"─── USER PROMPT ({len(user_prompt)} chars) ───")
    logger.debug(user_prompt[:3000] + ("..." if len(user_prompt) > 3000 else ""))
    logger.debug(f"─── RESPONSE ({output_chars} chars) ───")
    logger.debug(response_text[:3000] + ("..." if output_chars > 3000 else ""))
    logger.debug(f"{'─' * 60}")


# ── 시뮬레이션 이벤트 로깅 ────────────────────────────

def log_event(
    patient_id: str,
    day: int,
    event_type: str,
    detail: dict | str,
) -> None:
    """시뮬레이션 이벤트를 구조화하여 로깅한다.

    Args:
        patient_id: 환자 ID
        day: 시뮬레이션 날짜
        event_type: 이벤트 유형 (ae_onset, ae_resolved, ae_grade_change, quiet_day, event_day, ...)
        detail: 상세 정보 (dict 또는 string)
    """
    logger = get_logger("event")

    detail_str = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)

    logger.debug(f"[{patient_id}] Day {day:3d} | {event_type:20s} | {detail_str}")


# ── Hazard 로깅 ──────────────────────────────────────

def log_hazard(
    patient_id: str,
    day: int,
    ae_term: str,
    hazard_value: float,
    triggered: bool,
) -> None:
    """일별 hazard 계산 결과를 로깅한다 (DEBUG 레벨).

    Args:
        patient_id: 환자 ID
        day: 시뮬레이션 날짜
        ae_term: AE 이름
        hazard_value: 계산된 hazard 확률
        triggered: 샘플링 결과 (발생 여부)
    """
    logger = get_logger("hazard")

    if triggered:
        logger.debug(f"[{patient_id}] Day {day:3d} | {ae_term:30s} | h={hazard_value:.6f} | ★ TRIGGERED")
    elif hazard_value > 0.001:
        # 의미 있는 확률만 로깅 (너무 작은 건 스킵)
        logger.debug(f"[{patient_id}] Day {day:3d} | {ae_term:30s} | h={hazard_value:.6f} |   (no onset)")


# ── 요약/통계 ────────────────────────────────────────

def log_summary() -> dict:
    """시뮬레이션 종료 시 전체 통계를 로깅하고 반환한다."""
    logger = get_logger("summary")

    elapsed = time.time() - (_sim_start_time or time.time())
    stats = dict(_llm_stats)
    stats["total_elapsed_sec"] = round(elapsed, 1)

    logger.info(f"\n{'═' * 60}")
    logger.info(f"SIMULATION STATISTICS")
    logger.info(f"{'═' * 60}")
    logger.info(f"  Total LLM calls:    {stats['total_calls']}")
    logger.info(f"  Total input chars:  {stats['total_input_chars']:,}")
    logger.info(f"  Total output chars: {stats['total_output_chars']:,}")
    logger.info(f"  Total LLM latency:  {stats['total_latency_ms'] / 1000:.1f}s")
    logger.info(f"  Avg latency/call:   {stats['total_latency_ms'] / max(stats['total_calls'], 1):.0f}ms")
    logger.info(f"  Total elapsed:      {elapsed:.1f}s")
    logger.info(f"  Log file:           {_log_file_path}")
    logger.info(f"{'═' * 60}")

    # 에이전트별 호출 통계
    agent_stats: dict[str, list] = {}
    for call in stats["calls"]:
        agent = call["agent"]
        if agent not in agent_stats:
            agent_stats[agent] = []
        agent_stats[agent].append(call)

    logger.debug(f"\nPer-agent breakdown:")
    for agent, calls in sorted(agent_stats.items()):
        total_lat = sum(c["latency_ms"] for c in calls)
        logger.debug(f"  {agent}: {len(calls)} calls, {total_lat:.0f}ms total")

    # JSON으로도 저장
    if _log_file_path:
        stats_path = _log_file_path.with_suffix(".stats.json")
        stats_json = {k: v for k, v in stats.items() if k != "calls"}
        stats_json["per_agent"] = {
            agent: {"calls": len(calls), "total_latency_ms": sum(c["latency_ms"] for c in calls)}
            for agent, calls in agent_stats.items()
        }
        stats_path.write_text(json.dumps(stats_json, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"Stats saved to {stats_path}")

    return stats


def get_log_file_path() -> Path | None:
    """현재 로그 파일 경로를 반환한다."""
    return _log_file_path
