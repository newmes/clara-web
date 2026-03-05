"""LLM Client — Google Gemini API 공통 호출 모듈

모든 Agent가 이 모듈을 통해 LLM을 호출한다.
모델 변경 시 이 파일만 수정하면 됨.
모든 호출은 자동으로 logs/에 기록된다.
"""

import json
import os
import threading
import time
import traceback

from google import genai
from sim.logger import log_llm_call, get_logger

_logger = get_logger("llm_client")

DEFAULT_MODEL = "gemini-2.0-flash"

_thread_local = threading.local()

MAX_RETRIES = 10
API_TIMEOUT_SEC = 120
RETRY_BACKOFF_BASE = 5

_client: genai.Client | None = None
_client_lock = threading.Lock()


def set_caller(agent_name: str | None = None):
    """현재 호출자 에이전트를 설정한다 (스레드 안전). 로그에 표시된다."""
    _thread_local.caller_context = agent_name


def set_api_key(api_key: str):
    """현재 스레드에 API 키를 설정한다 (스레드 격리).

    각 스레드(시뮬레이션, 룰셋 생성 등)가 독립적인 클라이언트를 사용하므로
    다른 스레드의 키를 덮어쓰지 않는다.
    """
    _thread_local.api_key = api_key
    _thread_local.client = genai.Client(api_key=api_key)


def get_api_key() -> str | None:
    """현재 스레드에 설정된 API 키를 반환한다. 없으면 None."""
    return getattr(_thread_local, "api_key", None)


def _get_client() -> genai.Client:
    """Gemini Client를 반환한다. 스레드 로컬 → 전역 싱글톤 → 환경변수 순으로 탐색."""
    # 1) 스레드별 클라이언트 (set_api_key로 설정된 것)
    thread_client = getattr(_thread_local, "client", None)
    if thread_client is not None:
        return thread_client

    # 2) 전역 싱글톤 (환경변수 기반, 최초 1회 생성)
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        _client = genai.Client(api_key=api_key)
        return _client


def _repair_truncated_json(raw: str) -> str | None:
    """잘린 JSON을 복구 시도한다.

    LLM이 max_output_tokens에 도달하면 JSON이 중간에 잘릴 수 있음.
    열린 bracket/brace를 닫아서 최대한 유효한 JSON을 만든다.

    Returns:
        복구된 JSON 문자열, 또는 복구 불가 시 None
    """
    # 1) 잘린 문자열 내부에서 마지막 유효 위치 찾기
    in_string = False
    escape = False
    last_good = 0
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            if in_string:
                in_string = False
                last_good = i
            else:
                in_string = True
            continue
        if not in_string and ch in "{}[],":
            last_good = i

    # 열린 문자열 내에서 잘렸으면 마지막 유효 위치까지 자르기
    if in_string:
        raw = raw[: last_good + 1]

    raw = raw.rstrip()
    if raw.endswith(","):
        raw = raw[:-1]

    # 2) 남은 열린 괄호를 닫기
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
            continue
        if ch == "[":
            stack.append("]")
            continue
        if ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    if not stack:
        return None  # 이미 균형 맞음 — 복구 불필요

    repaired = raw + "".join(reversed(stack))
    return repaired


def generate_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    caller: str | None = None,
) -> dict:
    """LLM에게 JSON 응답을 요청하고 파싱하여 반환한다.

    JSON이 잘릴 경우 자동 복구를 시도하고, 실패 시 재시도한다.

    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 프롬프트
        model: Gemini 모델 ID
        max_tokens: 최대 출력 토큰
        caller: 호출한 에이전트 이름 (로깅용, None이면 _caller_context 사용)

    Returns:
        파싱된 JSON dict

    Raises:
        json.JSONDecodeError: JSON 파싱 실패 시 (재시도 후에도 실패)
    """
    if not caller:
        agent = getattr(_thread_local, "caller_context", "unknown")
    else:
        agent = caller

    client = _get_client()
    last_error = None

    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                    "response_mime_type": "application/json",
                },
            )

            elapsed_ms = int((time.time() - t0) * 1000)
            raw = response.text or ""

            log_llm_call(
                agent=agent,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=raw,
                model=model,
                latency_ms=elapsed_ms,
                max_tokens=max_tokens,
            )

            # JSON 파싱
            try:
                result = json.loads(raw)
                return result
            except json.JSONDecodeError:
                # 복구 시도
                repaired = _repair_truncated_json(raw)
                if repaired:
                    try:
                        result = json.loads(repaired)
                        _logger.warning(
                            f"[{agent}] JSON repaired (truncated output). "
                            f"Original {out_chars} chars."
                        )
                        return result
                    except json.JSONDecodeError:
                        pass

                # 복구 실패 → 재시도
                _logger.warning(
                    f"[{agent}] JSON parse failed (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Raw length={out_chars}"
                )
                last_error = json.JSONDecodeError(
                    "LLM output is not valid JSON", raw[:200], 0
                )
                time.sleep(min(RETRY_BACKOFF_BASE * (attempt + 1), 30))
                continue

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            error_name = type(e).__name__
            _logger.warning(
                f"[{agent}] LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                f"{error_name}: {e}"
            )
            last_error = e

            # Rate limit / server error → backoff and retry
            wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), 60)
            _logger.info(f"  Retrying in {wait}s...")
            time.sleep(wait)
            continue

    # 모든 재시도 실패
    raise last_error or RuntimeError(f"LLM call failed after {MAX_RETRIES} retries")
