# HANDOFF — CLARA Clinical Trial Simulation Engine

## Goal
Django 웹 뷰어의 외부 접속, 성능 최적화, 안정성 개선. Cloudflare Named Tunnel로 고정 URL 제공.

## Current Progress

### 1. Cloudflare Named Tunnel 설정 완료
- **URL**: `https://clara.parrotvox.com` (고정, 만료 없음)
- **Tunnel ID**: `a4d3b6a4-32d4-43fb-bfac-c85866c557ae` (parrotvox.com 소유 계정에서 생성)
- **토큰 기반 실행**: credentials file 불필요, 토큰만으로 실행
- **Keepalive 래퍼**: `/home/gideon/cloudflared-clara-keepalive.sh` — 죽으면 3초 후 자동 재시작
- **실행 명령**: `nohup /home/gideon/cloudflared-clara-keepalive.sh > /dev/null 2>&1 &`
- **로그**: `/home/gideon/cloudflared-clara.log`

### 2. Django 설정 (tunnel/proxy 지원)
- `settings.py`: `CSRF_TRUSTED_ORIGINS`에 `*.trycloudflare.com`, `*.parrotvox.com` 추가
- `USE_X_FORWARDED_HOST = True`, `SECURE_PROXY_SSL_HEADER` 설정
- `CORS_ALLOW_ALL_ORIGINS = True`

### 3. 성능 최적화
- **JSONL/JSON 파싱 캐시**: `crf_aggregator.py`에 `_read_jsonl_cached`, `_load_patient_json` (lru_cache, mtime 기반 무효화)
- **Aggregate 결과 캐시**: `_aggregate_cache` dict — 전체 결과 캐시 후 pagination 적용
- **성능**: Cold 2-3s → Cached 0.01-0.05s (100x+ 개선)
- **Doc Hub**: `views.py`의 `_load_patient_data`도 crf_aggregator 캐시 활용

### 4. 404 폴링 폭주 해결
- **원인**: `PINNED_RUN_ID`가 삭제된 Etoposide run을 가리킴 → 모든 landing 페이지에서 404 폴링 폭주
- **해결**:
  - `PINNED_RUN_ID`를 Padcev run으로 변경
  - `BlockStalePollingMiddleware` 추가 (middleware.py) — 차단된 run ID에 204 반환
  - `trial.html`에 `livePollErrors` 카운터 (3회 실패 시 폴링 중지)

### 5. 버그 수정
- **SAE 'dyspnoea' not found**: doc_hub에서 Open Report 링크에 `&mode={{ mode }}` 누락 → 추가
- **CM탭 500 에러**: `CMTRT` 필드가 dict인 경우 처리 (`isinstance` 체크)
- **Landing 하드코딩 IP**: 절대 경로 → 상대 경로 (`/demo/data-collection-agent/`)
- **CRF fetch timeout**: 30초로 증가 + 2회 자동 재시도 + 수동 Retry 링크

### 6. 커밋 & 푸시
- 커밋 `aa2d0e8` on `feature/doc-agent-enhancement` → origin 푸시 완료

## What Worked
- **토큰 기반 Named Tunnel**: parrotvox.com 소유 계정(Victoria@newmes.io)의 대시보드에서 tunnel 생성 → 토큰 복사 → 서버에서 토큰으로 실행. credentials file이나 cloudflared login 불필요.
- **DNS CNAME**: 대시보드에서 `clara` → `{tunnel-id}.cfargotunnel.com` CNAME 추가 (Proxied)
- **lru_cache + mtime**: 파일 변경 시 자동 무효화, 변경 없으면 캐시 히트
- **Keepalive 래퍼**: while true + sleep 3 패턴으로 프로세스 자동 재시작

## What Didn't Work
- **ngrok 무료**: IP가 URL에 노출됨, 대역폭 제한(ERR_NGROK_725)
- **Cloudflare Quick Tunnel**: URL이 매번 바뀜, 자주 만료됨
- **다른 계정(Hj.choi)에서 tunnel 생성 → parrotvox.com DNS 연결**: Error 1033 — tunnel과 DNS가 같은 Cloudflare 계정에 있어야 함
- **cloudflared login**: "Cloudflare One Connector: cloudflared Write" 권한 필요 — Individual Domain 스코프에서는 불가, Account 스코프 필요
- **whitenoise 미들웨어**: Docker 컨테이너에 패키지 미설치 → ImproperlyConfigured

## Next Steps
1. **서버 재부팅 대비**: keepalive 스크립트를 systemd service나 crontab @reboot로 등록
2. **이전 계정(Hj.choi) tunnel 정리**: 사용하지 않는 tunnel "clara" (ID: 57b90521...) 삭제
3. **이전 DNS 레코드 정리**: 잘못 붙은 CNAME 레코드 확인 및 삭제
4. **Quick Tunnel 프로세스 정리**: `/home/gideon/cloudflared-django-keepalive.sh` 비활성화 (더 이상 불필요)

## Key Files
| 파일 | 역할 |
|------|------|
| `/home/gideon/cloudflared-clara-keepalive.sh` | Named Tunnel 자동 재시작 래퍼 |
| `/home/gideon/cloudflared-clara.log` | Tunnel 로그 |
| `/home/gideon/.cloudflared/config-clara.yml` | 이전 config (토큰 방식에서는 미사용) |
| `/data2/workspace/vital/frontend/trial_server/settings.py` | Django 설정 |
| `/data2/workspace/vital/frontend/viewer/middleware.py` | BlockStalePollingMiddleware |
| `/data2/workspace/vital/frontend/viewer/crf_aggregator.py` | 캐시 포함 CRF 데이터 집계 |
| `/data2/workspace/vital/frontend/viewer/views.py` | Django 뷰 (PINNED_RUN_ID 등) |

## Tunnel 재실행 방법
```bash
# 프로세스 확인
ps aux | grep cloudflared | grep -v grep

# 죽어있으면 실행
nohup /home/gideon/cloudflared-clara-keepalive.sh > /dev/null 2>&1 &

# 로그 확인
tail -f /home/gideon/cloudflared-clara.log
```
