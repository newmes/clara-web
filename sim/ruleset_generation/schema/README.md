# Rule Set JSON Schema — 약물 유형별 스키마

`data/rule_set.json` 파일의 유효성 검증을 위한 JSON Schema 모음.

## 구조

```
data/schema/
├── README.md                    ← 지금 읽고 있는 파일
├── base.json                    ← 공통 베이스 스키마 (모든 유형이 공유)
│
├── iv_combination.json          ← IV 병용요법 (현재 Padcev+Pembro)
├── iv_monotherapy.json          ← IV 단독요법
├── oral_monotherapy.json        ← 경구 단독요법
├── oral_iv_combination.json     ← 경구 + IV 병용요법
├── subcutaneous_monotherapy.json ← 피하주사 단독요법
│
├── biomarker_targeted.json      ← [확장] 바이오마커 의존적 약물
└── maintenance_therapy.json     ← [확장] 유지요법 (유도→유지 2단계)
```

## 유형 분류 기준

| 분류 축 | 선택지 |
|---------|--------|
| **투여 경로 (Route)** | INTRAVENOUS, ORAL, SUBCUTANEOUS |
| **병용 여부** | Monotherapy, Combination |
| **바이오마커 의존** | 없음, 있음 (확장 스키마 합성) |
| **유지요법 여부** | 없음, 있음 (확장 스키마 합성) |

## 스키마 사용법

### 1. 기본 유형 선택 (route + 병용 여부)

| 약물 예시 | 기본 스키마 |
|-----------|------------|
| Padcev + Pembrolizumab (IV+IV) | `iv_combination.json` |
| Pembrolizumab 단독 (IV) | `iv_monotherapy.json` |
| Osimertinib 80mg QD (PO) | `oral_monotherapy.json` |
| CapeOx: Capecitabine(PO) + Oxaliplatin(IV) | `oral_iv_combination.json` |
| Trastuzumab SC 600mg | `subcutaneous_monotherapy.json` |

### 2. 확장 스키마 합성 (필요 시)

확장 스키마는 기본 유형 위에 추가 필드를 얹는 방식:

```
iv_monotherapy.json + biomarker_targeted.json
→ Trastuzumab IV (HER2+ 필수)

oral_monotherapy.json + biomarker_targeted.json
→ Osimertinib (EGFR mutation 필수)

iv_combination.json + maintenance_therapy.json
→ Carboplatin+Pemetrexed 유도 → Pemetrexed 유지
```

### 3. 필드 차이 요약

| 필드 | IV | Oral | SC | 비고 |
|------|:--:|:----:|:--:|------|
| `infusion_duration_minutes` | **필수** | 없음 | 없음 | IV 전용 |
| `daily_dosing_schedule` | 없음 | **필수** | 없음 | QD/BID/TID |
| `continuous_days_per_cycle` | 없음 | 선택 | 없음 | 간헐적 경구투여 시 (e.g., 14/21) |
| `injection_volume_ml` | 없음 | 없음 | 선택 | 대용량 SC 시 ISR 위험 |
| `injection_site_specific` | 없음 | 없음 | 선택 | AE 항목 확장 |
| `drug_interaction_rules` | 병용시 선택 | - | - | 경구+IV 병용 전용 |
| `biomarker` (in disease_baseline) | - | - | - | biomarker_targeted.json 합성 시 |
| `phases` (in trial_design) | - | - | - | maintenance_therapy.json 합성 시 |

## 변경이 필요 없는 필드 (약물 불문)

- `lab_reference_ranges`: 사람의 정상 검사 수치이므로 약물 무관
- `ecog_model`: 일반적인 ECOG 계산 가중치이므로 약물 무관
- `demographics`, `comorbidities`: **필드 구조**는 동일, **값**만 적응증에 맞게 변경
- `ae_profile`, `dose_modification_rules`, `supportive_care_rules`: **필드 구조**는 동일, **값**은 약물별 변경
- `ae_cascade_rules`: **필드 구조**는 동일, 약물별 AE 연쇄 규칙만 변경

## 검증 예시 (Python)

```python
import json
from jsonschema import validate, RefResolver

with open("data/schema/base.json") as f:
    base_schema = json.load(f)

with open("data/schema/iv_combination.json") as f:
    iv_combo_schema = json.load(f)

with open("data/rule_set.json") as f:
    rule_set = json.load(f)

resolver = RefResolver.from_schema(base_schema, store={
    "base.json": base_schema
})

validate(instance=rule_set, schema=iv_combo_schema, resolver=resolver)
print("Valid!")
```
