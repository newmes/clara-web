# Multimodal 코드 → 시뮬레이션 엔진 연동 변경사항

> **작성일**: 2026-02-18  
> **대상**: hyena  
> **목적**: `sim/multimodal/` 코드를 기존 시뮬레이션 엔진의 데이터 구조에 맞춰 수정한 내역 정리

---

## 요약

기존 multimodal 코드의 입출력 인터페이스가 시뮬레이션 엔진의 데이터 구조와 달라서,
**시뮬레이션 쪽은 건드리지 않고** multimodal 코드만 수정하여 직접 연동 가능하도록 변경했습니다.

핵심 변경:
1. **AE term 네이밍** — `maculopapular_rash` → `rash_maculopapular` (시뮬레이션 convention)
2. **Patient profile** — 시뮬레이션 JSON을 직접 받아 자동 변환
3. **Care record → 음성 텍스트** — 시뮬레이션 care_record에서 환자 대사 자동 추출
4. **AE → cough_config** — AE 목록으로부터 기침 설정 자동 유도
5. **새로운 시각적 AE 추가** — `stomatitis`, `pruritus`, `alopecia` CTCAE 기준 추가

---

## 파일별 변경 상세

### 1. `config.py`

#### CTCAE_CRITERIA 키 이름 변경
```
Before:  "maculopapular_rash", "acneiform_rash", "periorbital_edema", "sjs_prodrome"
After:   "rash_maculopapular", "rash_acneiform", "periorbital_edema", "sjs_prodrome",
         "stomatitis", "pruritus", "alopecia"  ← 3종 추가
```

- 시뮬레이션 엔진의 `rule_set.json`에 정의된 `ae_term` 형식 (`noun_first_snake_case`) 사용
- `stomatitis`: 입술/구강 변화 (Grade 1–3) — 시뮬레이션에서 32% 발생
- `pruritus`: 긁은 자국/피부 자극 (Grade 1–3) — 시뮬레이션에서 41% 발생
- `alopecia`: 탈모 (Grade 1–2만, CTCAE상 Grade 3 없음) — 시뮬레이션에서 35% 발생

#### SIGLIP_CLASSES 확장
```
Before: 13 classes (normal + 4 AE × 3 grades)
After:  21 classes (normal + 6 AE × 3 grades + alopecia × 2 grades)
```
`siglip_num_classes` 도 `13` → `21`로 변경.

> **⚠️ 주의**: 기존에 학습된 classifier head가 있으면 재학습 필요!

#### VOICE_MAP 확장
시뮬레이션은 sex를 `"M"` / `"F"`로 저장하므로, lowercase `"m"` / `"f"` 키 추가:
```python
("elderly", "m"): "Orus",
("elderly", "f"): "Leda",
```

#### 새 상수 추가
| 상수 | 설명 |
|------|------|
| `FACE_RENDERABLE_AES` | 얼굴 이미지로 렌더링 가능한 AE set (CTCAE_CRITERIA 키 전체) |
| `AE_COUGH_MAP` | AE term → cough config 매핑 (현재 `pneumonitis`만) |
| `RESPIRATORY_AES` | TTS 목소리 스타일에 영향주는 호흡기계 AE set |

#### cough_clips_dir 변경
```
Before: Path("/data2/workspace/AlphaRaven/old/MedGemma/...")  ← 외부 절대경로
After:  Path("<project_root>/data/cough_clips")                ← 프로젝트 내부 상대경로
```

> **TODO**: 기존 cough clip 파일들을 `data/cough_clips/dry/`, `data/cough_clips/wet/` 으로 복사 필요.

---

### 2. `schemas.py`

#### 새 dataclass: `SimPatientProfile`
시뮬레이션의 `patients/PT-XXX.json`에서 바로 생성 가능:
```python
from multimodal import SimPatientProfile

profile = SimPatientProfile.from_sim_patient(patient_json)  # raw dict → SimPatientProfile
profile.to_face_profile()   # → {"age": 73, "sex": "male", "race": "white"}
profile.to_voice_profile()  # → {"age": 73, "sex": "m"}
```

자동 처리하는 변환:
- `emr.demographics.age` → `age`
- `DM.SEX` (= `"M"`) → `sex` (= `"M"`)
- `emr.demographics.race` (= `"White"`) → `race`
- `to_face_profile()`에서 `"M"` → `"male"` 변환 수행

#### 새 dataclass: `SimAE`
시뮬레이션 AE 레코드 2종 모두 지원:
```python
from multimodal import SimAE

# CDASH 형식 (day JSONL의 AE[] 배열)
ae = SimAE.from_cdash({"AETERM": "rash_maculopapular", "_grade": 2, "_status": "active", ...})

# Hospital record 형식 (hospital_record.active_aes[])
ae = SimAE.from_hr_active({"ae": "rash_maculopapular", "grade": 2, "onset_day": 26, ...})

ae.to_face_ae()  # → {"ae": "rash_maculopapular", "grade": 2}
```

#### DetectedAE 변경 없음
`ae_term` 필드는 그대로이나, docstring에 시뮬레이션 convention 사용 명시.

---

### 3. `face_generator.py`

#### `generate_patient_face()` 시그니처 변경
```python
# Before
def generate_patient_face(
    patient_profile: dict[str, Any],       # {"age": 73, "sex": "male", "race": "white"}
    active_aes: list[dict[str, Any]],      # [{"ae": "maculopapular_rash", "grade": 2}]
    ...
)

# After — 3가지 입력 형식 모두 수용
def generate_patient_face(
    patient_profile,                        # SimPatientProfile | raw patient JSON | flat dict
    active_aes: list | None = None,         # [SimAE] | [CDASH dict] | [{"ae": ..., "grade": ...}]
    ...
)
```

#### 내부 추가 함수
| 함수 | 역할 |
|------|------|
| `_normalize_profile()` | SimPatientProfile / raw JSON / flat dict → face-prompt용 dict |
| `_normalize_aes()` | SimAE / CDASH / flat dict → face-renderable AE만 필터링 |

**자동 필터링**: `fatigue`, `nausea` 등 시각적으로 얼굴에 표현 불가한 AE는 자동 제외.
`FACE_RENDERABLE_AES`에 포함된 AE만 이미지 편집에 사용됨.

#### 사용 예시 (시뮬레이션 데이터 직접 사용)
```python
import json
from multimodal import generate_patient_face, SimPatientProfile

# 환자 프로필 로드
with open("data/runs/.../patients/PT-050.json") as f:
    patient_json = json.load(f)

# Day record 로드
with open("data/runs/.../simulations/PT-050_natural.jsonl") as f:
    for line in f:
        day_record = json.loads(line)
        if day_record["day"] == 30:
            break

# 방법 1: raw dict 직접 전달
result = generate_patient_face(
    patient_profile=patient_json,           # raw patients/*.json 그대로
    active_aes=day_record["AE"],            # CDASH AE[] 배열 그대로
    day=30,
)

# 방법 2: adapter 사용
profile = SimPatientProfile.from_sim_patient(patient_json)
result = generate_patient_face(profile, day_record["AE"], day=30)
```

---

### 4. `voice_generator.py`

#### 새 함수: `extract_patient_speech()`
시뮬레이션 care_record에서 환자 대사를 자동 추출:
```python
from multimodal import extract_patient_speech

text = extract_patient_speech(day_record["care_record"])
# → "Oh, hello there. It's me again. I feel mostly okay..."
```

추출 소스:
- `turns[].role == "patient"` 인 turn에서
- `content.greeting` + `content.general_wellbeing` + `content.responses[].answer`

#### 새 함수: `derive_cough_config()`
AE 목록에서 기침 설정을 자동 유도:
```python
from multimodal import derive_cough_config

cough = derive_cough_config(day_record["AE"])
# pneumonitis Grade 2 → {"type": "dry", "frequency": "frequent"}
# AE 없음 → {"type": "dry", "frequency": "none"}
```

매핑 규칙:
- `pneumonitis` → `type: "dry"`, frequency는 grade별 (`g1: occasional`, `g2: frequent`, `g3: severe`)
- `fatigue` / `anemia` Grade ≥ 2 → `frequency: "occasional"` (약한 기침/피로한 목소리)
- 기타 AE → 기침 없음

#### `generate_patient_voice()` 시그니처 변경
```python
# Before
def generate_patient_voice(
    text: str,
    patient_profile: dict[str, Any],       # {"age": 73, "sex": "male"}
    cough_config: dict[str, Any] | None,
)

# After
def generate_patient_voice(
    text: str,
    patient_profile,                        # SimPatientProfile | raw JSON | flat dict
    cough_config: dict[str, Any] | None,
    *,
    active_aes: list | None = None,         # ← 새 파라미터
)
```

`cough_config=None` + `active_aes` 전달 시 자동으로 `derive_cough_config()` 호출.

#### 전체 사용 예시
```python
from multimodal import generate_patient_voice, extract_patient_speech

text = extract_patient_speech(day_record["care_record"])
result = generate_patient_voice(
    text=text,
    patient_profile=patient_json,       # raw JSON 그대로
    cough_config=None,                  # auto-derive from active_aes
    active_aes=day_record["AE"],        # CDASH AE[] 그대로
)
```

---

### 5. `face_analyzer.py`

코드 변경 없음. 출력의 `DetectedAE.ae_term`이 이제 시뮬레이션과 같은 naming convention 사용
(`rash_maculopapular`, `rash_acneiform` 등).

> **⚠️ 주의**: `SIGLIP_CLASSES`가 21개로 확장되었으므로, 기존 13-class classifier head는
> 새 class set으로 재학습 필요.

---

### 6. `voice_analyzer.py`

docstring만 업데이트. 코드/시그니처 변경 없음.

---

### 7. `__init__.py`

새로 export된 심볼:
```python
# Simulation adapters
SimPatientProfile, SimAE

# Helper functions
extract_patient_speech, derive_cough_config

# Config constants
FACE_RENDERABLE_AES, AE_COUGH_MAP, RESPIRATORY_AES
```

---

## 데이터 구조 매핑 요약

### 시뮬레이션 → Face Generation

```
patients/PT-XXX.json                    → patient_profile (auto-converted)
  └ emr.demographics.{age, sex, race}

simulations/PT-XXX_*.jsonl (Day N)      → active_aes (auto-filtered)
  └ AE[].{AETERM, _grade}              → face-renderable AE만 통과
  └ hospital_record.active_aes[]         (fatigue, nausea 등 자동 제외)
```

### 시뮬레이션 → Voice Generation

```
patients/PT-XXX.json                    → patient_profile (age, sex → voice 선택)

simulations/PT-XXX_care_ai.jsonl (Day N)
  └ care_record[].turns[role=patient]   → text (via extract_patient_speech)
  └ AE[].{AETERM, _grade}              → cough_config (via derive_cough_config)
```

### Analysis → 시뮬레이션

```
FaceAnalysisResult.detected_aes[]
  └ ae_term: "rash_maculopapular"       ← 시뮬레이션과 동일한 term
  └ grade: 2                            ← 시뮬레이션과 동일한 scale

VoiceAnalysisResult
  └ cough_events[].{timestamp, type, severity}
  └ respiratory_assessment.{has_cough, overall_severity}
  └ transcript                          ← 환자 발화 텍스트
```

---

## TODO (후속 작업)

| 항목 | 설명 | 우선순위 |
|------|------|---------|
| cough clips 복사 | 기존 경로의 wav 파일을 `data/cough_clips/{dry,wet}/` 으로 이동 | 높음 |
| SigLIP head 재학습 | 21 classes로 변경되었으므로 `train_classifier()` 재실행 필요 | 높음 |
| Django API endpoint | `/api/multimodal/face/generate`, `/api/multimodal/voice/generate` 등 | 중간 |
| 비디오콜 UI 프론트엔드 | 얼굴 이미지 + 음성 재생 + 대화 UI 컴포넌트 | 중간 |
| MedGemma 27B 로컬 테스트 | GPU 6,7에서 정상 로딩/추론 확인 | 높음 |
| HeAR 모델 로컬 테스트 | GPU 5에서 TF SavedModel 로딩 확인 | 중간 |

---

## 시뮬레이션 AE term 전체 목록 (참조)

`rule_set.json`에 정의된 14종:

| ae_term | 발생률 | 얼굴 렌더링 |
|---------|--------|------------|
| `peripheral_neuropathy` | 67% | ❌ |
| `rash_maculopapular` | 50% | ✅ |
| `fatigue` | 51% | ❌ |
| `pruritus` | 41% | ✅ (긁은 자국) |
| `diarrhea` | 38% | ❌ |
| `alopecia` | 35% | ✅ (탈모) |
| `decreased_appetite` | 33% | ❌ |
| `stomatitis` | 32% | ✅ (입술) |
| `nausea` | 25% | ❌ |
| `anemia` | 18% | ❌ |
| `hyperglycemia` | 7% | ❌ |
| `colitis` | 5% | ❌ |
| `pneumonitis` | 4.5% | ❌ (기침으로 표현) |
| `infusion_related_reaction` | 3% | ❌ |

얼굴에 직접 렌더링 가능한 AE: **7종 중 4종** (rash, pruritus, stomatitis, alopecia)  
+ 추가 2종 (rash_acneiform, periorbital_edema, sjs_prodrome) — 향후 약물에서 발생 가능
