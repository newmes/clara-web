# 05. Observation Model — GT/HR 분리 + AE 감지

> **파일:** `sim/engine/observation.py` (829 lines), `sim/engine/mood.py` (556 lines)
> **역할:** Ground Truth를 Hospital Record로 필터링, 환자 심리가 보고 정확도에 미치는 영향 모델링
> **의학 근거:** Basch et al. (2006) — 환자-의사 AE 등급 일치율 ~50%; Di Maio et al. (2015) — 의사가 57% AE를 과소평가
>
> 🔗 **웹에서 확인:** [Trial Viewer (GT/HR 토글)](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) — Dashboard에서 GT ↔ HR 전환으로 정보 비대칭을 시각적으로 확인

---

## 1. 설계 철학: 정보 비대칭

```
Ground Truth (GT):    실제로 환자에게 일어나고 있는 모든 것
                      ↓ [필터링]
Hospital Record (HR): 병원이 알고 있는 것만

차이 = 정보 비대칭 = Care AI가 메울 수 있는 가치
```

### GT와 HR의 차이가 발생하는 이유

| 이유 | 예시 | 영향 |
|------|------|------|
| **방문 간격** | 3주 사이클 → 20일간 관찰 없음 | AE가 발생해도 다음 방문까지 모름 |
| **Lab 요구** | 호중구감소증은 혈액검사 필요 | 방문 없으면 절대 감지 불가 |
| **환자 심리** | stoic_minimizer → 증상 축소 | 보고 시 grade 과소평가 |
| **도구 한계** | 청진으로 폐 문제 청취, 시진으로 발진 확인 | 비대면에서 물리적 검사 불가 |
| **스캔 일정** | RECIST 매 6-9주 | 종양 변화를 그 사이 모름 |

---

## 2. 5개 관찰점 (Observation Points)

### 정의 및 발생 조건

| 관찰점 | 언제 발생 | 무엇을 캡처 |
|--------|----------|------------|
| `scheduled_visit` | 투약일 (cycle day에 해당) | **전체**: labs, vitals, 진찰, AE 평가, ECOG, 체중, 환자 면담 |
| `scheduled_scan` | RECIST 스케줄 (6-9주마다) | 종양 데이터만 |
| `self_report` | 비방문일, 확률적 (mood+grade 의존) | 환자가 보고한 AE만 |
| `video_call` | Care AI 모드에서 매일 | 환자 보고 AE + 영상 감지 AE + 환자 보고 vitals |
| `er_visit` | 응급 기준 충족 시 (자동) | **전체**: labs, vitals, 진찰, AE 평가, ECOG |

### OBSERVATION_CAPTURE 매트릭스

```
                     labs  vitals  physical  ae_assess  ecog  weight  interview  tumor  ae_patient  ae_video  vitals_patient
scheduled_visit       ✓      ✓       ✓         ✓        ✓      ✓        ✓       —        —          —          —
scheduled_scan        —      —       —         —        —      —        —       ✓        —          —          —
self_report           —      —       —         —        —      —        —       —        ✓          —          —
video_call            —      —       —         —        —      —        —       —        ✓          ✓          ✓
er_visit              ✓      ✓       ✓         ✓        ✓      —        —       —        —          —          —
```

**핵심 차이:**
- `scheduled_visit`/`er_visit` → "full exam" → 모든 AE 감지 (clinical_assessment)
- `self_report` → 환자가 인지한 AE만, mood에 의해 왜곡 가능
- `video_call` → self_report + 영상 시각 감지 (rash, 탈모, 피로 등)

---

## 3. AE 감지 채널 (40개 AE 등록)

### 3.1 채널 유형

| 채널 | 의미 | Care AI 관련 |
|------|------|-------------|
| `lab` | 혈액검사/소변검사 필요 | 방문 시에만 |
| `patient_reported` | 환자가 증상 인지 + 보고 | mood 의존 |
| `video_detectable` | 카메라/마이크로 시각/청각 감지 | ★ Care AI 핵심 가치 |
| `physical_exam` | 신체 검진 (촉진, 청진 등) | 방문 시에만 |

### 3.2 주요 AE별 채널 매핑

**Lab 전용 (방문 없으면 불가):**

| AE | 채널 | 환자 인지 시작 | 의미 |
|-----|------|-------------|------|
| `neutropenia` | [lab] | Grade 3 | G2까지 완전 무증상 — "silent killer" |
| `leukopenia` | [lab] | Grade 3 | 마찬가지 |
| `proteinuria` | [lab] | Grade 3 | 소변검사 필요 |
| `alt_increased` | [lab] | Grade 3 | 간수치만으로 확인 |

**영상 감지 가능 (Care AI 가치):**

| AE | 채널 | Video Signs | Grade 1부터 감지? |
|-----|------|------------|-----------------|
| `rash` | [patient_reported, **video_detectable**, physical_exam] | visible_rash, erythema, papules | Yes |
| `alopecia` | [**video_detectable**, patient_reported] | visible_hair_loss, thinning_hair | Yes |
| `fatigue` | [patient_reported, **video_detectable**] | visible_fatigue, slow_movements | Yes |
| `thrombocytopenia` | [lab, **video_detectable**] | bruising, petechiae | Grade 2 |
| `dyspnea` | [patient_reported, **video_detectable**] | labored_breathing, tachypnea | Yes |
| `cough` | [patient_reported, **video_detectable**] | audible_cough | Yes |

**환자 보고 전용:**

| AE | 채널 | 인지 시작 | 비고 |
|-----|------|---------|------|
| `nausea` | [patient_reported] | Grade 1 | 영상으로 안 보임 |
| `diarrhea` | [patient_reported] | Grade 1 | 환자만 알 수 있음 |
| `pruritus` | [patient_reported] | Grade 1 | 가려움은 보이지 않음 |
| `arthralgia` | [patient_reported] | Grade 1 | 통증은 보이지 않음 |

### 3.3 미등록 AE Fallback

```python
def get_ae_channels(ae_term):
    # 1. 정확 매칭
    # 2. 부분 문자열 매칭 (양방향)
    # 3. Fallback → WARNING 로그 + {"channels": ["patient_reported"], "patient_aware_threshold": 1}
```

경고 로그가 출력되지만 시뮬레이션은 계속. "patient_reported, threshold 1"은 가장 보수적인 기본값.

---

## 4. AE 감지 확률 모델

### 4.1 Priority Cascade (process_day에서)

각 활성 AE에 대해 순차적으로:

```
Priority A: Full exam (scheduled_visit or er_visit)
  → 100% 감지, clinical_assessment 채널, 정확한 grade

Priority B: Lab-required AE + no lab access
  → 0% 감지 (환자 인지 threshold 미만이면)

Priority C: Patient-reported
  → self_report 이벤트 존재 시: 감지
  → video_call 시: engagement × (1 - under_report_prob) 확률

Priority D: Video-detectable (Care AI only)
  → base_detect = min(0.3 + grade × 0.15, 0.90)
  → P(detect) = base_detect × video_cooperation
```

### 4.2 Video Detection 수학

$$P(\text{video detect}) = \min(0.3 + \text{grade} \times 0.15, \, 0.90) \times \text{video\_cooperation}$$

| Grade | Base Detect | × coop=0.5 | × coop=0.8 | × coop=1.0 |
|-------|-----------|------------|------------|------------|
| 1 | 0.45 | 0.225 | 0.360 | 0.450 |
| 2 | 0.60 | 0.300 | 0.480 | 0.600 |
| 3 | 0.75 | 0.375 | 0.600 | 0.750 |
| 4 | 0.90 | 0.450 | 0.720 | 0.900 |

**video_cooperation** (mood 기반):
$$\text{energy} \times 0.30 + (1-\text{defensiveness}) \times 0.35 + \text{trust\_in\_ai} \times 0.20 + (1-\text{irritability}) \times 0.15$$

shame_avoidant 환자: 높은 defensiveness → 낮은 coop → 카메라에 rash 안 보여줌.

---

## 5. Grade Distortion 모델

### 5.1 감지 채널별 정확도

| 채널 | Grade 보고 | 이유 |
|------|----------|------|
| `clinical_assessment` | 정확 | 의사 직접 평가 |
| `patient_reported` | ±1~2 distortion | mood 기반 과소/과대 보고 |
| `patient_reported_via_video` | ±1~2 distortion | 마찬가지 |
| `video_detected` | ±1~2 distortion | AI 추론 (시각적 징후 기반) |

### 5.2 Distortion 계산 (mood.compute_grade_distortion)

$$\text{over\_report} = 0.6 \times \text{anxiety} + 0.2 \times (1 - \text{cognitive\_clarity})$$
$$\text{under\_report} = 0.5 \times \text{defensiveness} + 0.2 \times \text{depression} + 0.1 \times (1 - \text{energy})$$
$$\text{net} = \text{over} - \text{under}$$

| net 범위 | Distortion | 의미 |
|---------|-----------|------|
| > +0.35 | +1 | 1등급 과대 보고 |
| -0.30 ~ +0.35 | 0 | 정확 보고 |
| < -0.30 | -1 | 1등급 과소 보고 |

### 5.3 Persona별 예상 Distortion

| Persona | over | under | net | 예상 |
|---------|------|-------|-----|------|
| `stoic_minimizer` | 0.15 | 0.415 | -0.265 | 0 (거의 -1) |
| `anxious_reporter` | 0.44 | 0.17 | +0.27 | 0 (거의 +1) |
| `catastrophizer` | 0.53 | 0.165 | **+0.365** | **+1** |
| `health_literate` | 0.27 | 0.145 | +0.125 | 0 (정확) |
| `confused_elderly` | 0.36 | 0.285 | +0.075 | 0 |

---

## 6. 해소된 AE 처리

```python
# GT에서 AE가 사라졌지만, 마지막 방문 이후라면:
for ae_term in known_aes:
    if ae_term not in active_gt_aes:
        if full_exam:
            known_aes[ae_term]["status"] = "resolved"
        # else: 병원은 여전히 활성으로 알고 있음!
```

**실질적 영향:**
- 방문 없이 AE가 해소되면, HR에는 여전히 "active"로 표시
- 다음 방문에서 불필요한 dose hold/reduction 결정 가능
- Care AI가 있으면 영상통화에서 환자가 "좋아졌다"고 보고 → 조기 반영

---

## 7. Hospital Record 구조

```json
{
  "patient_id": "PT-001",
  "day": 42,
  "observation_types": ["video_call"],
  "objective": {
    "location": "HOME",
    "treatment_status": "on_treatment",
    "labs": {"ANC": {"value": 4.2, ...}},       // ← 21일 전 값 (stale!)
    "vitals": {"BT": 37.2},                      // ← 환자 보고 체온만
    "active_aes": [
      {
        "ae": "rash",
        "grade": 2,                              // ← distortion 적용된 값
        "onset_day": 35,
        "detected_day": 42,
        "detection_delay": 7,
        "channel": "video_detected",
        "status": "active_stable"
      }
    ],
    "ecog": 1,                                   // ← 21일 전 값 (stale!)
    "tumor": null,                               // ← 스캔 전이면 null
    "labs_stale_days": 21,
    "vitals_stale_days": 21
  }
}
```

**staleness indicator** (`labs_stale_days`, `vitals_stale_days`): HR이 얼마나 오래된 데이터를 기반으로 하는지 표시.

---

## 8. 환자 심리 모델 (MoodState)

### 8.1 7차원 벡터

| 차원 | 방향성 | 주요 영향 |
|------|--------|----------|
| `anxiety` | ↑ = 과대보고 | self_report 확률 ↑, grade +1 distortion |
| `depression` | ↑ = 무반응 | self_report 확률 ↓, engagement ↓ |
| `irritability` | ↑ = 조기종료 | 통화 종료 확률 ↑ |
| `energy` | ↓ = 참여도 감소 | engagement ↓, 영상 협조 ↓ |
| `cognitive_clarity` | ↓ = 부정확 | report_accuracy ↓, distortion ↑ |
| `trust_in_ai` | ↓ = 비협조 | compliance ↓, 영상 협조 ↓ |
| `defensiveness` | ↑ = 과소보고 | under_report ↑, grade -1 distortion |

### 8.2 Mood Update — Ornstein-Uhlenbeck

매일의 mood 업데이트 (4단계):

**Phase 1: 이벤트 기반 delta 축적**
```python
for event in daily_events:
    if event.startswith("_ongoing:"):
        delta *= ONGOING_EVENT_DAMPING (0.15)  # 지속 이벤트는 15% 효과
    delta += EVENT_MOOD_DELTAS[event]
```

**Phase 2: 적응적 감쇠 (mean-reversion)**
$$\text{diff} = \text{baseline}[d] - \text{state}[d]$$
$$\text{decay\_rate} = 0.05 + |\text{diff}| \times 0.15$$
$$\delta[d] \mathrel{+}= \text{diff} \times \text{decay\_rate}$$

멀리 벗어날수록 빠르게 baseline으로 복귀 (최대 20%/일).

**Phase 3: Gaussian 노이즈**
$$\delta[d] \mathrel{+}= \mathcal{N}(0, 0.02)$$

**Phase 4: Inertia smoothing**
$$\text{state}[d] = \text{clamp}\Big(\text{state}[d] \times 0.75 + (\text{state}[d] + \delta[d]) \times 0.25, 0, 1\Big)$$

실질적으로 delta의 25%만 반영 → 안정적 변화.

### 8.3 Turn-Level Update (대화 중)

대화 중에는 inertia가 0.60 (vs 일별 0.75) → 40% 반영. 대화 중 감정 변화가 더 빠름.

### 8.4 Defensiveness Override

중증 AE가 심리 방어를 무너뜨림:

| 최고 Grade | 효과 | 의학적 근거 |
|-----------|------|-----------|
| Grade 4+ | defensiveness -= 0.30 (최소 0.05) | 생명위협 → 방어 불가 |
| Grade 3 | defensiveness -= 0.15 (최소 0.10) | 심각한 증상 → 인정 |
| Grade 1-2 | 변화 없음 | 경미 → 방어 유지 |

### 8.5 Self-Report 확률

$$P = \text{clamp}\Big((\text{grade\_factor} + \max(\text{personality\_factor}, 0)) \times \text{time\_factor}, 0, 0.95\Big)$$

| max_grade | grade_factor |
|-----------|-------------|
| 0 | 0.00 |
| 1 | 0.02 |
| 2 | 0.08 |
| 3 | 0.40 |
| 4 | 0.85 |

$$\text{personality\_factor} = 0.4 \times \text{anxiety} - 0.25 \times \text{defensiveness} - 0.15 \times \text{depression} + 0.1 \times (1 - \text{irritability})$$

$$\text{time\_factor} = \begin{cases} 1.0 & \text{days\_since\_visit} \leq 7 \\ \min(1.0 + 0.03 \times (\text{days} - 7), 1.5) & \text{otherwise} \end{cases}$$

### 8.6 ER 방문 기준 (mood 무관, 자동)

| 조건 | 기준 | 근거 |
|------|------|------|
| AE Grade 4+ | CTCAE Grade 4 = 생명위협 | 즉시 응급실 |
| 발열성 호중구감소증 | 체온 ≥ 38.3°C AND ANC < 1000 | ASCO/IDSA 가이드라인 |
| 중증 저산소 | SpO2 < 90% | 호흡 응급 |

---

## 9. Detection Log — 성과 측정

```python
detection_log.append({
    "ae_term": "peripheral_neuropathy",
    "day": 42,                    # 감지일
    "actual_onset_day": 35,       # GT의 실제 발생일
    "detection_delay": 7,         # 지연일
    "channel": "video_detected",
    "actual_grade": 2,            # GT grade
    "reported_grade": 2,          # HR grade (distortion 후)
})
```

### 성과 지표

```python
def compute_detection_delay_summary(detection_log):
    return {
        "count": 12,              # 총 감지 AE 수
        "mean_delay_days": 3.2,   # 평균 감지 지연 (Natural: ~10일, Care AI: ~2일)
        "max_delay_days": 14,     # 최악 지연
        "zero_delay_count": 5,    # 당일 감지 건수
        "by_channel": {
            "clinical_assessment": 4,
            "video_detected": 3,
            "patient_reported_via_video": 5
        }
    }
```

이것이 A/B 비교의 핵심 지표. Care AI의 가치 = `mean_delay_days` 감소 + `zero_delay_count` 증가.