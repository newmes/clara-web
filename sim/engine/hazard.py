# Source Generated with Decompyle++
# File: hazard.cpython-310.pyc (Python 3.10)

'''Hazard — rule_set 분포에서 일별 이벤트 확률을 계산하는 순수 수학 모듈

LLM 호출 없음. rule_set의 onset_day, duration_days 분포를 입력으로 받아
매일의 조건부 확률(hazard)을 계산한다.

핵심 아이디어:
  - Fate table 없이, 매일 "오늘 이 AE가 시작될 확률"을 rule_set에서 직접 계산
  - 혼합 모델(mixture model): 환자가 이 AE를 \'겪을 확률(incidence)\' ×
    \'겪는다면 오늘 시작될 확률(hazard)\'
  - P(onset on day t | no onset before t) = I·f(t) / (1 − I·F(t−1))
    여기서 I = incidence, F = onset CDF, f = onset PMF

지원 분포: normal, lognormal, uniform (rule_set.ae_profile.onset_day에서 사용)

하드코딩 상수는 config/defaults.py에 집중 관리.
'''
import math
from typing import Any
from sim.config.defaults import GRADE_TRANSITION_BASE_WORSEN, GRADE_TRANSITION_BASE_IMPROVE, GRADE_TIME_STABILIZE_DAY, GRADE_HIGH_WORSEN_DAMPING, GRADE_4_TO_5_DAMPING, GRADE_HIGH_IMPROVE_BOOST, GRADE_4_IMPROVE_BOOST, GRADE_CUMULATIVE_MAX_TIME_FACTOR, MAX_GRADE_TRANSITION_PROB, TUMOR_RATE, ECOG_MORTALITY_MAP, TREATMENT_DISCONTINUED_MORTALITY_MULT, MAX_DAILY_MORTALITY, TREATMENT_DISCONTINUED_ECOG_PENALTY, ECOG_AE_PENALTY_CAP, ECOG_MAX_DAILY_CHANGE, MIN_AE_BURDEN_WEIGHT, ECOG_TREATMENT_FATIGUE_PER_CYCLE, MAX_AE_CASCADE_HAZARD, MAX_DISCONTINUATION_PATIENT, MAX_DISCONTINUATION_PHYSICIAN, DISCONTINUATION_BACKGROUND_DAILY_RATE, INTERVENTION_EFFECTS

def _normal_cdf(x, mean, std):
    '''정규분포 CDF. math.erf 기반.'''
    if std <= 0:
        if x >= mean:
            return 1
        return 0
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


def _normal_pdf(x, mean, std):
    '''정규분포 PDF.'''
    if std <= 0:
        if x == mean:
            return float('inf')
        return 0
    return math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


def _lognormal_cdf(x, mu, sigma):
    '''로그정규분포 CDF. X ~ LogNormal(mu, sigma) → ln(X) ~ Normal(mu, sigma).'''
    if x <= 0:
        return 0
    return _normal_cdf(math.log(x), mu, sigma)


def _uniform_cdf(x, lo, hi):
    '''균등분포 CDF.'''
    if hi <= lo:
        if x >= lo:
            return 1
        return 0
    if x <= lo:
        return 0
    if x >= hi:
        return 1
    return (x - lo) / (hi - lo)


def _truncated_cdf(raw_cdf_fn, x, lo, hi, **dist_params):
    '''임의 분포의 절단(truncated) CDF.

    P(X ≤ x | lo ≤ X ≤ hi) = (F(x) − F(lo)) / (F(hi) − F(lo))
    '''
    f_lo = raw_cdf_fn(lo, **dist_params)
    f_hi = raw_cdf_fn(hi, **dist_params)
    denom = f_hi - f_lo
    if denom <= 1e-12:
        if x >= (lo + hi) / 2:
            return 1.0
        return 0.0
    f_x = raw_cdf_fn(min(max(x, lo), hi), **dist_params)
    return max(0.0, min(1.0, (f_x - f_lo) / denom))


def _distribution_cdf(x, distribution='normal', params=None):
    '''rule_set의 onset_day/duration_days 스펙에서 CDF를 계산한다.

    Args:
        x: 평가 지점 (day)
        distribution: "normal" | "lognormal" | "uniform"
        params: {"mean", "std", "min", "max"} 등

    Returns:
        CDF 값 [0, 1]
    '''
    lo = params.get('min', 0)
    hi = params.get('max', 365)
    if distribution == 'normal':
        mean = params.get('mean', 30)
        std = params.get('std', 10)
        return _truncated_cdf(_normal_cdf, x, lo, hi, mean=mean, std=std)
    if distribution == 'lognormal':
        if 'mu' in params:
            mu = params['mu']
            sigma = params.get('sigma', 0.5)
        elif 'mean' in params:
            mean = max(params['mean'], 1)
            std = max(params.get('std', mean * 0.5), 0.1)
            sigma_sq = math.log(1 + (std / mean) ** 2)
            mu = math.log(mean) - sigma_sq / 2
            sigma = math.sqrt(sigma_sq)
        else:
            mu = math.log(30)
            sigma = 0.5
        return _truncated_cdf(_lognormal_cdf, x, lo, hi, mu=mu, sigma=sigma)
    if distribution == 'uniform':
        return _uniform_cdf(x, lo, hi)
    return _uniform_cdf(x, lo, hi)


def daily_onset_hazard(day, incidence, onset_spec, is_drug_held=False, is_dose_reduced=False):
    '''이 AE가 오늘 시작될 확률 (아직 발생하지 않았다는 조건 하).

    혼합 모델(mixture model):
      - 환자가 이 AE를 겪을 확률 = incidence
      - 겪는다면 onset 시점은 onset_spec 분포를 따름
      - P(onset on day t | no onset before t) = I·f(t) / (1 − I·F(t−1))

    Args:
        day: 현재 시뮬레이션 날짜 (1-based)
        incidence: 이 환자의 조정된 AE 발생률 (0-1)
        onset_spec: rule_set의 onset_day 스펙
            {"distribution": "normal", "params": {"mean": 63, "std": 21, "min": 7, "max": 180}}
        is_drug_held: 원인 약물이 현재 hold 상태인가
        is_dose_reduced: 원인 약물이 감량 상태인가

    Returns:
        일별 onset 확률 [0, 1]. 이 값으로 Sampler.boolean()을 호출한다.
    '''
    if incidence <= 0 or day < 1:
        return 0
    if not onset_spec:
        onset_spec = {"distribution": "normal", "params": {"mean": 42, "std": 14, "min": 1, "max": 180}}
    dist = onset_spec.get('distribution', 'normal')
    params = onset_spec.get('params', onset_spec)
    F_t = _distribution_cdf(day, dist, params)
    F_prev = _distribution_cdf(day - 1, dist, params)
    f_t = max(0, F_t - F_prev)
    p_no_onset_yet = 1 - incidence * F_prev
    if p_no_onset_yet <= 1e-12:
        return 0
    hazard = incidence * f_t / p_no_onset_yet
    if is_drug_held:
        hazard *= INTERVENTION_EFFECTS['dose_hold']['onset_hazard_mult']
    elif is_dose_reduced:
        hazard *= INTERVENTION_EFFECTS['dose_reduction']['onset_hazard_mult']
    return min(max(hazard, 0), 1)


def daily_resolution_hazard(days_active, duration_spec=None, is_drug_held=False, is_dose_reduced=False, has_active_conmed=False, conmed_tier=3):
    '''활성 AE가 오늘 해소될 확률.

    Args:
        days_active: AE 발생 이후 경과일
        duration_spec: rule_set의 duration_days 스펙
            {"distribution": "normal", "params": {"mean": 30, "std": 10, "min": 7}}
            None이면 비가역적(irreversible) → 0 반환
        is_drug_held: 원인 약물이 hold 상태인가
        is_dose_reduced: 원인 약물이 감량 상태인가
        has_active_conmed: 이 AE에 대한 활성 보조약이 있는가
        conmed_tier: 보조약 효과 Tier

    Returns:
        일별 해소 확률 [0, 1]
    '''
    if duration_spec is None:
        return 0
    if not isinstance(duration_spec, dict) or 'distribution' not in duration_spec:
        if isinstance(duration_spec, (int, float)):
            if days_active >= duration_spec:
                return 1
            return 0
        return 0
    dist = duration_spec.get('distribution', 'normal')
    params = duration_spec.get('params', duration_spec)
    F_d = _distribution_cdf(days_active, dist, params)
    F_prev = _distribution_cdf(days_active - 1, dist, params)
    f_d = max(0, F_d - F_prev)
    p_still_active = 1 - F_prev
    if p_still_active <= 1e-12:
        return 0.8
    hazard = f_d / p_still_active
    resolution_mult = 1
    if is_drug_held:
        resolution_mult *= INTERVENTION_EFFECTS['dose_hold']['resolution_mult']
    elif is_dose_reduced:
        resolution_mult *= INTERVENTION_EFFECTS['dose_reduction']['resolution_mult']
    if has_active_conmed:
        tier_key = f'''conmed_tier{conmed_tier}'''
        fx = INTERVENTION_EFFECTS.get(tier_key, INTERVENTION_EFFECTS['conmed_tier3'])
        resolution_mult *= fx['resolution_mult']
    hazard *= resolution_mult
    return min(max(hazard, 0), 1)


def grade_transition_probs(current_grade, days_active, is_cumulative=False, has_active_conmed=False, conmed_tier=3, is_drug_held=False, is_dose_reduced=False):
    '''활성 AE의 grade 변화 확률을 계산한다.

    Args:
        current_grade: 현재 AE grade (1-5)
        days_active: AE 활성 경과일
        is_cumulative: 누적 독성 여부 (True면 시간 경과에 따라 악화 확률 증가)
        has_active_conmed: 이 AE에 대한 활성 보조약이 있는가
        conmed_tier: 보조약의 효과 Tier (2=표적, 3=대증, 4=비약물)
        is_drug_held: 원인 약물이 현재 hold 상태인가
        is_dose_reduced: 원인 약물이 감량 상태인가

    Returns:
        {"worsen": float, "improve": float, "stable": float}
        세 확률의 합 = 1.0

    모든 상수는 config/defaults.py에서 관리.

    핵심 원칙: Care AI가 직접 확률을 바꾸는 것이 아니라,
    Care AI 또는 Natural 방문의 결과로 처방된 action(conmed, dose hold)이
    grade 전이 확률을 변경한다. Natural이든 Care AI든 동일한 규칙.
    '''
    if current_grade >= 5:
        return {
            'worsen': 0,
            'improve': 0,
            'stable': 1 }
    base_worsen = GRADE_TRANSITION_BASE_WORSEN
    base_improve = GRADE_TRANSITION_BASE_IMPROVE
    time_factor = min(days_active / 60, GRADE_CUMULATIVE_MAX_TIME_FACTOR)
    if is_cumulative:
        base_worsen *= 1 + time_factor
        base_improve *= max(0.3, 1 - time_factor * 0.3)
    elif days_active > GRADE_TIME_STABILIZE_DAY:
        base_worsen *= 0.6
        base_improve *= 1.3
    if current_grade >= 3:
        base_worsen *= GRADE_HIGH_WORSEN_DAMPING
        base_improve *= GRADE_HIGH_IMPROVE_BOOST
    if current_grade >= 4:
        base_worsen *= GRADE_4_TO_5_DAMPING
        base_improve *= GRADE_4_IMPROVE_BOOST
    worsen_mult = 1
    improve_mult = 1
    if is_drug_held:
        fx = INTERVENTION_EFFECTS['dose_hold']
        worsen_mult *= fx['worsen_mult']
        improve_mult *= fx['improve_mult']
    elif is_dose_reduced:
        fx = INTERVENTION_EFFECTS['dose_reduction']
        worsen_mult *= fx['worsen_mult']
        improve_mult *= fx['improve_mult']
    if has_active_conmed:
        tier_key = f'''conmed_tier{conmed_tier}'''
        fx = INTERVENTION_EFFECTS.get(tier_key, INTERVENTION_EFFECTS['conmed_tier3'])
        if is_drug_held or is_dose_reduced:
            worsen_mult *= 1 + (fx['worsen_mult'] - 1) * 0.5
            improve_mult *= 1 + (fx['improve_mult'] - 1) * 0.5
        else:
            worsen_mult *= fx['worsen_mult']
            improve_mult *= fx['improve_mult']
    base_worsen *= worsen_mult
    base_improve *= improve_mult
    worsen = min(base_worsen, MAX_GRADE_TRANSITION_PROB)
    improve = min(base_improve, MAX_GRADE_TRANSITION_PROB)
    stable = max(0, 1 - worsen - improve)
    return {
        'worsen': worsen,
        'improve': improve,
        'stable': stable }


def tumor_change_pct(day, best_response=None, response_onset_day=None, baseline_sum_mm=50, patient_scale=1, effective_treatment_weeks=None):
    '''종양 반응 카테고리에 따른 예상 크기 변화율을 계산한다 (v2.3 시그모이드).

    의학적 근거:
      - 항암 효과는 첫 투약 직후 분자 수준에서 시작
      - 영상에서 관측 가능한 변화까지 lag_weeks 지연 (세포 사멸→괴사→리모델링)
      - 시그모이드(1 - exp(-t/τ)) 커브: 초기 lag → 가속 → 포화
      - patient_scale (lognormal): 동일 response category 내 환자별 속도 차이
      - Dose hold 시: effective_treatment_weeks가 calendar_weeks보다 느리게 증가
        → 종양 축소 속도 둔화 (약 효과가 줄어들므로)
      - PD (약제 내성): 종양 성장은 약물 투여와 무관 → calendar_weeks 사용

    RECIST 1.1 기준:
      - CR: 100% 감소 (target lesion disappearance)
      - PR: ≥30% 감소
      - PD: ≥20% 증가
      - SD: 그 사이

    Args:
        day: 현재 시뮬레이션 날짜
        best_response: "CR" | "PR" | "SD" | "PD"
        response_onset_day: (레거시 호환, 현재 미사용 — 시그모이드가 대체)
        baseline_sum_mm: 기저 종양 직경 합 (mm)
        patient_scale: 환자별 속도 계수 (lognormal, median=1.0)
        effective_treatment_weeks: 유효 치료 시간 (weeks).
            None이면 calendar_weeks 사용 (하위 호환).
            dose hold 시 calendar보다 느리게 증가 → 종양 축소 둔화.

    Returns:
        기저 대비 변화 퍼센트. 음수=축소, 양수=증가.
    '''
    T = TUMOR_RATE
    calendar_weeks = max(0, day - 1) / 7
    tx_weeks = effective_treatment_weeks if effective_treatment_weeks is not None else calendar_weeks
    ps = max(0.2, min(3, patient_scale))
    if best_response == 'CR':
        lag = T.get('CR_lag_weeks', 2) / ps
        plateau = T.get('CR_plateau', -95) * ps
        plateau = max(-100, plateau)
        blend = 1 - math.exp(-max(0, tx_weeks) / max(lag, 0.5))
        return round(blend * plateau, 1)
    if best_response == 'PR':
        lag = T.get('PR_lag_weeks', 2.5) / ps
        plateau = T.get('PR_plateau', -55) * ps
        plateau = max(-80, min(-30, plateau))
        blend = 1 - math.exp(-max(0, tx_weeks) / max(lag, 0.5))
        return round(blend * plateau, 1)
    if best_response == 'SD':
        lag = T.get('SD_lag_weeks', 3)
        base = T.get('SD_plateau', -5) * ps
        base = max(-15, min(10, base))
        amp = T.get('SD_amplitude', 4)
        blend = 1 - math.exp(-max(0, tx_weeks) / max(lag, 0.5))
        oscillation = amp * math.sin(tx_weeks * 0.4) * blend
        pct = base * blend + oscillation
        return round(max(-29, min(19, pct)), 1)
    if best_response == 'PD':
        lag = T.get('PD_lag_weeks', 1.5) / ps
        rate = T.get('PD_rate', 3.5) * ps
        pd_max = T.get('PD_max', 200)
        blend = 1 - math.exp(-max(0, calendar_weeks) / max(lag, 0.5))
        pct = blend * rate * calendar_weeks
        return round(min(pd_max, pct), 1)


def adjust_incidence_by_risk_modifiers(base_incidence, risk_modifiers, patient_conditions, patient_age=None):
    '''rule_set의 risk_modifiers를 적용하여 환자별 AE incidence를 조정한다.

    이 함수는 코드 기반 조정만 수행한다.
    LLM 기반 세밀한 조정은 daily_agent에서 별도로 한다.

    Args:
        base_incidence: 기본 발생률 (rule_set.ae_profile.incidence_all_grade)
        risk_modifiers: [{"condition": "baseline diabetes", "incidence_multiplier": 1.5}]
        patient_conditions: 환자의 동반질환 set (소문자)
        patient_age: 환자 나이

    Returns:
        조정된 incidence (0-1 범위 클램프)
    '''
    adjusted = base_incidence
    if not risk_modifiers:
        return min(max(adjusted, 0.0), 1.0)
    for modifier in risk_modifiers:
        condition = modifier.get('condition', '').lower()
        multiplier = modifier.get('incidence_multiplier', 1.0)
        if 'age' in condition and patient_age is not None:
            try:
                parts = condition.replace('age', '').strip().split()
                if len(parts) >= 2:
                    op, threshold = parts[0], float(parts[1])
                    if op == '>' and patient_age > threshold:
                        adjusted *= multiplier
                    elif op == '>=' and patient_age >= threshold:
                        adjusted *= multiplier
                    elif op == '<' and patient_age < threshold:
                        adjusted *= multiplier
            except (ValueError, IndexError):
                pass
        else:
            for pc in patient_conditions:
                if condition in pc or pc in condition:
                    adjusted *= multiplier
                    break
    return min(max(adjusted, 0.0), 0.99)


def _check_threshold(condition_key, current_val, uln=None, lln=None, baseline=None):
    '''threshold 조건 문자열을 파싱해 현재값이 조건을 만족하는지 판정.

    지원 형식:
      "ALT_gt_5xULN"        → current ALT > 5 × ULN
      "Platelets_lt_50000"  → current Platelets < 50,000
      "SBP_lt_90"           → current SBP < 90
      "Creatinine_gt_3xBL"  → current Creatinine > 3 × baseline
    '''
    parts = condition_key.split('_')
    op_idx = -1
    for i, p in enumerate(parts):
        if p in ('gt', 'lt', 'gte', 'lte'):
            op_idx = i
            break
    if op_idx < 1:
        return False
    op = parts[op_idx]
    val_str = '_'.join(parts[op_idx + 1:])
    threshold = None
    if 'xULN' in val_str and uln and uln > 0:
        try:
            threshold = float(val_str.replace('xULN', '')) * uln
        except ValueError:
            return False
    elif 'xBL' in val_str and baseline and baseline > 0:
        try:
            threshold = float(val_str.replace('xBL', '')) * baseline
        except ValueError:
            return False
    elif 'xLLN' in val_str and lln and lln > 0:
        try:
            threshold = float(val_str.replace('xLLN', '')) * lln
        except ValueError:
            return False
    else:
        try:
            threshold = float(val_str)
        except ValueError:
            return False
    if threshold is None:
        return False
    if op == 'gt':
        return current_val > threshold
    if op == 'lt':
        return current_val < threshold
    if op == 'gte':
        return current_val >= threshold
    if op == 'lte':
        return current_val <= threshold
    return False


def compute_daily_mortality(day, active_aes, tumor_status, ecog, treatment_discontinued=False, response_onset_day=None, risk_config=None, cumulative_doses=None, discontinuation_day=None):
    '''Log-linear hazard mortality model.

    Uses log(hazard) = log(base) + severity_weight * log(HR) to avoid
    double-counting between correlated factors (ECOG, AE grade, tumor status).

    Components:
      - base_risk: guaranteed daily hazard from underlying disease
      - severity_risk: state-dependent hazard scaled by clinical severity
      - HR (hazard ratio): combined from disease_progression, treatment_toxicity, ECOG

    The severity_weight (0~1) prevents the compounding problem where
    correlated factors (PD↔ECOG, AE↔ECOG, AE↔CSF) multiply each other.

    Returns:
        (daily_mortality_probability, {channel_name: value})
    '''
    annual = risk_config.get('baseline_annual_mortality', 0.25)
    if annual <= 0:
        return (0, {})
    daily_base = 1 - (1 - min(annual, 0.99)) ** (1 / 365)
    channels_cfg = risk_config.get('channels') or {}
    contributions = {}

    # ── Channel 1: Disease Progression ──
    dp_cfg = channels_cfg.get('disease_progression') or {}
    dp_mult = 1.0
    if tumor_status == 'PD':
        dp_mult = float(dp_cfg.get('pd_multiplier', 4))
    elif tumor_status in ('CR', 'PR') and response_onset_day and day > response_onset_day:
        lag = int(dp_cfg.get('response_lag_days', 21))
        reduction = float(dp_cfg.get('response_reduction', 0.3))
        elapsed = day - response_onset_day
        if elapsed > lag:
            progress = min((elapsed - lag) / 28, 1)
            dp_mult = 1 - (1 - reduction) * progress
    contributions['disease_progression'] = round(dp_mult, 4)

    # ── Channel 2: Treatment Toxicity (no coherence cap) ──
    tt_cfg = channels_cfg.get('treatment_toxicity', {})
    tt_mult = 1.0
    ae_grade_mults = tt_cfg.get('ae_grade_multipliers', {'3': 1.5, '4': 3})
    concurrent_threshold = int(tt_cfg.get('concurrent_ae_threshold', 3))
    concurrent_mult = float(tt_cfg.get('concurrent_ae_multiplier', 2))
    significant_count = 0
    for ae in active_aes:
        if ae.get('status') == 'resolved':
            continue
        g = ae.get('grade', 1)
        if g >= 2:
            significant_count += 1
        gm = float(ae_grade_mults.get(str(g), 1))
        if gm > 1:
            tt_mult *= gm
    if significant_count >= concurrent_threshold:
        tt_mult *= concurrent_mult
    if cumulative_doses:
        total_doses = sum(cumulative_doses.values())
        cum_factor = 1 + min(total_doses / 5000, 0.5)
        if treatment_discontinued and discontinuation_day is not None:
            days_since = max(day - discontinuation_day, 0)
            decay = math.exp(-0.05 * days_since)
            cum_factor = 1 + (cum_factor - 1) * decay
        tt_mult *= cum_factor
    contributions['treatment_toxicity'] = round(tt_mult, 4)

    # ── Channel 3: ECOG ──
    ecog_mult = ECOG_MORTALITY_MAP.get(min(ecog, 4), 1)
    contributions['ecog'] = ecog_mult

    # ── Channel 4: Treatment Discontinued ──
    if treatment_discontinued:
        discont_mult = TREATMENT_DISCONTINUED_MORTALITY_MULT
        if discontinuation_day is not None:
            days_since = max(day - discontinuation_day, 0)
            decay = math.exp(-0.023 * days_since)
            discont_mult = 1 + (discont_mult - 1) * decay
        contributions['treatment_discontinued'] = round(discont_mult, 4)

    # ── Compute combined HR (multiplicative) ──
    hr = 1.0
    for k, v in contributions.items():
        if not k.startswith('_'):
            hr *= v

    # ── Severity weight: how "clinically endangered" is this patient ──
    # Uses the max of three orthogonal severity signals.
    # Unlike old CSF, this only scales the HR exponent, not the base.
    max_ae_grade = max(
        (ae.get('grade', 0) for ae in active_aes if ae.get('status') != 'resolved'),
        default=0,
    )
    n_active_aes = sum(1 for ae in active_aes if ae.get('status') != 'resolved')

    if tumor_status == 'PD':
        sw_tumor = 0.8
    elif tumor_status == 'SD':
        sw_tumor = 0.3
    elif tumor_status in ('CR', 'PR'):
        sw_tumor = 0.15
    else:
        sw_tumor = 0.2

    sw_ecog_map = {0: 0.05, 1: 0.15, 2: 0.45, 3: 0.75, 4: 1.0}
    sw_ecog = sw_ecog_map.get(min(ecog, 4), 0.5)

    if max_ae_grade >= 4:
        sw_ae = 1.0
    elif max_ae_grade >= 3:
        sw_ae = 0.4
    elif n_active_aes >= 3:
        sw_ae = 0.3
    elif max_ae_grade == 2:
        sw_ae = 0.15
    else:
        sw_ae = 0.05

    severity_weight = max(sw_tumor, sw_ecog, sw_ae)
    contributions['_sw_tumor'] = round(sw_tumor, 4)
    contributions['_sw_ecog'] = round(sw_ecog, 4)
    contributions['_sw_ae'] = round(sw_ae, 4)
    contributions['_sw_combined'] = round(severity_weight, 6)

    # ── Log-linear hazard: exp(log(base) + severity_weight * log(HR)) ──
    log_base = math.log(max(daily_base, 1e-12))
    log_hr = math.log(max(hr, 1e-12))
    daily_risk = math.exp(log_base + severity_weight * log_hr)

    base_floor = daily_base * 0.15
    daily_risk = max(daily_risk, base_floor)

    # Grace period: patients screened into trial are stable enough to start.
    # Ramp mortality linearly over first cycle (21 days) unless G4+ AE present.
    GRACE_DAYS = 21
    has_g4 = any(ae.get('grade', 0) >= 4 for ae in active_aes)
    if day <= GRACE_DAYS and not has_g4:
        ramp = day / GRACE_DAYS
        daily_risk *= ramp

    daily_risk = min(daily_risk, MAX_DAILY_MORTALITY)
    return daily_risk, contributions


def compute_dynamic_ecog(
    baseline_ecog, current_ecog, active_aes, tumor_status,
    response_onset_day, day, comorbidities, treatment_discontinued, ecog_config,
):
    '''환자 상태 기반 동적 ECOG Performance Status.

    AE 부담, 질환 진행, 동반질환, 치료 피로도를 종합하여
    매일 ECOG 점수를 재계산한다.
    Daily change는 ±ECOG_MAX_DAILY_CHANGE로 제한.
    '''
    score = float(baseline_ecog)

    # ── AE burden → ECOG 악화 ──
    ae_w = float(ecog_config.get('ae_burden_weight', 0.15))
    ae_w = max(ae_w, MIN_AE_BURDEN_WEIGHT)
    ae_penalty = 0.0
    max_ae_grade = 0
    for ae in active_aes:
        if ae.get('status') == 'resolved':
            continue
        g = ae.get('grade', 1)
        max_ae_grade = max(max_ae_grade, g)
        ae_penalty += max(0, g - 1) * ae_w
    score += min(ae_penalty, ECOG_AE_PENALTY_CAP)

    # AE 등급별 ECOG 최저 하한
    ae_ecog_floor = 0
    if max_ae_grade >= 5:
        ae_ecog_floor = 4
    elif max_ae_grade >= 4:
        ae_ecog_floor = max(baseline_ecog + 2, 2)
    elif max_ae_grade >= 3:
        ae_ecog_floor = max(baseline_ecog + 1, 1)
    score = max(score, float(ae_ecog_floor))

    # ── 치료 피로도 ──
    cycle_n = day / 21.0
    fatigue = cycle_n * ECOG_TREATMENT_FATIGUE_PER_CYCLE
    score += min(fatigue, 0.5)

    # ── 질환 진행 ──
    dw = float(ecog_config.get('disease_weight', 0.3))
    if tumor_status == 'PD':
        score += dw
    elif tumor_status in ('CR', 'PR') and response_onset_day and day > response_onset_day:
        lag = int(ecog_config.get('response_lag_days', 21))
        benefit = float(ecog_config.get('response_benefit', -0.3))
        elapsed = day - response_onset_day
        if elapsed > lag:
            progress = min((elapsed - lag) / 28.0, 1.0)
            score += benefit * progress

    # ── 동반질환 ──
    co_pen = float(ecog_config.get('comorbidity_penalty', 0.1))
    n_co = len(comorbidities) if comorbidities else 0
    score += min(n_co * co_pen, 1.0)

    # ── 치료 중단 패널티 ──
    if treatment_discontinued:
        score += TREATMENT_DISCONTINUED_ECOG_PENALTY

    # ── Daily change cap + 범위 제한 ──
    new_ecog = int(max(0, min(4, round(score))))
    new_ecog = max(current_ecog - ECOG_MAX_DAILY_CHANGE,
                   min(current_ecog + ECOG_MAX_DAILY_CHANGE, new_ecog))
    return max(0, min(4, new_ecog))


def compute_causal_lab_target(
    lab_name, baseline_value, active_aes, cumulative_doses,
    ae_lab_links, cumulative_dose_effects, active_cms=None, cm_lab_effects=None,
):
    '''AE + 누적 약물 노출 + CM 치료 효과에 기반한 lab 값의 인과적 목표값 계산.

    이 함수는 "lab이 얼마가 되어야 하는지" 인과적 목표를 정하며,
    실제 값은 이 목표를 향해 점진적으로 이동한다 (daily_agent에서 처리).
    '''
    target = baseline_value

    # AE → lab 변화
    for link in ae_lab_links:
        if link.get('lab', '') != lab_name:
            continue
        ae_term = link.get('ae_term', '')
        for ae_state in active_aes:
            if ae_state.get('ae_term', '') != ae_term:
                continue
            if ae_state.get('status') == 'resolved':
                continue
            grade = ae_state.get('grade', 1)
            grade_fx = link.get('grade_effects') or {}
            mult = float(grade_fx.get(str(grade), 1.0))
            target *= mult

    # 누적 약물 노출 → lab 변화
    for eff in cumulative_dose_effects:
        if eff.get('lab', '') != lab_name:
            continue
        drug = eff.get('drug', '')
        per_100 = float(eff.get('per_100mg_multiplier', 1.0))
        dose = float(cumulative_doses.get(drug, 0))
        target *= per_100 ** (dose / 100.0)

    # CM 치료 효과 → lab 보정
    if active_cms and cm_lab_effects:
        active_indications = set()
        for cm in (active_cms or []):
            ind = cm.get('CMINDC', '')
            if cm.get('CMONGO', False) and ind:
                active_indications.add(ind.lower().strip())
        for effect in cm_lab_effects:
            if effect.get('lab', '') != lab_name:
                continue
            ind = effect.get('indication', '').lower().strip()
            if ind in active_indications:
                max_correction = float(effect.get('correction_factor', 1.0))
                target *= max_correction

    return max(0.0, target)


def compute_ae_cascade_multipliers(active_aes, cascade_rules):
    '''활성 AE가 다른 AE 발생 확률에 미치는 연쇄 효과.

    예: 호중구감소증 Grade 3+ → 감염 확률 ×3
    '''
    multipliers = {}
    for rule in cascade_rules:
        trigger = rule.get('trigger_ae', '')
        threshold = int(rule.get('grade_threshold', 3))
        target = rule.get('target_ae', '')
        mult = float(rule.get('multiplier', 1.0))
        for ae_state in active_aes:
            if ae_state.get('ae_term', '') != trigger:
                continue
            if ae_state.get('status') == 'resolved':
                continue
            if ae_state.get('grade', 0) >= threshold:
                multipliers[target] = max(multipliers.get(target, 1.0), mult)
    return multipliers


def compute_discontinuation_risk(
    day, active_aes, ecog, baseline_ecog, tumor_status,
    treatment_weeks, treatment_discontinued, dose_reductions, disposition_config,
):
    '''2-channel + background 중도탈락 확률 모델.

    Channels:
      1. patient_withdrawal (동의 철회) — AE 부담, ECOG, 치료 기간
      2. physician_decision (의사 결정) — ECOG 심각, dose reduction 반복, 종양 반응 불량

    Returns:
        {
            'patient_withdrawal': float,
            'physician_decision': float,
            'background': float,
            'independent_hazards': True,  # 독립 hazard로 조합
        }
    '''
    hazards = {}

    # Channel 1: 환자 동의 철회
    pw_cfg = disposition_config.get('consent_withdrawal') or {}
    pw_base = float(pw_cfg.get('base_daily_rate', 0.0004))
    pw_rf = pw_cfg.get('risk_factors') or {}
    pw_mult = 1.0
    has_severe_ae = any(
        ae.get('grade', 0) >= 3 and ae.get('status') != 'resolved'
        for ae in active_aes
    )
    if has_severe_ae:
        pw_mult *= float(pw_rf.get('active_ae_grade_3_plus', 2.5))
    ecog_delta = ecog - baseline_ecog
    if ecog_delta >= 1:
        pw_mult *= float(pw_rf.get('ecog_worsened', 2.0))
    if treatment_weeks > 12:
        pw_mult *= float(pw_rf.get('treatment_weeks_gt_12', 1.5))
    if tumor_status in ('SD', 'PD'):
        pw_mult *= float(pw_rf.get('poor_response', 1.3))
    patient_withdrawal = min(pw_base * pw_mult, MAX_DISCONTINUATION_PATIENT)
    hazards['patient_withdrawal'] = round(patient_withdrawal, 6)

    # Channel 2: 의사 결정
    pd_cfg = disposition_config.get('physician_decision') or {}
    pd_base = float(pd_cfg.get('base_daily_rate', 0.00012))
    pd_rf = pd_cfg.get('risk_factors') or {}
    pd_mult = 1.0
    if ecog >= 3:
        pd_mult *= float(pd_rf.get('ecog_ge_3', 3.0))
    if dose_reductions >= 2:
        pd_mult *= float(pd_rf.get('multiple_dose_reductions', 2.0))
    if tumor_status == 'PD':
        pd_mult *= float(pd_rf.get('poor_tumor_response', 2.0))
    if has_severe_ae:
        pd_mult *= float(pd_rf.get('severe_ae', 1.5))
    physician_decision = min(pd_base * pd_mult, MAX_DISCONTINUATION_PHYSICIAN)
    hazards['physician_decision'] = round(physician_decision, 6)

    # Background: protocol violation, other 등 일정 확률
    hazards['background'] = DISCONTINUATION_BACKGROUND_DAILY_RATE
    hazards['independent_hazards'] = True

    return hazards