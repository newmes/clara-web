'''Orchestrator — Fate table 없는 3-Phase 확률적 시뮬레이션

Phase 0: Rule Agent → rule_set (약물당 1회)
Phase 1: Patient Agent → N명 환자 (LLM→rand→LLM)
Phase 2: Daily Agent → 일별 시뮬레이션 (hazard function 기반 동적 이벤트 결정)

핵심 차이 (이전 4-Phase 대비):
  - Fate table 없음: 어떤 AE가 언제 발생할지 미리 정하지 않는다.
  - Hazard function: rule_set의 onset 분포에서 매일의 발생 확률을 코드로 계산한다.
  - 환자 상태 반영: 현재 AE, 개입 이력에 따라 확률이 동적으로 변한다.
  - Care AI 친화적: 개입(care_record)이 실제로 이벤트를 "예방"할 수 있다.
'''
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sim.agents import rule_agent, patient_agent
from sim.agents.daily_agent import DailySimulator, create_simulator
from sim.agents.care_agent import CareAgent, apply_interventions
from sim.engine.sampler import Sampler
from sim.engine.mood import MoodState
from sim.engine.observation import ObservationModel, compute_detection_delay_summary
from sim.logger import get_logger, log_event, log_summary, init_logging
from sim.crf_mapper import map_day_record, map_patient_record
from sim.hospital_record_mapper import extract_hospital_record
_print_lock = threading.Lock()
_logger = get_logger('orchestrator')


def _is_hospital_day(day, cycle_length, admin_cycle_days=None):
    '''투약 스케줄에 기반한 병원 방문일 판정.

    admin_cycle_days가 제공되면 해당 cycle day에 해당하는 모든 날을 병원일로 표시.
    없으면 각 사이클 Day 1만 (기존 동작).
    '''
    if day <= 0:
        return False
    cycle_day = (day - 1) % cycle_length + 1
    if admin_cycle_days:
        return cycle_day in admin_cycle_days
    return cycle_day == 1


def _get_cycle_info(day, cycle_length):
    '''(cycle_number, cycle_day) 반환. 1-based.'''
    cycle = (day - 1) // cycle_length + 1
    cycle_day = (day - 1) % cycle_length + 1
    return (cycle, cycle_day)


def _build_day0_record(patient: dict, simulator) -> dict:
    """Day 0 (screening/pre-treatment baseline) record from EMR data."""
    emr = patient.get("emr", {})
    pid = patient.get("patient_id", "")

    labs = {}
    for lab_name, lab_data in emr.get("baseline_labs", {}).items():
        if isinstance(lab_data, dict):
            labs[lab_name] = {
                "value": lab_data.get("value"),
                "unit": lab_data.get("unit", ""),
                "trend": "baseline",
                "LBBLFL": "Y",
            }
        elif isinstance(lab_data, (int, float)):
            labs[lab_name] = {
                "value": lab_data,
                "unit": "",
                "trend": "baseline",
                "LBBLFL": "Y",
            }
        elif isinstance(lab_data, str):
            m = __import__('re').search(r'-?\d+\.?\d*', lab_data)
            if m:
                labs[lab_name] = {
                    "value": float(m.group()),
                    "unit": "",
                    "trend": "baseline",
                    "LBBLFL": "Y",
                }

    vitals = dict(emr.get("baseline_vitals", {}))

    baseline_ecog = emr.get("baseline_ecog", 1)

    baseline_cm = []
    for mh in (emr.get("medical_history") or []):
        med = mh.get("medication")
        if med and mh.get("ongoing", False):
            baseline_cm.append({
                "CMTRT": med,
                "CMINDC": mh.get("condition", ""),
                "CMSTDAT": 0,
                "_baseline": True,
            })

    return {
        "patient_id": pid,
        "day": 0,
        "cycle": 0,
        "cycle_day": 0,
        "is_baseline": True,
        "objective": {
            "location": "OUTPATIENT",
            "treatment_status": "screening",
            "labs": labs,
            "vitals": vitals,
            "active_aes": [],
            "ecog": baseline_ecog,
        },
        "subjective": {
            "overall_awareness": "UNAWARE",
            "symptoms_patient_perceives": [],
        },
        "cm_records": baseline_cm,
        "ec_records": [],
        "care_record": [],
    }


def _find_last_hospital_record(day_results: list[dict]) -> dict | None:
    """day_results에서 가장 최근 병원 방문일의 hospital_record를 역순 검색."""
    for dr in reversed(day_results):
        hr = dr.get("hospital_record")
        if hr and isinstance(hr, dict) and hr.get("objective"):
            return hr
    return None


class SimulationRunner:
    '''3-Phase 확률적 시뮬레이션 러너.

    Fate table 없이, hazard function으로 매일의 이벤트를 동적 결정.
    '''

    def __init__(self, drug_name, indication, model='gemini-2.0-flash', data_dir='data', cycle_length=21, max_retries=2, seed=None):
        self.drug_name = drug_name
        self.indication = indication
        self.model = model
        self.data_dir = Path(data_dir)
        self.cycle_length = cycle_length
        self.max_retries = max_retries
        self.seed = seed
        self.rule_set = None
        self._progress = {}  # {patient_id: {day, total_days, status}}
        self._cancelled = False
        self._log_lines: list[str] = []
        self._log_max = 500  # keep last N lines in memory
        init_logging(run_name=f'''{drug_name.replace(' ', '_')}_{indication.replace(' ', '_')}''')

    def cancel(self):
        """Signal the simulation to stop."""
        self._cancelled = True

    @property
    def is_cancelled(self):
        return self._cancelled

    def log(self, msg: str):
        """Append a log message to the in-memory buffer and log file."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > self._log_max:
            self._log_lines = self._log_lines[-self._log_max:]
        # Also write to disk log
        log_path = self.data_dir / "sim_log.txt"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def get_log(self, since: int = 0) -> list[str]:
        """Return log lines from index `since`."""
        return self._log_lines[since:]

    def write_run_meta(self, n_patients: int, total_days: int, mode: str,
                       status: str = 'running', extra: dict | None = None):
        """Write/update run_meta.json for live status tracking."""
        from datetime import datetime
        meta_path = self.data_dir / 'run_meta.json'
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        meta.update({
            'drug_name': self.drug_name,
            'indication': self.indication,
            'n_patients': n_patients,
            'total_days': total_days,
            'mode': mode,
            'status': status,
            'model': self.model,
            'seed': self.seed,
            'updated_at': datetime.now().isoformat(),
            'progress': dict(self._progress),
        })
        if status == 'running' and 'started_at' not in meta:
            meta['started_at'] = datetime.now().isoformat()
        if status == 'completed' and 'completed_at' not in meta:
            meta['completed_at'] = datetime.now().isoformat()
        if extra:
            meta.update(extra)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    def update_progress(self, patient_id: str, day: int, total_days: int,
                        status: str = 'running'):
        """Update progress for a patient (called per-day)."""
        self._progress[patient_id] = {
            'day': day, 'total_days': total_days, 'status': status,
        }
        # Write meta periodically (every 10 days to avoid I/O overhead)
        if day % 10 == 0 or day == 1 or day == total_days or status != 'running':
            n_patients = len(self._progress)
            self.write_run_meta(n_patients, total_days, 'mixed', 'running')

    def discover_rules(self):
        '''Rule Agent: 약물/적응증의 시뮬레이션 규칙을 발견한다.'''
        _logger.info(f'''Phase 0: Discovering rules for {self.drug_name} ({self.indication})''')
        save_path = self.data_dir / 'rule_set.json'
        self.rule_set = rule_agent.discover_rules(drug_name=self.drug_name, indication=self.indication, model=self.model, save_path=str(save_path))
        return self.rule_set

    def load_rules(self, path=None):
        '''기존 rule_set 로드.'''
        if not path:
            path = str(self.data_dir / 'rule_set.json')
        self.rule_set = rule_agent.load_rules(path)
        return self.rule_set

    def create_patients(self, n):
        '''Patient Agent: N명 환자를 LLM→rand→LLM으로 생성.'''
        _logger.info(f'''Phase 1: Generating {n} patients''')
        if not self.rule_set:
            raise RuntimeError('rule_set이 없습니다. discover_rules() 또는 load_rules()를 먼저 호출하세요.')
        save_dir = self.data_dir / 'patients'
        patients = patient_agent.generate_patients(rule_set=self.rule_set, n=n, model=self.model, save_dir=str(save_dir), seed=self.seed)
        return patients

    def create_patients_parallel(self, n, max_workers):
        '''Patient Agent: N명 환자를 병렬로 생성.'''
        _logger.info(f'''Phase 1: Generating {n} patients (parallel, {max_workers} workers)''')
        if not self.rule_set:
            raise RuntimeError('rule_set이 없습니다. discover_rules() 또는 load_rules()를 먼저 호출하세요.')
        save_dir = self.data_dir / 'patients'
        save_dir.mkdir(parents=True, exist_ok=True)
        from sim.agents.patient_agent import generate_patient
        from sim.engine.sampler import Sampler

        generated_count = [0]

        def _gen_one(i):
            if self._cancelled:
                return None
            # 부모 스레드의 API 키를 워커 스레드에 전파
            from sim.agents.llm_client import get_api_key, set_api_key as _set_key
            if not get_api_key() and hasattr(self, '_api_key') and self._api_key:
                _set_key(self._api_key)
            patient_seed = (self.seed or 0) + i
            sampler = Sampler(seed=patient_seed)
            patient = generate_patient(self.rule_set, i, n, sampler, self.model)
            cdash_patient = map_patient_record(patient)
            out_path = save_dir / f'''{cdash_patient['patient_id']}.json'''
            out_path.write_text(json.dumps(cdash_patient, indent=2, ensure_ascii=False), encoding='utf-8')
            generated_count[0] += 1
            pid = cdash_patient.get('patient_id', f'PT-{i:03d}')
            age = cdash_patient.get('emr', {}).get('demographics', {}).get('age', '?')
            sex = cdash_patient.get('emr', {}).get('demographics', {}).get('sex', '?')
            self.log(f"👤 Generated {pid} ({age}{sex}) [{generated_count[0]}/{n}]")
            return cdash_patient

        actual_workers = min(max_workers, n)
        self.log(f"Phase 1: Generating {n} patients ({actual_workers} workers)")
        t0 = time.time()
        patients = [None] * n
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_gen_one, i): i for i in range(1, n + 1)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    patients[idx - 1] = result
        patients = [p for p in patients if p is not None]
        elapsed = time.time() - t0
        self.log(f"✅ {len(patients)} patients generated in {elapsed:.1f}s")
        print(f'''  {len(patients)} patients generated in {elapsed:.1f}s ({elapsed / max(len(patients),1):.1f}s/patient)''')
        return patients

    def run_natural(self, patient, total_days, save=True):
        '''무개입 시뮬레이션.

        Hazard function으로 매일의 이벤트를 동적 결정.
        이벤트 날: LLM 호출
        조용한 날: 코드만 (LLM 호출 없음)
        '''
        pid = patient['patient_id']
        cycle_length = self.cycle_length
        if self.rule_set:
            cycle_length = self.rule_set.get('trial_design', {}).get('cycle_length_days', self.cycle_length)
        print(f'''\n[Simulation] {pid} — Natural course, {total_days} days (no fate table)''')
        patient_num = int(pid.split('-')[1]) if '-' in pid else 1
        sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num)
        simulator = create_simulator(
            rule_set=self.rule_set,
            patient=patient,
            sampler=sampler,
            model=self.model,
            actual_duration=total_days,
        )
        persona = patient.get('persona', {})
        persona_type = persona.get('type', 'minimizer')
        mood = MoodState(persona_type=persona_type, seed=(self.seed or 0) + 200 + patient_num + 30000)
        obs_sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + 50000)
        observation_model = ObservationModel(mood=mood, sampler=obs_sampler, care_ai_enabled=False)
        all_admin_days = set()
        for drug in simulator.admin_schedule:
            all_admin_days.update(drug.get('cycle_days', [1]))
        admin_cycle_days = sorted(all_admin_days)
        day_results = []
        sim_path = self.data_dir / 'simulations' / f'''{pid}_natural.jsonl'''
        hr_path = self.data_dir / 'simulations' / f'''{pid}_natural_hospital.jsonl'''
        if save:
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            sim_path.write_text('', encoding='utf-8')
            hr_path.write_text('', encoding='utf-8')
        llm_calls = 0
        quiet_days = 0

        # Day 0: pre-treatment baseline from EMR
        day0 = _build_day0_record(patient, simulator)
        day_results.append(day0)
        if save:
            cdash_day0 = map_day_record(day0, patient)
            with open(sim_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cdash_day0, ensure_ascii=False) + '\n')

        for day in range(1, total_days + 1):
            if self._cancelled:
                self.log(f"⛔ {pid} — Cancelled at Day {day}")
                break

            cycle, cycle_day = _get_cycle_info(day, cycle_length)
            is_hospital = _is_hospital_day(day, cycle_length, admin_cycle_days)
            is_admin_day = simulator.is_administration_day(cycle_day)

            day_result = self._generate_day_with_retry(
                simulator, day_results, day, cycle, cycle_day, is_hospital,
            )

            # Observation model: GT → Hospital Record 변환
            is_visit, observed = observation_model.process_day(
                day=day, day_result=day_result, is_hospital=is_hospital,
                is_admin_day=is_admin_day, simulator=simulator,
            )

            # 병원 방문일: HR 기반 dose modification + conmed 처방
            if is_visit:
                observed_aes = observed.get('objective', {}).get('active_aes', [])
                dose_changes = simulator.apply_hospital_dose_modifications(
                    observed_aes, day, cycle, cycle_day,
                )
                if dose_changes:
                    simulator.patch_day_treatment_status(day_result)
                    new_ts = day_result.get('objective', {}).get('treatment_status', 'active')
                    observation_model.update_treatment_status(new_ts)
                # 관찰된 AE에 대해 보조약 처방
                simulator.prescribe_conmeds_for_aes(observed_aes, day)
                # cm_records 갱신
                day_result["cm_records"] = simulator._get_active_cm_records(day)

            # Observation model 결과를 day_result에 merge (JSONL에 기록)
            day_result["hospital_record"] = observed.get("hospital_record")
            day_result["observation_events"] = observed.get("observation_events")
            day_result["mood_state"] = observed.get("mood_state")

            # 자연 경과: care_record 항상 비어 있음
            day_result["care_record"] = []
            day_results.append(day_result)

            if save:
                cdash_record = map_day_record(day_result, patient)
                with open(sim_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cdash_record, ensure_ascii=False) + "\n")
                # Hospital-only record
                hr_record = extract_hospital_record(cdash_record)
                with open(hr_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(hr_record, ensure_ascii=False) + "\n")

            self.update_progress(pid, day, total_days)

            # 통계 추적 + 로그
            mode = day_result.get("_generation_mode", "?")
            aes = day_result.get("objective", {}).get("active_aes", [])
            ae_str = ", ".join(f'''{a.get('ae', '')} G{a.get('grade', '?')}''' for a in aes) if aes else "none"
            location = day_result.get("objective", {}).get("location", "?")
            mode_tag = "[Q]" if mode == "quiet" else "[E]"
            if mode == "quiet":
                quiet_days += 1
            else:
                llm_calls += 1
            events_str = day_result.get("_events_summary", "")
            ecog = day_result.get("objective", {}).get("ecog", "?")
            mort_risk = day_result.get("_mortality_risk", 0)
            extra_info = f" mort={mort_risk:.3f}" if mort_risk > 0.001 else ""
            day_log = f'''{pid} Day {day:3d} C{cycle}D{cycle_day:2d} {mode_tag} [{location}] AE:[{ae_str}] ECOG:{ecog}{extra_info} {events_str}'''
            print(f"  {day_log}")
            if events_str or day == 1 or day % 21 == 0:
                self.log(day_log)

            # 사망/중도탈락 처리
            ds = day_result.get("ds_record")
            if location == "DECEASED":
                print(f"  ✖ Patient deceased on Day {day}")
                break
            if ds and ds.get("DSDECOD") not in ("DEATH",):
                cause = ds.get("DSDECOD", "?")
                print(f"  ✖ Discontinued: {cause} on Day {day}")
                remaining = total_days - day
                if remaining > 0:
                    fu_count = min(remaining, 30)
                    for fu_day in range(day + 1, day + 1 + fu_count):
                        fu_cycle, fu_cycle_day = _get_cycle_info(fu_day, cycle_length)
                        fu_hospital = _is_hospital_day(fu_day, cycle_length, admin_cycle_days)
                        fu_is_admin = simulator.is_administration_day(fu_cycle_day)
                        fu_result = self._generate_day_with_retry(
                            simulator, day_results,
                            fu_day, fu_cycle, fu_cycle_day, fu_hospital,
                        )
                        fu_is_visit, fu_observed = observation_model.process_day(
                            day=fu_day, day_result=fu_result, is_hospital=fu_hospital,
                            is_admin_day=fu_is_admin, simulator=simulator,
                        )
                        if fu_is_visit:
                            fu_obs_aes = fu_observed.get('objective', {}).get('active_aes', [])
                            fu_dose = simulator.apply_hospital_dose_modifications(
                                fu_obs_aes, fu_day, fu_cycle, fu_cycle_day,
                            )
                            if fu_dose:
                                simulator.patch_day_treatment_status(fu_result)
                                fu_ts = day_result.get('objective', {}).get('treatment_status', 'active')
                                observation_model.update_treatment_status(fu_ts)
                        fu_result["care_record"] = []
                        fu_result["cm_records"] = simulator._get_active_cm_records(fu_day)
                        fu_result["hospital_record"] = fu_observed.get("hospital_record")
                        fu_result["observation_events"] = fu_observed.get("observation_events")
                        fu_result["mood_state"] = fu_observed.get("mood_state")
                        day_results.append(fu_result)
                        if save:
                            cdash_fu = map_day_record(fu_result, patient)
                            with open(sim_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(cdash_fu, ensure_ascii=False) + "\n")
                            hr_fu = extract_hospital_record(cdash_fu)
                            with open(hr_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(hr_fu, ensure_ascii=False) + "\n")
                        # follow-up 중 사망 체크
                        if fu_result.get("objective", {}).get("location") == "DECEASED":
                            print(f"  ✖ Patient deceased during follow-up on Day {fu_day}")
                            break
                break

        self._print_patient_summary(pid, simulator, day_results, llm_calls, quiet_days)
        return day_results

    def run_care_ai(self, patient, total_days, save=True):
        '''Care AI 개입 시뮬레이션.

        Natural과 동일한 생물학적 시뮬레이션에 매일 영상통화를 추가.
        Care AI가 감지한 이상에 대해 개입(dose hold, conmed 등)을 수행.
        '''
        pid = patient['patient_id']
        cycle_length = self.cycle_length
        if self.rule_set:
            cycle_length = self.rule_set.get('trial_design', {}).get('cycle_length_days', self.cycle_length)
        print(f'''\n[Simulation] {pid} — Care AI mode, {total_days} days''')
        patient_num = int(pid.split('-')[1]) if '-' in pid else 1
        sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + 10000)
        simulator = create_simulator(
            rule_set=self.rule_set,
            patient=patient,
            sampler=sampler,
            model=self.model,
            actual_duration=total_days,
        )
        persona = patient.get('persona', {})
        persona_type = persona.get('type', 'minimizer')
        mood = MoodState(persona_type=persona_type, seed=(self.seed or 0) + 200 + patient_num + 20000)
        care_sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + 30000)
        obs_sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + 40000)
        care_agent = CareAgent(patient=patient, rule_set=self.rule_set, mood=mood, sampler=care_sampler, model=self.model)
        observation_model = ObservationModel(mood=mood, sampler=obs_sampler, care_ai_enabled=True)
        all_admin_days = set()
        for drug in simulator.admin_schedule:
            all_admin_days.update(drug.get('cycle_days', [1]))
        admin_cycle_days = sorted(all_admin_days)
        day_results = []
        sim_path = self.data_dir / 'simulations' / f'''{pid}_care_ai.jsonl'''
        hr_path = self.data_dir / 'simulations' / f'''{pid}_care_ai_hospital.jsonl'''
        if save:
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            sim_path.write_text('', encoding='utf-8')
            hr_path.write_text('', encoding='utf-8')
        llm_calls = 0
        quiet_days = 0
        care_interventions = 0
        force_hospital_tomorrow = False
        last_forced_hospital_day = -999
        HOSPITAL_COOLDOWN_DAYS = 3

        # Day 0: pre-treatment baseline from EMR
        day0 = _build_day0_record(patient, simulator)
        day_results.append(day0)
        if save:
            cdash_day0 = map_day_record(day0, patient)
            with open(sim_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cdash_day0, ensure_ascii=False) + '\n')

        for day in range(1, total_days + 1):
            cycle, cycle_day = _get_cycle_info(day, cycle_length)
            is_hospital = _is_hospital_day(day, cycle_length, admin_cycle_days)
            is_admin_day = simulator.is_administration_day(cycle_day)

            # Care AI 강제 내원
            if force_hospital_tomorrow and (day - last_forced_hospital_day) >= HOSPITAL_COOLDOWN_DAYS:
                is_hospital = True
                force_hospital_tomorrow = False
                last_forced_hospital_day = day

            day_result = self._generate_day_with_retry(
                simulator, day_results, day, cycle, cycle_day, is_hospital,
            )

            # Care AI 영상통화 (매일)
            care_result = care_agent.conduct_video_call(
                day=day, day_result=day_result, day_results=day_results,
                last_hospital_record=_find_last_hospital_record(day_results),
            )
            day_result["care_record"] = [care_result]

            # Care AI 개입 적용: conmed 추가 등
            applied = apply_interventions(care_result, simulator, day)
            if applied:
                care_interventions += len(applied)

            # Care AI가 병원 방문을 권고했는지 확인 (actions 리스트 내부 확인)
            _escalation_actions = {"recommend_early_visit", "recommend_hospital_visit", "escalate_to_physician"}
            if any(a.get("action") in _escalation_actions for a in care_result.get("actions", [])):
                force_hospital_tomorrow = True

            # Observation model: GT → Hospital Record 변환
            is_visit, observed = observation_model.process_day(
                day=day, day_result=day_result, is_hospital=is_hospital,
                is_admin_day=is_admin_day, simulator=simulator,
                care_record=[care_result] if care_result else None,
            )

            # 병원 방문일: HR 기반 dose modification → 투약 → conmed 처방
            if is_visit:
                observed_aes = observed.get('objective', {}).get('active_aes', [])
                dose_changes = simulator.apply_hospital_dose_modifications(
                    observed_aes, day, cycle, cycle_day,
                )
                # 투약 전 AE 평가 완료 → 이제 투약 시도 (held면 자동 skip)
                simulator._process_drug_administration(day, cycle_day)
                # EC 레코드 재생성 (dose modification 결과 반영)
                simulator._enrich_ec_records(day_result, day, cycle_day, is_hospital)
                # 치료 상태 업데이트
                simulator.patch_day_treatment_status(day_result)
                new_ts = day_result.get('objective', {}).get('treatment_status', 'active')
                observation_model.update_treatment_status(new_ts)
                # 관찰된 AE에 대해 보조약 처방
                simulator.prescribe_conmeds_for_aes(observed_aes, day)
                # cm_records 갱신
                day_result["cm_records"] = simulator._get_active_cm_records(day)
                # active_aes를 re-sync (AEACN 귀인 반영)
                day_result["objective"]["active_aes"] = simulator._get_active_aes_list(include_resolved_today=True)

            # Observation model 결과를 day_result에 merge
            day_result["hospital_record"] = observed.get("hospital_record")
            day_result["observation_events"] = observed.get("observation_events")
            day_result["mood_state"] = observed.get("mood_state")

            day_results.append(day_result)

            if save:
                cdash_record = map_day_record(day_result, patient)
                with open(sim_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cdash_record, ensure_ascii=False) + "\n")
                hr_record = extract_hospital_record(cdash_record)
                with open(hr_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(hr_record, ensure_ascii=False) + "\n")

            # 통계 추적
            mode = day_result.get("_generation_mode", "?")
            aes = day_result.get("objective", {}).get("active_aes", [])
            ae_str = ", ".join(f'''{a.get('ae', '')} G{a.get('grade', '?')}''' for a in aes) if aes else "none"
            location = day_result.get("objective", {}).get("location", "?")
            mode_tag = "[Q]" if mode == "quiet" else "[E]"
            if mode == "quiet":
                quiet_days += 1
            else:
                llm_calls += 1
            events_str = day_result.get("_events_summary", "")
            ecog = day_result.get("objective", {}).get("ecog", "?")
            mort_risk = day_result.get("_mortality_risk", 0)
            extra_info = f" mort={mort_risk:.3f}" if mort_risk > 0.001 else ""
            print(f'''  Day {day:3d} C{cycle}D{cycle_day:2d} {mode_tag} [{location}] AE:[{ae_str}] ECOG:{ecog}{extra_info} {events_str}''')

            # 사망/중도탈락 처리
            ds = day_result.get("ds_record")
            if location == "DECEASED":
                print(f"  ✖ Patient deceased on Day {day}")
                break
            if ds and ds.get("DSDECOD") not in ("DEATH",):
                dsdecod = ds.get("DSDECOD", "?")
                print(f"  ✖ Discontinued: {dsdecod} on Day {day}")
                remaining = total_days - day
                if remaining > 0:
                    fu_count = min(remaining, 30)
                    for fu_day in range(day + 1, day + 1 + fu_count):
                        fu_cycle, fu_cycle_day = _get_cycle_info(fu_day, cycle_length)
                        fu_hospital = _is_hospital_day(fu_day, cycle_length, admin_cycle_days)
                        fu_result = self._generate_day_with_retry(
                            simulator, day_results,
                            fu_day, fu_cycle, fu_cycle_day, fu_hospital,
                        )
                        fu_care = care_agent.conduct_video_call(
                            day=fu_day, day_result=fu_result, day_results=day_results,
                            last_hospital_record=_find_last_hospital_record(day_results),
                        )
                        fu_result["care_record"] = [fu_care]
                        day_results.append(fu_result)
                        if save:
                            cdash_fu = map_day_record(fu_result, patient)
                            with open(sim_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(cdash_fu, ensure_ascii=False) + "\n")
                            hr_fu = extract_hospital_record(cdash_fu)
                            with open(hr_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(hr_fu, ensure_ascii=False) + "\n")
                        if fu_result.get("objective", {}).get("location") == "DECEASED":
                            print(f"  ✖ Patient deceased during follow-up on Day {fu_day}")
                            break
                break

        self._print_patient_summary(pid, simulator, day_results, llm_calls, quiet_days)
        return day_results

    def _generate_day_with_retry(self, simulator, day_results, day, cycle, cycle_day, is_hospital):
        '''DailySimulator 호출 + 재시도.'''
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = simulator.generate_day(day_results, day, cycle, cycle_day, is_hospital)
                if isinstance(result, list):
                    _logger.warning(f'''Day {day}: generate_day returned list, using first element''')
                    result = result[0] if result else {}
                if not isinstance(result, dict):
                    _logger.warning(f'''Day {day}: generate_day returned {type(result).__name__}, wrapping''')
                    result = {}
                return result
            except Exception as e:
                last_error = e
                _tb = f'''{type(e).__name__}: {e}'''
                pid = simulator.patient.get('patient_id', '?')
                _logger.error(f'''[{pid}] Day {day} attempt {attempt + 1} failed: {_tb}''')
                if attempt < self.max_retries:
                    time.sleep(1)
        raise RuntimeError(f'''Day {day} failed after {self.max_retries + 1} attempts: {last_error}''')

    def _print_patient_summary(self, pid, simulator, day_results, llm_calls, quiet_days):
        '''시뮬레이션 완료 후 환자별 요약.'''
        print(f'''\n  --- {pid} Summary ---''')
        print(f'''  Tumor response: {simulator.tumor_response} (onset ~Day {simulator.response_onset_day})''')
        print(f'''  AEs occurred: {list(simulator.occurred_aes.keys()) or 'none'}''')
        print(f'''  AEs resolved: {list(simulator.resolved_aes) or 'none'}''')
        print(f'''  Final ECOG: {simulator.current_ecog} (baseline: {simulator.baseline_ecog})''')
        if simulator.ds_record:
            ds = simulator.ds_record
            print(f'''  Disposition: {ds.get('DSDECOD', '?')} on Day {ds.get('DSSTDTC', '?')}''')
            print(f'''    Reason: {ds.get('DSTERM', '?')}''')
        elif simulator.is_deceased:
            print(f'''  Disposition: DEATH on Day {simulator.death_day}''')
        else:
            print(f'''  Disposition: ON STUDY (completed {len(day_results)} days)''')
        print(f'''  LLM calls: {llm_calls}, Quiet days: {quiet_days} ({(quiet_days / max(len(day_results), 1)) * 100:.0f}% quiet)''')

    def _run_single_patient(self, patient, total_days, mode='natural', save=True, verbose=False):
        '''단일 환자를 시뮬레이션하고 결과 요약 dict를 반환한다 (스레드 안전).

        Returns:
            {pid, results, llm_calls, quiet_days, summary_lines, error}
        '''
        # 부모 스레드의 API 키를 워커 스레드에 전파
        from sim.agents.llm_client import get_api_key, set_api_key as _set_key
        if not get_api_key() and hasattr(self, '_api_key') and self._api_key:
            _set_key(self._api_key)
        pid = patient['patient_id']
        lines = []
        try:
            results = self._run_patient_silent(
                patient, total_days, mode, save, lines, verbose,
            )
            total_aes = 0
            max_grade = 0
            for r in results:
                for ae in r.get('objective', {}).get('active_aes', []):
                    total_aes += 1
                    g = ae.get('grade', 0)
                    if g > max_grade:
                        max_grade = g
            llm_calls = sum(1 for r in results if r.get('_generation_mode') != 'quiet')
            quiet_days = sum(1 for r in results if r.get('_generation_mode') == 'quiet')
            final_status = results[-1].get('objective', {}).get('location', '?') if results else '?'
            return {
                'pid': pid,
                'results': results,
                'llm_calls': llm_calls,
                'quiet_days': quiet_days,
                'summary_lines': lines,
                'error': None,
            }
        except Exception as e:
            import traceback
            _logger.error(f'''[{pid}] Simulation failed: {e}\n{traceback.format_exc()}''')
            return {
                'pid': pid,
                'results': [],
                'llm_calls': 0,
                'quiet_days': 0,
                'summary_lines': lines,
                'error': str(e),
            }

    def _run_patient_silent(self, patient, total_days, mode, save, lines, verbose=False):
        '''단일 환자 시뮬레이션 (print 대신 lines 버퍼에 쓰기).'''
        pid = patient['patient_id']
        cycle_length = self.cycle_length
        if self.rule_set:
            cycle_length = self.rule_set.get('trial_design', {}).get('cycle_length_days', self.cycle_length)
        patient_num = int(pid.split('-')[1]) if '-' in pid else 1
        seed_offset = 0 if mode == 'natural' else 10000
        sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + seed_offset)
        simulator = create_simulator(
            rule_set=self.rule_set,
            patient=patient,
            sampler=sampler,
            model=self.model,
            actual_duration=total_days,
        )
        persona = patient.get('persona', {})
        persona_type = persona.get('type', 'minimizer')
        mood = MoodState(
            persona_type=persona_type,
            seed=(self.seed or 0) + 200 + patient_num + (20000 if mode == 'care_ai' else 30000),
        )
        obs_sampler = Sampler(
            seed=(self.seed or 0) + 200 + patient_num + (40000 if mode == 'care_ai' else 50000),
        )
        observation_model = ObservationModel(mood=mood, sampler=obs_sampler, care_ai_enabled=(mode == 'care_ai'))
        care_agent = None
        if mode == 'care_ai':
            care_sampler = Sampler(seed=(self.seed or 0) + 200 + patient_num + 30000)
            care_agent = CareAgent(patient=patient, rule_set=self.rule_set, mood=mood, sampler=care_sampler, model=self.model)
        all_admin_days = set()
        for drug in simulator.admin_schedule:
            all_admin_days.update(drug.get('cycle_days', [1]))
        admin_cycle_days = sorted(all_admin_days)
        day_results = []
        sim_path = self.data_dir / 'simulations' / f'''{pid}_{mode}.jsonl'''
        hr_path = self.data_dir / 'simulations' / f'''{pid}_{mode}_hospital.jsonl'''
        if save:
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            sim_path.write_text('', encoding='utf-8')
            hr_path.write_text('', encoding='utf-8')
        force_hospital_tomorrow = False
        last_forced_hospital_day = -999
        HOSPITAL_COOLDOWN_DAYS = 3

        # Day 0: pre-treatment baseline from EMR
        day0 = _build_day0_record(patient, simulator)
        day_results.append(day0)
        if save:
            cdash_day0 = map_day_record(day0, patient)
            with open(sim_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cdash_day0, ensure_ascii=False) + '\n')

        for day in range(1, total_days + 1):
            if self._cancelled:
                lines.append(f"  ⛔ {pid} — Cancelled at Day {day}")
                self.log(f"⛔ {pid} — Cancelled at Day {day}")
                break

            cycle, cycle_day = _get_cycle_info(day, cycle_length)
            is_hospital = _is_hospital_day(day, cycle_length, admin_cycle_days)
            is_admin_day = simulator.is_administration_day(cycle_day)

            if care_agent and force_hospital_tomorrow and (day - last_forced_hospital_day) >= HOSPITAL_COOLDOWN_DAYS:
                is_hospital = True
                force_hospital_tomorrow = False
                last_forced_hospital_day = day

            day_result = self._generate_day_with_retry(
                simulator, day_results, day, cycle, cycle_day, is_hospital,
            )

            # Care AI 영상통화
            if care_agent:
                care_result = care_agent.conduct_video_call(
                    day=day, day_result=day_result, day_results=day_results,
                    last_hospital_record=_find_last_hospital_record(day_results),
                )
                day_result["care_record"] = [care_result]
                # Care AI 개입 적용: conmed 추가 등
                apply_interventions(care_result, simulator, day)
                # 병원 방문 권고 확인
                _esc = {"recommend_early_visit", "recommend_hospital_visit", "escalate_to_physician"}
                if any(a.get("action") in _esc for a in care_result.get("actions", [])):
                    force_hospital_tomorrow = True
            else:
                day_result["care_record"] = []

            # Observation model
            is_visit, observed = observation_model.process_day(
                day=day, day_result=day_result, is_hospital=is_hospital,
                is_admin_day=is_admin_day, simulator=simulator,
                care_record=day_result.get("care_record"),
            )

            # 병원 방문일: HR 기반 dose modification → 투약 → conmed 처방
            if is_visit:
                observed_aes = observed.get('objective', {}).get('active_aes', [])
                dose_changes = simulator.apply_hospital_dose_modifications(
                    observed_aes, day, cycle, cycle_day,
                )
                # 투약 전 AE 평가 완료 → 이제 투약 시도 (held면 자동 skip)
                simulator._process_drug_administration(day, cycle_day)
                # EC 레코드 재생성 (dose modification 결과 반영)
                simulator._enrich_ec_records(day_result, day, cycle_day, is_hospital)
                # 치료 상태 업데이트
                simulator.patch_day_treatment_status(day_result)
                new_ts = day_result.get('objective', {}).get('treatment_status', 'active')
                observation_model.update_treatment_status(new_ts)
                # 관찰된 AE에 대해 보조약 처방
                simulator.prescribe_conmeds_for_aes(observed_aes, day)
                day_result["cm_records"] = simulator._get_active_cm_records(day)
                # active_aes를 re-sync (AEACN 귀인 반영)
                day_result["objective"]["active_aes"] = simulator._get_active_aes_list(include_resolved_today=True)

            # Observation model 결과를 day_result에 merge
            day_result["hospital_record"] = observed.get("hospital_record")
            day_result["observation_events"] = observed.get("observation_events")
            day_result["mood_state"] = observed.get("mood_state")

            day_results.append(day_result)

            if save:
                cdash_record = map_day_record(day_result, patient)
                with open(sim_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cdash_record, ensure_ascii=False) + "\n")
                hr_record = extract_hospital_record(cdash_record)
                with open(hr_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(hr_record, ensure_ascii=False) + "\n")

            # 로그 (이벤트 있는 날만)
            events_str = day_result.get("_events_summary", "")
            gen_mode = day_result.get("_generation_mode", "?")
            aes = day_result.get("objective", {}).get("active_aes", [])
            ae_str = ", ".join(f'''{a.get('ae', '')} G{a.get('grade', '?')}''' for a in aes) if aes else ""
            ecog = day_result.get("objective", {}).get("ecog", "?")
            if events_str or day == 1 or day % 21 == 0:
                tag = "[Q]" if gen_mode == "quiet" else "[E]"
                log_msg = f"{pid} Day {day} C{cycle}D{cycle_day} {tag} AE:[{ae_str}] ECOG:{ecog}"
                if events_str:
                    log_msg += f" {events_str}"
                self.log(log_msg)
            self.update_progress(pid, day, total_days)

            # 사망/중도탈락 처리
            location = day_result.get("objective", {}).get("location", "?")
            ds = day_result.get("ds_record")

            if location == "DECEASED":
                self.log(f"☠ {pid} — Deceased on Day {day}")
                lines.append(f"  ✖ Deceased on Day {day}")
                break

            if ds and ds.get("DSDECOD") not in ("DEATH",):
                dsdecod = ds.get("DSDECOD", "?")
                self.log(f"🚫 {pid} — Discontinued ({dsdecod}) on Day {day}")
                lines.append(f"  ✖ Discontinued: {dsdecod} on Day {day}")
                remaining = total_days - day
                if remaining > 0:
                    fu_count = min(remaining, 30)
                    for fu_day in range(day + 1, day + 1 + fu_count):
                        fu_cycle, fu_cycle_day = _get_cycle_info(fu_day, cycle_length)
                        fu_hospital = _is_hospital_day(fu_day, cycle_length, admin_cycle_days)
                        fu_result = self._generate_day_with_retry(
                            simulator, day_results,
                            fu_day, fu_cycle, fu_cycle_day, fu_hospital,
                        )
                        if care_agent:
                            fu_care = care_agent.conduct_video_call(
                                day=fu_day, day_result=fu_result, day_results=day_results,
                                last_hospital_record=_find_last_hospital_record(day_results),
                            )
                            fu_result["care_record"] = [fu_care]
                        else:
                            fu_result["care_record"] = []
                        day_results.append(fu_result)
                        if save:
                            cdash_fu = map_day_record(fu_result, patient)
                            with open(sim_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(cdash_fu, ensure_ascii=False) + "\n")
                        if fu_result.get("objective", {}).get("location") == "DECEASED":
                            lines.append(f"  ✖ Deceased during follow-up on Day {fu_day}")
                            break
                break

        # 환자 요약 라인
        aes_list = [ae.get('ae', '') for ae in day_results[-1].get('objective', {}).get('active_aes', [])] if day_results else []
        lines.append(f"  {pid}: {len(day_results)} days, AEs: {aes_list or 'none'}, ECOG: {simulator.current_ecog}")
        return day_results

    def run_parallel(self, patients, total_days, mode='natural', max_workers=10, save=True):
        '''여러 환자를 병렬로 시뮬레이션한다.

        Args:
            patients: 환자 리스트
            total_days: 시뮬레이션 기간
            mode: "natural" 또는 "care_ai"
            max_workers: 최대 동시 실행 수
            save: 결과를 파일에 저장할지
        Returns:
            {pid: day_results} dict
        '''
        n = len(patients)
        actual_workers = min(max_workers, n)
        self.log(f"Phase 2: {mode.upper()} — {n} patients × {total_days} days ({actual_workers} workers)")
        print(f'''\n  🚀 Parallel simulation: {n} patients × {total_days} days, {actual_workers} workers ({mode})''')
        print(f'''  {'──────────────────────────────────────────────────'}''')
        all_results = {}
        errors = []
        completed_count = [0]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(self._run_single_patient, patient, total_days, mode, save): patient['patient_id'] for patient in patients}
            for future in as_completed(futures):
                pid = futures[future]
                outcome = future.result()
                all_results[outcome['pid']] = outcome['results']
                completed_count[0] += 1
                n_days_done = len(outcome['results'])
                if outcome.get('error'):
                    errors.append(f'''{pid}: {outcome['error']}''')
                    self.log(f"❌ {pid} failed: {outcome['error']}")
                else:
                    self.log(f"✅ {pid} done ({n_days_done}d) [{completed_count[0]}/{n}]")
                self.update_progress(pid, n_days_done, total_days, 'done')
                with _print_lock:
                    for line in outcome.get('summary_lines', []):
                        print(line)
        elapsed = time.time() - t0
        self.log(f"🏁 {mode.upper()} complete: {n} patients in {elapsed:.1f}s")
        print(f'''\n  ✅ {mode} simulation complete: {n} patients in {elapsed:.1f}s ({elapsed / max(n, 1):.1f}s/patient)''')
        if errors:
            self.log(f"⚠ {len(errors)} errors during {mode}")
            print(f'''  ⚠️ {len(errors)} errors:''')
            for err in errors:
                print(f'''    - {err}''')
        return all_results
