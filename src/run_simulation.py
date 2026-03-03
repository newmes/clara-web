"""메인 실행 스크립트 — Fate table 없는 확률적 시뮬레이션 엔진

3-Phase 아키텍처:
  Phase 0: Rule Agent → rule_set (약물당 1회)
  Phase 1: Patient Agent → N명 환자 (LLM→rand→LLM)
  Phase 2: Daily Agent → 일별 시뮬레이션 (hazard function 기반, fate table 없음)

Usage:
    # 환자 1명 × 21일 quick test
    python src/run_simulation.py --patients 1 --days 21

    # 환자 10명 × Phase 1만 (환자 생성까지만)
    python src/run_simulation.py --patients 10 --patients-only

    # 기존 규칙 재사용 + 환자 3명 × 142일
    python src/run_simulation.py --patients 3 --days 142 --skip-rules

    # 다른 약물로 확장
    python src/run_simulation.py --drug "Ozempic" --indication "type 2 diabetes" --patients 5 --days 90

    # 시드 고정 (재현 가능)
    python src/run_simulation.py --patients 10 --days 42 --seed 42
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# .env 로드
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set.")
    print("Set it in .env or as environment variable.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import SimulationRunner
from src.logger import log_summary, get_log_file_path


class _TeeWriter:
    """stdout을 터미널과 로그 파일에 동시에 출력한다."""

    def __init__(self, terminal, log_file):
        self._terminal = terminal
        self._log_file = log_file

    def write(self, text):
        self._terminal.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._terminal.flush()
        self._log_file.flush()


def _make_run_dir(base_dir: str, drug_name: str, n_patients: int, n_days: int) -> Path:
    """실험별 고유 폴더 생성: data/runs/{timestamp}_{drug}_{N}pt_{D}d/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_drug = drug_name.replace(" ", "_").replace("+", "_")[:30]
    run_name = f"{ts}_{safe_drug}_{n_patients}pt_{n_days}d"
    run_dir = Path(base_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Simulation Engine v2 — 3-Phase Probabilistic (No Fate Table)"
    )
    parser.add_argument("--drug", default="Padcev + Pembrolizumab",
                        help="Drug name (default: Padcev + Pembrolizumab)")
    parser.add_argument("--indication", default="metastatic urothelial carcinoma",
                        help="Indication")
    parser.add_argument("--patients", type=int, default=1,
                        help="Number of patients to generate")
    parser.add_argument("--days", type=int, default=21,
                        help="Simulation days per patient (default: 21 = 1 cycle)")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Gemini model ID")
    parser.add_argument("--patients-only", action="store_true",
                        help="Only run Phase 0-1 (Rules + Patients, no daily sim)")
    parser.add_argument("--skip-rules", action="store_true",
                        help="Skip Rule Agent (reuse existing data/rule_set.json)")
    parser.add_argument("--rule-set", default=None,
                        help="Path to a specific rule_set JSON (implies --skip-rules)")
    parser.add_argument("--data-dir", default="data",
                        help="Base data directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--mode", choices=["natural", "care_ai", "both"],
                        default="natural",
                        help="Simulation mode: natural (default), care_ai, or both (A/B comparison)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel workers (default: 5). Set to 1 for sequential.")
    args = parser.parse_args()

    run_dir = _make_run_dir(args.data_dir, args.drug, args.patients, args.days)
    print(f"Run directory: {run_dir}")

    # 콘솔 로그 Tee
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    console_log_path = log_path / f"console_{run_dir.name}.log"
    log_file = open(console_log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout, log_file)

    try:
        runner = SimulationRunner(
            drug_name=args.drug,
            indication=args.indication,
            model=args.model,
            data_dir=str(run_dir),
            seed=args.seed,
        )

        print("=" * 60)
        print("Phase 0: Rule Discovery")
        print(f"Drug: {args.drug}")
        print(f"Indication: {args.indication}")
        print(f"Model: {args.model}")
        print(f"Seed: {args.seed or 'random'}")
        print("Architecture: 3-Phase (no fate table)")
        print("=" * 60)

        if args.rule_set:
            import shutil
            rs_path = Path(args.rule_set)
            runner.load_rules(str(rs_path))
            shutil.copy2(rs_path, run_dir / "rule_set.json")
        elif args.skip_rules:
            base_rule_path = Path(args.data_dir) / "rule_set.json"
            runner.load_rules(str(base_rule_path))
            import shutil
            shutil.copy2(base_rule_path, run_dir / "rule_set.json")
        else:
            runner.discover_rules()

        use_parallel = args.workers > 1
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Generating {args.patients} patients (LLM→rand→LLM)")
        if use_parallel:
            print(f"  Parallel: {args.workers} workers")
        print("=" * 60)

        if use_parallel and args.patients > 1:
            patients = runner.create_patients_parallel(args.patients, max_workers=args.workers)
        else:
            patients = runner.create_patients(args.patients)

        if args.patients_only:
            print(f"\n{'=' * 60}")
            print("Phase 0-1 complete.")
            print(f"  {len(patients)} patients generated.")
            print(f"  Saved to: {run_dir}/")
            print("=" * 60)
            _print_cohort_summary(patients)
            log_summary()
            return

        runner.write_run_meta(args.patients, args.days, args.mode, 'running')

        modes = [args.mode] if args.mode != "both" else ["natural", "care_ai"]

        for mode in modes:
            print(f"\n{'=' * 60}")
            print(f"Phase 2: Daily simulation — {mode.upper()} mode ({args.days} days)")
            print("  Event determination: hazard function (code) + rand")
            print("  Event days: LLM generates detailed state")
            print("  Quiet days: code only (no LLM call)")
            if mode == "care_ai":
                print("  Care AI: daily 4-turn video call (T1:Patient → T2:Nurse → T3:Patient → T4:Nurse)")
            if use_parallel:
                print(f"  Parallel: {args.workers} workers")
            print("=" * 60)

            if use_parallel:
                all_results = runner.run_parallel(
                    patients, total_days=args.days, mode=mode, max_workers=args.workers
                )
            else:
                all_results = {}
                for patient in patients:
                    if mode == "natural":
                        results = runner.run_natural(patient, total_days=args.days)
                    else:
                        results = runner.run_care_ai(patient, total_days=args.days)
                    all_results[patient["patient_id"]] = results

            print(f"\n{'=' * 60}")
            print(f"Simulation Complete — {mode.upper()}")
            print("=" * 60)
            for pid in sorted(all_results.keys()):
                results = all_results[pid]
                total_aes: set[str] = set()
                max_grade = 0
                llm_days = 0
                quiet_days = 0
                for r in results:
                    for ae in r.get("objective", {}).get("active_aes", []):
                        total_aes.add(ae.get("ae", ""))
                        max_grade = max(max_grade, ae.get("grade", 0))
                    if r.get("_generation_mode") == "quiet_day":
                        quiet_days += 1
                    else:
                        llm_days += 1
                final_status = (
                    results[-1].get("objective", {}).get("treatment_status", "?")
                    if results
                    else "?"
                )
                print(
                    f"  {pid}: {len(results)} days, {len(total_aes)} AEs "
                    f"(max G{max_grade}), status={final_status}, "
                    f"LLM calls={llm_days}, quiet={quiet_days}"
                )

        if args.mode == "both":
            print(f"\n{'=' * 60}")
            print("Phase 3: A/B Comparison — Natural vs Care AI")
            print("=" * 60)
            try:
                from src.evaluator import run_evaluation
                comparison = run_evaluation(run_dir)
            except Exception as eval_err:
                print(f"⚠ Evaluation failed (non-fatal): {eval_err}")
                print("  Simulation data is intact. Run evaluation separately if needed.")

        runner.write_run_meta(args.patients, args.days, args.mode, 'completed')

        log_summary()
        print(f"\nAll results saved to: {run_dir}/")
        print(f"Run directory: {run_dir}")

    finally:
        sys.stdout = original_stdout
        log_file.close()


def _print_cohort_summary(patients: list[dict]):
    """코호트 요약 통계를 출력한다."""
    print("\n--- Cohort Summary ---")
    ages = [p.get("emr", {}).get("demographics", {}).get("age", 0) for p in patients]
    sexes = [p.get("emr", {}).get("demographics", {}).get("sex", "?") for p in patients]
    races = [p.get("emr", {}).get("demographics", {}).get("race", "?") for p in patients]
    if ages:
        print(f"  Age: {min(ages)}-{max(ages)} (mean {sum(ages) / len(ages):.0f})")
    print(f"  Sex: {dict((s, sexes.count(s)) for s in set(sexes))}")
    print(f"  Race: {dict((r, races.count(r)) for r in set(races))}")
    all_conds = []
    for p in patients:
        for h in p.get("emr", {}).get("medical_history", []):
            all_conds.append(h.get("condition", ""))
    if all_conds:
        from collections import Counter
        top_conds = Counter(all_conds).most_common(5)
        print(f"  Top comorbidities: {top_conds}")


if __name__ == "__main__":
    main()
