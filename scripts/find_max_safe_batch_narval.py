from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = [
    "1dmrs:deep_d96_b6_r2",
    "1dmrs:wide_d128_b4_r2",
    "1dmrs:mlpwide_d96_b4_r4",
    "2dmrs:deep_d96_b6_r2",
    "2dmrs:wide_d128_b4_r2",
    "2dmrs:mlpwide_d96_b4_r4",
]


def _parse_int_list(text: str, search_order: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one batch size is required.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Batch sizes must be positive: {values}")
    values = list(dict.fromkeys(values))
    if search_order == "ascending":
        return sorted(values)
    if search_order == "descending":
        return sorted(values, reverse=True)
    return values


def _parse_cases(text: str) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected dmrs_case:variant, got {item!r}")
        dmrs_case, variant = item.split(":", 1)
        cases.append((dmrs_case.strip(), variant.strip()))
    if not cases:
        raise ValueError("At least one probe case is required.")
    return cases


def _classify_failure(text: str, returncode: int) -> str:
    lower = text.lower()
    if "resourceexhausted" in lower or "out of memory" in lower or " oom " in lower:
        return "oom"
    if returncode < 0:
        return f"signal_{abs(returncode)}"
    return "failed"


def _run_case(
    *,
    phase: str,
    config: str,
    batch_size: int,
    dmrs_case: str,
    variant: str,
    steps: int,
    eval_every: int,
    val_steps: int,
    log_dir: Path,
    timeout_s: int | None,
    eval_num_users: int,
    eval_ebno_points: str,
    eval_num_batches: int,
    eval_cov_batches: int,
    eval_receiver_microbatch_size: int,
    num_rx_ant: int | None,
) -> dict[str, Any]:
    rx_tag = f"rx{num_rx_ant}" if num_rx_ant is not None else "rx_config"
    tag = f"{phase}_{rx_tag}_bs{batch_size}_{dmrs_case}_{variant}"
    stdout_path = log_dir / f"{tag}.out"
    stderr_path = log_dir / f"{tag}.err"
    if phase == "training":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "probe_training_memory_narval.py"),
            "--config",
            config,
            "--dmrs-case",
            dmrs_case,
            "--variant",
            variant,
            "--batch-size",
            str(batch_size),
            "--steps",
            str(steps),
            "--eval-every",
            str(eval_every),
            "--val-steps",
            str(val_steps),
        ]
        if num_rx_ant is not None:
            cmd.extend(["--num-rx-ant", str(num_rx_ant)])
    elif phase == "evaluation":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "probe_evaluation_memory_narval.py"),
            "--config",
            config,
            "--dmrs-case",
            dmrs_case,
            "--variant",
            variant,
            "--batch-size",
            str(batch_size),
            "--num-users",
            str(eval_num_users),
            "--ebno-points",
            str(eval_ebno_points),
            "--num-batches",
            str(eval_num_batches),
            "--cov-batches",
            str(eval_cov_batches),
        ]
        if eval_receiver_microbatch_size > 0:
            cmd.extend(["--receiver-microbatch-size", str(eval_receiver_microbatch_size)])
        if num_rx_ant is not None:
            cmd.extend(["--num-rx-ant", str(num_rx_ant)])
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    start = time.monotonic()
    print(f"[BATCHPROBE] start phase={phase} batch={batch_size} dmrs_case={dmrs_case} variant={variant}", flush=True)
    with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_s,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            returncode = 124
            stderr.write(f"\n[BATCHPROBE] timed out after {timeout_s} seconds\n")
    elapsed_s = time.monotonic() - start
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    combined = stdout_text + "\n" + stderr_text
    ok = returncode == 0
    reason = "passed" if ok else _classify_failure(combined, returncode)
    print(
        f"[BATCHPROBE] done phase={phase} batch={batch_size} dmrs_case={dmrs_case} variant={variant} "
        f"status={reason} returncode={returncode} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )
    return {
        "phase": phase,
        "batch_size": int(batch_size),
        "dmrs_case": dmrs_case,
        "variant": variant,
        "passed": bool(ok),
        "reason": reason,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 3),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the largest Narval-safe UPAIR training/evaluation batch sizes.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--batch-sizes", default="64,32,16,8,4")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--phase", choices=["training", "evaluation", "both"], default="both")
    parser.add_argument("--search-order", choices=["ascending", "descending", "given"], default="descending")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--eval-every", type=int, default=175)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--continue-after-fail", action="store_true")
    parser.add_argument("--continue-after-pass", action="store_true")
    parser.add_argument("--per-case-timeout-min", type=float, default=0.0)
    parser.add_argument("--eval-num-users", type=int, default=4)
    parser.add_argument("--eval-ebno-points", default="0,8")
    parser.add_argument("--eval-num-batches", type=int, default=32)
    parser.add_argument("--eval-cov-batches", type=int, default=2)
    parser.add_argument("--eval-receiver-microbatch-size", type=int, default=4)
    parser.add_argument("--num-rx-ant", type=int, default=16, help="Override channel.num_rx_ant for the probe. Use 0 to keep the config value.")
    args = parser.parse_args()

    batch_sizes = _parse_int_list(args.batch_sizes, search_order=args.search_order)
    cases = _parse_cases(args.cases)
    timeout_s = None if args.per_case_timeout_min <= 0 else int(args.per_case_timeout_min * 60)
    phases = ["training", "evaluation"] if args.phase == "both" else [args.phase]
    num_rx_ant = None if int(args.num_rx_ant) <= 0 else int(args.num_rx_ant)

    probe_name = f"batch_probe_rx{num_rx_ant}" if num_rx_ant is not None else "batch_probe"
    output_root = PROJECT_ROOT / "TWC_plots_comprehensive" / probe_name
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    safe_batch_by_phase: dict[str, int | None] = {}

    for phase in phases:
        phase_safe_batch: int | None = None
        for batch_size in batch_sizes:
            batch_passed = True
            for dmrs_case, variant in cases:
                result = _run_case(
                    phase=phase,
                    config=args.config,
                    batch_size=batch_size,
                    dmrs_case=dmrs_case,
                    variant=variant,
                    steps=int(args.steps),
                    eval_every=int(args.eval_every),
                    val_steps=int(args.val_steps),
                    log_dir=log_dir,
                    timeout_s=timeout_s,
                    eval_num_users=int(args.eval_num_users),
                    eval_ebno_points=str(args.eval_ebno_points),
                    eval_num_batches=int(args.eval_num_batches),
                    eval_cov_batches=int(args.eval_cov_batches),
                    eval_receiver_microbatch_size=int(args.eval_receiver_microbatch_size),
                    num_rx_ant=num_rx_ant,
                )
                results.append(result)
                if not result["passed"]:
                    batch_passed = False
                    if not args.continue_after_fail:
                        break
            if batch_passed:
                if phase_safe_batch is None or int(batch_size) > phase_safe_batch:
                    phase_safe_batch = int(batch_size)
                if args.search_order == "descending" and not args.continue_after_pass:
                    break
            elif args.search_order in {"ascending", "given"} and not args.continue_after_fail:
                break
        safe_batch_by_phase[phase] = phase_safe_batch

    if all(safe_batch_by_phase[phase] is not None for phase in phases):
        safe_batch = min(int(safe_batch_by_phase[phase]) for phase in phases)
    else:
        safe_batch = None

    summary = {
        "safe_batch_size": safe_batch,
        "safe_batch_size_by_phase": safe_batch_by_phase,
        "batch_sizes_tested": batch_sizes,
        "cases": [f"{dmrs_case}:{variant}" for dmrs_case, variant in cases],
        "phase": args.phase,
        "phases_run": phases,
        "search_order": args.search_order,
        "num_rx_ant": num_rx_ant,
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "val_steps": int(args.val_steps),
        "eval_num_users": int(args.eval_num_users),
        "eval_ebno_points": str(args.eval_ebno_points),
        "eval_num_batches": int(args.eval_num_batches),
        "eval_cov_batches": int(args.eval_cov_batches),
        "eval_receiver_microbatch_size": int(args.eval_receiver_microbatch_size),
        "results": results,
    }
    summary_path = output_root / "batch_probe_summary.json"
    csv_path = output_root / "batch_probe_results.csv"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phase", "batch_size", "dmrs_case", "variant", "passed", "reason", "returncode", "elapsed_s", "stdout", "stderr"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"[BATCHPROBE] safe_batch_size={safe_batch}")
    for phase in phases:
        print(f"[BATCHPROBE] safe_batch_size_{phase}={safe_batch_by_phase[phase]}")
    print(f"[BATCHPROBE] wrote {summary_path}")
    print(f"[BATCHPROBE] wrote {csv_path}")
    if safe_batch is None:
        raise SystemExit("No candidate batch size passed all selected cases.")


if __name__ == "__main__":
    main()
