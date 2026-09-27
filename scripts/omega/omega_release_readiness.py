#!/usr/bin/env python3
"""Synthesize Omega Sprint 1 release readiness with hard+rollover semantics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "sounio.omega.sprint1-release-readiness.v1"
PTX_SCHEMA = "sounio.omega.ptx-launch-report.v1"
SASS_SCHEMA = "sounio.omega.sass-patch-report.v2"
FPGA_SCHEMA = "sounio.omega.fpga-seed-report.v1"
QUANTUM_SCHEMA = "sounio.quantum.conformance.v1"
BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"
POLICY_GUARD_SCHEMA = "sounio.omega.policy-mode-guard.v1"
NUM_EQ_SCHEMA = "sounio.omega.numeric-equivalence.v1"
PERF_SCHEMA = "sounio.omega.performance-summary.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Omega Sprint 1 release-readiness verdict artifact",
    )
    parser.add_argument(
        "--contract",
        default="benchmarks/independence/contract.v2.json",
        help="Sprint contract path",
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Optimization policy path",
    )
    parser.add_argument(
        "--ptx-report",
        default="artifacts/ptx/omega/ptx_launch_report.json",
        help="PTX launch report path",
    )
    parser.add_argument(
        "--sass-report",
        default="artifacts/sass/omega/sass_patch_report.json",
        help="SASS patch report path",
    )
    parser.add_argument(
        "--fpga-report",
        default="artifacts/fpga/fpga_seed_report.json",
        help="FPGA seed report path",
    )
    parser.add_argument(
        "--quantum-report",
        default="artifacts/quantum/omega/quantum_conformance.json",
        help="Quantum conformance report path",
    )
    parser.add_argument(
        "--rl-bridge",
        default="artifacts/omega/rl_readiness_bridge.v1.json",
        help="RL readiness bridge path",
    )
    parser.add_argument(
        "--policy-guard",
        default="artifacts/omega/policy_mode_guard.v1.json",
        help="Policy mode guard artifact path",
    )
    parser.add_argument(
        "--performance-report",
        default="artifacts/omega/performance_summary.v1.json",
        help="Optional performance summary report path",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/sprint1_release_readiness.v1.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require expected schemas for required artifacts",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="Fail when hard gates are not fully green",
    )
    parser.add_argument(
        "--require-success-criteria",
        action="store_true",
        help="Fail when all success criteria are not fully green",
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read {label} ({path}): {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid payload in {label} ({path}): expected object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def normalize_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def normalize_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def evaluate_l1(ptx: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "pass_marker_seen": bool(ptx.get("pass_marker_seen", False)),
        "launch_accepted": bool(ptx.get("launch_accepted", False)),
        "used_fallback": bool(ptx.get("used_fallback", True)),
    }
    reasons: list[str] = []
    if not checks["pass_marker_seen"]:
        reasons.append("ptx pass marker not seen")
    if not checks["launch_accepted"]:
        reasons.append("ptx launch not accepted")
    if checks["used_fallback"]:
        reasons.append("ptx launch used fallback path")
    return {
        "pass": not reasons,
        "checks": checks,
        "reasons": reasons,
    }


def evaluate_l2(sass: dict[str, Any], expected_kernels: list[str]) -> dict[str, Any]:
    kernel_count = normalize_int(sass.get("kernel_count", 0))
    patched_count = normalize_int(sass.get("patched_count", 0))
    eq_pass_count = normalize_int(sass.get("equivalence_pass_count", 0))
    fallback_count = normalize_int(sass.get("fallback_count", 0))
    results = sass.get("results", [])
    if not isinstance(results, list):
        results = []

    expected_count = len(expected_kernels)
    by_name = {
        item.get("name"): item
        for item in results
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    missing = sorted(kernel for kernel in expected_kernels if kernel not in by_name)

    reasons: list[str] = []
    if kernel_count != expected_count:
        reasons.append(f"kernel_count {kernel_count} != expected {expected_count}")
    if patched_count != expected_count:
        reasons.append(f"patched_count {patched_count} != expected {expected_count}")
    if eq_pass_count != expected_count:
        reasons.append(f"equivalence_pass_count {eq_pass_count} != expected {expected_count}")
    if fallback_count != 0:
        reasons.append(f"fallback_count {fallback_count} != 0")
    if missing:
        reasons.append(f"missing kernels in SASS report: {missing}")

    kernel_checks: dict[str, Any] = {}
    for kernel in expected_kernels:
        item = by_name.get(kernel)
        if not isinstance(item, dict):
            continue
        patched = bool(item.get("patched", False))
        fallback = bool(item.get("fallback_to_ptx", True))
        status = str(item.get("status", "unknown"))
        ok = patched and not fallback and status.startswith("patched")
        kernel_checks[kernel] = {
            "patched": patched,
            "fallback_to_ptx": fallback,
            "status": status,
            "pass": ok,
        }
        if not ok:
            reasons.append(f"kernel {kernel} not fully patched/equivalent ({status})")

    return {
        "pass": not reasons,
        "expected_kernels": expected_kernels,
        "counts": {
            "kernel_count": kernel_count,
            "patched_count": patched_count,
            "equivalence_pass_count": eq_pass_count,
            "fallback_count": fallback_count,
        },
        "kernel_checks": kernel_checks,
        "reasons": reasons,
    }


def evaluate_l3(fpga: dict[str, Any]) -> dict[str, Any]:
    if fpga.get("stale"):
        reason = fpga.get("stale_reason", "fpga report marked stale")
        return {
            "pass": False,
            "checks": {},
            "reasons": [f"fpga report is stale: {reason}"],
        }
    required_status = (
        "sim_status",
        "synth_status",
        "k_axi_sim_status",
        "k_axi_synth_status",
        "epistemic_power_accumulator_sim_status",
        "epistemic_power_accumulator_synth_status",
    )
    reasons: list[str] = []
    checks: dict[str, str] = {}
    for field in required_status:
        value = str(fpga.get(field, "missing"))
        checks[field] = value
        if value != "pass":
            reasons.append(f"{field}={value} (expected pass)")
    return {
        "pass": not reasons,
        "checks": checks,
        "reasons": reasons,
    }


def evaluate_bit_exact(
    contract: dict[str, Any],
    sass_report: dict[str, Any],
    sass_report_path: Path,
) -> dict[str, Any]:
    bit_exact = contract.get("bit_exact", {})
    if not isinstance(bit_exact, dict):
        bit_exact = {}

    required = bool(bit_exact.get("required", False))
    min_samples = normalize_int(bit_exact.get("samples", 50), default=50)
    expected_kernels = contract.get("kernel_sets", {}).get("l2_official", [])
    if not isinstance(expected_kernels, list):
        expected_kernels = []
    expected_kernels = [k for k in expected_kernels if isinstance(k, str)]

    if not required:
        return {
            "status": "pass",
            "required": False,
            "reason": "contract bit_exact.required=false",
            "details": [],
        }

    mismatches = normalize_int(sass_report.get("fallback_count", 0))
    per_kernel: list[dict[str, Any]] = []
    reasons: list[str] = []

    for kernel in expected_kernels:
        eq_path = sass_report_path.parent / f"{kernel}.numeric_equivalence.json"
        kernel_result: dict[str, Any] = {
            "kernel": kernel,
            "path": str(eq_path),
            "present": eq_path.exists(),
        }
        if not eq_path.exists():
            reasons.append(f"missing numeric equivalence report for {kernel}")
            per_kernel.append(kernel_result)
            continue

        eq = load_json(eq_path, f"numeric equivalence report ({kernel})")
        kernel_result["schema"] = str(eq.get("schema", ""))
        kernel_result["passed"] = bool(eq.get("passed", False))
        kernel_result["samples"] = normalize_int(eq.get("samples", 0))
        kernel_result["mismatches"] = normalize_int(eq.get("mismatches", 0))
        per_kernel.append(kernel_result)

        if eq.get("schema") != NUM_EQ_SCHEMA:
            reasons.append(
                f"{kernel} numeric equivalence schema mismatch ({eq.get('schema')})"
            )
        if not kernel_result["passed"]:
            reasons.append(f"{kernel} numeric equivalence failed")
        if kernel_result["samples"] < min_samples:
            reasons.append(
                f"{kernel} numeric equivalence samples {kernel_result['samples']} < {min_samples}"
            )
        if kernel_result["mismatches"] != 0:
            reasons.append(
                f"{kernel} numeric equivalence mismatches {kernel_result['mismatches']} != 0"
            )
            mismatches += kernel_result["mismatches"]

    if mismatches != 0:
        reasons.append(f"aggregated bit_exact mismatches {mismatches} != 0")

    status = "pass" if not reasons else "fail"
    return {
        "status": status,
        "required": True,
        "required_samples": min_samples,
        "mismatches": mismatches,
        "details": per_kernel,
        "reasons": reasons,
    }


def evaluate_compile_overhead(
    contract: dict[str, Any],
    bridge: dict[str, Any],
) -> dict[str, Any]:
    budget_cfg = contract.get("compile_time_budget", {})
    if not isinstance(budget_cfg, dict):
        budget_cfg = {}
    max_overhead = normalize_float(budget_cfg.get("max_overhead", 0.25), default=0.25)
    evidence = bridge.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    overhead = normalize_float(evidence.get("compile_overhead", -1.0), default=-1.0)
    if overhead < 0.0:
        return {
            "status": "fail",
            "max_overhead": max_overhead,
            "observed_overhead": overhead,
            "reason": "missing compile_overhead in RL readiness bridge evidence",
        }
    passed = overhead <= max_overhead
    return {
        "status": "pass" if passed else "fail",
        "max_overhead": max_overhead,
        "observed_overhead": overhead,
    }


def evaluate_quantum(
    contract: dict[str, Any],
    quantum: dict[str, Any],
) -> dict[str, Any]:
    cfg = contract.get("quantum_conformance", {})
    if not isinstance(cfg, dict):
        cfg = {}
    ks_min = normalize_float(cfg.get("ks_pvalue_min", 0.05), default=0.05)
    coverage_range = cfg.get("coverage_range", [0.9, 0.98])
    if (
        not isinstance(coverage_range, list)
        or len(coverage_range) != 2
        or not all(isinstance(x, (int, float)) for x in coverage_range)
    ):
        coverage_range = [0.9, 0.98]
    coverage_lo = float(coverage_range[0])
    coverage_hi = float(coverage_range[1])
    min_shots = normalize_int(cfg.get("min_shots", 50), default=50)

    ks = normalize_float(quantum.get("ks_pvalue", -1.0), default=-1.0)
    coverage = normalize_float(quantum.get("coverage_95", -1.0), default=-1.0)
    shots = normalize_int(quantum.get("shots", 0))
    conformant = bool(quantum.get("conformant", False))

    passed = (
        conformant
        and ks >= ks_min
        and coverage >= coverage_lo
        and coverage <= coverage_hi
        and shots >= min_shots
    )
    reasons: list[str] = []
    if not conformant:
        reasons.append("quantum report conformant=false")
    if ks < ks_min:
        reasons.append(f"ks_pvalue {ks:.6f} < ks_pvalue_min {ks_min:.6f}")
    if coverage < coverage_lo or coverage > coverage_hi:
        reasons.append(
            f"coverage_95 {coverage:.6f} outside [{coverage_lo:.6f}, {coverage_hi:.6f}]"
        )
    if shots < min_shots:
        reasons.append(f"shots {shots} < min_shots {min_shots}")
    return {
        "status": "pass" if passed else "fail",
        "ks_pvalue": ks,
        "ks_pvalue_min": ks_min,
        "coverage_95": coverage,
        "coverage_range": [coverage_lo, coverage_hi],
        "shots": shots,
        "min_shots": min_shots,
        "conformant": conformant,
        "reasons": reasons,
    }


def evaluate_rl_governance(
    policy: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    mode = str(policy.get("policy_mode", "unknown"))
    allow_active = bool(guard.get("allow_active", False))
    guard_pass = bool(guard.get("guard_pass", False))
    stable_runs_satisfied = bool(guard.get("stable_runs_satisfied", False))

    reasons: list[str] = []
    if mode == "shadow":
        passed = True
    elif mode == "active":
        passed = guard_pass and allow_active and stable_runs_satisfied
        if not guard_pass:
            reasons.append("policy_mode=active but policy mode guard failed")
        if not allow_active:
            reasons.append("policy_mode=active but allow_active=false")
        if not stable_runs_satisfied:
            reasons.append("policy_mode=active but stable_runs_satisfied=false")
    else:
        passed = False
        reasons.append(f"unsupported policy_mode {mode}")

    return {
        "status": "pass" if passed else "fail",
        "policy_mode": mode,
        "guard_pass": guard_pass,
        "allow_active": allow_active,
        "stable_runs_satisfied": stable_runs_satisfied,
        "reasons": reasons,
    }


def evaluate_geomean(
    contract: dict[str, Any],
    performance_path: Path,
) -> dict[str, Any]:
    perf_cfg = contract.get("performance_gate", {})
    if not isinstance(perf_cfg, dict):
        perf_cfg = {}
    threshold = normalize_float(perf_cfg.get("threshold", 1.2), default=1.2)
    min_samples = normalize_int(perf_cfg.get("minimum_samples", 50), default=50)

    if not performance_path.exists():
        return {
            "status": "pending",
            "threshold": threshold,
            "minimum_samples": min_samples,
            "reason": f"performance report missing ({performance_path})",
        }

    perf = load_json(performance_path, "performance report")
    schema = str(perf.get("schema", ""))
    geomean = normalize_float(perf.get("geomean_speedup", -1.0), default=-1.0)
    samples = normalize_int(perf.get("samples", 0))
    substitutions = perf.get("baseline_substitutions", [])
    if not isinstance(substitutions, list):
        substitutions = []

    reasons: list[str] = []
    if schema != PERF_SCHEMA:
        reasons.append(f"performance schema mismatch ({schema})")
    if geomean < 0.0:
        reasons.append("missing geomean_speedup in performance report")
    if samples < min_samples:
        reasons.append(f"samples {samples} < minimum_samples {min_samples}")

    substitution_policy = contract.get("baseline_substitution_policy", {})
    if not isinstance(substitution_policy, dict):
        substitution_policy = {}
    require_justification = bool(substitution_policy.get("require_justification", False))
    required_fields = substitution_policy.get("fields", [])
    if not isinstance(required_fields, list):
        required_fields = []
    if require_justification:
        for idx, entry in enumerate(substitutions):
            if not isinstance(entry, dict):
                reasons.append(f"baseline_substitutions[{idx}] must be object")
                continue
            for field in required_fields:
                if field not in entry or entry[field] in ("", None):
                    reasons.append(f"baseline_substitutions[{idx}] missing field {field}")

    passed = not reasons and geomean >= threshold
    if not reasons and geomean < threshold:
        reasons.append(f"geomean_speedup {geomean:.6f} < threshold {threshold:.6f}")
    return {
        "status": "pass" if passed else "fail",
        "threshold": threshold,
        "minimum_samples": min_samples,
        "geomean_speedup": geomean,
        "samples": samples,
        "baseline_substitutions": substitutions,
        "reasons": reasons,
    }


def expected_schema_map() -> dict[str, str]:
    return {
        "ptx": PTX_SCHEMA,
        "sass": SASS_SCHEMA,
        "fpga": FPGA_SCHEMA,
        "quantum": QUANTUM_SCHEMA,
        "rl_bridge": BRIDGE_SCHEMA,
        "policy_guard": POLICY_GUARD_SCHEMA,
    }


def verify_required_schemas(required: dict[str, tuple[dict[str, Any], Path]], strict: bool) -> None:
    if not strict:
        return
    expected = expected_schema_map()
    for key, (payload, path) in required.items():
        schema = str(payload.get("schema", ""))
        if schema != expected[key]:
            raise SystemExit(
                f"{path}: schema mismatch for {key} (expected {expected[key]}, got {schema})"
            )


def build_rollover_items(
    hard_gates: dict[str, dict[str, Any]],
    success_criteria: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for gate_name, gate in hard_gates.items():
        if gate.get("pass") is True:
            continue
        reasons = gate.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        items.append(
            {
                "id": gate_name,
                "scope": "hard_gate",
                "severity": "hard",
                "reasons": [str(reason) for reason in reasons],
            }
        )
    for criterion_name, criterion in success_criteria.items():
        status = str(criterion.get("status", "unknown"))
        if status == "pass":
            continue
        reasons = criterion.get("reasons")
        if reasons is None and "reason" in criterion:
            reasons = [criterion["reason"]]
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons is not None else [f"{criterion_name} status={status}"]
        items.append(
            {
                "id": criterion_name,
                "scope": "success_criteria",
                "severity": "soft",
                "status": status,
                "reasons": [str(reason) for reason in reasons],
            }
        )
    return items


def main() -> int:
    args = parse_args()

    contract_path = Path(args.contract)
    policy_path = Path(args.policy)
    ptx_path = Path(args.ptx_report)
    sass_path = Path(args.sass_report)
    fpga_path = Path(args.fpga_report)
    quantum_path = Path(args.quantum_report)
    bridge_path = Path(args.rl_bridge)
    guard_path = Path(args.policy_guard)
    performance_path = Path(args.performance_report)
    out_path = Path(args.out)

    contract = load_json(contract_path, "contract")
    policy = load_json(policy_path, "policy")
    ptx = load_json(ptx_path, "ptx report")
    sass = load_json(sass_path, "sass report")
    fpga = load_json(fpga_path, "fpga report")
    quantum = load_json(quantum_path, "quantum report")
    bridge = load_json(bridge_path, "rl readiness bridge")
    guard = load_json(guard_path, "policy mode guard")

    verify_required_schemas(
        {
            "ptx": (ptx, ptx_path),
            "sass": (sass, sass_path),
            "fpga": (fpga, fpga_path),
            "quantum": (quantum, quantum_path),
            "rl_bridge": (bridge, bridge_path),
            "policy_guard": (guard, guard_path),
        },
        strict=args.strict,
    )

    l2_kernels = contract.get("kernel_sets", {}).get("l2_official", [])
    if not isinstance(l2_kernels, list):
        l2_kernels = []
    l2_kernels = [k for k in l2_kernels if isinstance(k, str)]
    if not l2_kernels:
        l2_kernels = ["gemm", "attention", "epistemic_elementwise", "monte_carlo"]

    hard_gates = {
        "l1_ptx_epistemic": evaluate_l1(ptx),
        "l2_sass_patch_4_kernels": evaluate_l2(sass, l2_kernels),
        "l3_fpga_seed": evaluate_l3(fpga),
    }
    hard_gate_pass = all(item.get("pass") is True for item in hard_gates.values())

    success_criteria = {
        "geomean_plus20": evaluate_geomean(contract, performance_path),
        "compile_overhead_lte_25pct": evaluate_compile_overhead(contract, bridge),
        "bit_exact_50_samples": evaluate_bit_exact(contract, sass, sass_path),
        "quantum_conformance_ks_coverage": evaluate_quantum(contract, quantum),
        "rl_shadow_default_or_active_ready": evaluate_rl_governance(policy, guard),
    }
    success_criteria_pass = all(
        str(item.get("status", "unknown")) == "pass"
        for item in success_criteria.values()
    )

    rollover_items = build_rollover_items(hard_gates, success_criteria)
    release_ready = hard_gate_pass
    if hard_gate_pass and success_criteria_pass:
        verdict = "release-hard-pass-success-pass"
    elif hard_gate_pass:
        verdict = "release-hard-pass-rollover-soft"
    else:
        verdict = "release-hard-fail-rollover"

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sprint_window": contract.get("sprint_window", {}),
        "hard_rollover_policy": True,
        "release_ready": release_ready,
        "hard_gate_pass": hard_gate_pass,
        "success_criteria_pass": success_criteria_pass,
        "verdict": verdict,
        "policy_mode": str(policy.get("policy_mode", "unknown")),
        "hard_gates": hard_gates,
        "success_criteria": success_criteria,
        "rollover_items": rollover_items,
        "artifacts": {
            "contract": str(contract_path),
            "policy": str(policy_path),
            "ptx_report": str(ptx_path),
            "sass_report": str(sass_path),
            "fpga_report": str(fpga_path),
            "quantum_report": str(quantum_path),
            "rl_bridge": str(bridge_path),
            "policy_guard": str(guard_path),
            "performance_report": str(performance_path),
            "performance_report_present": performance_path.exists(),
        },
    }

    write_json(out_path, payload)

    print(
        "omega_release_readiness: "
        f"verdict={verdict} "
        f"release_ready={str(release_ready).lower()} "
        f"success_criteria_pass={str(success_criteria_pass).lower()} "
        f"rollover_items={len(rollover_items)} "
        f"out={out_path}"
    )

    if args.require_release_ready and not release_ready:
        return 2
    if args.require_success_criteria and not success_criteria_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
