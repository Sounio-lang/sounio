#!/usr/bin/env python3
"""Generate weekly Omega governance drift report (JSON + Markdown)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.weekly-drift-report.v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega weekly governance drift report generator"
    )
    parser.add_argument(
        "--hw-trend",
        default="artifacts/fpga/hardware_epistemic_power_live_trend.v1.json",
        help="Hardware live-read trend path",
    )
    parser.add_argument(
        "--resource-trend",
        default="artifacts/fpga/hardware_resource_trend.v2.json",
        help="Hardware resource trend path",
    )
    parser.add_argument(
        "--rl-trend",
        default="artifacts/omega/rl_readiness_trend.v1.json",
        help="RL readiness trend path",
    )
    parser.add_argument(
        "--attestation",
        default="artifacts/omega/governance_attestation.v1.json",
        help="Governance attestation path",
    )
    parser.add_argument(
        "--release-readiness",
        default="artifacts/omega/sprint1_release_readiness.v1.json",
        help="Sprint 1 release-readiness artifact path",
    )
    parser.add_argument(
        "--performance-summary",
        default="artifacts/omega/performance_summary.v1.json",
        help="Performance summary artifact path",
    )
    parser.add_argument(
        "--external-baseline-collection",
        default="artifacts/omega/external_baseline_collection.v1.json",
        help="External baseline collection artifact path",
    )
    parser.add_argument(
        "--baseline-freeze",
        default="artifacts/omega/baseline_freeze.v1.json",
        help="Baseline freeze artifact path",
    )
    parser.add_argument(
        "--hw-live",
        default="artifacts/fpga/hardware_epistemic_power_live.v1.json",
        help="Hardware live-read artifact path",
    )
    parser.add_argument(
        "--fpga-report",
        default="artifacts/fpga/fpga_seed_report.json",
        help="FPGA seed report path",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/omega/weekly_drift_report.v4.json",
        help="Weekly report JSON output path",
    )
    parser.add_argument(
        "--out-md",
        default="artifacts/omega/weekly_drift_report.md",
        help="Weekly report markdown output path",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Number of recent runs to summarize",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read JSON artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid JSON payload in {path}: expected object")
    return payload


def load_json_optional(path: Path) -> tuple[dict, bool]:
    if not path.exists():
        return {}, False
    return load_json(path), True


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def summarize_hw_trend(trend: dict, window: int) -> dict:
    runs = trend.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    recent = [run for run in runs if isinstance(run, dict)][-window:]
    gains = [float(run.get("gain_ratio", 0.0)) for run in recent]
    drift_flags = [bool(run.get("drift_violation", False)) for run in recent]
    return {
        "total_runs": len(runs),
        "window_runs": len(recent),
        "avg_gain_ratio": average(gains),
        "drift_violations_in_window": sum(1 for flag in drift_flags if flag),
        "last_status": str(trend.get("last_status", "unknown")),
    }


def summarize_rl_trend(trend: dict, window: int) -> dict:
    runs = trend.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    recent = [run for run in runs if isinstance(run, dict)][-window:]
    gains = [float(run.get("epistemic_power_gain", 0.0)) for run in recent]
    overheads = [float(run.get("compile_overhead", 0.0)) for run in recent]
    drift_flags = [bool(run.get("drift_violation", False)) for run in recent]
    passes = [bool(run.get("policy_rl_readiness_pass", False)) for run in recent]
    return {
        "total_runs": len(runs),
        "window_runs": len(recent),
        "avg_epistemic_gain": average(gains),
        "avg_compile_overhead": average(overheads),
        "drift_violations_in_window": sum(1 for flag in drift_flags if flag),
        "policy_pass_rate": average([1.0 if p else 0.0 for p in passes]),
        "last_status": str(trend.get("last_status", "unknown")),
    }


def summarize_attestation(attestation: dict) -> dict:
    artifacts = attestation.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    missing = attestation.get("missing_artifacts", [])
    if not isinstance(missing, list):
        missing = []
    return {
        "artifact_count": int(attestation.get("artifact_count", len(artifacts))),
        "missing_count": len(missing),
        "signed": bool(attestation.get("signed", False)),
        "aggregate_sha256": str(attestation.get("aggregate_sha256", "")),
        "qir_emitter_pass": bool(attestation.get("qir_emitter_pass", False)),
        "merkle_lane_pass": bool(attestation.get("merkle_lane_pass", False)),
        "resource_trend_pass": bool(attestation.get("resource_trend_pass", False)),
        "genesis_manifest_signed": bool(attestation.get("genesis_manifest_signed", False)),
        "genesis_cold_boot_ready": bool(attestation.get("genesis_cold_boot_ready", False)),
        "genesis_manifest_aggregate_sha256": str(
            attestation.get("genesis_manifest_aggregate_sha256", "")
        ),
    }


def summarize_release_readiness(readiness: dict) -> dict:
    hard_gate_pass = bool(readiness.get("hard_gate_pass", False))
    success_pass = bool(readiness.get("success_criteria_pass", False))
    release_ready = bool(readiness.get("release_ready", False))
    rollover_items = readiness.get("rollover_items", [])
    if not isinstance(rollover_items, list):
        rollover_items = []
    verdict = str(readiness.get("verdict", "unknown"))
    return {
        "hard_gate_pass": hard_gate_pass,
        "success_criteria_pass": success_pass,
        "release_ready": release_ready,
        "verdict": verdict,
        "rollover_item_count": len(rollover_items),
    }


def summarize_performance_summary(summary: dict) -> dict:
    geomean = summary.get("geomean_speedup", 0.0)
    threshold = summary.get("threshold", 0.0)
    samples = summary.get("samples", 0)
    substitutions = summary.get("baseline_substitutions", [])
    if not isinstance(substitutions, list):
        substitutions = []
    return {
        "measurement_mode": str(summary.get("measurement_mode", "unknown")),
        "geomean_speedup": float(geomean) if isinstance(geomean, (int, float)) else 0.0,
        "threshold": float(threshold) if isinstance(threshold, (int, float)) else 0.0,
        "samples": int(samples) if isinstance(samples, int) else 0,
        "substitution_count": len(substitutions),
    }


def summarize_external_baselines(summary: dict) -> dict:
    required = summary.get("baseline_count", 0)
    available = summary.get("available_count", 0)
    execute = summary.get("execute", False)
    missing = summary.get("missing_reports", [])
    if not isinstance(missing, list):
        missing = []
    return {
        "baseline_count": int(required) if isinstance(required, int) else 0,
        "available_count": int(available) if isinstance(available, int) else 0,
        "missing_reports": len(missing),
        "execute": bool(execute),
    }


def summarize_baseline_freeze(summary: dict) -> dict:
    baseline_count = summary.get("baseline_count", 0)
    available_count = summary.get("available_count", 0)
    frozen_rows = summary.get("frozen_rows", [])
    if not isinstance(frozen_rows, list):
        frozen_rows = []
    return {
        "status": str(summary.get("status", "unknown")),
        "measurement_mode": str(summary.get("measurement_mode", "unknown")),
        "baseline_count": int(baseline_count) if isinstance(baseline_count, int) else 0,
        "available_count": int(available_count) if isinstance(available_count, int) else 0,
        "frozen_rows": len(frozen_rows),
        "freeze_digest_sha256": str(summary.get("freeze_digest_sha256", "")),
    }


def summarize_canonical_bootstrap(attestation: dict, freeze: dict) -> dict:
    fpr = str(attestation.get("canonical_pubkey_fingerprint", ""))
    ts = str(attestation.get("canonical_bootstrap_timestamp", ""))
    policy_sign_fingerprint = str(attestation.get("policy_sign_fingerprint", ""))
    policy_sign_timestamp = str(attestation.get("policy_sign_timestamp", ""))
    policy_pinned_digest = str(attestation.get("policy_pinned_digest_sha256", ""))
    baseline_freeze_digest = str(attestation.get("baseline_freeze_digest_sha256", ""))
    baseline_freeze_policy_pinned = str(
        attestation.get("baseline_freeze_policy_pinned_digest_sha256", "")
    )
    pinned_digest_match = bool(attestation.get("pinned_digest_match", False))
    if not ts:
        ts = str(freeze.get("canonical_bootstrap_timestamp", ""))
    if not baseline_freeze_digest:
        baseline_freeze_digest = str(freeze.get("freeze_digest_sha256", ""))
    return {
        "canonical_pubkey_fingerprint": fpr,
        "canonical_bootstrap_timestamp": ts,
        "policy_sign_fingerprint": policy_sign_fingerprint,
        "policy_sign_timestamp": policy_sign_timestamp,
        "policy_pinned_digest_sha256": policy_pinned_digest,
        "baseline_freeze_digest_sha256": baseline_freeze_digest,
        "baseline_freeze_policy_pinned_digest_sha256": baseline_freeze_policy_pinned,
        "pinned_digest_match": pinned_digest_match,
    }


def summarize_sprint3_hardware(hw_live: dict, fpga_report: dict) -> dict:
    variance = hw_live.get("hardware_epistemic_power_variance_q32_32", 0)
    overhead_us = hw_live.get("poll_overhead_us", 0)
    fpga_stale = bool(fpga_report.get("stale"))
    # hardware/** has never been versioned in this repository -- a stale
    # report's status fields describe an environment this checkout doesn't
    # have. Report "stale" plainly rather than deriving a pass from them.
    bidir_sim = "stale" if fpga_stale else fpga_report.get("k_axi_return_sim_status", "unknown")
    bidir_synth = "stale" if fpga_stale else fpga_report.get("k_axi_return_synth_status", "unknown")
    return {
        "live_read_conformant": bool(hw_live.get("live_read_conformant", False)),
        "hardware_epistemic_power_log_q32_32": int(
            hw_live.get("hardware_epistemic_power_log_q32_32", 0)
        )
        if isinstance(hw_live.get("hardware_epistemic_power_log_q32_32", 0), int)
        else 0,
        "hybrid_epistemic_power_log_q32_32": int(
            hw_live.get("hybrid_epistemic_power_log_q32_32", 0)
        )
        if isinstance(hw_live.get("hybrid_epistemic_power_log_q32_32", 0), int)
        else 0,
        "variance_q32_32": int(variance) if isinstance(variance, int) else 0,
        "poll_overhead_us": int(overhead_us) if isinstance(overhead_us, int) else 0,
        "sub_1ms_overhead": isinstance(overhead_us, int) and overhead_us >= 0 and overhead_us < 1000,
        "bidir_kaxi_sim_status": str(bidir_sim),
        "bidir_kaxi_synth_status": str(bidir_synth),
        "bidir_kaxi_pass": bidir_sim == "pass" and bidir_synth == "pass",
    }


def summarize_resource_trend(trend: dict) -> dict:
    runs = trend.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    last = runs[-1] if runs else {}
    if not isinstance(last, dict):
        last = {}
    max_drift = last.get("max_relative_drift", trend.get("last_max_relative_drift", 0.0))
    threshold = trend.get("drift_threshold", 0.05)
    return {
        "last_status": str(trend.get("last_status", "")),
        "run_count": len(runs),
        "max_relative_drift": float(max_drift) if isinstance(max_drift, (int, float)) else 0.0,
        "drift_threshold": float(threshold) if isinstance(threshold, (int, float)) else 0.05,
    }


def make_markdown(payload: dict) -> str:
    hw = payload["hardware_live_read"]
    rl = payload["rl_readiness"]
    att = payload["attestation"]
    release = payload["release_readiness"]
    perf = payload["performance_summary"]
    ext = payload["external_baselines"]
    freeze = payload["baseline_freeze"]
    canonical = payload["canonical_bootstrap"]
    sprint3 = payload["sprint3_hardware"]
    resource_trend = payload["resource_trend_gate"]
    lines = [
        "# Omega Weekly Governance Drift Report",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Window (runs): `{payload['window']}`",
        "",
        "## Hardware Live-Read Trend",
        f"- Total runs: `{hw['total_runs']}`",
        f"- Window runs: `{hw['window_runs']}`",
        f"- Avg gain ratio: `{hw['avg_gain_ratio']:.6f}`",
        f"- Drift violations (window): `{hw['drift_violations_in_window']}`",
        f"- Last status: `{hw['last_status']}`",
        "",
        "## RL Readiness Trend",
        f"- Total runs: `{rl['total_runs']}`",
        f"- Window runs: `{rl['window_runs']}`",
        f"- Avg epistemic gain: `{rl['avg_epistemic_gain']:.6f}`",
        f"- Avg compile overhead: `{rl['avg_compile_overhead']:.6f}`",
        f"- Policy pass rate (window): `{rl['policy_pass_rate']:.3f}`",
        f"- Drift violations (window): `{rl['drift_violations_in_window']}`",
        f"- Last status: `{rl['last_status']}`",
        "",
        "## Governance Attestation",
        f"- Artifact count: `{att['artifact_count']}`",
        f"- Missing artifacts: `{att['missing_count']}`",
        f"- Signed: `{str(att['signed']).lower()}`",
        f"- Aggregate SHA256: `{att['aggregate_sha256']}`",
        f"- QIR emitter pass: `{str(att['qir_emitter_pass']).lower()}`",
        f"- Merkle lane pass: `{str(att['merkle_lane_pass']).lower()}`",
        f"- Resource trend pass: `{str(att['resource_trend_pass']).lower()}`",
        f"- Genesis manifest signed: `{str(att['genesis_manifest_signed']).lower()}`",
        f"- Genesis cold boot ready: `{str(att['genesis_cold_boot_ready']).lower()}`",
        f"- Genesis aggregate SHA256: `{att['genesis_manifest_aggregate_sha256']}`",
        "",
        "## Sprint 1 Release Readiness",
        f"- Verdict: `{release['verdict']}`",
        f"- Hard gates pass: `{str(release['hard_gate_pass']).lower()}`",
        f"- Success criteria pass: `{str(release['success_criteria_pass']).lower()}`",
        f"- Release ready (hard gate semantics): `{str(release['release_ready']).lower()}`",
        f"- Rollover items: `{release['rollover_item_count']}`",
        "",
        "## Performance Summary",
        f"- Mode: `{perf['measurement_mode']}`",
        f"- Geomean speedup: `{perf['geomean_speedup']:.6f}`",
        f"- Threshold: `{perf['threshold']:.6f}`",
        f"- Samples: `{perf['samples']}`",
        f"- Baseline substitutions: `{perf['substitution_count']}`",
        "",
        "## External Baselines",
        f"- Required baselines: `{ext['baseline_count']}`",
        f"- Available baselines: `{ext['available_count']}`",
        f"- Missing reports: `{ext['missing_reports']}`",
        f"- Execute mode: `{str(ext['execute']).lower()}`",
        "",
        "## Baseline Freeze",
        f"- Status: `{freeze['status']}`",
        f"- Mode: `{freeze['measurement_mode']}`",
        f"- Required baselines: `{freeze['baseline_count']}`",
        f"- Available baselines: `{freeze['available_count']}`",
        f"- Frozen rows: `{freeze['frozen_rows']}`",
        f"- Freeze digest: `{freeze['freeze_digest_sha256']}`",
        "",
        "## Canonical Bootstrap Key",
        f"- Fingerprint: `{canonical['canonical_pubkey_fingerprint']}`",
        f"- Last bootstrap timestamp: `{canonical['canonical_bootstrap_timestamp']}`",
        f"- Policy sign fingerprint: `{canonical['policy_sign_fingerprint']}`",
        f"- Policy sign timestamp: `{canonical['policy_sign_timestamp']}`",
        f"- Policy pinned digest: `{canonical['policy_pinned_digest_sha256']}`",
        f"- Baseline freeze digest: `{canonical['baseline_freeze_digest_sha256']}`",
        f"- Baseline freeze policy-pinned digest: `{canonical['baseline_freeze_policy_pinned_digest_sha256']}`",
        f"- Pinned digest match: `{str(canonical['pinned_digest_match']).lower()}`",
        "",
        "## Sprint 3 Hardware",
        f"- Live read conformant: `{str(sprint3['live_read_conformant']).lower()}`",
        f"- Hardware log Q32.32: `{sprint3['hardware_epistemic_power_log_q32_32']}`",
        f"- Hybrid log Q32.32: `{sprint3['hybrid_epistemic_power_log_q32_32']}`",
        f"- Variance Q32.32: `{sprint3['variance_q32_32']}`",
        f"- Poll overhead (us): `{sprint3['poll_overhead_us']}`",
        f"- Sub-1ms overhead: `{str(sprint3['sub_1ms_overhead']).lower()}`",
        f"- Bidir K-AXI sim: `{sprint3['bidir_kaxi_sim_status']}`",
        f"- Bidir K-AXI synth: `{sprint3['bidir_kaxi_synth_status']}`",
        f"- Bidir K-AXI pass: `{str(sprint3['bidir_kaxi_pass']).lower()}`",
        "",
        "## Sprint 4 Resource Trend",
        f"- Last status: `{resource_trend['last_status']}`",
        f"- Run count: `{resource_trend['run_count']}`",
        f"- Max relative drift: `{resource_trend['max_relative_drift']:.6f}`",
        f"- Drift threshold: `{resource_trend['drift_threshold']:.6f}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be > 0")

    hw_trend, hw_present = load_json_optional(Path(args.hw_trend))
    resource_trend, resource_trend_present = load_json_optional(Path(args.resource_trend))
    rl_trend, rl_present = load_json_optional(Path(args.rl_trend))
    attestation, att_present = load_json_optional(Path(args.attestation))
    release_readiness, release_present = load_json_optional(Path(args.release_readiness))
    performance_summary, performance_present = load_json_optional(Path(args.performance_summary))
    external_baseline_collection, external_baseline_collection_present = load_json_optional(
        Path(args.external_baseline_collection)
    )
    baseline_freeze, baseline_freeze_present = load_json_optional(Path(args.baseline_freeze))
    hw_live, hw_live_present = load_json_optional(Path(args.hw_live))
    fpga_report, fpga_report_present = load_json_optional(Path(args.fpga_report))

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window": args.window,
        "sources": {
            "hardware_live_read_trend_present": hw_present,
            "hardware_resource_trend_present": resource_trend_present,
            "rl_readiness_trend_present": rl_present,
            "governance_attestation_present": att_present,
            "release_readiness_present": release_present,
            "performance_summary_present": performance_present,
            "external_baseline_collection_present": external_baseline_collection_present,
            "baseline_freeze_present": baseline_freeze_present,
            "hw_live_present": hw_live_present,
            "fpga_report_present": fpga_report_present,
            "hardware_live_read_trend_path": args.hw_trend,
            "hardware_resource_trend_path": args.resource_trend,
            "rl_readiness_trend_path": args.rl_trend,
            "governance_attestation_path": args.attestation,
            "release_readiness_path": args.release_readiness,
            "performance_summary_path": args.performance_summary,
            "external_baseline_collection_path": args.external_baseline_collection,
            "baseline_freeze_path": args.baseline_freeze,
            "hw_live_path": args.hw_live,
            "fpga_report_path": args.fpga_report,
        },
        "hardware_live_read": summarize_hw_trend(hw_trend, args.window),
        "rl_readiness": summarize_rl_trend(rl_trend, args.window),
        "attestation": summarize_attestation(attestation),
        "release_readiness": summarize_release_readiness(release_readiness),
        "performance_summary": summarize_performance_summary(performance_summary),
        "external_baselines": summarize_external_baselines(external_baseline_collection),
        "baseline_freeze": summarize_baseline_freeze(baseline_freeze),
        "canonical_bootstrap": summarize_canonical_bootstrap(attestation, baseline_freeze),
        "sprint3_hardware": summarize_sprint3_hardware(hw_live, fpga_report),
        "resource_trend_gate": summarize_resource_trend(resource_trend),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_text(out_json, json.dumps(payload, indent=2))
    write_text(out_md, make_markdown(payload))

    print(
        "omega_weekly_drift_report: "
        f"json={out_json} "
        f"md={out_md} "
        f"window={args.window}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
