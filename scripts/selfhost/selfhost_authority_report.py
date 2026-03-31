#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a machine-readable selfhost authority summary.")
    parser.add_argument("--policy-summary", required=True)
    parser.add_argument("--drift-summary", required=True)
    parser.add_argument("--fixed-summary", required=True)
    parser.add_argument("--fallback-summary", required=True)
    parser.add_argument("--abi-summary", required=True)
    parser.add_argument("--aarch64-summary", required=True)
    parser.add_argument("--aarch64-runtime-summary", required=True)
    parser.add_argument("--parity-json", required=True)
    parser.add_argument("--legacy-summary")
    parser.add_argument("--legacy-status", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-md", required=True)
    return parser.parse_args()


def parse_summary(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def parse_results(path: str) -> dict[str, dict[str, int]]:
    support_class_counts: dict[str, int] = {}
    ok_by_support_class: dict[str, int] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            support_class = row.get("support_class", "unknown") or "unknown"
            support_class_counts[support_class] = support_class_counts.get(support_class, 0) + 1
            if row.get("status") == "ok":
                ok_by_support_class[support_class] = ok_by_support_class.get(support_class, 0) + 1
    return {
        "support_class_counts": support_class_counts,
        "ok_by_support_class": ok_by_support_class,
    }


def parse_inventory(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()

    policy_summary = parse_summary(args.policy_summary)
    drift_summary = parse_summary(args.drift_summary)
    fixed = parse_summary(args.fixed_summary)
    fallback = parse_summary(args.fallback_summary)
    abi = parse_summary(args.abi_summary)
    aarch64 = parse_summary(args.aarch64_summary)
    aarch64_runtime = parse_summary(args.aarch64_runtime_summary)
    legacy = parse_summary(args.legacy_summary) if args.legacy_summary else {}
    parity = json.loads(Path(args.parity_json).read_text(encoding="utf-8"))
    comparison = parity["comparison"]

    fixed_pass = fixed.get("gen2_md5", "") != "" and fixed.get("gen2_md5") == fixed.get("gen3_md5")
    abi_fail = int(abi.get("summary_fail", "1"))
    aarch64_fail = int(aarch64.get("summary_fail", "1"))
    aarch64_runtime_fail = int(aarch64_runtime.get("summary_fail", "1"))
    fallback_inventory = parse_inventory(fallback["report_json"])
    fallback_fenced = True
    for entry in fallback_inventory["entries"]:
        if entry["classification"] != "unsupported_but_fenced":
            fallback_fenced = False
            break

    blocking_checks = [
        {
            "name": "promotion_policy",
            "blocking": True,
            "status": "pass",
            "summary_file": args.policy_summary,
            "report_json": policy_summary.get("report_json", ""),
        },
        {
            "name": "release_drift",
            "blocking": True,
            "status": "pass",
            "summary_file": args.drift_summary,
            "report_json": drift_summary.get("report_json", ""),
        },
        {
            "name": "fixed_point",
            "blocking": True,
            "status": "pass" if fixed_pass else "fail",
            "summary_file": args.fixed_summary,
            "gen2_md5": fixed.get("gen2_md5", ""),
            "gen3_md5": fixed.get("gen3_md5", ""),
        },
        {
            "name": "fallback_inventory",
            "blocking": True,
            "status": "pass" if fallback_fenced else "fail",
            "summary_file": args.fallback_summary,
            "classification_counts": fallback_inventory["classification_counts"],
            "entries": fallback_inventory["entries"],
        },
        {
            "name": "abi_parity_regressions",
            "blocking": True,
            "status": "pass" if abi_fail == 0 else "fail",
            "summary_file": args.abi_summary,
            "pass_count": int(abi.get("summary_pass", "0")),
            "fail_count": abi_fail,
            "skip_count": int(abi.get("summary_skip", "0")),
            "results": parse_results(abi["results_file"]),
        },
        {
            "name": "aarch64_compile_proof",
            "blocking": True,
            "status": "pass" if aarch64_fail == 0 else "fail",
            "summary_file": args.aarch64_summary,
            "pass_count": int(aarch64.get("summary_pass", "0")),
            "fail_count": aarch64_fail,
            "skip_count": int(aarch64.get("summary_skip", "0")),
            "results": parse_results(aarch64["results_file"]),
        },
        {
            "name": "aarch64_runtime",
            "blocking": True,
            "status": "pass" if aarch64_runtime_fail == 0 else "fail",
            "summary_file": args.aarch64_runtime_summary,
            "pass_count": int(aarch64_runtime.get("summary_pass", "0")),
            "fail_count": aarch64_runtime_fail,
            "skip_count": int(aarch64_runtime.get("summary_skip", "0")),
            "runner_path": aarch64_runtime.get("runner_path", ""),
            "runner_source": aarch64_runtime.get("runner_source", ""),
            "results": parse_results(aarch64_runtime["results_file"]),
        },
        {
            "name": "source_artifact_parity",
            "blocking": True,
            "status": "pass" if comparison["parity_pass"] else "fail",
            "summary_file": args.parity_json,
            "accepted_equals_rebuild": comparison["accepted_equals_rebuild"],
            "accepted_equals_head": comparison["accepted_equals_head"],
            "rebuild_equals_head": comparison["rebuild_equals_head"],
            "introduced_by_head_count": len(comparison["failures_introduced_by_current_head_vs_artifact"]),
            "rebuild_only_count": len(comparison["failures_present_only_in_rebuildable_source_baseline"]),
            "artifact_only_count": len(comparison["failures_present_only_in_artifact_baseline"]),
        },
    ]

    nonblocking_surfaces = [
        {
            "name": "legacy_native_acceptance",
            "blocking": False,
            "status": args.legacy_status,
            "summary_file": args.legacy_summary or "",
            "reason": (
                "accepted artifact baseline is not green on this legacy surface"
                if args.legacy_status == "skipped"
                else ""
            ),
        }
    ]

    overall_status = "pass"
    for check in blocking_checks:
        if check["status"] != "pass":
            overall_status = "fail"
            break

    report = {
        "schema": "sounio.selfhost_authority_gate.v2",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "entrypoint": "scripts/selfhost/selfhost_authority_gate.sh",
        "exit_semantics": {
            "0": "all blocking authority checks passed",
            "1": "at least one blocking authority check failed",
        },
        "blocking_checks": blocking_checks,
        "nonblocking_surfaces": nonblocking_surfaces,
        "baseline_context": {
            "accepted_artifact_fail_count": parity["accepted_artifact_baseline"]["fail"],
            "rebuild_from_source_fail_count": parity["rebuild_from_source_baseline"]["fail"],
            "current_head_fail_count": parity["current_head"]["fail"],
            "inherited_baseline_failure_count": len(comparison["inherited_baseline_failures"]),
            "artifact_only_failure_count": len(comparison["failures_present_only_in_artifact_baseline"]),
            "rebuild_only_failure_count": len(comparison["failures_present_only_in_rebuildable_source_baseline"]),
            "head_only_failure_count": len(comparison["failures_present_only_in_current_head"]),
            "introduced_by_head_vs_artifact_count": len(comparison["failures_introduced_by_current_head_vs_artifact"]),
            "removed_by_head_vs_artifact_count": len(comparison["failures_removed_by_current_head_vs_artifact"]),
            "parity_pass": comparison["parity_pass"],
        },
        "overall_status": overall_status,
        "legacy_summary_present": bool(legacy),
    }

    Path(args.summary_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Authority Gate",
        "",
        f"- overall_status: {overall_status}",
        f"- git_commit: {report['git']['commit']}",
        f"- git_branch: {report['git']['branch']}",
        "",
        "## Blocking Checks",
        "",
    ]
    for check in blocking_checks:
        md_lines.append(f"- {check['name']}: {check['status']}")
    md_lines.extend(
        [
            "",
            "## Baseline Context",
            "",
            f"- accepted artifact fail count: {report['baseline_context']['accepted_artifact_fail_count']}",
            f"- rebuild-from-source fail count: {report['baseline_context']['rebuild_from_source_fail_count']}",
            f"- current head fail count: {report['baseline_context']['current_head_fail_count']}",
            f"- inherited baseline failure count: {report['baseline_context']['inherited_baseline_failure_count']}",
            f"- rebuild-only failure count: {report['baseline_context']['rebuild_only_failure_count']}",
            f"- introduced-by-head vs artifact count: {report['baseline_context']['introduced_by_head_vs_artifact_count']}",
            "",
            "## Nonblocking Surfaces",
            "",
        ]
    )
    for surface in nonblocking_surfaces:
        reason = f" ({surface['reason']})" if surface["reason"] else ""
        md_lines.append(f"- {surface['name']}: {surface['status']}{reason}")
    Path(args.summary_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
