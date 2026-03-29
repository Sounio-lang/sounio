#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RESULT_RE = re.compile(r"^\s*(Pass|Fail|Skip):\s*(\d+)\s*$")
FAIL_RE = re.compile(r"^\s*FAIL\s+([^\s]+)\s+\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse three suite logs and classify failure deltas.")
    parser.add_argument("--accepted-log", required=True)
    parser.add_argument("--rebuild-log", required=True)
    parser.add_argument("--head-log", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    counts = {"Pass": None, "Fail": None, "Skip": None}
    failures: list[str] = []
    for line in text.splitlines():
        result_match = RESULT_RE.match(line.strip())
        if result_match:
            counts[result_match.group(1)] = int(result_match.group(2))
        fail_match = FAIL_RE.match(line)
        if fail_match:
            failures.append(fail_match.group(1))
    if counts["Pass"] is None or counts["Fail"] is None or counts["Skip"] is None:
        raise SystemExit(f"unable to parse suite summary from {path}")
    unique_failures = sorted(set(failures))
    return {
        "log_path": str(path),
        "pass": counts["Pass"],
        "fail": counts["Fail"],
        "skip": counts["Skip"],
        "total": counts["Pass"] + counts["Fail"] + counts["Skip"],
        "failures": unique_failures,
    }


def sorted_set(values: set[str]) -> list[str]:
    return sorted(values)


def main() -> int:
    args = parse_args()
    accepted = parse_log(Path(args.accepted_log))
    rebuild = parse_log(Path(args.rebuild_log))
    head = parse_log(Path(args.head_log))

    accepted_set = set(accepted["failures"])
    rebuild_set = set(rebuild["failures"])
    head_set = set(head["failures"])

    report = {
        "schema": "sounio.selfhost_suite_delta.v1",
        "accepted_artifact_baseline": accepted,
        "rebuild_from_source_baseline": rebuild,
        "current_head": head,
        "comparison": {
            "inherited_baseline_failures": sorted_set(accepted_set & rebuild_set & head_set),
            "failures_present_only_in_artifact_baseline": sorted_set(accepted_set - rebuild_set - head_set),
            "failures_present_only_in_rebuildable_source_baseline": sorted_set(rebuild_set - accepted_set - head_set),
            "failures_present_only_in_current_head": sorted_set(head_set - accepted_set - rebuild_set),
            "failures_present_in_artifact_and_rebuild_only": sorted_set((accepted_set & rebuild_set) - head_set),
            "failures_present_in_artifact_and_head_only": sorted_set((accepted_set & head_set) - rebuild_set),
            "failures_present_in_rebuild_and_head_only": sorted_set((rebuild_set & head_set) - accepted_set),
            "failures_introduced_by_current_head_vs_artifact": sorted_set(head_set - accepted_set),
            "failures_removed_by_current_head_vs_artifact": sorted_set(accepted_set - head_set),
            "failures_removed_by_rebuild_vs_artifact": sorted_set(accepted_set - rebuild_set),
            "accepted_equals_rebuild": accepted_set == rebuild_set,
            "accepted_equals_head": accepted_set == head_set,
            "rebuild_equals_head": rebuild_set == head_set,
            "parity_pass": accepted_set == rebuild_set == head_set,
            "harness_environment_mismatch_candidates": [],
        },
    }

    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Suite Delta",
        "",
        "## Totals",
        "",
        f"- accepted artifact baseline: {accepted['pass']} pass / {accepted['fail']} fail / {accepted['skip']} skip / {accepted['total']} total",
        f"- rebuild-from-source baseline: {rebuild['pass']} pass / {rebuild['fail']} fail / {rebuild['skip']} skip / {rebuild['total']} total",
        f"- current head: {head['pass']} pass / {head['fail']} fail / {head['skip']} skip / {head['total']} total",
        "",
        "## Key Result",
        "",
        f"- parity_pass: {report['comparison']['parity_pass']}",
        f"- accepted_equals_rebuild: {report['comparison']['accepted_equals_rebuild']}",
        f"- accepted_equals_head: {report['comparison']['accepted_equals_head']}",
        f"- rebuild_equals_head: {report['comparison']['rebuild_equals_head']}",
        "",
        "## Delta Sets",
        "",
    ]

    for key in (
        "inherited_baseline_failures",
        "failures_present_only_in_artifact_baseline",
        "failures_present_only_in_rebuildable_source_baseline",
        "failures_present_only_in_current_head",
        "failures_present_in_artifact_and_rebuild_only",
        "failures_present_in_artifact_and_head_only",
        "failures_present_in_rebuild_and_head_only",
        "failures_introduced_by_current_head_vs_artifact",
        "failures_removed_by_current_head_vs_artifact",
        "failures_removed_by_rebuild_vs_artifact",
    ):
        values = report["comparison"][key]
        md_lines.append(f"### {key}")
        if values:
            for value in values:
                md_lines.append(f"- {value}")
        else:
            md_lines.append("- none")
        md_lines.append("")

    Path(args.md_out).write_text("\n".join(md_lines), encoding="utf-8")

    return 0 if report["comparison"]["parity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
