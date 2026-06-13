#!/usr/bin/env python3
"""Completion contract for the native visual frontend plan.

This audit is deliberately stricter than a marker smoke test: it groups the
original visual-frontend plan into coarse acceptance categories and checks that
the current manifest plus the plan-coverage checker still prove each category.
The Sounio programs in the manifest remain the semantic proof; this script keeps
the top-level "no caveats" contract visible and machine-checkable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--manifest",
        default="scripts/ci/native_visual_frontend_gate.manifest.tsv",
        help="Gate manifest path, relative to root unless absolute.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_manifest(path: Path) -> tuple[set[str], dict[str, int]]:
    labels: set[str] = set()
    counts = {"check": 0, "run": 0, "compile": 0, "script": 0}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"bad manifest row: {raw}")
        mode, label, _rel_path, _marker = parts
        labels.add(label)
        counts[mode] = counts.get(mode, 0) + 1
    return labels, counts


def coverage_requirements(text: str) -> set[str]:
    return set(re.findall(r'Evidence\(\s*"([^"]+)"', text))


def run_checker(root: Path, manifest: Path) -> str:
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/ci/check_native_visual_frontend_plan_coverage.py"),
            "--root",
            str(root),
            "--manifest",
            str(manifest),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    return proc.stdout.strip()


def require_all(name: str, actual: set[str], required: tuple[str, ...], failures: list[str]) -> None:
    missing = sorted(set(required) - actual)
    if missing:
        failures.append(f"{name}: missing {', '.join(missing)}")


def require_tokens(name: str, text: str, tokens: tuple[str, ...], failures: list[str]) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        failures.append(f"{name}: missing tokens {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest

    failures: list[str] = []
    labels, counts = parse_manifest(manifest)
    coverage_text = read_text(root / "scripts/ci/check_native_visual_frontend_plan_coverage.py")
    requirements = coverage_requirements(coverage_text)
    checker_out = run_checker(root, manifest)
    roadmap = read_text(root / "docs/design/native_visual_frontend_roadmap.md")
    readme = read_text(root / "stdlib/viz/README.md")

    groups: dict[str, tuple[str, ...]] = {
        "visual_ir": (
            "visual-ir-node-kinds",
            "visual-ir-style-layout-scene",
            "visual-ir-fixed-capacity",
        ),
        "renderers": (
            "canvas-renderer",
            "html-svg-renderer",
            "html-svg-passive-export-checker",
            "seed-canvas-extension",
            "seed-sci-effects",
        ),
        "native_controls": (
            "native-controls",
            "interaction-state-in-sounio",
            "workbench-native-package-controls",
            "workbench-native-project-controls",
            "workbench-project-workspace-controls",
        ),
        "physchem": (
            "physchem-core-structs",
            "physchem-extended-surfaces",
            "physchem-units",
            "physchem-demo",
        ),
        "authoring_persistence": (
            "scene-authoring-snapshots",
            "scene-authoring-history",
            "scene-authoring-package",
            "scene-project-format",
            "workbench-project-workspace",
            "workbench-project-diff-panel",
        ),
        "workbench_replay_export": (
            "workbench-session-timeline",
            "workbench-session-archive",
            "workbench-visual-notebook",
            "workbench-experiment-export-packet",
            "workbench-experiment-import-replay",
            "workbench-experiment-evidence-certificate",
            "workbench-experiment-evidence-bundle-manifest",
        ),
        "demos_and_ci": (
            "lab-demo-rich-scene",
            "native-window-demo",
            "gate-headless-run",
            "headless-pixel-assertions",
            "gpu-deferred-roadmap",
        ),
    }
    for name, required in groups.items():
        require_all(f"coverage group {name}", requirements, required, failures)

    label_groups: dict[str, tuple[str, ...]] = {
        "headless_pixels": ("viz_headless", "viz_canvas_ext_surface", "viz_chart_builders"),
        "passive_html": ("viz_html_static", "viz_html_passive_export", "viz_workbench_html_physchem"),
        "physchem": ("viz_physchem", "viz_physchem_dynamics", "viz_physchem_demo_run"),
        "workbench": (
            "viz_workbench_roundtrip",
            "viz_workbench_app_events",
            "viz_workbench_replay_session",
            "viz_workbench_demo_run",
        ),
        "evidence_chain": (
            "viz_workbench_experiment_export_packet",
            "viz_workbench_experiment_import_replay",
            "viz_workbench_experiment_packet_ledger_replay",
            "viz_workbench_experiment_evidence_certificate",
            "viz_workbench_experiment_evidence_bundle_manifest",
        ),
        "demos": (
            "viz_hello_demo",
            "viz_lab_demo",
            "viz_lab_window_demo",
            "viz_physchem_demo",
            "viz_workbench_demo",
            "viz_lab_run",
            "viz_physchem_demo_run",
            "viz_workbench_demo_run",
        ),
    }
    for name, required in label_groups.items():
        require_all(f"manifest group {name}", labels, required, failures)

    require_tokens(
        "roadmap completion notes",
        roadmap,
        (
            "V1: CPU Canvas",
            "V1.8j: Canonical Headless Visual Gate",
            "V1.8m: Scene Package Export/Import",
            "V2: GPU Renderer",
            "VizWorkbenchExperimentEvidenceBundleManifest",
        ),
        failures,
    )
    require_tokens(
        "readme completion notes",
        readme,
        (
            "Scientific meaning stays in Sounio",
            "canonical headless acceptance gate",
            "VizWorkbenchExperimentEvidenceBundleManifest",
            "GPU rendering is deferred",
        ),
        failures,
    )

    if counts.get("check", 0) < 2:
        failures.append("manifest count: expected at least 2 check entries")
    if counts.get("run", 0) < 62:
        failures.append("manifest count: expected at least 62 run entries")
    if counts.get("compile", 0) < 5:
        failures.append("manifest count: expected at least 5 compile entries")
    if counts.get("script", 0) < 2:
        failures.append("manifest count: expected at least 2 script entries")
    if len(requirements) < 107:
        failures.append("coverage count: expected at least 107 requirements")

    if failures:
        for failure in failures:
            print(f"error: native visual frontend completion audit: {failure}", file=sys.stderr)
        return 1

    print(checker_out)
    print(
        "NATIVE_VISUAL_FRONTEND_COMPLETION_AUDIT_PASS "
        f"groups={len(groups) + len(label_groups)} "
        f"requirements={len(requirements)} "
        f"check={counts.get('check', 0)} "
        f"run={counts.get('run', 0)} "
        f"compile={counts.get('compile', 0)} "
        f"script={counts.get('script', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
