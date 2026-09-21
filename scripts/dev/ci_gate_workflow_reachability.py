#!/usr/bin/env python3
"""Transitive workflow reachability of scripts/ci/*_gate.sh.

This is a CENSUS instrument, not a wirer. It answers: can a GitHub
workflow execute this gate, at any call depth, by following real
invocations (bash/sh/source), not comments and not inventory globs.

Positive control (must be non-zero on this repo): at least one gate is
reachable (ci.yml names several) AND at least one named leftover is not
(madaros_f128_f256_ladder_gate.sh). A census that reports 0 leftovers,
or 0 reachable, has not measured. The 2026-08-18 named three plus the
F2 bitcast/sitofp boundary gate are wired.

Usage:
  python3 scripts/dev/ci_gate_workflow_reachability.py
  python3 scripts/dev/ci_gate_workflow_reachability.py --tsv PATH --json PATH
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Conservative invoke: a shell actually runs the path.
INVOKE_RE = re.compile(
    r"""(?x)
    (?:^|&&|\|\||;|`|\(|\s)
    (?:bash|sh|source|\.)
    \s+
    (?:[A-Za-z0-9_]+=\S+\s+)*          # env assignments
    (?P<q>["']?)
    (?:\$\{?(?:ROOT_DIR|ROOT|PWD|GITHUB_WORKSPACE)\}?/? )?
    (?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+\.sh)
    (?P=q)
    """
)

# YAML `run:` sometimes has the script as the only token after bash.
YAML_RUN_RE = re.compile(
    r"""(?x)
    run:\s*(?:\|)?\s*
    (?:bash|sh)
    \s+
    (?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+\.sh)
    """
)

# YAML `run:` blocks sometimes list gate paths on their own continuation
# lines, then `bash "$g"`. A listed *_gate.sh on a non-comment YAML line
# is an invoke. Do NOT apply this to .sh files: scan-lists (sigpipe
# hygiene SHAPE_BAN_* arrays, gate_vacuity `git ls-files`) name gates
# without executing them. Treating those as invokes made
# mli_s3_bit_identity_gate.sh look workflow-reachable and REFUTE'd the
# leftover instrument after #1880. Pass-2 already drew this line.
BARE_GATE_RE = re.compile(
    r"(?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+_gate\.sh)"
)

MAKE_RE = re.compile(r"""(?x)(?:^|&&|\s)make\s+(?P<target>[A-Za-z0-9_.-]+)""")

# Inventory, not invoke — gate_vacuity_gate.sh lists every gate.
INVENTORY_MARKERS = (
    "git ls-files",
    "mapfile",
    "find ",
    "rglob",
)

# Remaining forgotten leftovers after the 2026-08-18 named-three landing
# plus the F2 bitcast/sitofp boundary wire. Wiring one of these without
# removing it here is a REFUTE, not a silent pass.
NAMED_DIRECT_ORPHANS = (
    "madaros_f128_f256_ladder_gate.sh",
    "madaros_f128_f256_v0c_wire_gate.sh",
    "madaros_f128_f256_v0d_softfloat_gate.sh",
    "madaros_print_f64_negative_gate.sh",
    "mli_s3_bit_identity_gate.sh",
    "stdlib_source_byte_ceiling_gate.sh",
)

DISSERTATION_GATES = (
    "dissertation_confidence_gate_gate.sh",
    "dissertation_dossier_gate.sh",
    "dissertation_frontend_parity_gate.sh",
    "dissertation_pbpk28_parity_gate.sh",
    "dissertation_pbpk_hessian_gate.sh",
    "dissertation_pbpk_suite_gate.sh",
)


def is_comment_line(path: Path, line: str) -> bool:
    s = line.lstrip()
    if not s or s.startswith("#!"):
        return False
    if path.suffix in {".yml", ".yaml"}:
        return s.startswith("#")
    if path.suffix == ".sh" or path.name in {"Makefile", "makefile"}:
        return s.startswith("#")
    if path.suffix == ".py":
        return s.startswith("#")
    return s.startswith("#")


def extract_invokes(path: Path, text: str) -> set[str]:
    found: set[str] = set()
    scanners = [INVOKE_RE, YAML_RUN_RE]
    # Bare gate paths are invoke only in workflow YAML. In .sh they are
    # almost always a scan-list or a comment-adjacent inventory.
    if path.suffix in {".yml", ".yaml"}:
        scanners.append(BARE_GATE_RE)
    for i, line in enumerate(text.splitlines(), 1):
        if is_comment_line(path, line):
            continue
        if any(m in line for m in INVENTORY_MARKERS) and "_gate.sh" in line:
            # Listing gates is not running them (gate_vacuity_gate.sh).
            continue
        for rx in scanners:
            for m in rx.finditer(line):
                found.add(m.group("path"))
    return found


def extract_make_targets(path: Path, text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        if is_comment_line(path, line):
            continue
        for m in MAKE_RE.finditer(line):
            tgt = m.group("target")
            if tgt not in {"-C", "-f", "-j", "build", "clean"}:
                found.add(tgt)
    return found


def parse_makefile(makefile: Path) -> dict[str, set[str]]:
    """target -> invoked script paths (best-effort, no $(fn) expansion)."""
    if not makefile.is_file():
        return {}
    text = makefile.read_text(errors="replace")
    recipes: dict[str, set[str]] = defaultdict(set)
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("\t"):
            for t in current:
                recipes[t] |= extract_invokes(makefile, line)
            continue
        if "=" in line and not line.startswith(".") and ":" not in line.split("=")[0]:
            continue
        if ":" in line and not line.startswith("\t"):
            left = line.split(":", 1)[0]
            current = [x.strip() for x in left.split() if x.strip()]
    return recipes


def header_blob(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()[:n]
    except OSError:
        return ""
    return "\n".join(lines)


def classify_leftover(name: str, gate_path: Path, makefile_hit: bool, umbrella_hit: bool) -> str:
    """Classify a workflow-unreachable gate. Evidence-only buckets."""
    if name in NAMED_DIRECT_ORPHANS:
        return "forgotten"
    if name in DISSERTATION_GATES or name.startswith("dissertation_"):
        return "manual-by-design"
    if name == "bootstrap_chain_gate.sh":
        return "manual-by-design"
    comments = "\n".join(
        ln for ln in header_blob(gate_path).splitlines() if ln.lstrip().startswith("#")
    ).lower()
    head = comments
    if any(k in head for k in ("obsolete", "superseded", "deprecated", "do not use", "historical only")):
        return "obsolete"
    if any(
        k in head
        for k in (
            "not wired",
            "not in ci",
            "not in github",
            "operator-run",
            "run on slurm",
            "slurm-only",
            "manual gate",
            "run by hand",
            "not a ci job",
        )
    ):
        return "manual-by-design"
    if makefile_hit or umbrella_hit:
        return "manual-by-design"
    if "positive control" in head or "GATE_CONTRACT" in header_blob(gate_path):
        return "forgotten"
    return "unclassified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tsv",
        default="",
        help="Write the per-gate table here. Default: do not overwrite the 2026-08-18 snapshot.",
    )
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    gates = sorted((REPO / "scripts/ci").glob("*_gate.sh"))
    if len(gates) < 400:
        print(f"REFUTE: expected >=400 scripts/ci/*_gate.sh, got {len(gates)}", file=sys.stderr)
        return 2

    workflows = sorted((REPO / ".github/workflows").glob("*.yml")) + sorted(
        (REPO / ".github/workflows").glob("*.yaml")
    )
    if not workflows:
        print("REFUTE: no workflows", file=sys.stderr)
        return 2

    makefile_recipes = parse_makefile(REPO / "Makefile")

    # BFS over invoked scripts, starting at workflow files.
    invoked_by: dict[str, set[str]] = defaultdict(set)
    queue: deque[Path] = deque(workflows)
    seen: set[Path] = set()
    root_of: dict[str, str] = {}
    depth_of: dict[str, int] = {}

    for wf in workflows:
        rel = str(wf.relative_to(REPO))
        depth_of[rel] = 0
        root_of[rel] = wf.name

    while queue:
        cur = queue.popleft()
        if cur in seen or not cur.is_file():
            continue
        seen.add(cur)
        try:
            text = cur.read_text(errors="replace")
        except OSError:
            continue
        cur_rel = str(cur.relative_to(REPO))
        d = depth_of.get(cur_rel, 0)
        origin = root_of.get(cur_rel, cur.name)

        for inv in extract_invokes(cur, text):
            child = REPO / inv
            invoked_by[inv].add(cur_rel)
            if inv not in depth_of:
                depth_of[inv] = d + 1
                root_of[inv] = origin
            if child.is_file() and child not in seen:
                queue.append(child)

        for tgt in extract_make_targets(cur, text):
            for inv in makefile_recipes.get(tgt, ()):
                child = REPO / inv
                invoked_by[inv].add(f"{cur_rel}:make:{tgt}")
                if inv not in depth_of:
                    depth_of[inv] = d + 1
                    root_of[inv] = origin
                if child.is_file() and child not in seen:
                    queue.append(child)

    # Makefile-only (operator) and umbrella-only (operator) — not workflow roots.
    makefile_names = set()
    for scripts in makefile_recipes.values():
        for p in scripts:
            makefile_names.add(Path(p).name)

    umbrella = REPO / "scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh"
    umbrella_names: set[str] = set()
    if umbrella.is_file():
        for inv in extract_invokes(umbrella, umbrella.read_text(errors="replace")):
            umbrella_names.add(Path(inv).name)

    # Direct basename mention in .github (the 78-number, including comments).
    gh_text = ""
    for p in (REPO / ".github").rglob("*"):
        if p.is_file() and p.suffix in {".yml", ".yaml", ".md"}:
            try:
                gh_text += p.read_text(errors="replace") + "\n"
            except OSError:
                pass
    direct_mention = {g.name for g in gates if g.name in gh_text}

    rows = []
    n_reach = 0
    classes = defaultdict(int)
    for g in gates:
        rel = f"scripts/ci/{g.name}"
        reachable = rel in invoked_by or rel in depth_of and depth_of.get(rel, 0) > 0
        # depth_of includes workflows at 0; a gate must have been invoked.
        reachable = rel in invoked_by
        if reachable:
            n_reach += 1
            klass = "workflow-reachable"
            how = ",".join(sorted(invoked_by[rel])[:6])
            depth = depth_of.get(rel, 0)
            origin = root_of.get(rel, "")
        else:
            klass = classify_leftover(
                g.name, g, g.name in makefile_names, g.name in umbrella_names
            )
            how = ""
            if g.name in makefile_names:
                how = "makefile"
            if g.name in umbrella_names:
                how = (how + "+umbrella") if how else "umbrella"
            depth = 0
            origin = ""
        classes[klass] += 1
        rows.append(
            {
                "gate": g.name,
                "reachable": "yes" if reachable else "no",
                "class": klass,
                "depth": depth,
                "origin_workflow": origin,
                "via": how,
                "direct_github_mention": "yes" if g.name in direct_mention else "no",
            }
        )

    # Instrument checks.
    reach_names = {r["gate"] for r in rows if r["reachable"] == "yes"}
    if not reach_names:
        print("REFUTE: 0 reachable gates — instrument is dead", file=sys.stderr)
        return 2
    for orphan in NAMED_DIRECT_ORPHANS:
        if orphan in reach_names:
            print(f"NOTE: named orphan {orphan} is reachable under invoke-trace", file=sys.stderr)
        else:
            pass
    missing_named = [o for o in NAMED_DIRECT_ORPHANS if o in reach_names]
    # Remaining forgotten leftovers must stay unreachable. Wiring one
    # without removing it from NAMED_DIRECT_ORPHANS is a REFUTE.
    if missing_named:
        print(
            f"REFUTE: named direct orphans resolved as reachable: {missing_named}",
            file=sys.stderr,
        )
        return 2

    # #1880: these two are in Contracts. If they drop out of the invoke
    # graph, the leftover census is lying the other way.
    wired_must_reach = (
        "dissertation_confidence_gate_gate.sh",
        "dissertation_frontend_parity_gate.sh",
    )
    missing_wired = [n for n in wired_must_reach if n not in reach_names]
    if missing_wired:
        print(
            f"REFUTE: #1880 wires missing from invoke graph: {missing_wired}",
            file=sys.stderr,
        )
        return 2

    leftover = len(gates) - n_reach
    summary = {
        "gates_total": len(gates),
        "workflows": [w.name for w in workflows],
        "direct_github_mention": len(direct_mention),
        "direct_mention_orphan_upper_bound": len(gates) - len(direct_mention),
        "workflow_reachable_transitive": n_reach,
        "leftover": leftover,
        "class_counts": dict(classes),
        "named_direct_orphans_unreachable": list(NAMED_DIRECT_ORPHANS),
        "dissertation_leftover": [
            n for n in DISSERTATION_GATES if n not in reach_names
        ],
        "bootstrap_chain_reachable": "bootstrap_chain_gate.sh" in reach_names,
        "positive_control": {
            "reachable_nonzero": n_reach > 0,
            "named_orphans_still_unreachable": all(
                o not in reach_names for o in NAMED_DIRECT_ORPHANS
            ),
            "scan_list_not_invoke_mli_s3": "mli_s3_bit_identity_gate.sh"
            not in reach_names,
            "dissertation_1880_wires_reachable": all(
                n in reach_names for n in wired_must_reach
            ),
        },
        "tsv": "",
    }
    if args.tsv:
        tsv_path = Path(args.tsv)
        if not tsv_path.is_absolute():
            tsv_path = REPO / tsv_path
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "gate",
            "reachable",
            "class",
            "depth",
            "origin_workflow",
            "via",
            "direct_github_mention",
        ]
        with tsv_path.open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        try:
            summary["tsv"] = str(tsv_path.relative_to(REPO))
        except ValueError:
            summary["tsv"] = str(tsv_path)
    if args.json:
        jp = REPO / args.json
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
