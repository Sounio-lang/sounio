#!/usr/bin/env python3
"""Second-pass leftover classification for scripts/ci/*_gate.sh.

Pass 1 (ci_gate_workflow_reachability.py) answered: can a workflow execute
this gate? This pass answers, for every leftover: obsolete, manual-by-design,
forgotten, or still unclassified.

Evidence-only. A filename prefix is not a class. Quoted historical phrases
("the old header said SUPERSEDED") are not obsolete. A scan-list of other
gates (sigpipe hygiene) is not an invoke.

Positive controls (must fire or the instrument is dead):

  * leftover set non-empty
  * the four still-unwired dissertation leftovers stay leftover
    (dossier, hessian, pbpk28, suite)
  * the two #1880 wires stay reachable
    (confidence, frontend_parity) — they were leftover on 64924d371a
  * mli_s3_bit_identity_gate.sh leftover (listed by sigpipe, never executed)
  * lean_single_fixed_point_gate.sh is NOT obsolete (it quotes a dead header)

The 6/6 leftover fact is SHA-bound: true on 64924d371a, false as a
present-tense claim after 12ebda238d (#1880). Do not re-introduce a
control that REFUTEs because those two are now in Contracts.

Usage:
  python3 scripts/dev/ci_gate_leftover_class_pass2.py
  python3 scripts/dev/ci_gate_leftover_class_pass2.py --tsv PATH --json PATH
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

INVOKE_RE = re.compile(
    r"""(?x)
    (?:^|&&|\|\||;|`|\(|\s)
    (?:bash|sh|source|\.)
    \s+
    (?:[A-Za-z0-9_]+=\S+\s+)*
    (?P<q>["']?)
    (?:\$\{?(?:ROOT_DIR|ROOT|PWD|GITHUB_WORKSPACE)\}?/? )?
    (?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+\.sh)
    (?P=q)
    """
)
YAML_RUN_RE = re.compile(
    r"""(?x)run:\s*(?:\|)?\s*(?:bash|sh)\s+(?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+\.sh)"""
)
BARE_GATE_RE = re.compile(r"(?P<path>scripts/(?:ci|dev)/[A-Za-z0-9_./-]+_gate\.sh)")
INVENTORY_MARKERS = ("git ls-files", "mapfile", "find ", "rglob")

DISSERTATION_GATES = (
    "dissertation_confidence_gate_gate.sh",
    "dissertation_dossier_gate.sh",
    "dissertation_frontend_parity_gate.sh",
    "dissertation_pbpk28_parity_gate.sh",
    "dissertation_pbpk_hessian_gate.sh",
    "dissertation_pbpk_suite_gate.sh",
)
# 6/6 leftover was measured on this SHA. After #1880 (12ebda238d) the
# first and third names are in Contracts. A control that demands all
# six leftover expires the moment that wire lands.
DISSERTATION_SIX_ALL_LEFTOVER_SHA = "64924d371a"
DISSERTATION_WIRED_AS_OF_1880 = (
    "dissertation_confidence_gate_gate.sh",
    "dissertation_frontend_parity_gate.sh",
)
DISSERTATION_STILL_LEFTOVER = (
    "dissertation_dossier_gate.sh",
    "dissertation_pbpk28_parity_gate.sh",
    "dissertation_pbpk_hessian_gate.sh",
    "dissertation_pbpk_suite_gate.sh",
)

# Quoted / historical — do not treat as a current obsolete assertion.
OBSOLETE_QUOTE_MARKERS = (
    "old header",
    "earlier draft",
    "which read as if",
    "was equally wrong",
    "the old header said",
)

OBSOLETE_ASSERTIONS = (
    "this gate is obsolete",
    "this script is obsolete",
    "do not use this gate",
    "do not use this script",
    "retired; use",
    "historical only — do not",
)

MANUAL_MARKERS = (
    "not wired",
    "not in ci",
    "not in github",
    "operator-run",
    "run on slurm",
    "slurm-only",
    "manual gate",
    "run by hand",
    "not a ci job",
    "reachable only by hand",
    "handoff only",
    "make check only",
)

FORGOTTEN_MARKERS = (
    "gate_contract",
    "positive control",
    "must fail if",
    "evidence gate",
    "acceptance:",
    "exit 0 = pass",
    "hard gate",
    "hard path",
    "fail-closed",
)


def is_comment_line(path: Path, line: str) -> bool:
    s = line.lstrip()
    if not s or s.startswith("#!"):
        return False
    return s.startswith("#")


def extract_invokes(path: Path, text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        if is_comment_line(path, line):
            continue
        if any(m in line for m in INVENTORY_MARKERS) and "_gate.sh" in line:
            continue
        for rx in (INVOKE_RE, YAML_RUN_RE):
            for m in rx.finditer(line):
                found.add(m.group("path"))
        # Bare paths in YAML for-loops are invokes. Bare paths in .sh are
        # often scan lists (sigpipe hygiene) and are not execution.
        if path.suffix in {".yml", ".yaml"}:
            for m in BARE_GATE_RE.finditer(line):
                found.add(m.group("path"))
    return found


def parse_makefile(makefile: Path) -> set[str]:
    names: set[str] = set()
    if not makefile.is_file():
        return names
    text = makefile.read_text(errors="replace")
    current: list[str] = []
    recipes: dict[str, set[str]] = defaultdict(set)
    for line in text.splitlines():
        if line.startswith("\t"):
            for t in current:
                recipes[t] |= extract_invokes(makefile, line)
            continue
        if ":" in line and not line.startswith("\t"):
            left = line.split(":", 1)[0]
            current = [x.strip() for x in left.split() if x.strip()]
    for scripts in recipes.values():
        for p in scripts:
            names.add(Path(p).name)
    return names


def header_comments(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()[:n]
    except OSError:
        return ""
    return "\n".join(ln for ln in lines if ln.lstrip().startswith("#")).lower()


def classify_leftover(
    name: str, gate_path: Path, makefile_hit: bool, umbrella_hit: bool
) -> str:
    head = header_comments(gate_path)
    if any(k in head for k in OBSOLETE_QUOTE_MARKERS):
        pass
    elif any(k in head for k in OBSOLETE_ASSERTIONS):
        return "obsolete"
    if name == "bootstrap_chain_gate.sh" or makefile_hit or umbrella_hit:
        return "manual-by-design"
    if any(k in head for k in MANUAL_MARKERS):
        return "manual-by-design"
    blob_head = ""
    try:
        blob_head = gate_path.read_text(errors="replace")[:3000]
    except OSError:
        pass
    if "GATE_CONTRACT" in blob_head or any(k in head for k in FORGOTTEN_MARKERS):
        return "forgotten"
    return "unclassified"


def github_text() -> str:
    out = []
    gh = REPO / ".github"
    if not gh.is_dir():
        return ""
    for p in gh.rglob("*"):
        if p.is_file() and p.suffix in {".yml", ".yaml", ".md"}:
            try:
                out.append(p.read_text(errors="replace"))
            except OSError:
                pass
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tsv",
        default="docs/audit/CI_GATE_LEFTOVER_CLASS_PASS2_2026-08-18.tsv",
    )
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    gates = sorted((REPO / "scripts/ci").glob("*_gate.sh"))
    if len(gates) < 400:
        print(f"REFUTE: expected >=400 *_gate.sh, got {len(gates)}", file=sys.stderr)
        return 2

    workflows = sorted((REPO / ".github/workflows").glob("*.yml"))
    sh_files = list((REPO / "scripts/ci").glob("*.sh")) + list(
        (REPO / "scripts/dev").glob("*.sh")
    )
    edges: dict[str, set[str]] = defaultdict(set)
    for p in list(workflows) + sh_files:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(REPO))
        edges[rel] |= extract_invokes(p, text)

    invoked_by: dict[str, set[str]] = defaultdict(set)
    depth_of: dict[str, int] = {}
    q: deque[str] = deque()
    for w in workflows:
        rel = str(w.relative_to(REPO))
        depth_of[rel] = 0
        q.append(rel)
    while q:
        cur = q.popleft()
        for nxt in edges.get(cur, ()):
            invoked_by[nxt].add(cur)
            if nxt not in depth_of:
                depth_of[nxt] = depth_of[cur] + 1
                q.append(nxt)

    makefile_names = parse_makefile(REPO / "Makefile")
    umbrella = REPO / "scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh"
    umbrella_names: set[str] = set()
    if umbrella.is_file():
        for inv in extract_invokes(umbrella, umbrella.read_text(errors="replace")):
            umbrella_names.add(Path(inv).name)

    gh = github_text()
    rows = []
    classes: dict[str, int] = defaultdict(int)
    n_reach = 0
    for g in gates:
        rel = f"scripts/ci/{g.name}"
        reachable = rel in invoked_by
        if reachable:
            n_reach += 1
            klass = "workflow-reachable"
            via = ",".join(sorted(invoked_by[rel])[:6])
        else:
            klass = classify_leftover(
                g.name, g, g.name in makefile_names, g.name in umbrella_names
            )
            via = ""
            if g.name in makefile_names:
                via = "makefile"
            if g.name in umbrella_names:
                via = (via + "+umbrella") if via else "umbrella"
        classes[klass] += 1
        rows.append(
            {
                "gate": g.name,
                "reachable": "yes" if reachable else "no",
                "class": klass,
                "via": via,
                "direct_github_mention": "yes" if g.name in gh else "no",
                "dissertation_six": "yes" if g.name in DISSERTATION_GATES else "no",
            }
        )

    leftover = len(gates) - n_reach
    if leftover == 0:
        print("REFUTE: leftover=0 — instrument or population is dead", file=sys.stderr)
        return 2
    if n_reach == 0:
        print("REFUTE: reachable=0 — instrument is dead", file=sys.stderr)
        return 2

    diss_rows = [r for r in rows if r["dissertation_six"] == "yes"]
    if len(diss_rows) != 6:
        print(f"REFUTE: expected 6 dissertation gates, got {len(diss_rows)}", file=sys.stderr)
        return 2
    by_name = {r["gate"]: r for r in diss_rows}
    ci_yml = (REPO / ".github/workflows/ci.yml").read_text(errors="replace")
    # Dual-era control: the 6/6 leftover fact is true on 64924d371a and
    # false as a present-tense claim after 12ebda238d. Detect which era
    # this tree is in by whether the #1880 run: lines exist.
    wired_in_this_tree = all(
        f"scripts/ci/{n}" in ci_yml for n in DISSERTATION_WIRED_AS_OF_1880
    )
    if wired_in_this_tree:
        missing_wires = [
            n for n in DISSERTATION_WIRED_AS_OF_1880 if by_name[n]["reachable"] != "yes"
        ]
        if missing_wires:
            print(
                f"REFUTE: #1880 wires missing from invoke graph: {missing_wires}",
                file=sys.stderr,
            )
            return 2
        leaked_reds = [
            n for n in DISSERTATION_STILL_LEFTOVER if by_name[n]["reachable"] == "yes"
        ]
        if leaked_reds:
            print(
                f"REFUTE: still-red dissertation leftover is now reachable: {leaked_reds}",
                file=sys.stderr,
            )
            return 2
    else:
        if any(r["reachable"] == "yes" for r in diss_rows):
            print("REFUTE: a dissertation gate is workflow-reachable", file=sys.stderr)
            return 2
        if any(r["direct_github_mention"] == "yes" for r in diss_rows):
            print("REFUTE: a dissertation gate is named in .github/", file=sys.stderr)
            return 2

    mli = next(r for r in rows if r["gate"] == "mli_s3_bit_identity_gate.sh")
    if mli["reachable"] == "yes":
        print(
            "REFUTE: mli_s3 treated as reachable via a scan-list (sigpipe)",
            file=sys.stderr,
        )
        return 2

    lean_fp = next(r for r in rows if r["gate"] == "lean_single_fixed_point_gate.sh")
    if lean_fp["class"] == "obsolete":
        print(
            "REFUTE: lean_single_fixed_point classified obsolete "
            "(header quotes a dead SUPERSEDED line)",
            file=sys.stderr,
        )
        return 2

    tsv_path = Path(args.tsv)
    if not tsv_path.is_absolute():
        tsv_path = REPO / tsv_path
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "gate",
        "reachable",
        "class",
        "via",
        "direct_github_mention",
        "dissertation_six",
    ]
    with tsv_path.open("w") as f:
        f.write(
            "# leftover class pass-2. 6/6 leftover measured on "
            f"{DISSERTATION_SIX_ALL_LEFTOVER_SHA}; "
            "at 12ebda238d (#1880) two of the six are in Contracts "
            "(confidence, frontend_parity).\n"
        )
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    leftover_classes = {
        k: v for k, v in classes.items() if k != "workflow-reachable"
    }
    summary = {
        "gates_total": len(gates),
        "workflow_reachable_transitive": n_reach,
        "leftover": leftover,
        "class_counts": dict(classes),
        "leftover_class_counts": leftover_classes,
        "dissertation_six": {
            r["gate"]: {
                "reachable": r["reachable"],
                "class": r["class"],
                "via": r["via"],
                "direct_github_mention": r["direct_github_mention"],
            }
            for r in diss_rows
        },
        "positive_control": {
            "leftover_nonzero": leftover > 0,
            "dissertation_six_all_leftover_on": DISSERTATION_SIX_ALL_LEFTOVER_SHA,
            "dissertation_era": (
                "post-12ebda238d" if wired_in_this_tree else "pre-12ebda238d"
            ),
            "dissertation_1880_wires_reachable": wired_in_this_tree,
            "dissertation_still_leftover_unreachable": True,
            "mli_s3_not_reachable_via_scan_list": True,
            "lean_single_fixed_point_not_obsolete": True,
        },
        "tsv": (
            str(tsv_path.relative_to(REPO))
            if tsv_path.is_relative_to(REPO)
            else str(tsv_path)
        ),
    }
    if args.json:
        jp = Path(args.json)
        if not jp.is_absolute():
            jp = REPO / jp
        jp.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
