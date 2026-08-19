#!/usr/bin/env bash
# gate_reachability_gate.sh — a gate that does not run is not a pass.
#
# Enumerates scripts/ci/*.sh (non-recursive — the founder 547) and
# requires a manifest row for each. States:
#   admitido     must be named under .github/ (git grep -F <basename>)
#   nao-admitido written, deliberately not wired; owner AND reason required
#   obsoleto     candidate to delete; owner AND reason required
# Missing row = red. Malformed (empty owner/reason, unknown state,
# duplicate, stale row) is redder than absent.
#
# Visibility: counts and the full nao-admitido list print even on green.
# No age expiry. The list is the debt.
#
# usage:
#   bash scripts/ci/gate_reachability_gate.sh
#   bash scripts/ci/gate_reachability_gate.sh --bootstrap > scripts/ci/gate_reachability.manifest.tsv
#   bash scripts/ci/gate_reachability_gate.sh --self-test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="check"
MANIFEST="${GATE_REACHABILITY_MANIFEST:-$ROOT_DIR/scripts/ci/gate_reachability.manifest.tsv}"
SCRIPTS_DIR="${GATE_REACHABILITY_SCRIPTS_DIR:-$ROOT_DIR/scripts/ci}"
GITHUB_DIR="${GATE_REACHABILITY_GITHUB_DIR:-$ROOT_DIR/.github}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap) MODE="bootstrap"; shift ;;
    --self-test) MODE="self-test"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --scripts-dir) SCRIPTS_DIR="$2"; shift 2 ;;
    --github-dir) GITHUB_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "gate_reachability: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

export GATE_REACHABILITY_ROOT="$ROOT_DIR"
export GATE_REACHABILITY_MANIFEST="$MANIFEST"
export GATE_REACHABILITY_SCRIPTS_DIR="$SCRIPTS_DIR"
export GATE_REACHABILITY_GITHUB_DIR="$GITHUB_DIR"
export GATE_REACHABILITY_MODE="$MODE"

python3 - <<'PY'
import os, sys, tempfile, shutil, pathlib, subprocess

ROOT = pathlib.Path(os.environ["GATE_REACHABILITY_ROOT"])
MANIFEST = pathlib.Path(os.environ["GATE_REACHABILITY_MANIFEST"])
SCRIPTS_DIR = pathlib.Path(os.environ["GATE_REACHABILITY_SCRIPTS_DIR"])
GITHUB_DIR = pathlib.Path(os.environ["GATE_REACHABILITY_GITHUB_DIR"])
MODE = os.environ["GATE_REACHABILITY_MODE"]

STATES = ("admitido", "nao-admitido", "obsoleto")
BOOTSTRAP_OWNER = "por atribuir"
BOOTSTRAP_REASON = "herdado no bootstrap 2026-08-19, estado por rever"
SELF = "gate_reachability_gate.sh"

def list_gates(scripts_dir: pathlib.Path):
    if not scripts_dir.is_dir():
        return []
    return sorted(p.name for p in scripts_dir.glob("*.sh") if p.is_file())

def github_blob(github_dir: pathlib.Path) -> str:
    """Same instrument as `git grep -F <basename> -- .github/`: substring over .github/."""
    if not github_dir.is_dir():
        return ""
    chunks = []
    for p in sorted(github_dir.rglob("*")):
        if not p.is_file():
            continue
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)

def named_set(gates, blob: str):
    return {g for g in gates if g in blob}

def parse_manifest(path: pathlib.Path):
    """Return (rows, errors). rows: list of dicts. Missing file → (None, [missing])."""
    if not path.is_file():
        return None, [f"missing-manifest={path}"]
    rows = []
    errors = []
    seen = {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [], [f"empty-manifest={path}"]
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts[0].strip() == "gate" and len(parts) > 1 and parts[1].strip() == "state":
            continue
        if len(parts) < 4:
            errors.append(
                f"malformed:{path}:{lineno}: want gate<TAB>state<TAB>owner<TAB>reason, got {len(parts)} field(s)"
            )
            continue
        gate, state, owner, reason = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        extra = [p for p in parts[4:] if p.strip()]
        if extra:
            errors.append(f"malformed:{path}:{lineno}: extra fields after reason")
        if not gate:
            errors.append(f"malformed:{path}:{lineno}: empty gate")
            continue
        if gate in seen:
            errors.append(f"duplicate:{gate}: lines {seen[gate]} and {lineno}")
        else:
            seen[gate] = lineno
        if state not in STATES:
            errors.append(f"malformed:{gate}: unknown-state={state!r} (want admitido|nao-admitido|obsoleto)")
        if state in ("nao-admitido", "obsoleto"):
            if not owner:
                errors.append(f"malformed:{gate}: {state} missing owner")
            if not reason:
                errors.append(f"malformed:{gate}: {state} missing reason")
        rows.append({"gate": gate, "state": state, "owner": owner, "reason": reason, "line": lineno})
    return rows, errors

def classify(gates, rows, named):
    by_gate = {r["gate"]: r for r in (rows or [])}
    fails = []
    counts = {s: 0 for s in STATES}
    debt = []  # nao-admitido / obsoleto rows, for the always-on print
    for g in gates:
        row = by_gate.get(g)
        if row is None:
            fails.append(f"unlisted={g}")
            continue
        st = row["state"]
        if st in counts:
            counts[st] += 1
        if st in ("nao-admitido", "obsoleto"):
            debt.append(row)
        is_named = g in named
        if st == "admitido" and not is_named:
            fails.append(f"admitido-unwired={g}")
        if st in ("nao-admitido", "obsoleto") and is_named:
            fails.append(f"declared-{st}-but-named={g}")
    if rows is not None:
        live = set(gates)
        for r in rows:
            if r["gate"] not in live:
                fails.append(f"stale={r['gate']}")
    return fails, counts, debt

def print_report(gates, named, counts, debt, fails, extra_errors):
    unnamed = [g for g in gates if g not in named]
    print("GATE_REACHABILITY")
    print(f"scripts={len(gates)} named={len(named)} unnamed={len(unnamed)}")
    print(
        "manifest_"
        + " ".join(f"{s}={counts.get(s, 0)}" for s in STATES)
    )
    print(f"nao-admitido ({sum(1 for r in debt if r['state']=='nao-admitido')}):")
    for r in debt:
        if r["state"] != "nao-admitido":
            continue
        print(f"  {r['gate']}  owner={r['owner']}  reason={r['reason']}")
    obs = [r for r in debt if r["state"] == "obsoleto"]
    print(f"obsoleto ({len(obs)}):")
    for r in obs:
        print(f"  {r['gate']}  owner={r['owner']}  reason={r['reason']}")
    all_err = list(extra_errors) + list(fails)
    if all_err:
        print("GATE_REACHABILITY_FAIL")
        for e in all_err:
            print(f"  {e}")
        return 1
    print("GATE_REACHABILITY_OK")
    return 0

def bootstrap_tsv(gates, named):
    lines = [
        "# gate_reachability.manifest.tsv",
        "# Columns: gate<TAB>state<TAB>owner<TAB>reason",
        "# state ∈ admitido | nao-admitido | obsoleto",
        "# admitido: must be named under .github/; owner/reason may be empty",
        "# nao-admitido / obsoleto: owner AND reason must be non-empty",
        "# A scripts/ci/*.sh with no row is RED. Malformed is redder than absent.",
        f"# Bootstrap 2026-08-19 on {len(gates)} top-level scripts/ci/*.sh.",
        "# named → admitido; unnamed → nao-admitido owner=por atribuir",
        "gate\tstate\towner\treason",
    ]
    for g in gates:
        if g in named:
            lines.append(f"{g}\tadmitido\t\t")
        else:
            lines.append(f"{g}\tnao-admitido\t{BOOTSTRAP_OWNER}\t{BOOTSTRAP_REASON}")
    return "\n".join(lines) + "\n"

def run_check(scripts_dir, github_dir, manifest, require_manifest=True):
    gates = list_gates(scripts_dir)
    blob = github_blob(github_dir)
    named = named_set(gates, blob)
    extra = []
    if require_manifest and not manifest.is_file():
        extra.append(f"missing-manifest={manifest}")
        # Still classify: every live script is unlisted.
        fails, counts, debt = classify(gates, [], named)
        return print_report(gates, named, counts, debt, fails, extra)
    rows, parse_err = parse_manifest(manifest) if manifest.is_file() else ([], [f"missing-manifest={manifest}"])
    extra.extend(parse_err)
    fails, counts, debt = classify(gates, rows or [], named)
    return print_report(gates, named, counts, debt, fails, extra)

def write_fixture(base: pathlib.Path):
    scripts = base / "scripts" / "ci"
    github = base / ".github" / "workflows"
    scripts.mkdir(parents=True)
    github.mkdir(parents=True)
    (scripts / "on_ci.sh").write_text("#!/bin/bash\n")
    (scripts / "off_ci.sh").write_text("#!/bin/bash\n")
    (scripts / "ghost.sh").write_text("#!/bin/bash\n")
    (github / "ci.yml").write_text("run: bash scripts/ci/on_ci.sh\n")
    return scripts, github

def self_test():
    failures = []

    def expect(rc, pred, msg):
        if not pred:
            failures.append(msg + f" (rc={rc})")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gate-reach-"))
    try:
        scripts, github = write_fixture(tmp)
        man = tmp / "manifest.tsv"

        # 1. no manifest → red, and every script unlisted
        rc = run_check(scripts, github, man, require_manifest=True)
        expect(rc, rc == 1, "no-manifest must be red")

        # 2. well-formed bootstrap-shaped manifest → green
        man.write_text(
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tadmitido\t\t\n"
            "off_ci.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
            "ghost.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 0, "bootstrap-shaped fixture must be green")

        # header after comments — the committed bootstrap file looks like this
        man.write_text(
            "# comment\n"
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tadmitido\t\t\n"
            "off_ci.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
            "ghost.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 0, "header-after-comments must be green")

        # 3. admitido but unnamed → red (the negative phase)
        man.write_text(
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tadmitido\t\t\n"
            "off_ci.sh\tadmitido\t\t\n"
            "ghost.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 1, "admitido-unwired must be red")

        # 4. nao-admitido missing owner → redder than a review
        man.write_text(
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tadmitido\t\t\n"
            "off_ci.sh\tnao-admitido\t\therdado\n"
            "ghost.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 1, "nao-admitido without owner must be red")

        # 5. missing row → red
        man.write_text(
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tadmitido\t\t\n"
            "off_ci.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 1, "unlisted ghost.sh must be red")

        # 6. declared nao-admitido but named → red
        man.write_text(
            "gate\tstate\towner\treason\n"
            "on_ci.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
            "off_ci.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
            "ghost.sh\tnao-admitido\tpor atribuir\therdado no bootstrap 2026-08-19, estado por rever\n"
        )
        rc = run_check(scripts, github, man)
        expect(rc, rc == 1, "named-but-nao-admitido must be red")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("GATE_REACHABILITY_SELFTEST_FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("GATE_REACHABILITY_SELFTEST_OK")
    return 0

if MODE == "self-test":
    sys.exit(self_test())

gates = list_gates(SCRIPTS_DIR)
blob = github_blob(GITHUB_DIR)
named = named_set(gates, blob)

if MODE == "bootstrap":
    sys.stdout.write(bootstrap_tsv(gates, named))
    sys.exit(0)

sys.exit(run_check(SCRIPTS_DIR, GITHUB_DIR, MANIFEST))
PY
