#!/usr/bin/env bash
# Derive TypeKind ladder positions for family E from fixtures.
# The index does not store a position. This gate prints one.
#
# Derivation (protocol v3):
#   no pass path and no refuse path           -> Garden
#   pass runs AND refuse matches expect_diag  -> Claim-ready
#   pass runs AND refuse check succeeds       -> Executable (refuse XPASS)
#   pass fails AND refuse matches expect_diag -> Reserved
#   otherwise                                 -> Hypothesis
#
# Fail the gate when a registered pass does not run, or a registered refuse
# starts passing / misses its named diagnostic. Garden rows never fail.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=../lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

INDEX="${TYPEKIND_E_INDEX:-$ROOT_DIR/tests/typekind/e/index.tsv}"
OUT="${TYPEKIND_E_OUT:-}"
SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"

[[ -x "$SOUC" ]] || { echo "TYPEKIND_E_FAIL reason=missing_souc" >&2; exit 1; }
[[ -f "$INDEX" ]] || { echo "TYPEKIND_E_FAIL reason=missing_index path=$INDEX" >&2; exit 1; }

export TYPEKIND_E_SOUC="$SOUC"
export TYPEKIND_E_ROOT="$ROOT_DIR"
export TYPEKIND_E_INDEX="$INDEX"
export TYPEKIND_E_OUT="$OUT"

python3 - <<'PY'
import os, subprocess, sys
from pathlib import Path

root = Path(os.environ["TYPEKIND_E_ROOT"])
souc = os.environ["TYPEKIND_E_SOUC"]
index = Path(os.environ["TYPEKIND_E_INDEX"])
out = os.environ.get("TYPEKIND_E_OUT") or ""

pass_run_fail = 0
refuse_xpass = 0
refuse_nodiag = 0
rows = 0

print("kind\tpass_rc\trefuse_rc\texpect_diag\tdeepest_layer\tderived")


def run_cmd(args):
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


for raw in index.read_text().splitlines():
    if not raw.strip() or raw.startswith("#"):
        continue
    cols = raw.split("\t")
    while len(cols) < 5:
        cols.append("")
    kind, pass_path, refuse_path, expect_diag, deepest = cols[:5]
    rows += 1
    pass_rc, refuse_rc = "-", "-"

    if not pass_path and not refuse_path:
        print(f"{kind}\t{pass_rc}\t{refuse_rc}\t{expect_diag}\t{deepest}\tGarden")
        continue

    pass_ok = False
    if pass_path:
        p = root / pass_path
        if not p.is_file():
            print(f"TYPEKIND_E_FAIL reason=missing_pass_file kind={kind} path={pass_path}", file=sys.stderr)
            sys.exit(1)
        rc, log = run_cmd([souc, "run", str(p)])
        pass_rc = str(rc)
        if rc == 0:
            pass_ok = True
        else:
            pass_run_fail += 1
            print(f"TYPEKIND_E_PASS_FAIL kind={kind} path={pass_path}", file=sys.stderr)
            print("\n".join(log.splitlines()[-8:]), file=sys.stderr)

    refuse_ok = False
    refuse_passed = False
    if refuse_path:
        p = root / refuse_path
        if not p.is_file():
            print(f"TYPEKIND_E_FAIL reason=missing_refuse_file kind={kind} path={refuse_path}", file=sys.stderr)
            sys.exit(1)
        rc, log = run_cmd([souc, "check", str(p)])
        refuse_rc = str(rc)
        if rc == 0:
            refuse_passed = True
            refuse_xpass += 1
            print(f"TYPEKIND_E_REFUSE_XPASS kind={kind} path={refuse_path}", file=sys.stderr)
        elif expect_diag and expect_diag in log:
            refuse_ok = True
        else:
            refuse_nodiag += 1
            print(f"TYPEKIND_E_REFUSE_NODIAG kind={kind} expect={expect_diag} path={refuse_path}", file=sys.stderr)
            print("\n".join(log.splitlines()[-12:]), file=sys.stderr)

    if pass_ok and refuse_ok:
        derived = "Claim-ready"
    elif pass_ok and refuse_passed:
        derived = "Executable"
    elif (not pass_ok) and pass_path and refuse_ok:
        derived = "Reserved"
    elif pass_ok:
        derived = "Executable"
    else:
        derived = "Hypothesis"

    print(f"{kind}\t{pass_rc}\t{refuse_rc}\t{expect_diag}\t{deepest}\t{derived}")

print(f"TYPEKIND_E_DERIVED rows={rows} pass_fail={pass_run_fail} refuse_xpass={refuse_xpass} refuse_nodiag={refuse_nodiag}")
if out:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        f"TYPEKIND_E_DERIVED rows={rows} pass_fail={pass_run_fail} refuse_xpass={refuse_xpass} refuse_nodiag={refuse_nodiag}\n"
    )

if pass_run_fail or refuse_xpass or refuse_nodiag:
    print(
        f"TYPEKIND_E_FAIL pass_fail={pass_run_fail} refuse_xpass={refuse_xpass} refuse_nodiag={refuse_nodiag}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"TYPEKIND_E_PASS rows={rows}")
PY
