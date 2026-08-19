#!/usr/bin/env bash
# kind_ladder_gate.sh — derive TypeKind ladder positions from fixtures.
#
# Protocol v3: the position is CALCULATED, not asserted. The index stores
# kind, pass path, refuse path, expected diagnostic, deepest named layer.
# It does not store a position. This script emits the table.
#
# Pattern copied from scripts/ci/known_failure_madaros_recheck.sh:
# a listed refuse that starts to pass is XPASS (blocker fell). A listed
# pass that starts to fail is a regression. Both stop the merge.
#
# Instrument: scripts/lib/gate_assert.sh. Artefact exists and is ELF;
# engine is read from the compile log (never --version); rc is file-backed
# (never grepped out of a pipe); skip is rc=77, not 0.
#
# Day-zero: judge only kinds that already have fixtures. A kind with no
# paths is Garden by default. Do not require 99 pairs.
#
# Positive control (must fire before the real index is trusted):
#   M1 — refuse fixture that type-checks  → gate must fail
#   M2 — pass fixture that is i64 = true  → gate must fail
# If either mutant is accepted, this is not a gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "kind_ladder_gate"

# Inherited SOUC_BIN on this pod points at the integration checkout.
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SOUC="${KIND_LADDER_SOUC:-$ROOT/bin/souc}"
INDEX="${KIND_LADDER_INDEX:-$ROOT/tests/archaeology/kind_ladder/index.tsv}"
OUT_DIR="${KIND_LADDER_OUT:-}"
SELFTEST_ONLY=0
SKIP_SELFTEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --index) INDEX="$2"; shift 2 ;;
    --souc) SOUC="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --selftest-only) SELFTEST_ONLY=1; shift ;;
    --skip-selftest) SKIP_SELFTEST=1; shift ;;
    -h|--help)
      echo "usage: $0 [--index PATH] [--souc PATH] [--out-dir DIR] [--selftest-only]"
      exit 0
      ;;
    *) gate_fail "unknown argument: $1" ;;
  esac
done

require_tool timeout "timeout(1) missing — cannot bound compile/check"
require_tool od "od(1) missing — cannot check ELF magic"
require_tool python3 "python3 missing — cannot parse the index"
require_executable "$SOUC" "souc not executable: $SOUC"
# bin/souc is a bash shim. The engine that must exist is the Madaros ELF
# the shim would exec (same candidate order as bin/souc).
MADAROS_ELF=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
    MADAROS_ELF="$cand"
    break
  fi
done
[[ -n "$MADAROS_ELF" ]] \
  || gate_skip "no Madaros ELF (build Madaros or set MADAROS_RAW_BIN) — not a green 0"
require_elf "$MADAROS_ELF" "Madaros engine is not a native ELF: $MADAROS_ELF"

OWN_OUT=0
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kind-ladder.XXXXXX")"
  OWN_OUT=1
  trap 'rm -rf "$OUT_DIR"' EXIT
else
  mkdir -p "$OUT_DIR"
fi

WORKDIR="$OUT_DIR/work"
mkdir -p "$WORKDIR" "$OUT_DIR/logs"
TABLE="$OUT_DIR/derived.tsv"

echo "KIND_LADDER_GATE_START"
echo "souc=$SOUC"
echo "madaros_elf=$MADAROS_ELF"
echo "index=$INDEX"
echo "out=$OUT_DIR"

# ---------------------------------------------------------------------------
# One kind. Writes a TSV line to stdout. Return 0 = polarity ok, 1 = block.
# Never read rc through a pipe.
# ---------------------------------------------------------------------------
blank() { [[ -z "${1:-}" || "$1" == "-" ]]; }

run_pass_fixture() {
  local label="$1" src="$2"
  local elf="$WORKDIR/${label}.elf"
  local clog="$OUT_DIR/logs/${label}.pass.compile.log"
  local rlog="$OUT_DIR/logs/${label}.pass.run.log"
  rm -f -- "$PWD/-o" "$elf"
  gate_capture_rc "$WORKDIR/${label}.pass.compile.rc" -- \
    timeout 120 "$SOUC" compile "$src" -o "$elf" >"$clog" 2>&1
  require_rc_file "$WORKDIR/${label}.pass.compile.rc"
  local crc engine
  crc="$(cat "$WORKDIR/${label}.pass.compile.rc")"
  engine="$(classify_compile_log "$clog")"
  if [[ "$crc" != "0" ]]; then
    printf '%s\t%s\n' "$crc" "$engine"
    return 0
  fi
  if [[ -e "$PWD/-o" ]]; then
    gate_fail "$label compile wrote a literal -o file"
  fi
  require_elf "$elf" "$label pass compile artefact"
  [[ "$engine" == "madaros" ]] \
    || gate_fail "$label pass compile log named engine=$engine (want madaros) — $clog"
  gate_capture_rc "$WORKDIR/${label}.pass.run.rc" -- \
    timeout 60 "$elf" >"$rlog" 2>&1
  require_rc_file "$WORKDIR/${label}.pass.run.rc"
  local rrc
  rrc="$(cat "$WORKDIR/${label}.pass.run.rc")"
  if [[ "$rrc" != "0" ]]; then
    printf '%s\t%s\n' "$rrc" "$engine"
    return 0
  fi
  printf '0\t%s\n' "$engine"
}

run_refuse_fixture() {
  local label="$1" src="$2"
  local clog="$OUT_DIR/logs/${label}.refuse.check.log"
  gate_capture_rc "$WORKDIR/${label}.refuse.rc" -- \
    timeout 120 "$SOUC" check "$src" >"$clog" 2>&1
  require_rc_file "$WORKDIR/${label}.refuse.rc"
  cat "$WORKDIR/${label}.refuse.rc"
}

# judge_kind <kind> <pass> <refuse> <diag> <layer>
# Prints one derived TSV line. Writes $WORKDIR/last.block (0|1).
# Callers must not rely on JUDGE_BLOCK across $(...) — that is a subshell.
judge_kind() {
  local kind="$1" pass="$2" refuse="$3" diag="$4" layer="$5"
  local pass_st="-" refuse_st="-" position note="-" block=0

  if blank "$pass" && blank "$refuse"; then
    printf '%s\tGarden\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$kind" "-" "-" "$diag" "$layer" "-" "-"
    printf '0\n' >"$WORKDIR/last.block"
    return 0
  fi

  if blank "$pass" || blank "$refuse"; then
    position="Hypothesis"
    if ! blank "$pass"; then
      require_file "$ROOT/$pass" "pass fixture missing: $pass"
      local pr
      pr="$(run_pass_fixture "$kind" "$ROOT/$pass")"
      pass_st="${pr%%$'\t'*}"
      if [[ "$pass_st" != "0" ]]; then
        note="REGRESS"
        block=1
      fi
    fi
    if ! blank "$refuse"; then
      require_file "$ROOT/$refuse" "refuse fixture missing: $refuse"
      refuse_st="$(run_refuse_fixture "$kind" "$ROOT/$refuse")"
      if [[ "$refuse_st" == "0" ]]; then
        note="XPASS"
        block=1
      elif [[ -n "$diag" && "$diag" != "-" ]]; then
        if [[ "$(count_matches "$diag" "$OUT_DIR/logs/${kind}.refuse.check.log" --fixed)" == "0" ]]; then
          note="DIAG"
          block=1
        fi
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$kind" "$position" "$pass" "$refuse" "$diag" "$layer" "$pass_st" "$refuse_st" "$note"
    printf '%s\n' "$block" >"$WORKDIR/last.block"
    return 0
  fi

  require_file "$ROOT/$pass" "pass fixture missing: $pass"
  require_file "$ROOT/$refuse" "refuse fixture missing: $refuse"

  local pr
  pr="$(run_pass_fixture "$kind" "$ROOT/$pass")"
  pass_st="${pr%%$'\t'*}"
  refuse_st="$(run_refuse_fixture "$kind" "$ROOT/$refuse")"

  local refuse_has_diag=0
  if [[ -n "$diag" && "$diag" != "-" ]]; then
    if [[ "$(count_matches "$diag" "$OUT_DIR/logs/${kind}.refuse.check.log" --fixed)" != "0" ]]; then
      refuse_has_diag=1
    fi
  fi

  if [[ "$pass_st" == "0" && "$refuse_st" != "0" ]]; then
    if [[ "$refuse_has_diag" -eq 1 || -z "$diag" || "$diag" == "-" ]]; then
      position="Claim-ready"
    else
      position="Hypothesis"
      note="DIAG"
      block=1
    fi
  elif [[ "$pass_st" != "0" && "$refuse_st" != "0" ]]; then
    if [[ "$refuse_has_diag" -eq 1 ]]; then
      position="Reserved"
      note="-"
    else
      position="Hypothesis"
      note="DIAG"
      block=1
    fi
  elif [[ "$pass_st" == "0" && "$refuse_st" == "0" ]]; then
    position="Executable"
    note="XPASS"
    block=1
  else
    position="Hypothesis"
    note="REGRESS"
    block=1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$position" "$pass" "$refuse" "$diag" "$layer" "$pass_st" "$refuse_st" "$note"
  printf '%s\n' "$block" >"$WORKDIR/last.block"
}

# ---------------------------------------------------------------------------
# Positive control. A helper that never fails is not a helper.
# ---------------------------------------------------------------------------
selftest_mutants() {
  local mdir="$WORKDIR/mutants"
  mkdir -p "$mdir"

  cat > "$mdir/ok.sio" <<'EOF'
fn main() -> i64 {
    let x: i64 = 1
    let _y = x + x
    0
}
EOF
  cat > "$mdir/bad.sio" <<'EOF'
fn main() -> i64 {
    let x: i64 = true
    0
}
EOF

  # M1: a valid program listed as refuse. Must XPASS / block.
  local line m1_refuse_rc
  m1_refuse_rc="$(run_refuse_fixture MUTANT_REFUSE_PASSES "$mdir/ok.sio")"
  if [[ "$m1_refuse_rc" == "0" ]]; then
    echo "selftest M1: refuse-that-passes produced rc=0 (the bad polarity exists)"
  else
    gate_fail "selftest M1: expected a passing program to check OK; got rc=$m1_refuse_rc — mutant is not a mutant"
  fi
  # The gate's rule: listed refuse rc==0 is XPASS and must block.
  if [[ "$m1_refuse_rc" == "0" ]]; then
    echo "selftest M1 FIRED: refuse fixture that passes is XPASS (would stop merge)"
  fi

  # M2: list a type error as the pass fixture. Must REGRESS / block.
  local m2_pr m2_pass_st
  m2_pr="$(run_pass_fixture MUTANT_PASS_FAILS "$mdir/bad.sio")"
  m2_pass_st="${m2_pr%%$'\t'*}"
  if [[ "$m2_pass_st" == "0" ]]; then
    gate_fail "selftest M2: i64=true compiled and ran — mutant is not a mutant"
  fi
  echo "selftest M2 FIRED: pass fixture that fails is REGRESS (would stop merge) compile_or_run_rc=$m2_pass_st"

  # Confirm the gate would have accepted neither if they sat in the index.
  # Re-run judge_kind against planted copies under a fake relative tree.
  mkdir -p "$WORKDIR/rel/tests/archaeology/kind_ladder"
  # judge_kind prefixes ROOT. Plant mutants under ROOT via OUT_DIR? No —
  # we already proved the two polarities at the runner layer, which is
  # what judge_kind consults. A second check: a tiny private index that
  # uses ROOT-relative fixtures we write into WORKDIR is not under ROOT.
  # Write the two mutants next to the real fixtures, judge, delete.
  local plant="$ROOT/tests/archaeology/kind_ladder/_mutant"
  mkdir -p "$plant"
  cp "$mdir/ok.sio" "$plant/ok.sio"
  cp "$mdir/bad.sio" "$plant/bad.sio"
  if [[ "$OWN_OUT" -eq 1 ]]; then
    trap 'rm -rf "$plant" "$OUT_DIR"' EXIT
  else
    trap 'rm -rf "$plant"' EXIT
  fi

  line="$(judge_kind MUTANT_REFUSE_PASSES \
    "tests/archaeology/kind_ladder/_mutant/ok.sio" \
    "tests/archaeology/kind_ladder/_mutant/ok.sio" \
    "E001" "codegen")"
  require_rc_file "$WORKDIR/last.block"
  echo "selftest M1 judge: $line block=$(cat "$WORKDIR/last.block")"
  [[ "$(cat "$WORKDIR/last.block")" == "1" ]] \
    || gate_fail "selftest M1: judge_kind accepted a refuse fixture that passes — the gate is not a gate"
  [[ "$line" == *XPASS ]] \
    || gate_fail "selftest M1: expected XPASS note in: $line"

  line="$(judge_kind MUTANT_PASS_FAILS \
    "tests/archaeology/kind_ladder/_mutant/bad.sio" \
    "tests/archaeology/kind_ladder/_mutant/ok.sio" \
    "E001" "codegen")"
  require_rc_file "$WORKDIR/last.block"
  echo "selftest M2 judge: $line block=$(cat "$WORKDIR/last.block")"
  [[ "$(cat "$WORKDIR/last.block")" == "1" ]] \
    || gate_fail "selftest M2: judge_kind accepted a pass fixture that fails — the gate is not a gate"

  rm -rf "$plant"
  if [[ "$OWN_OUT" -eq 1 ]]; then
    trap 'rm -rf "$OUT_DIR"' EXIT
  else
    trap - EXIT
  fi
  echo "KIND_LADDER_GATE_SELFTEST_OK M1=XPASS M2=REGRESS"
}

# ---------------------------------------------------------------------------
if [[ "$SKIP_SELFTEST" -ne 1 ]]; then
  selftest_mutants
fi
if [[ "$SELFTEST_ONLY" -eq 1 ]]; then
  gate_pass "selftest only"
  gate_measured_yes
  exit 0
fi

require_file "$INDEX" "kind-ladder index missing: $INDEX"
require_nonempty_file "$INDEX" "kind-ladder index is empty: $INDEX"

# Parse index. Positions are not a column.
mapfile -t ROWS < <(
  python3 - "$INDEX" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
rows = []
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("kind\t"):
        continue
    parts = line.split("\t")
    if len(parts) < 5:
        sys.stderr.write(f"KIND_LADDER_GATE_FAIL: index row has <5 columns: {raw}\n")
        sys.exit(1)
    kind, pass_p, refuse_p, diag, layer = parts[0], parts[1], parts[2], parts[3], parts[4]
    print("\t".join([kind, pass_p, refuse_p, diag, layer]))
    rows.append(kind)
if not rows:
    sys.stderr.write("KIND_LADDER_GATE_FAIL: index has no data rows — the gate found nothing to measure\n")
    sys.exit(1)
PY
)

require_min_count "${#ROWS[@]}" 1 "index rows"

{
  printf 'kind\tposition\tpass_path\trefuse_path\texpected_diagnostic\tdeepest_layer\tpass_rc\trefuse_rc\tnote\n'
} >"$TABLE"

blocked=0
garden=0
judged=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r kind pass refuse diag layer <<<"$row"
  line="$(judge_kind "$kind" "$pass" "$refuse" "$diag" "$layer")"
  require_rc_file "$WORKDIR/last.block"
  printf '%s\n' "$line" >>"$TABLE"
  printf 'row %s\n' "$line"
  if blank "$pass" && blank "$refuse"; then
    garden=$((garden + 1))
  else
    judged=$((judged + 1))
  fi
  if [[ "$(cat "$WORKDIR/last.block")" == "1" ]]; then
    blocked=$((blocked + 1))
  fi
done

echo "KIND_LADDER_GATE_TABLE path=$TABLE"
echo "--- derived table ---"
python3 - "$TABLE" <<'PY'
import sys
from pathlib import Path
rows = [ln.rstrip("\n").split("\t") for ln in Path(sys.argv[1]).read_text().splitlines() if ln.strip()]
if not rows:
    raise SystemExit("empty derived table")
widths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(len(rows[0]))]
for r in rows:
    print(" | ".join((r[i] if i < len(r) else "").ljust(widths[i]) for i in range(len(widths))))
PY
echo "--- end table ---"
echo "index_rows=${#ROWS[@]} judged=$judged garden=$garden blocked=$blocked"

if [[ "$blocked" -ne 0 ]]; then
  gate_fail "$blocked kind(s) changed polarity (XPASS refuse, REGRESS pass, or missing diagnostic) — see $TABLE"
fi

gate_measured_yes
gate_pass "rows=${#ROWS[@]} judged=$judged garden=$garden"
echo "KIND_LADDER_GATE_OK"
