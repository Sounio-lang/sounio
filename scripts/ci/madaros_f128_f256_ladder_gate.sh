#!/usr/bin/env bash
# madaros_f128_f256_ladder_gate.sh — staged f128/f256 ladder gate (V0-B first).
#
# Spec: docs/architecture/F128_F256_LADDER.md
# Semantic-Lane-ID: WS-G-V0B-LITERALS-CHECK
#
# V0-B intent: literals + type spellings accepted end-to-end through `check`.
# Arithmetic/casts/implicit conversion remain rejected.
#
# CRITICAL SHAPE: this gate must FAIL under current V0-A (parser E249 on
# f128/f256) and PASS only when V0-B is genuinely implemented. A silent no-op
# cannot pass — the positive control must fire, and positives must reach
# `check: OK` without error[E249].
#
# External oracle (not Sounio): tests/vectors/f128_f256/literal_boundary_*.jsonl
# from MPFR via gen/literal_boundary_gen.c (GENERATION_RECEIPT.md). Probes embed
# those source_literal strings and expected/via_f64 limb tables so a widen-f64
# shortcut cannot green-wash against self-consistency. Arithmetic corpora
# f128.jsonl/f256.jsonl are intentionally NOT consumed at V0-B (V0-D only).
#
# Note: Madaros may exit 0 while still printing E249 (diagnostic muting).
# The gate judges stdout/stderr content, not exit code alone.
#
# Usage:
#   bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b
#
# Environment scrub (required on Slurm compute nodes):
#   unset SOUC_BIN SOUNIO_SOUC_BIN
#   export SOUNIO_STDLIB_PATH="$ROOT/stdlib"   # must be node-visible
#   export SOUC="$ROOT/bin/souc"              # optional override

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STAGE="v0b"
if [[ "${1:-}" == "--stage" ]]; then
  STAGE="${2:-}"
  shift 2 || true
elif [[ "${1:-}" == --stage=* ]]; then
  STAGE="${1#--stage=}"
  shift || true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 --stage v0b|v0c|v0d" >&2
  exit 64
fi

if [[ "$STAGE" == "v0c" ]]; then
  exec bash "$ROOT_DIR/scripts/ci/madaros_f128_f256_v0c_wire_gate.sh" "$@"
fi

if [[ "$STAGE" == "v0d" ]]; then
  exec bash "$ROOT_DIR/scripts/ci/madaros_f128_f256_v0d_softfloat_gate.sh" "$@"
fi

if [[ "$STAGE" != "v0b" ]]; then
  echo "FAIL unsupported stage='$STAGE' (implemented: v0b, v0c, v0d)" >&2
  exit 64
fi

# Scrub pod-local env that compute nodes cannot see.
unset SOUC_BIN SOUNIO_SOUC_BIN || true
if [[ -n "${SOUNIO_STDLIB_PATH:-}" && ! -d "${SOUNIO_STDLIB_PATH}" ]]; then
  unset SOUNIO_STDLIB_PATH || true
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
if [[ ! -x "$SOUC" ]]; then
  echo "FAIL souc not executable: $SOUC" >&2
  exit 2
fi
if [[ ! -d "$SOUNIO_STDLIB_PATH" ]]; then
  echo "FAIL SOUNIO_STDLIB_PATH missing: $SOUNIO_STDLIB_PATH" >&2
  exit 2
fi

# V0-B is a Madaros check-path gate. lean_single does not print `check: OK` and
# does not own the E249 reserved-wide surface the same way — refuse silent
# engine fallback that would green-wash or mis-diagnose.
if [[ "${SOUNIO_SOUC_ENGINE:-}" == "lean_single" ]]; then
  echo "FAIL stage=v0b requires default Madaros check path; SOUNIO_SOUC_ENGINE=lean_single is refused" >&2
  exit 2
fi
# If MADAROS_RAW_BIN is set, it must be a real ELF (not a missing path).
if [[ -n "${MADAROS_RAW_BIN:-}" && ! -x "${MADAROS_RAW_BIN}" ]]; then
  echo "FAIL MADAROS_RAW_BIN not executable: $MADAROS_RAW_BIN" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0b.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()

note_pass() {
  PASS=$((PASS + 1))
  echo "PASS $1"
}

note_fail() {
  FAIL=$((FAIL + 1))
  FAILURES+=("$1")
  echo "FAIL $1" >&2
}

run_check() {
  local src="$1"
  local log="$2"
  # Absolute path: Madaros ancestor-walk needs it for some multi-module cases;
  # single-file V0-B probes are fine either way, but stay consistent.
  if [[ "$src" != /* ]]; then
    src="$ROOT_DIR/$src"
  fi
  "$SOUC" check "$src" >"$log" 2>&1 || true
}

has_e249() {
  grep -Fq 'error[E249' "$1"
}

has_check_ok() {
  grep -Fq 'check: OK' "$1"
}

RESERVED_MSG='f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A'

echo "=== madaros_f128_f256_ladder_gate stage=v0b ==="
echo "root=$ROOT_DIR"
echo "souc=$SOUC"
echo "stdlib=$SOUNIO_STDLIB_PATH"
echo "souc_version=$("$SOUC" --version 2>/dev/null | head -1 || echo unknown)"

# ---------------------------------------------------------------------------
# MPFR external oracle — vectors + probe embedding (not Sounio-derived).
# ---------------------------------------------------------------------------
VEC_DIR="$ROOT_DIR/tests/vectors/f128_f256"
F128_LIT="$VEC_DIR/literal_boundary_f128.jsonl"
F256_LIT="$VEC_DIR/literal_boundary_f256.jsonl"
# sha256 from GENERATION_RECEIPT.md Wave 3 section (grok-cli1 / docs/ws-g-ref-vectors)
F128_LIT_SHA256="4b7804f0d70016770fb8bda4b67beb9226be1cebf9dc02f3771a4a7fcdbfa52d"
F256_LIT_SHA256="574161d42fe10379d42198c03474a7b0b1daa39235111c45c2e8e28a7017a4c3"

if [[ ! -f "$F128_LIT" || ! -f "$F256_LIT" ]]; then
  note_fail "mpfr_literal_boundary_corpus_missing under tests/vectors/f128_f256/"
else
  got128="$(sha256sum "$F128_LIT" | awk '{print $1}')"
  got256="$(sha256sum "$F256_LIT" | awk '{print $1}')"
  if [[ "$got128" != "$F128_LIT_SHA256" ]]; then
    note_fail "mpfr_f128_literal_boundary_hash_mismatch got=$got128 expected=$F128_LIT_SHA256"
  else
    note_pass "mpfr_f128_literal_boundary_hash_ok"
  fi
  if [[ "$got256" != "$F256_LIT_SHA256" ]]; then
    note_fail "mpfr_f256_literal_boundary_hash_mismatch got=$got256 expected=$F256_LIT_SHA256"
  else
    note_pass "mpfr_f256_literal_boundary_hash_ok"
  fi
fi

# Corpus integrity: every double_rounds_differs row must have expected != via_f64
# (proves the trap set is real external ground truth, independent of Sounio).
ORACLE_PY="$TMP_DIR/oracle_embed_check.py"
cat >"$ORACLE_PY" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
pairs = [
    ("f128", root / "tests/vectors/f128_f256/literal_boundary_f128.jsonl",
     root / "tests/run-pass/f128_v0b_literal_smoke.sio"),
    ("f256", root / "tests/vectors/f128_f256/literal_boundary_f256.jsonl",
     root / "tests/run-pass/f256_v0b_literal_forms.sio"),
]
rc = 0
for fmt, corpus, probe in pairs:
    if not corpus.is_file() or not probe.is_file():
        print(f"FAIL missing corpus or probe for {fmt}")
        rc = 1
        continue
    rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
    text = probe.read_text()
    dr = [r for r in rows if r.get("double_rounds_differs")]
    # expected limbs must differ from via_f64 on every trap
    bad_eq = []
    for r in dr:
        if r["expected"]["limbs"] == r["via_f64"]["limbs"]:
            bad_eq.append(r["id"])
    if bad_eq:
        print(f"FAIL {fmt} double_rounds_differs but limbs equal: {bad_eq}")
        rc = 1
    else:
        print(f"PASS {fmt}_double_round_traps_distinct n={len(dr)}")

    # Every embeddable (no leading '-') double-round source_literal must appear
    # as a typed binding in the probe source.
    missing = []
    for r in dr:
        lit = r["source_literal"]
        if lit.startswith("-"):
            # Sounio has no unary minus — limb-oracle only; still require table.
            if f"ORACLE_{r['id']}_EXPECTED" not in text:
                missing.append(f"limb_table:{r['id']}")
            continue
        # binding form: let v_N: fXXX = <lit>
        if lit not in text:
            missing.append(f"source:{r['id']}:{lit}")
        if f"ORACLE_{r['id']}_EXPECTED" not in text:
            missing.append(f"expected_table:{r['id']}")
        if f"ORACLE_{r['id']}_VIA_F64" not in text:
            missing.append(f"via_f64_table:{r['id']}")
    if missing:
        print(f"FAIL {fmt}_probe_missing_oracle_embed count={len(missing)}")
        for m in missing[:12]:
            print(f"  missing {m}")
        rc = 1
    else:
        embeddable = sum(1 for r in dr if not r["source_literal"].startswith("-"))
        print(f"PASS {fmt}_probe_embeds_double_round_sources n={embeddable}")

    # Must NOT claim Sounio as oracle
    if "oracle: tests/vectors/f128_f256/literal_boundary_" not in text:
        print(f"FAIL {fmt}_probe_missing_oracle_header")
        rc = 1
    else:
        print(f"PASS {fmt}_probe_declares_mpfr_oracle")

# Arithmetic corpora present but explicitly unused at V0-B (V0-D only).
arith = list((root / "tests/vectors/f128_f256").glob("f128.jsonl")) + \
        list((root / "tests/vectors/f128_f256").glob("f256.jsonl"))
hard = list((root / "tests/vectors/f128_f256_v0d").glob("arith_hard_*.jsonl")) \
    if (root / "tests/vectors/f128_f256_v0d").is_dir() else []
if len(arith) == 2:
    print("NOTE arithmetic_corpora_present_but_not_consumed_at_v0b files=f128.jsonl,f256.jsonl (V0-D)")
else:
    print("NOTE arithmetic_corpora_absent_or_partial (ok for V0-B; required at V0-D)")
if hard:
    names = ",".join(p.name for p in sorted(hard))
    print(f"NOTE v0d_hard_case_corpora_present_but_not_consumed_at_v0b files={names} (V0-D)")
else:
    print("NOTE v0d_hard_case_corpora_absent (ok for V0-B; see PR #1761 f128_f256_v0d)")

sys.exit(rc)
PY

ORACLE_LOG="$TMP_DIR/oracle_embed.log"
if ! python3 "$ORACLE_PY" "$ROOT_DIR" >"$ORACLE_LOG" 2>&1; then
  note_fail "mpfr_oracle_probe_embedding_check"
  cat "$ORACLE_LOG" >&2 || true
else
  while IFS= read -r line; do
    case "$line" in
      PASS\ *) note_pass "${line#PASS }" ;;
      NOTE\ *) echo "$line" ;;
      *) echo "$line" ;;
    esac
  done <"$ORACLE_LOG"
fi

# ---------------------------------------------------------------------------
# Positive control — MUST fire. Proves the compiler check path is live and
# that a silent broken SOUC cannot masquerade as V0-B green.
# ---------------------------------------------------------------------------
CONTROL_SRC="tests/run-pass/hello.sio"
CONTROL_LOG="$TMP_DIR/positive_control_hello.log"
if [[ ! -f "$ROOT_DIR/$CONTROL_SRC" ]]; then
  note_fail "positive_control_missing:$CONTROL_SRC"
else
  run_check "$CONTROL_SRC" "$CONTROL_LOG"
  if has_check_ok "$CONTROL_LOG" && ! has_e249 "$CONTROL_LOG"; then
    note_pass "positive_control_f64_check_ok"
  else
    note_fail "positive_control_did_not_fire (expected check: OK without E249 on $CONTROL_SRC)"
    cat "$CONTROL_LOG" >&2 || true
  fi
fi

# ---------------------------------------------------------------------------
# V0-B positive witnesses — must check green WITHOUT E249.
# Under V0-A these print E249 → stage FAIL (correct today).
# ---------------------------------------------------------------------------
POSITIVE_SOURCES=(
  tests/run-pass/f128_v0b_literal_smoke.sio
  tests/run-pass/f256_v0b_literal_forms.sio
)

for src in "${POSITIVE_SOURCES[@]}"; do
  label="$(basename "$src" .sio)"
  log="$TMP_DIR/${label}.check.log"
  if [[ ! -f "$ROOT_DIR/$src" ]]; then
    note_fail "positive_missing:$src"
    continue
  fi
  run_check "$src" "$log"
  if has_e249 "$log"; then
    # Expected under V0-A — record exact diagnostic for the receipt.
    e249_line="$(grep -F 'error[E249' "$log" | head -1 || true)"
    reserved_hit=0
    grep -Fq "$RESERVED_MSG" "$log" && reserved_hit=1
    note_fail "positive_still_E249:$label :: ${e249_line} reserved_msg=${reserved_hit}"
    continue
  fi
  if has_check_ok "$log"; then
    note_pass "positive_check_ok:$label"
  else
    note_fail "positive_no_check_ok:$label (no E249 but check did not print check: OK)"
    tail -40 "$log" >&2 || true
  fi
done

# ---------------------------------------------------------------------------
# V0-B negative witnesses — must NOT reach check: OK.
# Pins the boundary so V0-B cannot silently grow into V0-D arithmetic.
# ---------------------------------------------------------------------------
NEGATIVE_SOURCES=(
  tests/compile-fail/f128_v0b_arithmetic_rejected.sio
  tests/compile-fail/f256_v0b_arithmetic_rejected.sio
  tests/compile-fail/f128_v0b_cast_rejected.sio
  tests/compile-fail/f128_v0b_implicit_conversion_rejected.sio
)

NEGATIVE_OK=0
for src in "${NEGATIVE_SOURCES[@]}"; do
  label="$(basename "$src" .sio)"
  log="$TMP_DIR/${label}.check.log"
  if [[ ! -f "$ROOT_DIR/$src" ]]; then
    note_fail "negative_missing:$src"
    continue
  fi
  run_check "$src" "$log"
  if has_check_ok "$log"; then
    note_fail "negative_incorrectly_check_ok:$label"
    cat "$log" >&2 || true
  else
    NEGATIVE_OK=$((NEGATIVE_OK + 1))
    note_pass "negative_still_rejected:$label"
  fi
done

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
echo "NEGATIVE_REJECTED=$NEGATIVE_OK/${#NEGATIVE_SOURCES[@]}"

if [[ "$FAIL" -eq 0 ]]; then
  # Exact success receipt from docs/architecture/F128_F256_LADDER.md
  echo "PASS f128_f256_v0b_literals check=green parser=E249_lifted typecheck=distinct_no_implicit literals=decimal+hex+binary negative_arithmetic=${#NEGATIVE_SOURCES[@]}"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0b"
  exit 0
fi

echo "FAIL madaros_f128_f256_ladder_gate stage=v0b" >&2
echo "first_failures:" >&2
for f in "${FAILURES[@]}"; do
  echo "  - $f" >&2
done

# Explicit V0-A diagnosis when positives are still E249 (today's expected state).
if printf '%s\n' "${FAILURES[@]}" | grep -q 'positive_still_E249'; then
  echo "diagnosis=V0-A_parser_E249_still_active (expected until V0-B implementation lands)" >&2
  echo "observed_reserved_message=$RESERVED_MSG" >&2
fi

exit 1
