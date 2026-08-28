#!/usr/bin/env bash
# scripts/madaros_cd_exact_e2e_gate.sh
#
# Wave12 showcase: tests/run-pass/cd_exact_generic_i64.sio under default Madaros.
#
# Science tokens required on stdout (rc=0):
#   ZD PROVED
#   SQ PASS
#   NONZERO PASS
#   COMP i 0   for i in 0..15  (16 lines)
#
# Residual closed by this gate (Wave12):
#   #1383 → multi-mod check specializes generics (check: OK)
#   Wave12 collapse → when generics instantiate, collapse specialized merged
#            items into programs[0] and lower as single-module (avoids SEGV in
#            multi-mod body lower of generic templates — measured: cd_add_exact
#            after summary_done / bodies_begin).
#   Wave12 STAR → freestanding specialized-items handoff for that collapse:
#            lean_single drops nested field-in-array store
#            `(*programs)[0].items = specialized_ir` (and whole-Program array
#            writeback SEGV'd at seed_begin). Fix: out-param monomorphized
#            list + module_frontend_lower_specialized_items_box (with-externs,
#            hollow-main guard → multi-mod fallback for dual gum+knowledge).
#            Collapse gated on Option occupancy (not (bool,i64)+out-param).
#
# Requires a current-source Madaros that includes both fixes
# (artifacts/self-hosted/madaros or MADAROS_RAW_BIN). The committed prebuilt
# may lag.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/cd_exact_generic_i64.sio"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_cd_exact_e2e_gate =="
"$SOUC" --version 2>&1 | head -2 || true

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros-w12-spec" \
            "$ROOT/artifacts/self-hosted/madaros" \
            "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "src=$SRC"

# --- check ---
echo "== check =="
if ! "$RAW" check "$SRC" >"$OUT/check.log" 2>&1; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=check_failed" >&2
  tail -40 "$OUT/check.log" >&2 || true
  exit 1
fi
if ! grep -q 'check: OK' "$OUT/check.log"; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=check_missing_ok" >&2
  cat "$OUT/check.log" >&2 || true
  exit 1
fi
if grep -qE 'error\[|verdict=1' "$OUT/check.log"; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=check_has_errors" >&2
  cat "$OUT/check.log" >&2 || true
  exit 1
fi
echo "check: OK"

# --- compile ---
echo "== compile =="
ELF="$OUT/cd_exact_generic_i64.elf"
set +e
"$RAW" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1
cc_rc=$?
set -e
if [[ $cc_rc -ne 0 || ! -f "$ELF" || ! -s "$ELF" ]]; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=compile_failed cc_rc=$cc_rc" >&2
  # Loud residual class markers for triage
  if grep -q 'dep_begin 1' "$OUT/compile.log" && ! grep -q 'dep_lower_done 1\|specialized_collapse\|lower_done' "$OUT/compile.log"; then
    echo "residual_class=body_lower_dep1_segv" >&2
  fi
  tail -60 "$OUT/compile.log" >&2 || true
  exit 1
fi
chmod +x "$ELF" 2>/dev/null || true
echo "compile: OK elf=$(stat -c%s "$ELF") bytes"
if grep -q 'specialized_collapse lower_count=1' "$OUT/compile.log"; then
  echo "path=specialized_collapse"
else
  echo "path=multi_mod_or_single (no collapse marker)"
fi

# --- run ---
echo "== run =="
set +e
"$ELF" >"$OUT/run.stdout" 2>"$OUT/run.stderr"
run_rc=$?
set -e
if [[ $run_rc -ne 0 ]]; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=run_failed run_rc=$run_rc" >&2
  cat "$OUT/run.stdout" >&2 || true
  cat "$OUT/run.stderr" >&2 || true
  exit 1
fi

# --- science tokens ---
echo "== science tokens =="
missing=0
for tok in "ZD PROVED" "SQ PASS" "NONZERO PASS"; do
  if ! grep -qxF "$tok" "$OUT/run.stdout"; then
    echo "MISSING token: $tok" >&2
    missing=1
  else
    echo "token_ok: $tok"
  fi
done

comp_ok=0
for i in $(seq 0 15); do
  if grep -qxF "COMP $i 0" "$OUT/run.stdout"; then
    comp_ok=$((comp_ok + 1))
  else
    echo "MISSING COMP $i 0" >&2
    missing=1
  fi
done
echo "comp_zero_lines=$comp_ok/16"

if [[ $missing -ne 0 ]]; then
  echo "MADAROS_CD_EXACT_E2E_GATE_BLOCKED reason=science_tokens" >&2
  echo "--- stdout ---" >&2
  cat "$OUT/run.stdout" >&2
  exit 1
fi

echo "MADAROS_CD_EXACT_E2E_GATE_OK"
echo "ZD PROVED / SQ PASS / NONZERO PASS / 16x COMP i 0  rc=0"
exit 0
