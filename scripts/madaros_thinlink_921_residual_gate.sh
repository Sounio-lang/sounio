#!/usr/bin/env bash
# scripts/madaros_thinlink_921_residual_gate.sh
#
# Wave14 Agent D residual / closeout gate for issue #921 (Defect B):
# multimodule thin-link rc=12 when importing math::rational alongside a second
# module (historically algebra::cayley_dickson::cd_sigma).
#
# Measured on origin/main (2026-07-21, Madaros v0.80.0):
#   * DEFAULT multi-module path uses full IR (compact experimental disabled).
#   * Handoff repro compiles + runs green (stdout "11\n").
#   * Compact opt-in (SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1) still fails
#     imported_simple_ir_emit_failed, then falls back to full IR and succeeds.
#
# This gate:
#   1. Proves the #921 *filed* fail class is CLOSED on the default path.
#   2. Classifies the experimental compact residual honestly (emit_failed +
#      fallback, not hard thin-link rc=12).
#   3. Keeps single-module controls green so a regression cannot hide behind
#      "both imports fail alone".
#
# Does NOT claim: compact emitter completeness, all D3 memory-wall residuals,
# or every multi-module pairing in the stdlib.
#
# Exit 0 + MADAROS_THINLINK_921_RESIDUAL_GATE_OK  → default #921 closed
# Exit 1 + MADAROS_THINLINK_921_RESIDUAL_GATE_FAIL → default regression
# Exit 2 + ..._BLOCKED                             → missing toolchain

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
unset SOUNIO_ENABLE_COMPACT_IMPORTED_IR || true

ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

REPRO="$ROOT/docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio"
if [[ ! -f "$REPRO" ]]; then
  echo "MADAROS_THINLINK_921_RESIDUAL_GATE_BLOCKED reason=missing_repro path=$REPRO" >&2
  exit 2
fi

echo "== madaros_thinlink_921_residual_gate =="
"$SOUC" --version 2>&1 | head -2 || true

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_THINLINK_921_RESIDUAL_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 2
fi

RAW_SHA256="$(sha256sum "$RAW" | awk '{print $1}')"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$RAW_SHA256"
echo "git_sha=$GIT_SHA"
echo "git_branch=$GIT_BRANCH"

# --- controls: each import alone ---
cat >"$OUT/rational_only.sio" <<'EOF'
use math::rational::{Rational, rat_one}
fn main() with IO, Mut, Panic, Div {
    let r = rat_one()
    print_int(r.num)
    print("\n")
}
EOF
cat >"$OUT/cd_sigma_only.sio" <<'EOF'
use algebra::cayley_dickson::{cd_sigma}
fn main() with IO, Mut, Panic, Div {
    print_int(cd_sigma(3, 6, 4) as i64)
    print("\n")
}
EOF

compile_run() {
  local name="$1" src="$2" elf="$3" expect_stdout="$4"
  echo "== default: $name =="
  if ! "$SOUC" compile "$src" -o "$elf" >"$OUT/${name}.compile.log" 2>&1; then
    echo "FAIL: compile $name (rc nonzero)"
    if grep -Fq 'multimodule native thin-link compilation failed' "$OUT/${name}.compile.log"; then
      echo "  fail_class=thinlink_hard_fail"
    fi
    if grep -Fq 'compact IR ELF write failed' "$OUT/${name}.compile.log"; then
      echo "  note=compact_path_touched (unexpected on default; compact is opt-in)"
    fi
    tail -30 "$OUT/${name}.compile.log" || true
    fail=1
    return
  fi
  if grep -Fq 'multimodule native thin-link compilation failed' "$OUT/${name}.compile.log"; then
    echo "FAIL: $name reported thin-link hard fail despite compile rc=0"
    fail=1
    return
  fi
  if ! grep -Eq 'using full IR path|imported_compile: begin|Written to' "$OUT/${name}.compile.log"; then
    echo "WARN: $name compile log missing full-IR markers (still checking run)"
  fi
  if [[ ! -x "$elf" ]]; then
    # lean_single historically skipped exec bit; Madaros should set it.
    chmod +x "$elf" 2>/dev/null || true
  fi
  if [[ ! -f "$elf" ]]; then
    echo "FAIL: $name missing ELF $elf"
    fail=1
    return
  fi
  # Capture raw stdout bytes (do not use $(...) — it strips trailing newlines).
  set +e
  "$elf" >"$OUT/${name}.stdout" 2>"$OUT/${name}.run.err"
  local run_rc=$?
  set -e
  if [[ $run_rc -ne 0 ]]; then
    echo "FAIL: run $name rc=$run_rc"
    cat "$OUT/${name}.run.err" || true
    fail=1
    return
  fi
  printf '%s' "$expect_stdout" >"$OUT/${name}.expect"
  if ! cmp -s "$OUT/${name}.stdout" "$OUT/${name}.expect"; then
    echo "FAIL: $name stdout mismatch"
    echo "  expected=$(od -An -tx1 "$OUT/${name}.expect" | tr -s ' ')"
    echo "  got=$(od -An -tx1 "$OUT/${name}.stdout" | tr -s ' ')"
    fail=1
    return
  fi
  echo "PASS: $name compile+run stdout_ok"
}

compile_run "rational_only" "$OUT/rational_only.sio" "$OUT/rational_only.elf" $'1\n'
compile_run "cd_sigma_only" "$OUT/cd_sigma_only.sio" "$OUT/cd_sigma_only.elf" $'1\n'
# Handoff repro: cd_sigma(3,6,4)=1, rat_one().num=1 → "11\n"
compile_run "handoff_repro_rational_plus_cd_sigma" "$REPRO" "$OUT/handoff.elf" $'11\n'

# --- experimental compact residual classification ---
echo "== compact_opt_in residual classification =="
COMPACT_LOG="$OUT/compact.compile.log"
COMPACT_ELF="$OUT/compact.elf"
set +e
SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1 "$SOUC" compile "$REPRO" -o "$COMPACT_ELF" >"$COMPACT_LOG" 2>&1
compact_rc=$?
set -e

compact_class="unknown"
if grep -Fq 'imported_simple_ir_emit_failed' "$COMPACT_LOG"; then
  compact_class="compact_emit_failed"
elif grep -Fq 'imported_simple_ir_unsupported_function_shape' "$COMPACT_LOG"; then
  compact_class="compact_unsupported_shape"
elif grep -Fq 'compact IR load failed' "$COMPACT_LOG"; then
  compact_class="compact_load_failed"
elif grep -Fq 'compact IR ELF write failed' "$COMPACT_LOG"; then
  compact_class="compact_elf_write_failed"
elif grep -Fq 'Native binary size:' "$COMPACT_LOG" && ! grep -Fq 'falling back to full IR path' "$COMPACT_LOG"; then
  compact_class="compact_emit_ok_no_fallback"
else
  compact_class="compact_no_fail_marker"
fi

fallback_ok=0
if grep -Fq 'falling back to full IR path' "$COMPACT_LOG" && [[ $compact_rc -eq 0 && -f "$COMPACT_ELF" ]]; then
  fallback_ok=1
fi

hard_thinlink=0
if grep -Fq 'multimodule native thin-link compilation failed' "$COMPACT_LOG"; then
  hard_thinlink=1
fi

echo "compact_fail_class=$compact_class"
echo "compact_compile_rc=$compact_rc"
echo "compact_fallback_to_full_ir=$fallback_ok"
echo "compact_hard_thinlink_fail=$hard_thinlink"

# Residual policy: compact may fail emit (experimental stub), but MUST fall
# back to full IR and succeed. A hard thin-link failure is a regression of the
# historical #921 filed class.
if [[ $hard_thinlink -eq 1 || $compact_rc -ne 0 ]]; then
  echo "FAIL: compact opt-in hard-failed without successful full-IR recovery"
  tail -40 "$COMPACT_LOG" || true
  fail=1
elif [[ $fallback_ok -eq 1 ]]; then
  chmod +x "$COMPACT_ELF" 2>/dev/null || true
  printf '%s' $'11\n' >"$OUT/compact.expect"
  set +e
  "$COMPACT_ELF" >"$OUT/compact.stdout" 2>/dev/null
  set -e
  if ! cmp -s "$OUT/compact.stdout" "$OUT/compact.expect"; then
    echo "FAIL: compact-fallback ELF stdout mismatch (got=$(od -An -tx1 "$OUT/compact.stdout" | tr -s ' '))"
    fail=1
  else
    echo "PASS: compact residual classified ($compact_class) + full-IR fallback run ok"
  fi
elif [[ "$compact_class" == "compact_emit_ok_no_fallback" ]]; then
  # Unexpected: compact path succeeded outright. Still require correct stdout.
  chmod +x "$COMPACT_ELF" 2>/dev/null || true
  printf '%s' $'11\n' >"$OUT/compact.expect"
  set +e
  "$COMPACT_ELF" >"$OUT/compact.stdout" 2>/dev/null
  set -e
  if ! cmp -s "$OUT/compact.stdout" "$OUT/compact.expect"; then
    echo "FAIL: compact-only ELF silent-corruption (stdout not 11\\n)"
    echo "  got=$(od -An -tx1 "$OUT/compact.stdout" | tr -s ' ')"
    fail=1
  else
    echo "PASS: compact path emitted correct ELF (stronger than residual)"
  fi
else
  echo "FAIL: compact path neither fell back nor emitted a valid ELF"
  tail -40 "$COMPACT_LOG" || true
  fail=1
fi

# Receipt
mkdir -p "$ROOT/artifacts/compiler"
RECEIPT="$ROOT/artifacts/compiler/madaros_thinlink_921_residual_receipt.v1.json"
DEFAULT_STATUS="FAIL"
if [[ $fail -eq 0 ]]; then
  DEFAULT_STATUS="PASS"
fi
# Prefer repo-relative path in the receipt when possible.
RAW_RECEIPT="$RAW"
case "$RAW" in
  "$ROOT"/*) RAW_RECEIPT="${RAW#"$ROOT"/}" ;;
esac

cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_thinlink_921_residual_receipt.v1",
  "issue": 921,
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "raw_elf": "$RAW_RECEIPT",
  "raw_elf_sha256": "$RAW_SHA256",
  "default_path_status": "$DEFAULT_STATUS",
  "default_fail_class": "none",
  "compact_fail_class": "$compact_class",
  "compact_fallback_to_full_ir": $fallback_ok,
  "compact_hard_thinlink_fail": $hard_thinlink,
  "handoff_repro": "docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio",
  "expected_stdout": "11\\n",
  "claim_boundary": "Default multi-module full-IR path closes filed #921 (rational+cd_sigma thin-link rc=12). Compact emitter remains experimental; residual is emit_failed→full-IR fallback, not hard rc=12."
}
EOF
echo "receipt: $RECEIPT"

if [[ $fail -ne 0 ]]; then
  echo "MADAROS_THINLINK_921_RESIDUAL_GATE_FAIL"
  exit 1
fi
echo "MADAROS_THINLINK_921_RESIDUAL_GATE_OK"
echo "fail_class_default=CLOSED (full IR; no thin-link rc=12)"
echo "fail_class_compact_residual=$compact_class (experimental; fallback required)"
exit 0
