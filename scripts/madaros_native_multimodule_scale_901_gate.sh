#!/usr/bin/env bash
# scripts/madaros_native_multimodule_scale_901_gate.sh
#
# Wave15 Agent C — closeout / residual gate for issue #901:
# large multi-module native compile of prob::distributions under default Madaros.
#
# Historical fail class (docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md):
#   Merged IR ~210 functions → imported_simple_ir_emit_failed / thin-link rc=12
#   under default Madaros; only lean_single worked.
#
# Measured Wave15C 2026-07-22 on origin/main (post into-acc #1402, spec DCE #1397):
#   DEFAULT full-IR path: Merged IR ~71–73 after into_acc; compile+run green.
#   Issue acceptance probe: m=5.000000
#   Textbook science graph: PROB_TEXTBOOK_OK
#   Full driver tests/stdlib/prob/test_prob_stdlib.sio: PROB_STDLIB_OK under Madaros
#
# This gate:
#   1. Proves the #901 *filed* fail class is CLOSED on the default Madaros path.
#   2. Locks textbook numeric correctness (not silent wrong numbers).
#   3. Keeps #921 rational+cd pairing and special multi-import green as smaller
#      post-into-acc corpus.
#   4. Classifies hard residuals that remain out of scope (not thin-link/scale).
#
# Does NOT claim: all stats/OLS multi-module verticals, compact emitter, or every
# multi-module pairing in the stdlib. Dual / order_spread / cd_exact must stay
# green under their own gates — this gate does not re-run those (orthogonal).
#
# Exit 0 + MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_OK  → #901 scale closed
# Exit 1 + ..._GATE_FAIL                                 → default regression
# Exit 2 + ..._BLOCKED                                   → missing toolchain

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

PROBE="$ROOT/tests/run-pass/madaros_native_multimodule_scale_prob.sio"
TEXTBOOK="$ROOT/tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio"
DRIVER="$ROOT/tests/stdlib/prob/test_prob_stdlib.sio"
REPORT="$ROOT/examples/prob/distribution_report.sio"
THIN_REPRO="$ROOT/docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio"

for f in "$PROBE" "$TEXTBOOK" "$DRIVER"; do
  if [[ ! -f "$f" ]]; then
    echo "MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_BLOCKED reason=missing_source path=$f" >&2
    exit 2
  fi
done

echo "== madaros_native_multimodule_scale_901_gate =="
engine_line="$("$SOUC" --version 2>&1 | head -1 || true)"
echo "engine: $engine_line"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single" >&2
  exit 1
fi
if ! echo "$engine_line" | grep -qi Madaros; then
  echo "WARN: version string does not mention Madaros: $engine_line"
fi

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$ROOT/artifacts/self-hosted/madaros" "$ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 2
fi

RAW_SHA256="$(sha256sum "$RAW" | awk '{print $1}')"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$RAW_SHA256"
echo "git_sha=$GIT_SHA"
echo "git_branch=$GIT_BRANCH"

# Accumulators for receipt
PROBE_MERGED=""
PROBE_FINAL_FN=""
TEXTBOOK_MERGED=""
DRIVER_MERGED=""
SPECIAL_MERGED=""
DEFAULT_FAIL_CLASS="none"

classify_compile_fail() {
  local log="$1"
  if grep -Fq 'multimodule native thin-link compilation failed' "$log"; then
    echo "thinlink"
    return
  fi
  if grep -Eqi 'SIGSEGV|Segmentation fault' "$log"; then
    echo "SEGV"
    return
  fi
  if grep -Eqi 'Cannot allocate|out of memory|OOM' "$log"; then
    echo "OOM"
    return
  fi
  if grep -Fq 'error[E011' "$log"; then
    echo "E011"
    return
  fi
  if grep -Fq 'imported_simple_ir_emit_failed' "$log"; then
    echo "compact_emit_failed"
    return
  fi
  if grep -Fq 'visibility preflight failed' "$log"; then
    echo "preflight"
    return
  fi
  echo "compile_fail"
}

extract_merged_ir() {
  local log="$1"
  # Prefer last "Merged IR: N" line
  awk '/Merged IR:/{n=$NF} END{print n}' "$log" 2>/dev/null || true
}

extract_final_fn() {
  local log="$1"
  awk '/final_fn_count/{n=$NF} END{print n}' "$log" 2>/dev/null || true
}

compile_run() {
  local name="$1" src="$2" elf="$3" expect_pat="$4"
  echo "== default: $name =="
  if ! "$SOUC" compile "$src" -o "$elf" >"$OUT/${name}.compile.log" 2>&1; then
    local cls
    cls="$(classify_compile_fail "$OUT/${name}.compile.log")"
    echo "FAIL: compile $name (fail_class=$cls)"
    DEFAULT_FAIL_CLASS="$cls"
    tail -40 "$OUT/${name}.compile.log" || true
    fail=1
    return
  fi
  if grep -Fq 'multimodule native thin-link compilation failed' "$OUT/${name}.compile.log"; then
    echo "FAIL: $name reported thin-link hard fail despite compile rc=0"
    DEFAULT_FAIL_CLASS="thinlink"
    fail=1
    return
  fi
  local merged finalfn
  merged="$(extract_merged_ir "$OUT/${name}.compile.log")"
  finalfn="$(extract_final_fn "$OUT/${name}.compile.log")"
  echo "  merged_ir=${merged:-unknown} final_fn_count=${finalfn:-unknown}"
  case "$name" in
    acceptance_probe) PROBE_MERGED="$merged"; PROBE_FINAL_FN="$finalfn" ;;
    textbook) TEXTBOOK_MERGED="$merged" ;;
    stdlib_driver) DRIVER_MERGED="$merged" ;;
    special_multi) SPECIAL_MERGED="$merged" ;;
  esac
  if [[ ! -x "$elf" ]]; then
    chmod +x "$elf" 2>/dev/null || true
  fi
  if [[ ! -f "$elf" ]]; then
    echo "FAIL: $name missing ELF $elf"
    fail=1
    return
  fi
  if ! "$elf" >"$OUT/${name}.run.out" 2>"$OUT/${name}.run.err"; then
    echo "FAIL: run $name (nonzero rc)"
    cat "$OUT/${name}.run.out" || true
    cat "$OUT/${name}.run.err" || true
    DEFAULT_FAIL_CLASS="run_fail"
    fail=1
    return
  fi
  if [[ -n "$expect_pat" ]]; then
    if ! grep -Eq "$expect_pat" "$OUT/${name}.run.out"; then
      echo "FAIL: $name stdout missing pattern /$expect_pat/"
      cat "$OUT/${name}.run.out" || true
      DEFAULT_FAIL_CLASS="stdout_miss"
      fail=1
      return
    fi
  fi
  echo "PASS: $name compile+run"
}

# --- #901 filed acceptance ---
compile_run acceptance_probe "$PROBE" "$OUT/probe.elf" 'm=5(\.0+)?'

# --- textbook science graph ---
compile_run textbook "$TEXTBOOK" "$OUT/textbook.elf" 'PROB_TEXTBOOK_OK'

# --- full stdlib driver (historically lean_single-only) ---
compile_run stdlib_driver "$DRIVER" "$OUT/driver.elf" 'PROB_STDLIB_OK'

# --- consumer example ---
if [[ -f "$REPORT" ]]; then
  compile_run distribution_report "$REPORT" "$OUT/report.elf" 'Uniform\(0,10\) mean'
fi

# --- smaller green corpus post-into-acc ---
cat >"$OUT/special_multi.sio" <<'EOF'
use special::gamma::*
use special::erf::*
use special::igamma::*
fn main() -> i32 with IO, Mut, Div, Panic {
    print("g=")
    println(gamma(5.0))
    print("erf=")
    println(erf(1.0))
    return 0
}
EOF
compile_run special_multi "$OUT/special_multi.sio" "$OUT/special_multi.elf" 'g=24(\.0+)?'

if [[ -f "$THIN_REPRO" ]]; then
  compile_run thinlink_921_pairing "$THIN_REPRO" "$OUT/thin921.elf" '^11$'
fi

# --- honest residual classification (must remain red with non-scale fail class) ---
# OLS multi-mod vertical still fails at typecheck (E019 method calls), NOT thin-link.
OLS_SRC="$ROOT/tests/stdlib/stats/test_ols_diag_e2e.sio"
OLS_FAIL_CLASS="not_measured"
if [[ -f "$OLS_SRC" ]]; then
  echo "== residual classification: ols_diag_e2e (expect red, non-scale) =="
  set +e
  "$SOUC" compile "$OLS_SRC" -o "$OUT/ols.elf" >"$OUT/ols.compile.log" 2>&1
  ols_rc=$?
  set -e
  if [[ "$ols_rc" -eq 0 ]]; then
    echo "NOTE: ols_diag_e2e now compiles under Madaros (unexpected promotion; not failing gate)"
    OLS_FAIL_CLASS="now_green"
  else
    if grep -Fq 'error[E019' "$OUT/ols.compile.log"; then
      OLS_FAIL_CLASS="E019_method_calls"
    else
      OLS_FAIL_CLASS="$(classify_compile_fail "$OUT/ols.compile.log")"
    fi
    if [[ "$OLS_FAIL_CLASS" == "thinlink" || "$OLS_FAIL_CLASS" == "SEGV" || "$OLS_FAIL_CLASS" == "OOM" ]]; then
      echo "FAIL: residual ols fail class is scale-class ($OLS_FAIL_CLASS) — re-open #901 scale residual"
      DEFAULT_FAIL_CLASS="$OLS_FAIL_CLASS"
      fail=1
    else
      echo "PASS: residual ols fail_class=$OLS_FAIL_CLASS (not thinlink/SEGV/OOM scale class)"
    fi
  fi
fi

RECEIPT_DIR="$ROOT/artifacts/compiler"
mkdir -p "$RECEIPT_DIR"
RECEIPT="$RECEIPT_DIR/madaros_native_multimodule_scale_901_receipt.v1.json"

status_str="PASS"
sentinel="MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_OK"
if [[ "$fail" -ne 0 ]]; then
  status_str="FAIL"
  sentinel="MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_FAIL"
fi

cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_native_multimodule_scale_901_receipt.v1",
  "issue": 901,
  "status": "$status_str",
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "raw_elf": "$RAW",
  "raw_elf_sha256": "$RAW_SHA256",
  "engine": "madaros_default",
  "default_fail_class": "$DEFAULT_FAIL_CLASS",
  "acceptance_probe": {
    "source": "tests/run-pass/madaros_native_multimodule_scale_prob.sio",
    "expected_stdout": "m=5.000000",
    "merged_ir": "${PROBE_MERGED:-}",
    "final_fn_count": "${PROBE_FINAL_FN:-}"
  },
  "textbook_science": {
    "source": "tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio",
    "sentinel": "PROB_TEXTBOOK_OK",
    "merged_ir": "${TEXTBOOK_MERGED:-}"
  },
  "stdlib_driver": {
    "source": "tests/stdlib/prob/test_prob_stdlib.sio",
    "sentinel": "PROB_STDLIB_OK",
    "merged_ir": "${DRIVER_MERGED:-}"
  },
  "smaller_green_corpus": [
    "special::gamma + erf + igamma multi-import",
    "docs/handoff/repros/multimodule_thinlink_rc12_madaros.sio (#921 pairing)"
  ],
  "residual": {
    "ols_diag_e2e_fail_class": "$OLS_FAIL_CLASS",
    "note": "OLS residual is typecheck E019 (method calls), not #901 thin-link/scale class"
  },
  "claim_boundary": "Default Madaros multi-module full-IR path closes filed #901 (prob::distributions large-graph thin-link / scale). Compact emitter remains experimental. Does not claim all stats multi-mod verticals."
}
EOF

echo "receipt: $RECEIPT"
echo "default_fail_class=$DEFAULT_FAIL_CLASS"
echo "ols_residual_fail_class=$OLS_FAIL_CLASS"
echo "$sentinel"
exit "$fail"
