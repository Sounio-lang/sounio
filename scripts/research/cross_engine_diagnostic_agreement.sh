#!/usr/bin/env bash
# Accept/reject diagnostic agreement between the two Sounio compilers.
#
# WHAT THIS MEASURES, AND WHY IT IS NOT SOMETHING ELSE
#
# lean_single (frozen bootstrap seed, single file, token-level recognition) and Madaros (modular
# Lexer→Parser→AST→Check→HIR→SIR→HLIR(SSA)→ELF) are two implementations of one language sharing a
# specification and differing in internal representation. For each versioned .sio file this script
# records whether each engine ACCEPTS or REJECTS it, and classifies the pair:
#
#   AGREE_ACCEPT   both accept
#   AGREE_REJECT   both reject
#   MADAROS_ONLY   Madaros rejects, lean_single accepts
#   LEAN_ONLY      lean_single rejects, Madaros accepts
#
# The last two are the signal. Sun, Le & Su (Epiphron, ICSE 2016) built the first randomised
# differential tester whose oracle is cross-compiler diagnostic inconsistency, but restricted it to
# WARNINGS on syntactically valid, compilable programs; the accept/reject configuration was left as
# future work. Csmith (PLDI 2011) excludes the diagnostic surface by construction — every generated
# program must pass the lexer, parser and typechecker before it is used.
#
# WHAT THIS IS NOT
#
# It is NOT a defect rate. A disagreement can be a defect in either engine OR an intentional
# divergence (lean_single mutes ~35 diagnostic classes for imported functions and has no strict
# mode; Madaros implements guarantees lean_single never had). Epiphron measured a false-positive
# rate in [10%, 47%] on its own warning-inconsistency oracle and needed dedicated filters. Any
# claim built on these counts MUST separate intentional from unintentional divergence by hand.
#
# It is also NOT calibration. Calibration needs a STATED confidence checked against an OBSERVED
# correctness rate. This supplies only the observed half.
#
# Neither engine is ground truth. Where they disagree, this says so; it does not say who is right.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${CROSS_ENGINE_OUT_DIR:-$(mktemp -d /tmp/sounio-cross-engine.XXXXXX)}"
mkdir -p "$OUT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Both engines are invoked THROUGH bin/souc, not directly. lean_single's raw CLI is
# `mini_native <source.sio> <output>` with no check-only mode, so calling it directly would compare
# a full compile against a Madaros --check -- different amounts of work, and rc=1 on files that are
# perfectly fine. The wrapper translates the verb for each engine. Measured while building this:
# the direct form reported rc=1 for two files that both engines actually accept.
MADAROS_BIN="${CROSS_ENGINE_MADAROS_BIN:-}"

if [[ -z "$MADAROS_BIN" ]]; then
  echo "[cross-engine] FAIL: set CROSS_ENGINE_MADAROS_BIN to a Madaros built from the source under test." >&2
  echo "[cross-engine] A prebuilt bin/madaros is not a baseline -- it lags source (measured 127 commits on 2026-07-26)." >&2
  exit 1
fi

[[ -x "$MADAROS_BIN" ]] || { echo "[cross-engine] FAIL: not executable: $MADAROS_BIN" >&2; exit 1; }
[[ -x "$ROOT_DIR/bin/souc" ]] || { echo "[cross-engine] FAIL: bin/souc missing" >&2; exit 1; }

# Madaros needs a large stack to run at all (its own build reports frames up to ~31 MB), so a
# 16384 KiB CI default kills it on a three-line program. Same block as the Madaros gates.
stack_kb="${CROSS_ENGINE_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null)" || { echo "[cross-engine] FAIL: soft stack limit unavailable" >&2; exit 1; }
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || { echo "[cross-engine] FAIL: could not raise soft stack to ${stack_kb} KiB" >&2; exit 1; }
fi
printf '[cross-engine] stack_kb before=%s after=%s\n' "$stack_before" "$(ulimit -S -s)"
printf '[cross-engine] lean=wrapper(SOUNIO_SOUC_ENGINE=lean_single)\n[cross-engine] madaros=%s\n[cross-engine] out=%s\n' \
  "$MADAROS_BIN" "$OUT_DIR"

# Positive control: the instrument must be able to REPORT a disagreement, or a clean sweep means
# nothing. These three are verified by hand -- both accept, Madaros-only rejects, both reject.
pc_fail=0
pc() {
  local f="$1" want="$2"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" check "$f" >/dev/null 2>&1; local l=$?
  SOUNIO_SOUC_BIN="$MADAROS_BIN" "$ROOT_DIR/bin/souc" check "$f" >/dev/null 2>&1; local m=$?
  local got="AGREE_ACCEPT"
  if [[ $l -ne 0 && $m -ne 0 ]]; then got="AGREE_REJECT"
  elif [[ $l -eq 0 && $m -ne 0 ]]; then got="MADAROS_ONLY"
  elif [[ $l -ne 0 && $m -eq 0 ]]; then got="LEAN_ONLY"; fi
  if [[ "$got" == "$want" ]]; then printf '[cross-engine] positive-control OK %s -> %s\n' "$(basename "$f")" "$got"
  else printf '[cross-engine] positive-control FAIL %s: want %s got %s\n' "$(basename "$f")" "$want" "$got" >&2; pc_fail=1; fi
}
pc tests/run-pass/struct_name_8byte_collision_ref.sio AGREE_ACCEPT
pc tests/run-pass/interval_outward_rounding_containment.sio MADAROS_ONLY
pc stdlib/verify/interval.sio AGREE_REJECT
if [[ $pc_fail -ne 0 ]]; then
  echo "[cross-engine] ABORT: the instrument cannot reproduce a known disagreement; a sweep would be meaningless." >&2
  exit 1
fi

TSV="$OUT_DIR/agreement.tsv"
printf 'file\tlean_rc\tmadaros_rc\tverdict\n' > "$TSV"

# Per-file timeout: a hung compile must not stall the sweep, and a timeout is NOT a rejection --
# it is recorded distinctly so it cannot be silently counted as a disagreement.
TMO="${CROSS_ENGINE_TIMEOUT:-25}"

n=0
while IFS= read -r f; do
  n=$((n + 1))
  lout="$OUT_DIR/l.$$"; mout="$OUT_DIR/m.$$"
  SOUNIO_SOUC_ENGINE=lean_single timeout "$TMO" "$ROOT_DIR/bin/souc" check "$f" >"$lout" 2>&1; lrc=$?
  SOUNIO_SOUC_BIN="$MADAROS_BIN" timeout "$TMO" "$ROOT_DIR/bin/souc" check "$f" >"$mout" 2>&1; mrc=$?

  if [[ "$lrc" -eq 124 || "$mrc" -eq 124 ]]; then
    verdict="TIMEOUT"
  elif [[ "$lrc" -eq 0 && "$mrc" -eq 0 ]]; then
    verdict="AGREE_ACCEPT"
  elif [[ "$lrc" -ne 0 && "$mrc" -ne 0 ]]; then
    verdict="AGREE_REJECT"
  elif [[ "$lrc" -eq 0 && "$mrc" -ne 0 ]]; then
    verdict="MADAROS_ONLY"
    cp "$mout" "$OUT_DIR/madaros_only.$(echo "$f" | tr '/' '_').log" 2>/dev/null || true
  else
    verdict="LEAN_ONLY"
    cp "$lout" "$OUT_DIR/lean_only.$(echo "$f" | tr '/' '_').log" 2>/dev/null || true
  fi
  printf '%s\t%s\t%s\t%s\n' "$f" "$lrc" "$mrc" "$verdict" >> "$TSV"
  rm -f "$lout" "$mout"

  if ((n % 250 == 0)); then printf '[cross-engine] %d files\n' "$n" >&2; fi
done < <(git ls-files '*.sio' | grep -vE '^(archive|bootstrap)/')

echo
echo "=== verdicts over $n files ==="
awk -F'\t' 'NR>1{c[$4]++} END{for (k in c) printf "%-14s %6d\n", k, c[k]}' "$TSV" | sort -k2 -rn
echo
echo "TSV: $TSV"
echo "Disagreement logs: $OUT_DIR/{madaros_only,lean_only}.*.log"
echo
echo "REMINDER: these counts mix genuine defects with intentional divergence. Do not report a"
echo "disagreement RATE as a defect rate without classifying the cases by hand."
