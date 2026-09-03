#!/usr/bin/env bash
# scripts/ci/track_a_nv2_parity_inventory.sh
#
# M1.2 inventory: for each .sio file in the given corpus, compile with both
# Track A (bin/souc → lean_single ELF) and Track N-v2 (native_compile_driver
# stage1 ELF) and record outcomes.
#
# Output: a TSV at $OUT_DIR/inventory.tsv with columns
#   file  a_compile  nv2_compile  a_run  nv2_run  stdout_match  exit_match  category
#
# Categories drive the M1.2 punch-list:
#   ok            both compilers succeed and the resulting ELFs agree
#   nv2_compile   A compiles but N-v2 fails to compile      (the punch-list -- wire features in)
#   nv2_run       both compile but N-v2's ELF behaves differently (codegen divergence)
#   a_only        A compiles, N-v2 cannot even attempt (e.g. parse error)
#   a_fail        A also fails -- out of scope for parity (likely an intentionally broken test)
#   both_fail     neither compiler accepts the input
#
# Usage:
#   bash scripts/ci/track_a_nv2_parity_inventory.sh [CORPUS_GLOB...]
# Defaults to examples/native/*.sio if no globs given. Pass directories or
# globs to widen, e.g.:
#   bash scripts/ci/track_a_nv2_parity_inventory.sh 'examples/native/*.sio' 'tests/selfhost-driver-output/*.sio'

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_PARITY_INVENTORY_DIR:-$(mktemp -d /tmp/sounio-parity-inventory.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ELF_DIR="$OUT_DIR/elfs"
RESULTS_TSV="$OUT_DIR/inventory.tsv"
SUMMARY="$OUT_DIR/summary.txt"
mkdir -p "$LOG_DIR" "$ELF_DIR"

# ── Build N-v2 stage1 driver ELF once ─────────────────────────────────────────
NV2_DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
NV2_STAGE1="$OUT_DIR/nv2_driver.stage1"

echo "[parity-inv] souc=$SOUC_BIN"
echo "[parity-inv] out=$OUT_DIR"
echo "[parity-inv] building N-v2 stage1 driver..."

if ! "$SOUC_BIN" run "$NV2_DRIVER_SRC" -- "$NV2_DRIVER_SRC" -o "$NV2_STAGE1" \
        >"$LOG_DIR/nv2_stage1_build.log" 2>&1; then
  echo "[parity-inv] FAIL: could not build N-v2 stage1 driver" >&2
  tail -n 40 "$LOG_DIR/nv2_stage1_build.log" >&2
  exit 1
fi
chmod +x "$NV2_STAGE1"
echo "[parity-inv] N-v2 stage1 ready ($(stat -c %s "$NV2_STAGE1") bytes)"

# ── Resolve corpus ────────────────────────────────────────────────────────────
declare -a INPUTS=()
if [[ $# -eq 0 ]]; then
  shopt -s nullglob
  INPUTS=(examples/native/*.sio)
  shopt -u nullglob
else
  for arg in "$@"; do
    if [[ -d "$arg" ]]; then
      while IFS= read -r -d '' f; do INPUTS+=("$f"); done < <(find "$arg" -name '*.sio' -type f -print0)
    else
      shopt -s nullglob
      for f in $arg; do INPUTS+=("$f"); done
      shopt -u nullglob
    fi
  done
fi

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "[parity-inv] no inputs found" >&2
  exit 1
fi

echo "[parity-inv] inputs: ${#INPUTS[@]} files"

# ── Header ────────────────────────────────────────────────────────────────────
printf 'file\ta_compile\tnv2_compile\ta_run\tnv2_run\tstdout_match\texit_match\tcategory\n' >"$RESULTS_TSV"

# ── Counters ──────────────────────────────────────────────────────────────────
total=0; ok=0; nv2_compile_fail=0; nv2_run_diverge=0; a_only=0; a_fail=0; both_fail=0

# Run a binary capturing stdout, stderr, exit. Time-bound to 10s to avoid
# infinite-loop tests hanging the inventory.
run_capture() {
  local bin="$1" outdir="$2" tag="$3"
  local stdout_path="$outdir/$tag.stdout"
  local stderr_path="$outdir/$tag.stderr"
  local rc=0
  set +e
  timeout 10 "$bin" >"$stdout_path" 2>"$stderr_path"
  rc=$?
  set -e
  echo "$rc"
}

for src in "${INPUTS[@]}"; do
  total=$((total + 1))
  rel="${src#$ROOT_DIR/}"
  rel="${rel#./}"
  slug="$(printf '%s' "$rel" | tr '/' '_' | tr '.' '_')"

  case_dir="$ELF_DIR/$slug"
  mkdir -p "$case_dir"

  a_elf="$case_dir/a.elf"
  nv2_elf="$case_dir/nv2.elf"
  a_log="$LOG_DIR/$slug.a.log"
  nv2_log="$LOG_DIR/$slug.nv2.log"

  # ── Compile with Track A (raw form, no chmod). ──────────────────────────────
  set +e
  timeout 30 "$SOUC_BIN" "$src" "$a_elf" >"$a_log" 2>&1
  a_compile=$?
  set -e
  if [[ $a_compile -eq 0 && -f "$a_elf" ]]; then chmod +x "$a_elf"; fi

  # ── Compile with Track N-v2. ────────────────────────────────────────────────
  nv2_src="$src"
  if grep -Eq '^[[:space:]]*use[[:space:]]+algebra::octonion(::|[[:space:]])' "$src"; then
    bundled_src="$case_dir/nv2.prebundled.sio"
    prebundle_log="$LOG_DIR/$slug.nv2.prebundle.log"
    set +e
    timeout 30 "$SOUC_BIN" run self-hosted/compiler/native_prebundle.sio -- "$src" -o "$bundled_src" --root stdlib >"$prebundle_log" 2>&1
    prebundle_rc=$?
    set -e
    if [[ $prebundle_rc -eq 0 && -s "$bundled_src" ]]; then
      nv2_src="$bundled_src"
    else
      {
        printf '[parity-inv] native_prebundle failed rc=%s for %s\n' "$prebundle_rc" "$src"
        cat "$prebundle_log" 2>/dev/null || true
      } >>"$nv2_log"
    fi
  fi
  set +e
  timeout 30 "$NV2_STAGE1" "$nv2_src" "$nv2_elf" >>"$nv2_log" 2>&1
  nv2_compile=$?
  set -e
  if [[ $nv2_compile -eq 0 && -f "$nv2_elf" ]]; then chmod +x "$nv2_elf"; fi

  # ── If both compiled, run both for behavioural parity. ──────────────────────
  a_run="-"; nv2_run="-"; stdout_match="-"; exit_match="-"
  if [[ $a_compile -eq 0 && -x "$a_elf" ]]; then
    a_run="$(run_capture "$a_elf" "$case_dir" a)"
  fi
  if [[ $nv2_compile -eq 0 && -x "$nv2_elf" ]]; then
    nv2_run="$(run_capture "$nv2_elf" "$case_dir" nv2)"
  fi
  if [[ $a_compile -eq 0 && $nv2_compile -eq 0 && -x "$a_elf" && -x "$nv2_elf" ]]; then
    # Strict + trim-trailing-newline comparison. Track A's print() emits a
    # spurious trailing newline; we accept that as parity for inventory
    # purposes but record it separately so it's visible in the TSV.
    if cmp -s "$case_dir/a.stdout" "$case_dir/nv2.stdout"; then
      stdout_match=1
    else
      # Strip trailing newlines from both and re-compare. Track A's print()
      # currently emits an extra trailing newline that N-v2 does not -- treat
      # that as parity for inventory purposes but mark it as "trim".
      a_trim="$(printf '%s' "$(cat "$case_dir/a.stdout")")"
      nv2_trim="$(printf '%s' "$(cat "$case_dir/nv2.stdout")")"
      if [[ "$a_trim" == "$nv2_trim" ]]; then
        stdout_match=trim
      else
        stdout_match=0
      fi
    fi
    if [[ "$a_run" == "$nv2_run" ]]; then
      exit_match=1
    else
      exit_match=0
    fi
  fi

  # ── Categorize. ─────────────────────────────────────────────────────────────
  # Track A exit 139 = SIGSEGV; treat that as a Track A bug, not an N-v2 punch-list item.
  if [[ $a_compile -eq 0 && $nv2_compile -eq 0 ]]; then
    if [[ "$stdout_match" == "1" || "$stdout_match" == "trim" ]] && [[ "$exit_match" == "1" ]]; then
      category="ok"; ok=$((ok + 1))
    elif [[ "$a_run" == "139" || "$a_run" == "134" || "$a_run" == "124" ]] && [[ "$nv2_run" == "0" ]]; then
      # Track A crashed/timed out, N-v2 ran cleanly.
      category="a_bug"; a_only=$((a_only + 1))
    else
      category="nv2_run"; nv2_run_diverge=$((nv2_run_diverge + 1))
    fi
  elif [[ $a_compile -eq 0 && $nv2_compile -ne 0 ]]; then
    category="nv2_compile"; nv2_compile_fail=$((nv2_compile_fail + 1))
  elif [[ $a_compile -ne 0 && $nv2_compile -eq 0 ]]; then
    category="a_only"; a_only=$((a_only + 1))
  elif [[ $a_compile -ne 0 && $nv2_compile -ne 0 ]]; then
    category="both_fail"; both_fail=$((both_fail + 1))
  else
    category="a_fail"; a_fail=$((a_fail + 1))
  fi

  printf '%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n' \
    "$rel" "$a_compile" "$nv2_compile" "$a_run" "$nv2_run" \
    "$stdout_match" "$exit_match" "$category" >>"$RESULTS_TSV"
done

# ── Summary ───────────────────────────────────────────────────────────────────
{
  printf 'parity inventory: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'corpus           : %d files\n' "$total"
  printf 'ok               : %d\n' "$ok"
  printf 'nv2_compile fail : %d  (the punch-list -- Track A accepts, N-v2 does not)\n' "$nv2_compile_fail"
  printf 'nv2_run diverge  : %d  (both compile, behaviour differs)\n' "$nv2_run_diverge"
  printf 'a_only           : %d  (only N-v2 fails -- investigate; possibly N-v2-specific features)\n' "$a_only"
  printf 'both_fail        : %d  (neither accepts; likely intentionally broken or unsupported by both)\n' "$both_fail"
  printf 'a_fail           : %d  (Track A also failed -- out of scope)\n' "$a_fail"
  printf '\nResults TSV     : %s\n' "$RESULTS_TSV"
} | tee "$SUMMARY"

# ── Exit code: 0 always; this script is diagnostic, not enforcing. ────────────
# A sibling gate (track_a_nv2_parity_gate.sh, M1.2 final step) will assert
# nv2_compile_fail == 0 once the punch-list is driven to zero.
exit 0
