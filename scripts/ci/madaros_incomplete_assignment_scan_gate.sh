#!/usr/bin/env bash
# The incomplete-assignment pre-scan, both directions.
#
# compiler/main.sio compile_scan_incomplete_assignment walks the input file BYTE BY BYTE looking
# for an `=` followed only by whitespace up to EOF. It had no comment, string or char-literal
# awareness, so any file whose last non-whitespace byte was `=` was rejected outright with
# "syntax appears incomplete after assignment" — which the `// =====` banner this repo puts at end
# of file makes routine. Five versioned .sio files were rejected that way, including
# self-hosted/hlir/lower.sio, and the scan was MASKING their real diagnostics.
#
# This lives in a gate rather than in tests/compile-fail because the check is MADAROS-ONLY: the
# pre-scan is in the Madaros driver, and lean_single accepts `let x =` outright. A compile-fail
# test therefore passes or fails depending on which engine bin/souc resolves to — measured, it
# passed locally against Madaros artifacts and failed in CI where the suite fell back to
# lean_single ("expected compile failure but passed", 1 of 2542). Pinning the engine is the fix.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-incomplete-assign] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-incomplete-assign] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

OUT_DIR="${SOUNIO_MADAROS_INCOMPLETE_ASSIGN_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-incomplete-assign.XXXXXX)}"
mkdir -p "$OUT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Subject must be a Madaros built from current source: the scan lives in compiler/main.sio.
# NOT wrapped in scripts/dev/souc-build-lock.sh — build_modular_madaros.sh takes that lock itself
# (lines 101 and 115) and flock(1) is not reentrant, so wrapping it deadlocks.
SOUC_BIN="${SOUNIO_MADAROS_INCOMPLETE_ASSIGN_GATE_BIN:-}"
if [[ -z "$SOUC_BIN" ]]; then
  SOUC_BIN="$OUT_DIR/madaros-from-source.elf"
  printf '[madaros-incomplete-assign] no *_GATE_BIN; building Madaros from source\n'
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$SOUC_BIN" \
        >"$OUT_DIR/build.log" 2>&1; then
    echo "[madaros-incomplete-assign] FAIL: could not build Madaros from source" >&2
    tail -n 30 "$OUT_DIR/build.log" >&2 || true
    exit 1
  fi
fi
chmod +x "$SOUC_BIN" 2>/dev/null || true
[[ -x "$SOUC_BIN" ]] || { echo "[madaros-incomplete-assign] FAIL: not executable: $SOUC_BIN" >&2; exit 1; }

printf '[madaros-incomplete-assign] souc=%s\n' "$SOUC_BIN"

fail=0
MARKER='syntax appears incomplete after assignment'

# want=reject expects the pre-scan to fire; want=accept expects it not to.
run_case() {
  local label="$1" want="$2" src="$3"
  local log="$OUT_DIR/$label.log"
  "$SOUC_BIN" --check "$src" >"$log" 2>&1 || true
  local n
  n="$(grep -c "$MARKER" "$log" || true)"
  if [[ "$want" == "reject" && "$n" -eq 0 ]]; then
    echo "[madaros-incomplete-assign] FAIL($label): a genuinely unfinished assignment was accepted" >&2
    tail -n 6 "$log" >&2 || true
    fail=1
    return
  fi
  if [[ "$want" == "accept" && "$n" -ne 0 ]]; then
    echo "[madaros-incomplete-assign] FAIL($label): the pre-scan fired on valid source" >&2
    tail -n 6 "$log" >&2 || true
    fail=1
    return
  fi
  printf '[madaros-incomplete-assign] PASS(%s) want=%s hits=%s\n' "$label" "$want" "$n"
}

# A real unfinished assignment must still be caught: making the scan comment-aware must not make
# it blind.
cat >"$OUT_DIR/real_incomplete.sio" <<'SIO'
fn main() with IO, Mut, Panic, Div {
    let x =
}
SIO
run_case real_incomplete reject "$OUT_DIR/real_incomplete.sio"

# The trigger that rejected five versioned files: a `=` banner comment at end of file, plus `=`
# inside a string and inside a char literal.
cat >"$OUT_DIR/banner_eof.sio" <<'SIO'
fn main() with IO, Mut, Panic, Div {
    let eq_in_string = "a = b ="
    let eq_char = '='
    print_int(str_len(eq_in_string))
    println(" BANNER")
}

// ============================================================================
// End of file
// ============================================================================
SIO
run_case banner_eof accept "$OUT_DIR/banner_eof.sio"

# In-tree file that the pre-scan used to reject outright.
run_case hlir_lower_in_tree accept "$ROOT_DIR/self-hosted/hlir/lower.sio"

if [[ "$fail" -ne 0 ]]; then
  echo "[madaros-incomplete-assign] GATE FAILED" >&2
  exit 1
fi

echo "[madaros-incomplete-assign] PASS: unfinished assignments rejected, '=' banners and literals accepted"
