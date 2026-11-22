#!/usr/bin/env bash
# A bare call is resolved through the calling module's own `use` items. A pub
# fn of a module the caller never imported -- loaded only because some other
# module used it -- is refused with error[E-NOT-IMPORTED] instead of being
# bound through the flat function table. Named imports, globs, `pub use`
# re-exports and same-spelled locals still compile and run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
FIX="$ROOT_DIR/tests/multimodule/import_hygiene"
TAG="[madaros-import-hygiene]"

fail() {
  echo "$TAG FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

WORK="$(mktemp -d /tmp/sounio-madaros-import-hygiene.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

if ! "$RAW" compile "$FIX/ok.sio" -o "$WORK/ok.elf" >"$WORK/ok.log" 2>&1 || [[ ! -s "$WORK/ok.elf" ]]; then
  tail -n 30 "$WORK/ok.log" >&2 || true
  fail "ok.sio does not compile"
fi
chmod +x "$WORK/ok.elf"
"$WORK/ok.elf" >"$WORK/ok.out" 2>&1 || fail "ok.sio exited non-zero"
diff -u "$FIX/expected_ok.txt" "$WORK/ok.out" || fail "ok.sio output differs"
echo "$TAG PASS(ok): named import, glob, pub use and a same-spelled local"

rc=0
sounio_capture_rc rc "$WORK/unimported.log" -- "$RAW" compile "$FIX/unimported.sio" -o "$WORK/unimported.elf"
sounio_expect_refusal "$rc" "$WORK/unimported.log" "" "error[E-NOT-IMPORTED]" || {
  cat "$WORK/unimported.log" >&2
  fail "unimported.sio: expected a clean E-NOT-IMPORTED refusal, got rc=$rc $(sounio_refusal_reason)"
}
grep -q '`hyg_double`' "$WORK/unimported.log" || fail "unimported.sio: refusal does not name hyg_double"
echo "$TAG PASS(unimported): a pub fn of an unimported module is refused"
echo "$TAG PASS"
