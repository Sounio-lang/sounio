#!/usr/bin/env bash
# #2831: a relative `use` that is not found next to the importing module must
# resolve against that module's ancestor directories (closest first) before
# stdlib/, self-hosted/ and the CWD, on both engines -- Madaros's order
# (module_frontend_resolve_import_relpath). Layout: main.sio -> `use sub::b` ->
# sub/b.sio -> `use a` with a.sio only next to main.sio. lean_single used to try
# only sub/a.sio, then stdlib/, self-hosted/ and the CWD, so the program
# compiled only when run from main.sio's directory (E224 "unreadable import"
# otherwise).
#
# Discriminator: a second tree also has sub/a.sio; the closer file (sub/) must
# win on both engines (prints 101, not 2).
#
# Engines: lean_single = the committed bin/souc-lean-single-x86_64 (refreshed by
# the PR that changes lean_single.sio), Madaros = bin/souc's default engine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-use-entry-dir.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || true

fail() { echo "[use-entry-dir] FAIL: $*" >&2; exit 1; }

mk_tree() {
  local d="$1"
  mkdir -p "$d/sub"
  printf 'pub fn one() -> i64 { 1 }\n' >"$d/a.sio"
  printf 'use sub::b::{two}\nfn main() -> i32 with IO { print_int(two()); print("\\n"); return 0 }\n' >"$d/main.sio"
  printf 'use a::{one}\npub fn two() -> i64 { one() + 1 }\n' >"$d/sub/b.sio"
}

mk_tree "$WORK/far"
mk_tree "$WORK/near"
printf 'pub fn one() -> i64 { 100 }\n' >"$WORK/near/sub/a.sio"

# A directory unrelated to the tree, so the CWD fallback cannot find a.sio.
mkdir -p "$WORK/elsewhere"

run_case() {
  local engine="$1" tree="$2" want="$3" out rc
  set +e
  if [[ "$engine" == lean_single ]]; then
    out="$(cd "$WORK/elsewhere" && "$ROOT_DIR/bin/souc-lean-single-x86_64" "$WORK/$tree/main.sio" "$WORK/$tree.$engine.elf" >"$WORK/$tree.$engine.log" 2>&1 && chmod +x "$WORK/$tree.$engine.elf" && "$WORK/$tree.$engine.elf")"
  else
    out="$(cd "$WORK/elsewhere" && timeout 600 "$ROOT_DIR/bin/souc" run "$WORK/$tree/main.sio" 2>"$WORK/$tree.$engine.log" | tail -n 1)"
  fi
  rc=$?
  set -e
  if [[ $rc -ne 0 || "$out" != "$want" ]]; then
    grep -E 'error|E224' "$WORK/$tree.$engine.log" | head -5 >&2 || true
    fail "$engine $tree: rc=$rc out='$out' want='$want'"
  fi
  echo "[use-entry-dir] PASS $engine $tree -> $out"
}

run_case lean_single far 2
run_case lean_single near 101
run_case madaros far 2
run_case madaros near 101
echo "[use-entry-dir] PASS"
