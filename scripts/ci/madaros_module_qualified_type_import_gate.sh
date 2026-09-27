#!/usr/bin/env bash
# Copilot review (PR #2515), comment 4114124500: a path-form TYPE import
# (`use pkg::mod::Type;`) must still register its module-qualifier suffix.
# Copilot review (PR #2515), comment 4114245201 (and its follow-up fix in
# self-hosted/ir/lower.sio's callee_path_module_stripped_name): the lowerer
# side of this exact shape is now fixed too, so this gate compiles and runs
# the fixture end-to-end instead of only `--check`ing it. See
# tests/multimodule/module_qualified_type_import/README.md for the history.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/module_qualified_type_import"
TAG="[madaros-module-qualified-type-import]"

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

if [[ -n "${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-module-qualified-type-import.XXXXXX)"
fi
if [[ -z "${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# compile_case <label> <source> -> $WORK/<label>.{log,elf,out}
compile_and_run() {
  local label="$1" src="$2"
  local log="$WORK/$label.log" elf="$WORK/$label.elf" out="$WORK/$label.out"
  [[ -f "$src" ]] || fail "$label: missing fixture $src"
  if ! "$RAW" --native-compile "$src" -o "$elf" >"$log" 2>&1; then
    tail -n 40 "$log" >&2 || true
    fail "$label: did not compile"
  fi
  [[ -s "$elf" ]] || fail "$label: compiler did not emit an ELF"
  chmod +x "$elf"
  if ! timeout 30 "$elf" >"$out" 2>&1; then
    cat "$out" >&2 || true
    fail "$label: compiled program did not run to completion"
  fi
}

expect_output() {
  local label="$1" expected="$2"
  if ! diff -u "$expected" "$WORK/$label.out" >"$WORK/$label.diff"; then
    cat "$WORK/$label.diff" >&2
    fail "$label: program output differs from $(basename "$(dirname "$expected")")/expected.txt"
  fi
}

# --- basic: a path-form type import must keep its module-qualifier suffix,
# AND the lowerer must resolve the resulting Type::method call to the real
# mangled impl-method body, not a body-less stub (see README history) ---
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): a path-form type import's qualified associated call compiles and runs correctly"

# --- oversized: Copilot review (PR #2515), comment 4114530613:
# module_frontend_named_import_terminal_is_type's byte scanner reads through
# read_file's fixed 1 MiB buffer (the same cap every read_file caller in
# this tree lives with) -- a real module file past that size (self-hosted/
# check/check.sio and self-hosted/ir/lower.sio both are) with its struct/
# enum declaration past the cutoff used to scan as "not found", silently
# misclassifying a genuine type import as a function/value one. Generated
# at run time rather than checked in, so the repository does not carry a
# multi-MiB fixture: a filler comment block pushes `pub struct Widget` past
# 1 MiB in "pkg_mod_big.sio", exactly mirroring the `basic` case above but
# at a size only the fail-closed fix (self-hosted/compiler/module_frontend.
# sio) can get right.
oversized_dir="$WORK/oversized"
mkdir -p "$oversized_dir"
{
  i=0
  # ~1.05 MiB of `// filler...\n` lines (18 bytes each) pushes the struct
  # declaration past the 1 MiB read_file cutoff.
  while ((i < 61000)); do
    printf '// filler filler filler\n'
    i=$((i + 1))
  done
  cat <<'EOF'
pub struct Widget {
    n: i64,
}

impl Widget {
    pub fn make() -> i64 {
        42
    }
}
EOF
} > "$oversized_dir/pkg_mod_big.sio"
[[ "$(wc -c <"$oversized_dir/pkg_mod_big.sio")" -gt 1048576 ]] || fail "oversized: generated fixture did not exceed 1 MiB"
cat <<'EOF' > "$oversized_dir/main.sio"
use pkg_mod_big::Widget;

fn main() -> i32 with IO, Mut, Panic {
    let v: i64 = pkg_mod_big::Widget::make()
    if v == 42 {
        println("OVERSIZED_MODULE_TYPE_IMPORT_QUALIFIER_OK")
        return 0
    }
    1
}
EOF
echo "OVERSIZED_MODULE_TYPE_IMPORT_QUALIFIER_OK" > "$oversized_dir/expected.txt"
compile_and_run oversized "$oversized_dir/main.sio"
expect_output oversized "$oversized_dir/expected.txt"
echo "$TAG PASS(oversized): a path-form type import past read_file's 1 MiB cutoff still keeps its module-qualifier suffix"

echo "$TAG PASS: path-form type import resolved correctly"
