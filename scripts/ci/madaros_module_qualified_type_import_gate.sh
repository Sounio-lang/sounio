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

# --- oversized: Copilot review (PR #2515), comment 4114530613 (correcting
# the prior round's own comment 4114530613 fix -- see this fixture's
# README): module_frontend_named_import_terminal_is_type now scans the
# file's real, full size (read_file is not actually 1 MiB-capped; only the
# lexer's genuine 16 MiB ceiling is), rather than either truncating at 1
# MiB or blindly guessing "is type" past it. Generated at run time rather
# than checked in, so the repository does not carry a multi-MiB fixture: a
# filler comment block pushes `pub struct Widget` past 1 MiB in
# "pkg_mod_big.sio", exactly mirroring the `basic` case above but at a size
# a real scan -- not a truncated one -- is needed to get right.
oversized_dir="$WORK/oversized"
mkdir -p "$oversized_dir"
{
  i=0
  # ~1.05 MiB of `// filler...\n` lines pushes the struct declaration well
  # past 1 MiB (still far under the lexer's real 16 MiB ceiling).
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
echo "$TAG PASS(oversized): a path-form type import past 1 MiB is still found by a real full-file scan, not a blind fail-closed guess"

# --- multiline: Copilot review, "Handle newlines when scanning path-form
# type imports": the lexer treats LF/CR as ordinary whitespace, so
# `struct\nWidget` is a valid declaration, but the scanner used to skip
# only space/tab after the keyword and missed it -- misclassifying the
# import as a function/value and reintroducing the W044/body-less-lowering
# bug this whole scanner exists to prevent.
multiline_dir="$WORK/multiline"
mkdir -p "$multiline_dir"
cat <<'EOF' > "$multiline_dir/pkg_mod_ml.sio"
pub struct
Widget {
    n: i64,
}

impl Widget {
    pub fn make() -> i64 {
        42
    }
}
EOF
cat <<'EOF' > "$multiline_dir/main.sio"
use pkg_mod_ml::Widget;

fn main() -> i32 with IO, Mut, Panic {
    let v: i64 = pkg_mod_ml::Widget::make()
    if v == 42 {
        println("MULTILINE_MODULE_TYPE_IMPORT_QUALIFIER_OK")
        return 0
    }
    1
}
EOF
echo "MULTILINE_MODULE_TYPE_IMPORT_QUALIFIER_OK" > "$multiline_dir/expected.txt"
compile_and_run multiline "$multiline_dir/main.sio"
expect_output multiline "$multiline_dir/expected.txt"
echo "$TAG PASS(multiline): a struct declaration split across a newline is still found by the type scanner"

echo "$TAG PASS: path-form type import resolved correctly"
