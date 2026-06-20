#!/usr/bin/env bash
# Focused regression gate for enum declaration + enum variant USE in Madaros.
#
# Why this exists: enum-variant use (E::A, `match` on an enum, `E::A as i64`)
# SIGSEGV'd the native_v2 compiler because lowerer_register_enum_variants_mut wrote
# the variant table with a two-level nested store
# (`(*(*lo).enum_variants).entries[idx] = ...`) that the codegen silently discards,
# leaving enum_variants empty so lookup_variant_discriminant fell through to a
# corrupt module.functions scan. See docs/audit/MADAROS_ROOT2_ENUM_FNCOUNT_2026-06-20.md.
#
# The generic full gate never exercised enum USE, so the bug shipped green. This
# gate builds AND runs each fixture and checks the exit code, so it FAILS on a
# pre-fix Madaros (compile SIGSEGV / wrong discriminant) and PASSES once fixed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
WORK="${SOUNIO_MADAROS_ENUM_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-enum.XXXXXX)}"
KEEP_WORK="${SOUNIO_MADAROS_ENUM_GATE_KEEP:-0}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK"

fail() { echo "[madaros-enum] FAIL: $*" >&2; exit 1; }
pass() { echo "[madaros-enum] PASS: $*"; }

# build_run <name> <expected-exit> <source>
# Builds the source to an ELF (compiler must not crash) and runs it, asserting the
# program's exit code. A compiler SIGSEGV shows up as a missing emit marker.
build_run() {
  local name="$1" expected="$2" src="$3"
  local sio="$WORK/$name.sio" elf="$WORK/$name.elf" log="$WORK/$name.log"
  printf '%s\n' "$src" > "$sio"
  rm -f "$elf"

  # Check ELF-produced (not a build-subcommand log marker): the production launcher
  # path prints "Output:" while the raw build subcommand prints "emitted path=";
  # both produce the ELF on success, and a compiler crash produces none.
  set +e
  "$MADAROS" build "$sio" -o "$elf" >"$log" 2>&1
  local crc=$?
  set -e
  if [[ ! -s "$elf" ]]; then
    echo "[madaros-enum] build log tail for $name (rc=$crc):" >&2
    tail -n 20 "$log" >&2 || true
    fail "$name: compiler produced no ELF (Root2 enum-registration regression?)"
  fi
  chmod +x "$elf"

  set +e
  "$elf"
  local rrc=$?
  set -e
  if [[ "$rrc" != "$expected" ]]; then
    fail "$name: expected exit $expected, got $rrc"
  fi
  pass "$name (exit $rrc)"
}

printf '[madaros-enum] madaros=%s\n' "$MADAROS"
printf '[madaros-enum] work=%s\n' "$WORK"

# Declaration only (never regressed, but pins the baseline).
build_run enum_decl_unused 5 'enum E { A, B }
fn main() -> i64 { 5 }'

# Variant USE in a let binding — the exact Root2 SIGSEGV repro.
build_run enum_let_use 5 'enum E { A, B }
fn main() -> i64 { let e = E::A
    5 }'

# Variant value via discriminant (default discriminants start at 0: A=0, B=1).
build_run enum_as_int 1 'enum E { A, B }
fn main() -> i64 { E::B as i64 }'

# Match on an enum scrutinee resolves to the right arm value.
build_run enum_match_b 20 'enum E { A, B }
fn main() -> i64 { let e = E::B
    match e { E::A => 10, E::B => 20 } }'

# Three-variant enum: discriminant ordering A=0,B=1,C=2 through a match.
build_run enum_match_three 2 'enum C { X, Y, Z }
fn main() -> i64 { let c = C::Z
    match c { C::X => 0, C::Y => 1, C::Z => 2 } }'

echo "[madaros-enum] all enum regression fixtures passed"
