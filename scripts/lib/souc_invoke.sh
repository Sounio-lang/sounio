#!/usr/bin/env bash
# Pick a compiler's argv by asking the compiler what it is.
#
# WHY THIS EXISTS. The two engines in this tree do not share a command line,
# and nothing in scripts/ knows that:
#
#   lean_single    <src> <out>            positional, no subcommand
#   Madaros        build <src> <out>      the literal word `build` is required
#
# Madaros's parse_options treats every bare positional as `input_file` and
# keeps the LAST one, with the output defaulting to `a.out` in $PWD. So handing
# Madaros lean_single's argv does not error: it compiles the wrong file to the
# wrong place and exits 0. The stray a.out that keeps reappearing at the repo
# root is exactly that. scripts/ci/build_modular_madaros.sh:134 uses the
# lean_single form, which is CORRECT for the seed it builds with and silently
# wrong for any Madaros passed in via SOUC_BIN.
#
# usage:
#   . scripts/lib/souc_invoke.sh
#   souc_banner   <bin>              -> "madaros" | "lean_single" | "unknown"
#   souc_compile  <bin> <src> <out>  -> compiles, argv chosen by banner

if [[ -n "${_SOUNIO_SOUC_INVOKE_SOURCED:-}" ]]; then return 0; fi
_SOUNIO_SOUC_INVOKE_SOURCED=1

souc_banner() {
  local bin="$1" out
  [[ -x "$bin" ]] || { printf 'unknown'; return 0; }
  # Both engines answer --version, and each says something only it says.
  # Madaros prints its three-line banner; lean_single prints its usage line,
  # which still carries the `mini_native` name it was bootstrapped under.
  out="$("$bin" --version 2>&1 | head -3 || true)"
  if printf '%s' "$out" | grep -qi 'madaros'; then
    printf 'madaros'
  elif printf '%s' "$out" | grep -qE 'Usage: mini_native <source\.sio> <output>'; then
    printf 'lean_single'
  else
    printf 'unknown'
  fi
}

# souc_compile <bin> <src> <out> [extra args...]
# Madaros recurses deeply in the parser; the caller may already have raised the
# soft stack, but raising it again in a subshell is free and CI's default
# 16 MiB is not enough.
souc_compile() {
  local bin="$1" src="$2" out="$3"; shift 3
  local kind
  kind="$(souc_banner "$bin")"
  case "$kind" in
    madaros)
      ( ulimit -s 524288 2>/dev/null || true; "$bin" build "$src" "$out" "$@" ) ;;
    lean_single)
      ( ulimit -s 524288 2>/dev/null || true; "$bin" "$src" "$out" "$@" ) ;;
    *)
      echo "souc_invoke: cannot tell what compiler $bin is; refusing to guess an argv" >&2
      return 78 ;;
  esac
}
