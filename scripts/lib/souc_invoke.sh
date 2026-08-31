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

# A compile must be about THIS tree. `SOUNIO_STDLIB_PATH` and `SOUC_BIN` are
# exported by the dev profile and point at the shared /workspace/sounio checkout,
# so a worktree compile silently resolves the stdlib from another commit. Measured
# 2026-08-27 by hand on the gen2 wall: with the inherited value, 80 E175 from a
# stdlib at one commit against a compiler at another, dying in type-check before
# lowering; with it cleared, the run reaches the real wall. AGENT_HANDOFF.md
# records two earlier wrong conclusions from the same confound.
#
# madaros_fixed_point_gate.sh ALREADY defends itself (it exports
# SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" at line 109, with a comment naming this
# exact hazard). I first wrote that it did not, and that was wrong. This guard is
# for every OTHER caller of souc_compile, which inherits whatever the profile
# exported and has no such line.
#
# Only a value pointing OUTSIDE the repo is dropped. Setting it deliberately to
# another path in-tree is a real use and is left alone; the accidental case is
# always the absolute path to a different checkout.
# Only SOUNIO_STDLIB_PATH is dropped. It is the RESOLUTION path -- it decides
# which stdlib the compile reads -- and a foreign value silently pairs one tree's
# stdlib with another tree's compiler. The binary-naming variables are NOT
# touched: MADAROS_BIN and SOUC_BIN legitimately point outside the repo, because
# a built compiler lives in /tmp or a scratch dir, and dropping them would break
# every caller that passes one. (I had them in this list; the first end-to-end run
# would have thrown away the very binary under test.) They are reported, not
# removed, so a surprising one is visible in the log.
souc_scrub_foreign_env() {
  local root="${1:-$PWD}" v p
  p="${SOUNIO_STDLIB_PATH:-}"
  if [[ -n "$p" ]]; then
    case "$p" in
      "$root"/*) : ;;
      *) echo "souc_invoke: dropping SOUNIO_STDLIB_PATH=$p (outside $root); it would pair a foreign stdlib with this tree's compiler" >&2
         unset SOUNIO_STDLIB_PATH ;;
    esac
  fi
  for v in SOUC_BIN SOUNIO_SOUC_BIN MADAROS_BIN MADAROS_RAW_BIN; do
    p="$(eval printf '%s' "\${$v:-}")"
    [[ -n "$p" ]] && echo "souc_invoke: $v=$p" >&2
  done
  return 0
}

# souc_compile <bin> <src> <out> [extra args...]
# Madaros recurses deeply in the parser; the caller may already have raised the
# soft stack, but raising it again in a subshell is free.
#
# 512 MiB was not enough and the shortfall was invisible: the gen2 self-compile
# needs ~1.1 GiB of stack, and below that it dies rc=42 at the cross-module DCE
# refusal instead of reaching the wall it is meant to measure. It also needs the
# virtual-memory limit lifted -- with the stack alone raised it still dies early.
# Both are requested with `|| true` so a constrained runner degrades rather than
# refusing to run, and the values actually granted are echoed, because `ulimit`
# silently declines above the hard limit and a number nobody printed is a number
# nobody checked.
_souc_limits() {
  ulimit -s 1572864 2>/dev/null || ulimit -s 524288 2>/dev/null || true
  ulimit -v unlimited 2>/dev/null || true
  echo "souc_invoke: stack=$(ulimit -s) virt=$(ulimit -v)" >&2
}

souc_compile() {
  local bin="$1" src="$2" out="$3"; shift 3
  local kind
  kind="$(souc_banner "$bin")"
  local root; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  case "$kind" in
    madaros)
      ( souc_scrub_foreign_env "$root"; _souc_limits; "$bin" build "$src" "$out" "$@" ) ;;
    lean_single)
      ( souc_scrub_foreign_env "$root"; _souc_limits; "$bin" "$src" "$out" "$@" ) ;;
    *)
      echo "souc_invoke: cannot tell what compiler $bin is; refusing to guess an argv" >&2
      return 78 ;;
  esac
}
