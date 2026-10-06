#!/usr/bin/env bash
# Region reclamation (docs/decisions/adr-010-madaros-region-reclamation.md):
# witnesses that cannot live in a plain run-pass file. Copilot review, PR #2737.
#
#  1. Streaming lane. The single-module streaming lane
#     (compiler/module_native_streaming.sio, reached by --probe-native-streaming)
#     compiles each function separately through compile_ir_function_v2_into.
#     Every region witness must compile there with a non-empty body for every
#     region builtin (they used to be zero-length call targets); the default
#     build of each witness is compiled and run.
#  2. Fail closed. SOUNIO_NV2_CORE_REFUSE_FN makes the core emitter refuse one
#     function exactly as it refuses an opcode it cannot emit. Both lanes must
#     refuse the program and write no ELF (the streaming lane used to discard
#     the failure and report success).
#  3. Mutation. SOUNIO_RGN_BARRIER_SABOTAGE removes ONE store path's barrier.
#     The witness for that path must then fail: a witness that passes with its
#     barrier removed proves nothing.
#  4. Opt-out. SOUNIO_NO_REGION_RECLAIM=1 restores the pre-region lowering:
#     no __rgn_* symbol is emitted.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="[madaros-region-reclaim]"
fail() { echo "$TAG FAIL: $*" >&2; exit 1; }

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2; exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

WORK="$(mktemp -d /tmp/sounio-madaros-region-reclaim.XXXXXX)"
if [[ "${SOUNIO_MADAROS_REGION_GATE_KEEP:-0}" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

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
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"
case "$RAW" in /*) ;; *) RAW="$PWD/$RAW" ;; esac

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
# Exercise the shipped default, even if the caller exported the A/B opt-out.
unset SOUNIO_NO_REGION_RECLAIM SOUNIO_RGN_BARRIER_SABOTAGE SOUNIO_NV2_CORE_REFUSE_FN SOUNIO_PROBE_STREAMING_OUT SOUNIO_NATIVE_STREAM_DIAG

# Match bin/madaros without changing the compiled program's stack limit.
raw_compile() (
  local stack_kb="${MADAROS_STACK_KB:-524288}"
  case "$stack_kb" in
    ''|*[!0-9]*) fail "invalid MADAROS_STACK_KB: $stack_kb" ;;
    0) stack_kb=unlimited ;;
  esac
  ulimit -s "$stack_kb" 2>/dev/null || fail "cannot configure compiler stack to $stack_kb KiB; no verdict"
  exec "$RAW" "$@"
)

# The default single-module build (what `souc run` / `madaros build` uses; the
# raw native-compile flag only serves multi-module programs).
default_build() { MADAROS_RAW_BIN="$RAW" "$ROOT_DIR/bin/madaros" build "$1" -o "$2"; }

RP="$ROOT_DIR/tests/run-pass"
expected_of() { sed -n 's|^//@ expect-stdout: *||p' "$1" | head -1; }

# run_elf <elf> <out>: exit status of the program, stdout captured.
run_elf() { local rc=0; timeout 300 "$1" >"$2" 2>&1 || rc=$?; echo "$rc"; }

STREAM_CASES=(
  madaros_region_reclaim_escape.sio
  madaros_region_reclaim_result.sio
  madaros_region_reclaim_encoded_scalar.sio
  madaros_region_reclaim_encoded_aggregate.sio
  madaros_region_reclaim_cast_xor.sio
  madaros_region_reclaim_barrier_paths.sio
  madaros_region_reclaim_syscall_escape.sio
)

# ---------------------------------------------------------------- 1 streaming
# The default build is compiled and RUN. The streaming lane's ELF cannot be
# run: on origin/main it already segfaults for a hello-world (measured
# 2026-10-06 with SOUNIO_PROBE_STREAMING_OUT), independently of regions. So
# for that lane the gate requires what the review asked for: every emitted
# call target -- each __rgn_* builtin in particular -- gets a non-empty body,
# and a region actually opens there (an __rgn_enter body exists).
for name in "${STREAM_CASES[@]}"; do
  src="$RP/$name"; base="${name%.sio}"
  [[ -f "$src" ]] || fail "missing fixture $src"
  want="$(expected_of "$src")"
  default_build "$src" "$WORK/$base.default.elf" >"$WORK/$base.default.log" 2>&1 \
    || { tail -n 30 "$WORK/$base.default.log" >&2; fail "$base: default compile failed"; }
  rc="$(run_elf "$WORK/$base.default.elf" "$WORK/$base.default.out")"
  [[ "$rc" == 0 ]] || fail "$base: default build exited $rc"
  grep -qxF "$want" "$WORK/$base.default.out" || fail "$base: default build did not print $want"

  SOUNIO_NATIVE_STREAM_DIAG=1 raw_compile --probe-native-streaming "$src" >"$WORK/$base.stream.log" 2>&1 \
    || { tail -n 30 "$WORK/$base.stream.log" >&2; fail "$base: streaming probe crashed"; }
  if ! grep -q "streaming_used=true ok=true" "$WORK/$base.stream.log"; then
    if grep -q "streaming_used=false" "$WORK/$base.stream.log"; then
      echo "$TAG note: $base: streaming lane declined: $(grep -o 'err=[^ ]*' "$WORK/$base.stream.log" | head -1)"
      continue
    fi
    tail -n 30 "$WORK/$base.stream.log" >&2
    fail "$base: streaming lane refused a program the default lane compiles"
  fi
  grep -q "native_streaming: emitted fn=__rgn_enter bytes=" "$WORK/$base.stream.log" \
    || fail "$base: streaming lane opened no region (no __rgn_enter body)"
  if grep -E "native_streaming: emitted fn=[^ ]+ bytes=0$" "$WORK/$base.stream.log" | grep -q "fn=__rgn_"; then
    grep -E "emitted fn=__rgn_[^ ]+ bytes=0$" "$WORK/$base.stream.log" >&2
    fail "$base: streaming lane emitted an empty region builtin"
  fi
  STREAMED=$(( ${STREAMED:-0} + 1 ))
  echo "$TAG PASS: streaming lane emits every region builtin: $base"
done
[[ "${STREAMED:-0}" -ge 3 ]] || fail "streaming lane covered fewer than 3 region witnesses (${STREAMED:-0})"

# ---------------------------------------------------------------- 2 fail closed
src="$RP/madaros_region_reclaim_escape.sio"
SOUNIO_NV2_CORE_REFUSE_FN=put_ref SOUNIO_PROBE_STREAMING_OUT="$WORK/refuse.stream.elf" \
  raw_compile --probe-native-streaming "$src" >"$WORK/refuse.stream.log" 2>&1 || true
grep -q "streaming_used=true ok=false err=streaming_native_v2_codegen_failed:put_ref" "$WORK/refuse.stream.log" \
  || { tail -n 20 "$WORK/refuse.stream.log" >&2; fail "streaming lane did not refuse a function the core emitter refused"; }
[[ ! -e "$WORK/refuse.stream.elf" ]] || fail "streaming lane wrote an ELF despite a refused function"
if SOUNIO_NV2_CORE_REFUSE_FN=put_ref default_build "$src" "$WORK/refuse.default.elf" >"$WORK/refuse.default.log" 2>&1; then
  fail "default lane compiled a program whose function the core emitter refused"
fi
[[ ! -e "$WORK/refuse.default.elf" ]] || fail "default lane wrote an ELF despite a refused function"
echo "$TAG PASS: both lanes fail closed on a refused function"

# ---------------------------------------------------------------- 3 mutation
# <sabotage> <fixture> <exit status of the check that must catch it>. Exit 1
# for seqpush: the reclaimed growth storage is caught by the Seq bounds check
# (exit 1) before the field check that would return 5.
MUTATIONS=(
  "field madaros_region_reclaim_escape.sio 1"
  "global madaros_region_reclaim_escape.sio 2"
  "storeptr madaros_region_reclaim_barrier_paths.sio 1"
  "index madaros_region_reclaim_barrier_paths.sio 2"
  "seqset madaros_region_reclaim_barrier_paths.sio 4"
  "seqpush madaros_region_reclaim_barrier_paths.sio 1"
  "rawwrite madaros_region_reclaim_barrier_paths.sio 6"
  "call madaros_region_reclaim_cast_xor.sio 1"
  "syscall madaros_region_reclaim_syscall_escape.sio 2"
)
for m in "${MUTATIONS[@]}"; do
  read -r path name want_rc <<<"$m"
  base="${name%.sio}.no-$path"
  SOUNIO_RGN_BARRIER_SABOTAGE="$path" default_build "$RP/$name" "$WORK/$base.elf" >"$WORK/$base.log" 2>&1 \
    || { tail -n 30 "$WORK/$base.log" >&2; fail "$base: compile failed"; }
  rc="$(run_elf "$WORK/$base.elf" "$WORK/$base.out")"
  [[ "$rc" == "$want_rc" ]] || fail "$base: without the $path barrier the witness exited $rc, expected $want_rc (witness has no teeth)"
  echo "$TAG PASS: removing the $path barrier breaks $name (exit $rc)"
done

# ---------------------------------------------------------------- 4 opt-out
src="$RP/madaros_region_reclaim_loop.sio"
SOUNIO_NO_REGION_RECLAIM=1 default_build "$src" "$WORK/optout.elf" >"$WORK/optout.log" 2>&1 \
  || { tail -n 30 "$WORK/optout.log" >&2; fail "opt-out compile failed"; }
if grep -a -q "__rgn_enter" "$WORK/optout.elf"; then
  fail "SOUNIO_NO_REGION_RECLAIM=1 still emitted region runtime symbols"
fi
default_build "$src" "$WORK/default.loop.elf" >"$WORK/default.loop.log" 2>&1 \
  || { tail -n 30 "$WORK/default.loop.log" >&2; fail "default loop compile failed"; }
grep -a -q "__rgn_enter" "$WORK/default.loop.elf" || echo "$TAG note: symbol names not present in ELF; opt-out check is vacuous"
echo "$TAG PASS: opt-out emits no region runtime"
echo "$TAG PASS"
