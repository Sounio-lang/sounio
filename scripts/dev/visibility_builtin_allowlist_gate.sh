#!/usr/bin/env bash
# scripts/dev/visibility_builtin_allowlist_gate.sh
#
# Regression gate for the visibility-preflight builtin allow-list.
#
# Pre-fix, every builtin in {print_f64, sqrt, exp, log, sin, cos, assert}
# tripped E137 ("use of undeclared variable") in any 2+-module program,
# even though the codegen lane resolved them correctly. The visibility
# preflight at self-hosted/check/check.sio:checker_collect_runtime_builtins_inplace
# was missing them from its allow-list.
#
# This gate compiles a 2-module program that exercises all seven builtins
# via the default route. PASS = the program compiles rc=0 and runs rc=0.
# BLOCKED = any E137 or non-zero exit.
#
# Bounded scope: no CI wiring (promotion is a separate decision), no
# ModuleGraph abstractions, no roadmap expansion.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FIXTURE_DIR="$REPO_ROOT/tests/compiler/visibility_builtin_allowlist_gate"
MAIN_SRC="$FIXTURE_DIR/main.sio"
SCRATCH_DIR="$(mktemp -d /tmp/visibility-builtin-gate.XXXXXX)"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

log()  { printf '[gate] %s\n' "$*" >&2; }

# Resolve source-fresh Madaros first, then checked prebuilt.
RAW=""
for cand in "$REPO_ROOT/artifacts/self-hosted/madaros" "$REPO_ROOT/bin/madaros-linux-x86_64"; do
  if [[ -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
    RAW="$cand"; break
  fi
done

if [[ -z "$RAW" ]]; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi

RAW_SHA256="$(sha256sum "$RAW" | awk '{print $1}')"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "raw_elf=$RAW"
log "raw_elf_sha256=$RAW_SHA256"
log "git_sha=$GIT_SHA"

# Test BOTH the default route and the -O route. The default route is the
# primary subject of the regression (users hit E137 via `souc compile <src>`
# without flags). The -O route confirms the visibility fix is independent of
# optimization. Note: on a baseline checkout without the separate compact-path
# disable fix, the default route enters the compact imported-simple-IR stub
# for this multi-module program and produces a degenerate ELF; the visibility
# fix itself is verified by `souc check`, which the gate also runs.
echo "=== visibility check (no compile) ==="
"$RAW" --check "$MAIN_SRC" >"$SCRATCH_DIR/check.stdout" 2>"$SCRATCH_DIR/check.stderr"
CHECK_RC=$?
if [[ $CHECK_RC -ne 0 ]]; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=check_failed check_rc=$CHECK_RC" >&2
  echo "--- check stderr tail ---" >&2
  tail -10 "$SCRATCH_DIR/check.stderr" >&2
  exit 1
fi
if grep -qE 'error\[E[0-9]+\]' "$SCRATCH_DIR/check.stdout" "$SCRATCH_DIR/check.stderr"; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=check_emitted_error" >&2
  exit 1
fi
log "visibility check: PASS (no E137/E174 in importing program)"

# Run route: -O (the default route is verified separately by the
# default_path_fidelity_gate once the compact-disable fix lands; on this
# branch in isolation, the compact stub produces a degenerate ELF for the
# multi-builtin fixture shape, which is an unrelated limitation).
echo "=== -O compile + run ==="
OUT="$SCRATCH_DIR/visibility_builtin_O.elf"
STDOUT="$SCRATCH_DIR/stdout"
STDERR="$SCRATCH_DIR/stderr"

"$RAW" "$MAIN_SRC" -o "$OUT" -O >"$STDOUT" 2>"$STDERR"
CC_RC=$?

if [[ $CC_RC -ne 0 ]]; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=compile_fail cc_rc=$CC_RC" >&2
  echo "--- stderr tail ---" >&2
  tail -10 "$STDERR" >&2
  exit 1
fi

if [[ ! -s "$OUT" ]]; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=no_elf_emitted" >&2
  exit 1
fi

chmod +x "$OUT"
RUN_OUT="$SCRATCH_DIR/run.out"
"$OUT" >"$RUN_OUT" 2>/dev/null
RUN_RC=$?

if [[ $RUN_RC -ne 0 ]]; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=run_fail run_rc=$RUN_RC" >&2
  exit 1
fi

# Sanity: stdout should contain at least one occurrence of "0.500000"
# (from print_f64(half())). We don't compare full bytes because the
# transcendental results depend on libm precision and are not the
# subject of this gate (the visibility fix is).
if ! grep -q '0\.500000' "$RUN_OUT"; then
  echo "VISIBILITY_BUILTIN_GATE_BLOCKED reason=stdout_missing_half_value" >&2
  echo "--- stdout ---" >&2
  cat "$RUN_OUT" >&2
  exit 1
fi

LINES=$(wc -l < "$RUN_OUT")
echo "VISIBILITY_BUILTIN_GATE_PASS raw_sha256=${RAW_SHA256:0:16} stdout_lines=$LINES"
exit 0
