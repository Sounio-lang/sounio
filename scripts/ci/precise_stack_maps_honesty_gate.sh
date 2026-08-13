#!/usr/bin/env bash
# Guards against precise_stack_maps regressing back to `true` under
# self-hosted/native without the recorder actually deriving roots from
# regalloc liveness.
#
# Measured 2026-07-27: the current stack-map recorder
# (self-hosted/native/codegen_x86_linux.sio, codegen.sio) sets
# stack_map_root_temp_counts[idx] = temp_count for every function — i.e. it
# marks every live temp as a GC root rather than computing the live root set
# from the register allocator's liveness. Publishing precise_stack_maps: true
# under that behaviour is a false signal: a panel or a downstream consumer
# reading the field learns nothing true about precision.
#
# `true` is legitimate again once the recorder stops using temp_count as a
# stand-in for the live root set — at that point flip this gate's pattern (or
# retire it) alongside the code change, in the same commit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# archive/** and the frozen bootstrap seed are historical/generated artifacts,
# not sites this gate polices — flipping them would perturb the bootstrap
# fixed point for no honesty gain.
PATTERN='precise_stack_maps:[[:space:]]*true'
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if grep -RIn --include='*.sio' -E "$PATTERN" self-hosted/native >"$TMP"; then
  echo "precise_stack_maps: true found under self-hosted/native:"
  cat "$TMP"
  echo
  echo "This field is currently only honest as false: the stack-map recorder" >&2
  echo "marks every live temp as a GC root (root_temp_counts == temp_count)" >&2
  echo "instead of deriving roots from regalloc liveness. If this change makes" >&2
  echo "the recorder genuinely precise, update this gate's PATTERN in the same" >&2
  echo "commit as the code change -- do not just flip the literal back." >&2
  exit 1
fi

echo "precise_stack_maps honesty check passed: no self-hosted/native site claims true."
