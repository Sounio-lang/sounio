#!/usr/bin/env bash
# lean_single_seed_freshness_gate.sh — guard the committed lean_single seed
# against provenance rot (#725).
#
# bin/souc-lean-single-x86_64 is the ELF behind SOUNIO_SOUC_ENGINE=lean_single,
# but no build target regenerates it: `make build` seeds from
# bin/souc-linux-x86_64 and leaves gen3.elf at the repo root. The committed
# seed therefore rots silently whenever lean_single.sio changes without a
# manual copy — exactly what happened between 2026-07-25 and Track B
# (2026-08-15), when the documented engine alias silently exited 1.
#
# Two assertions, both cheap:
#   1. identity  — the seed still answers with lean_single's own CLI banner
#      (mini_native; lean_single.sio:1), so it has not been swapped for a
#      different tool
#   2. freshness — the seed still carries the newest verified extern "C"
#      capability, by running scripts/ci/ffi_extern_c_gate.sh against it
#      (compile + execute + side-effect file). A stub added to
#      lean_single.sio without refreshing the seed turns this red.
#
# SOUNIO_LEAN_SEED_DEEP=1 adds a third, expensive assertion — the seed can
# compile current lean_single.sio — wrapped in the global build lock.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SEED="$ROOT_DIR/bin/souc-lean-single-x86_64"

fail() { echo "[lean-seed] FAIL: $*" >&2; exit 1; }

[[ -x "$SEED" ]] || fail "no executable seed at $SEED"

# 1. Identity: lean_single's own usage banner.
BANNER="$(set +e; "$SEED" 2>&1; set -e)"
echo "$BANNER" | grep -q 'Usage: mini_native <source.sio> <output>' \
  || fail "seed does not answer with the mini_native (lean_single) banner — wrong tool committed?
$(echo "$BANNER" | head -3)"

# 2. Freshness: the FFI capability gate, defaulting to the seed.
bash scripts/ci/ffi_extern_c_gate.sh

# 3. Deep mode (opt-in): the seed can still compile current lean_single.sio.
if [[ -n "${SOUNIO_LEAN_SEED_DEEP:-}" ]]; then
  WORK="$(mktemp -d /tmp/sounio-lean-seed-deep.XXXXXX)"
  trap 'rm -rf "$WORK"' EXIT
  scripts/dev/souc-build-lock.sh "$SEED" self-hosted/compiler/lean_single.sio "$WORK/lean_rebuild.elf" \
    || fail "seed can no longer compile current lean_single.sio — bootstrap chain broken, regenerate via make build"
  echo "[lean-seed] deep: seed compiles current lean_single.sio ($(md5sum "$WORK/lean_rebuild.elf" | cut -d' ' -f1))"
fi

echo "[lean-seed] PASS: committed seed is lean_single (banner) and fresh (extern \"C\" system() gate)"
