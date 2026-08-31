#!/usr/bin/env bash
# Rebuild bin/souc-lean-single-x86_64 from source and verify the committed blob.
#
# A 2.5 MB ELF cannot be reviewed by reading it. This reproduces it, and checks
# the two properties that make it trustworthy:
#
#   FIXED POINT   the artifact compiles lean_single.sio into ITSELF, bit for bit.
#                 Without this, the committed binary carries the code generation
#                 of whatever bootstrap ELF happened to build it rather than the
#                 code generation the source describes. #1606 nearly shipped a
#                 generation-1 artifact for exactly that reason.
#
#   DDC           diverse double-compilation. Starting from a DIFFERENT seed and
#                 iterating to ITS fixed point must land on the same bytes. A
#                 fixed point alone proves self-consistency, not correctness: a
#                 miscompile injected by the bootstrap and reproduced by the
#                 compiler it builds survives unnoticed. Two independent paths
#                 converging is what rules that out (trusting-trust).
#
# Both were run by hand for #1606; this makes them repeatable. The old-seed path
# there began at a binary carrying five known miscompiles (6f9c20b85) and still
# converged on the committed artifact, so those defects do not self-perpetuate.
#
# Usage:
#   scripts/ci/verify_lean_seed.sh                  # fixed point only (fast)
#   SOUNIO_SEED_DDC=1 scripts/ci/verify_lean_seed.sh   # + diverse double-compilation
#
# Environment:
#   SOUNIO_SEED_BOOTSTRAP  ELF used to derive generation 1 (default bin/souc-linux-x86_64)
#   SOUNIO_SEED_DDC        set to 1 to also run the DDC leg (slower: 3 more compiles)
#   SOUNIO_SEED_DDC_FROM   ELF to start the DDC leg from (default: the committed
#                          seed at HEAD~ if it differs, else skipped with a notice)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

SRC="self-hosted/compiler/lean_single.sio"
SEED="bin/souc-lean-single-x86_64"
BOOTSTRAP="${SOUNIO_SEED_BOOTSTRAP:-bin/souc-linux-x86_64}"

fail() { echo "[seed-verify] FAIL: $*" >&2; exit 1; }
note() { echo "[seed-verify] $*"; }

[ -f "$SRC" ]  || fail "missing $SRC"
[ -x "$SEED" ] || fail "missing or non-executable $SEED"
[ -x "$BOOTSTRAP" ] || fail "missing or non-executable $BOOTSTRAP (set SOUNIO_SEED_BOOTSTRAP)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/seed-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

sha() { sha256sum "$1" | cut -d' ' -f1; }
short() { sha "$1" | cut -c1-16; }

note "source:    $SRC"
note "committed: $SEED  $(short "$SEED")"
note "bootstrap: $BOOTSTRAP  $(short "$BOOTSTRAP")"

# --- fixed point ------------------------------------------------------------
# The committed seed compiling the source must reproduce the committed seed.
"$SEED" "$SRC" "$WORK/self.elf" >/dev/null 2>&1
[ -s "$WORK/self.elf" ] || fail "committed seed could not compile $SRC"
chmod +x "$WORK/self.elf"

if [ "$(sha "$WORK/self.elf")" != "$(sha "$SEED")" ]; then
    echo "[seed-verify] committed $(short "$SEED")" >&2
    echo "[seed-verify] rebuilt   $(short "$WORK/self.elf")" >&2
    fail "NOT a fixed point -- the committed seed is not what the source produces.
  Derive it properly:
    $BOOTSTRAP $SRC /tmp/gen1.elf && chmod +x /tmp/gen1.elf
    /tmp/gen1.elf $SRC /tmp/gen2.elf     # gen2 is the fixed point
    cp /tmp/gen2.elf $SEED"
fi
note "FIXED POINT ok: the committed seed reproduces itself bit for bit"

# Determinism: the same input twice must give the same bytes, or "bit for bit"
# above means nothing.
"$SEED" "$SRC" "$WORK/self2.elf" >/dev/null 2>&1
[ -s "$WORK/self2.elf" ] || fail "second self-compile produced no output"
[ "$(sha "$WORK/self2.elf")" = "$(sha "$WORK/self.elf")" ] \
    || fail "NON-DETERMINISTIC: two identical compiles differ"
note "DETERMINISTIC ok: two identical compiles agree"

# --- diverse double-compilation --------------------------------------------
if [ "${SOUNIO_SEED_DDC:-0}" != "1" ]; then
    note "PASS (set SOUNIO_SEED_DDC=1 to also run diverse double-compilation)"
    exit 0
fi

DDC_FROM="${SOUNIO_SEED_DDC_FROM:-}"
if [ -z "$DDC_FROM" ]; then
    if git rev-parse "HEAD~1:$SEED" >/dev/null 2>&1; then
        git show "HEAD~1:$SEED" > "$WORK/prev_seed.elf" 2>/dev/null
        chmod +x "$WORK/prev_seed.elf" 2>/dev/null
        if [ -s "$WORK/prev_seed.elf" ] \
           && [ "$(sha "$WORK/prev_seed.elf")" != "$(sha "$SEED")" ]; then
            DDC_FROM="$WORK/prev_seed.elf"
        fi
    fi
fi
if [ -z "$DDC_FROM" ]; then
    note "DDC skipped: no distinct second starting point (set SOUNIO_SEED_DDC_FROM)"
    exit 0
fi

note "DDC start: $(short "$DDC_FROM")"
"$DDC_FROM" "$SRC" "$WORK/a1.elf" >/dev/null 2>&1
[ -s "$WORK/a1.elf" ] || fail "DDC: starting seed could not compile $SRC"
chmod +x "$WORK/a1.elf"
"$WORK/a1.elf" "$SRC" "$WORK/a2.elf" >/dev/null 2>&1
[ -s "$WORK/a2.elf" ] || fail "DDC: generation 1 could not compile $SRC"
chmod +x "$WORK/a2.elf"

note "DDC path:  $(short "$DDC_FROM") -> $(short "$WORK/a1.elf") -> $(short "$WORK/a2.elf")"
if [ "$(sha "$WORK/a2.elf")" != "$(sha "$SEED")" ]; then
    fail "DDC MISMATCH: an independent path reaches a DIFFERENT fixed point.
  That is the trusting-trust signature -- a defect that survives self-compilation
  would look exactly like this. Do not ship the seed until it is explained."
fi
note "DDC ok: an independent starting point converges on the committed artifact"
note "PASS"
