#!/usr/bin/env bash
# Rebuild artifacts/bootstrap/boot4.elf from source and verify the committed blob.
#
# boot4 is self-hosting: it compiles bootstrap/boot4.sio, and the result compiles
# it again. So the committed artifact has an obligation its 393 KB cannot state
# for itself -- that it is what its own source produces.
#
# It was NOT, until #1631. The committed ELF was 393354 bytes where the source
# produces 393956, and nothing in CI noticed: every consumer just ran whatever
# blob was checked in. A prebuilt that lags its source is a compiler nobody has
# read, deciding whether other people's code compiles.
#
# Two properties, same shape as scripts/ci/verify_lean_seed.sh:
#
#   FIXED POINT   the artifact compiles boot4.sio into ITSELF, bit for bit.
#   DETERMINISTIC the same input twice gives the same bytes, or "bit for bit"
#                 above means nothing.
#
# Convergence is at generation 2, not 1 -- gen1 differs from gen2, and gen2 ==
# gen3 == gen4. That is why the derivation below iterates rather than building
# once.
#
# Usage:
#   scripts/ci/verify_boot4_seed.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

SRC="bootstrap/boot4.sio"
SEED="artifacts/bootstrap/boot4.elf"

fail() { echo "[boot4-verify] FAIL: $*" >&2; exit 1; }
note() { echo "[boot4-verify] $*"; }

[ -f "$SRC" ]  || fail "missing $SRC"
[ -x "$SEED" ] || fail "missing or non-executable $SEED"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/boot4-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

sha() { sha256sum "$1" | cut -d' ' -f1; }
short() { sha "$1" | cut -c1-16; }

note "source:    $SRC"
note "committed: $SEED  $(short "$SEED")"

# --- fixed point ------------------------------------------------------------
"$SEED" "$SRC" "$WORK/self.elf" >/dev/null 2>&1
[ -s "$WORK/self.elf" ] || fail "committed boot4 could not compile $SRC"
chmod +x "$WORK/self.elf"

if [ "$(sha "$WORK/self.elf")" != "$(sha "$SEED")" ]; then
    echo "[boot4-verify] committed $(short "$SEED")" >&2
    echo "[boot4-verify] rebuilt   $(short "$WORK/self.elf")" >&2
    fail "NOT a fixed point -- the committed artifact is not what the source produces.
  Derive it properly (convergence is at generation 2):
    ./$SEED $SRC /tmp/g1.elf && chmod +x /tmp/g1.elf
    /tmp/g1.elf $SRC /tmp/g2.elf && chmod +x /tmp/g2.elf
    /tmp/g2.elf $SRC /tmp/g3.elf     # g3 must equal g2
    cp /tmp/g2.elf $SEED"
fi
note "FIXED POINT ok: the committed artifact reproduces itself bit for bit"

# --- determinism ------------------------------------------------------------
"$SEED" "$SRC" "$WORK/self2.elf" >/dev/null 2>&1
[ -s "$WORK/self2.elf" ] || fail "second self-compile produced no output"
[ "$(sha "$WORK/self2.elf")" = "$(sha "$WORK/self.elf")" ] \
    || fail "NON-DETERMINISTIC: two identical compiles differ"
note "DETERMINISTIC ok: two identical compiles agree"

note "PASS"
