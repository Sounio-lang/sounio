#!/usr/bin/env bash
# Build the modular Stage1 Sounio compiler (Madaros) from self-hosted/compiler/main.sio.
#
# Usage:
#   bash scripts/ci/build_modular_madaros.sh [OUTPUT_PATH]
#
# Default OUTPUT_PATH: bin/madaros
#
# The build uses the checked-in Stage0 compiler (bin/souc or bin/souc-linux-x86_64)
# as the seed. Because compiling main.sio is CPU-heavy, this script serializes
# through scripts/dev/souc-build-lock.sh so multiple agents do not stampede the
# workspace pod.
#
# Environment:
#   SOUC_BIN / SOUNIO_SOUC_BIN — override the seed compiler
#   SOUNIO_STDLIB_PATH         — forwarded to the seed compiler
#   SOUNIO_BUILD_LOCK          — override the global build lock path

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-$ROOT_DIR/artifacts/self-hosted/madaros}"
if [[ "$OUT" == -* ]]; then
    echo "error: output path must not start with '-': $OUT" >&2
    exit 2
fi

# Resolve seed compiler. The seed MUST be the lean_single bootstrap ELF — never
# the bin/souc wrapper (a #!-script that now routes to Madaros). Seeding from the
# wrapper would make Madaros build itself: an unverified self-host fixed point,
# out of scope here. The `#!` guard skips any wrapper script; the lean_single ELF
# is preferred explicitly.
resolve_seed() {
    local cand
    for cand in "${SOUC_BIN:-}" "${SOUNIO_SOUC_BIN:-}" \
                "$ROOT_DIR/bin/souc-lean-single-x86_64" "$ROOT_DIR/bin/souc-linux-x86_64" \
                "$ROOT_DIR/bin/souc"; do
        if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
            echo "$cand"
            return 0
        fi
    done
    echo "error: build_modular_madaros: no lean_single seed ELF found" >&2
    echo "  tried (ELF only): SOUC_BIN, SOUNIO_SOUC_BIN, $ROOT_DIR/bin/souc-lean-single-x86_64, $ROOT_DIR/bin/souc-linux-x86_64, $ROOT_DIR/bin/souc" >&2
    return 1
}

SEED="$(resolve_seed)"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"

if [[ ! -f "$SRC" ]]; then
    echo "error: modular compiler source not found: $SRC" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

echo "Building Madaros (modular compiler):"
echo "  seed:  $SEED"
echo "  src:   $SRC"
echo "  out:   $OUT"

# Serialize heavy build via the global workspace lock.
#
# REPRODUCIBLE-BUILD HYGIENE: pin the address-space layout with setarch -R
# (ADDR_NO_RANDOMIZE, no privilege) so builds are byte-reproducible by
# construction — defensively immune to any future runtime-address leak into
# emission. The current seed (2e595371) has NO such leak: 20 no-setarch
# builds are byte-identical, and setarch vs no-setarch produce the SAME
# binary (md5 8e0e4197). The one 104MB gate-failing binary once observed was
# seed-VERSION drift (a pre-c647203a6 seed still emitting the rel8
# inline-string jump eb 01 at code 0x1740 where the current seed emits
# e9 01000000 rel32) from concurrent-build contamination, NOT an ASLR effect
# — initially mis-diagnosed, disproven by byte-bisection. Kept as cheap
# defensive hygiene; falls back cleanly if setarch is absent.
if command -v setarch >/dev/null 2>&1 && setarch "$(uname -m)" -R true >/dev/null 2>&1; then
    setarch "$(uname -m)" -R scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$OUT"
else
    echo "  note: setarch -R unavailable; build not guaranteed byte-reproducible" >&2
    scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$OUT"
fi

if [[ ! -s "$OUT" ]]; then
    echo "error: modular compiler build produced no output: $OUT" >&2
    exit 1
fi

chmod +x "$OUT"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
