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
EXPANDED_SEED_DIR=""

cleanup_expanded_seed() {
    if [[ -n "$EXPANDED_SEED_DIR" ]]; then
        rm -rf "$EXPANDED_SEED_DIR"
    fi
}
trap cleanup_expanded_seed EXIT

if [[ ! -f "$SRC" ]]; then
    echo "error: modular compiler source not found: $SRC" >&2
    exit 1
fi

# The checked lean_single seed is still the bootstrap root, but the recovered
# modular compiler now exceeds its 16 MiB concatenated SRC import buffer. Build a
# temporary source-equivalent lean_single with a 32 MiB SRC buffer, then use that
# as the actual Madaros seed. This keeps the bootstrap source-derived while
# avoiding a checked-in binary refresh just to change a fixed buffer size.
if [[ "${SOUNIO_MADAROS_BUILD_EXPANDED_LEAN_SEED:-1}" != "0" && "$(basename "$SEED")" == "souc-lean-single-x86_64" ]]; then
    EXPANDED_SEED_DIR="$(mktemp -d /tmp/sounio-expanded-lean-seed.XXXXXX)"
    EXPANDED_SEED_SRC="$EXPANDED_SEED_DIR/lean_single_32m.sio"
    EXPANDED_SEED_BIN="$EXPANDED_SEED_DIR/souc-lean-single-32m"
    cp "$ROOT_DIR/self-hosted/compiler/lean_single.sio" "$EXPANDED_SEED_SRC"
    python3 - "$EXPANDED_SEED_SRC" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = [
    (
        "var SRC: [i8; 16777216] = [0; 16777216]",
        "var SRC: [i8; 33554432] = [0; 33554432]",
    ),
    ("let SRC_CAP: i64 = 16777216", "let SRC_CAP: i64 = 33554432"),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f"expanded lean_single seed pattern not found: {old}")
    text = text.replace(old, new, 1)
path.write_text(text)
PY
    echo "Preparing expanded lean_single seed:"
    echo "  root:  $SEED"
    echo "  src:   $EXPANDED_SEED_SRC"
    echo "  out:   $EXPANDED_SEED_BIN"
    scripts/dev/souc-build-lock.sh "$SEED" "$EXPANDED_SEED_SRC" "$EXPANDED_SEED_BIN"
    chmod +x "$EXPANDED_SEED_BIN"
    SEED="$EXPANDED_SEED_BIN"
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

echo "Building Madaros (modular compiler):"
echo "  seed:  $SEED"
echo "  src:   $SRC"
echo "  out:   $OUT"

# Serialize heavy build via the global workspace lock.
scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$OUT"

if [[ ! -s "$OUT" ]]; then
    echo "error: modular compiler build produced no output: $OUT" >&2
    exit 1
fi

chmod +x "$OUT"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
