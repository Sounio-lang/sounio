#!/usr/bin/env bash
# Build the modular Stage1 Sounio compiler (Madaros) from self-hosted/compiler/main.sio.
#
# Usage:
#   bash scripts/ci/build_modular_madaros.sh [OUTPUT_PATH]
#
# Default OUTPUT_PATH: bin/madaros
#
# The build derives a *source-tracking* seed from the current lean_single.sio
# before compiling main.sio. The checked-in bin/souc-lean-single-x86_64 seed lags
# the source (it predates the #719 arena/vmem fixes: the 16MB resolve_imports SRC
# guard and __arena_*/__pool_* intrinsics) and fails main.sio at the import wall
# ("import too large for SRC buffer: patterns.sio"). Rather than freeze a new seed
# binary (bootstrap-chain provenance rot is tracked separately in #725), we compile
# the current lean_single.sio with a committed bootstrap ELF to obtain a fresh seed
# that carries the current source's features, then compile main.sio with THAT seed.
# Because these compiles are CPU-heavy, the script serializes through
# scripts/dev/souc-build-lock.sh so multiple agents do not stampede the workspace pod.
#
# Environment:
#   SOUC_BIN / SOUNIO_SOUC_BIN — override the bootstrap ELF. If it already points at
#                                a fresh source-tracking seed (e.g. a gen3.elf from
#                                `make build`), it is used directly for main.sio with
#                                no extra lean_single derivation.
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

# Resolve the bootstrap ELF used to derive the source-tracking seed. It MUST be an
# ELF — never the bin/souc wrapper (a #!-script that now routes to Madaros); the
# `#!` guard skips any wrapper. bin/souc-linux-x86_64 is preferred over the older
# bin/souc-lean-single-x86_64 seed because it tracks the current lean_single.sio
# source (it can compile it), whereas the committed lean_single seed lags (#725).
# An explicitly-provided SOUC_BIN/SOUNIO_SOUC_BIN wins (e.g. a fresh gen3.elf from
# `make build`).
resolve_bootstrap_elf() {
    local cand
    for cand in "${SOUC_BIN:-}" "${SOUNIO_SOUC_BIN:-}" \
                "$ROOT_DIR/bin/souc-linux-x86_64" "$ROOT_DIR/bin/souc-lean-single-x86_64" \
                "$ROOT_DIR/bin/souc"; do
        if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
            echo "$cand"
            return 0
        fi
    done
    echo "error: build_modular_madaros: no bootstrap ELF found" >&2
    echo "  tried (ELF only): SOUC_BIN, SOUNIO_SOUC_BIN, $ROOT_DIR/bin/souc-linux-x86_64, $ROOT_DIR/bin/souc-lean-single-x86_64, $ROOT_DIR/bin/souc" >&2
    return 1
}

BOOTSTRAP_ELF="$(resolve_bootstrap_elf)"
LEAN_SRC="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"

if [[ ! -f "$SRC" ]]; then
    echo "error: modular compiler source not found: $SRC" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# Derive a source-tracking seed. If SOUC_BIN/SOUNIO_SOUC_BIN was explicitly set we
# trust it as an already-fresh seed (e.g. a gen3.elf) and use it directly. Otherwise
# the bootstrap ELF is a committed binary that may lag the source, so we first
# compile the CURRENT lean_single.sio with it to obtain a fresh seed that carries the
# current source's features (arena/vmem #719 etc.), then compile main.sio with that.
TMP_SEED_DIR=""
cleanup() {
    if [[ -n "$TMP_SEED_DIR" && -d "$TMP_SEED_DIR" ]]; then
        rm -rf "$TMP_SEED_DIR"
    fi
}
trap cleanup EXIT

# The freshness decision must be made on the seed ACTUALLY resolved, not on
# whether the variable happened to be set. resolve_bootstrap_elf REJECTS `#!`
# wrappers, so SOUC_BIN=bin/souc -- the wrapper, and the default export in some
# agent shells -- used to fall through to a committed ELF that lags the source
# AND skip the derivation meant to compensate for exactly that lag. Result:
# "import too large for SRC buffer: codegen_plan.sio", then SIGSEGV, ~40 s in.
# Measured 2026-07-30 while rebuilding to re-confirm the FO GUM stack.
PROVIDED_SEED=0
for v in SOUC_BIN SOUNIO_SOUC_BIN; do
    val="${!v:-}"
    [[ -n "$val" ]] || continue
    if [[ "$val" == "$BOOTSTRAP_ELF" ]]; then
        PROVIDED_SEED=1
    else
        echo "warning: $v=$val is not usable as a bootstrap ELF (a '#!' wrapper," \
             "or missing/non-executable); ignoring it and deriving a seed instead" >&2
    fi
done

if [[ $PROVIDED_SEED -eq 1 ]]; then
    SEED="$BOOTSTRAP_ELF"
    echo "→ using provided seed directly (SOUC_BIN/SOUNIO_SOUC_BIN): $SEED"
else
    if [[ ! -f "$LEAN_SRC" ]]; then
        echo "error: lean_single source not found for seed derivation: $LEAN_SRC" >&2
        exit 1
    fi
    TMP_SEED_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-seed.XXXXXX")"
    SEED="$TMP_SEED_DIR/gen_seed.elf"
    echo "→ deriving source-tracking seed from lean_single.sio (committed seed lags; see #725)"
    echo "  bootstrap ELF: $BOOTSTRAP_ELF"
    echo "  lean src:      $LEAN_SRC"
    echo "  gen seed:      $SEED"
    # One generation is sufficient — it carries the current source's features.
    scripts/dev/souc-build-lock.sh "$BOOTSTRAP_ELF" "$LEAN_SRC" "$SEED"
    if [[ ! -s "$SEED" ]]; then
        echo "error: seed derivation produced no output: $SEED" >&2
        exit 1
    fi
    chmod +x "$SEED"
fi

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
