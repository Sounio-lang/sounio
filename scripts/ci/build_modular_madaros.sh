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
#   SOUC_BIN / SOUNIO_SOUC_BIN — override the bootstrap ELF used only when a
#                                source-tracking seed must be derived.
#   SOUNIO_MADAROS_SEED        — pin the exact executable used to compile main.sio.
#   SOUNIO_MADAROS_REQUIRE_PINNED_SEED=1
#                              — fail instead of deriving a replacement if the
#                                explicitly pinned seed cannot build main.sio.
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

BUILD_LOG="$(mktemp "${TMPDIR:-/tmp}/madaros-build.XXXXXX.log")"
# Derive a source-tracking seed. If SOUC_BIN/SOUNIO_SOUC_BIN was explicitly set we
# trust it as an already-fresh seed (e.g. a gen3.elf) and use it directly. Otherwise
# the bootstrap ELF is a committed binary that may lag the source, so we first
# compile the CURRENT lean_single.sio with it to obtain a fresh seed that carries the
# current source's features (arena/vmem #719 etc.), then compile main.sio with that.
TMP_SEED_DIR=""
cleanup() {
    rm -f "$BUILD_LOG"
    if [[ -n "$TMP_SEED_DIR" && -d "$TMP_SEED_DIR" ]]; then
        rm -rf "$TMP_SEED_DIR"
    fi
}
trap cleanup EXIT

# SEED SELECTION.
#
# SOUC_BIN / SOUNIO_SOUC_BIN used to short-circuit derivation and pin SEED to the
# committed prebuilt. Two things were wrong with that:
#
#   1. Those variables select the USER-FACING compiler (see bin/souc), not the
#      bootstrap seed, and the workspace context hook exports SOUC_BIN — so an
#      ordinary agent session silently opted out of derivation without asking.
#   2. The committed prebuilt lags main.sio, which the message below already
#      admits (#725). As of 2026-07-30 it SEGFAULTS compiling main.sio:
#      deterministic, exit 139, no ELF. Derivation from lean_single.sio works.
#
# So they no longer choose the seed. To pin one deliberately, set
# SOUNIO_MADAROS_SEED to its path — an explicit variable that means only this.
# A pinned seed that cannot build main.sio falls back to derivation with a
# warning rather than failing the build: a stale pin should cost time, not
# correctness. (#1559)
if [[ -n "${SOUNIO_MADAROS_SEED:-}" ]]; then
    SEED="$SOUNIO_MADAROS_SEED"
    if [[ ! -x "$SEED" ]]; then
        echo "error: SOUNIO_MADAROS_SEED is not an executable file: $SEED" >&2
        exit 1
    fi
    echo "→ using pinned seed (SOUNIO_MADAROS_SEED): $SEED"
    PINNED_SEED=1
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

run_seed_build() {
    local seed="$1"
    rm -f "$BUILD_LOG"
    set +e
    scripts/dev/souc-build-lock.sh "$seed" "$SRC" "$OUT" >"$BUILD_LOG" 2>&1
    local rc=$?
    set -e
    cat "$BUILD_LOG"
    if grep -Eq '^(error(\[[^]]+\])?:|Error:)' "$BUILD_LOG"; then
        echo "error: modular compiler emitted an error diagnostic despite exit $rc" >&2
        return 86
    fi
    return "$rc"
}

# Serialize heavy build via the global workspace lock.
#
# `|| true` so a seed that crashes is diagnosed here rather than aborting under
# set -e. The stale committed prebuilt segfaults on current main.sio, and the
# useful response is to say which seed died and try a good one — not to hand the
# caller a bare exit 139.
set +e
run_seed_build "$SEED"; BUILD_RC=$?
set -e

if [[ "$BUILD_RC" -ne 0 || ! -s "$OUT" ]]; then
    echo "warning: seed failed to build the compiler (exit $BUILD_RC, output $([[ -s "$OUT" ]] && echo present || echo absent))" >&2
    echo "warning:   seed was: $SEED" >&2
    if [[ "${PINNED_SEED:-0}" == "1" &&
          "${SOUNIO_MADAROS_REQUIRE_PINNED_SEED:-0}" == "1" ]]; then
        echo "error: pinned seed is required; refusing automatic seed derivation" >&2
        exit 1
    fi
    if [[ "${PINNED_SEED:-0}" == "1" && -f "$LEAN_SRC" ]]; then
        # A pinned seed that cannot compile main.sio is stale, not fatal: derive a
        # fresh one from lean_single.sio and retry once. This is the exact recovery
        # a caller would do by hand after reading the message above.
        echo "warning: pinned seed is unusable; deriving from lean_single.sio and retrying" >&2
        TMP_SEED_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-seed.XXXXXX")"
        SEED="$TMP_SEED_DIR/gen_seed.elf"
        scripts/dev/souc-build-lock.sh "$BOOTSTRAP_ELF" "$LEAN_SRC" "$SEED"
        if [[ ! -s "$SEED" ]]; then
            echo "error: fallback seed derivation produced no output: $SEED" >&2
            exit 1
        fi
        chmod +x "$SEED"
        rm -f "$OUT"
        set +e
        run_seed_build "$SEED"; BUILD_RC=$?
        set -e
    fi
fi

if [[ "$BUILD_RC" -ne 0 || ! -s "$OUT" ]]; then
    echo "error: modular compiler build produced no output: $OUT (seed exit $BUILD_RC)" >&2
    exit 1
fi

chmod +x "$OUT"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
