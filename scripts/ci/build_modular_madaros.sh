#!/usr/bin/env bash
# Build the modular Stage1 Sounio compiler (Madaros) from self-hosted/compiler/main.sio.
#
# Usage:
#   bash scripts/ci/build_modular_madaros.sh [OUTPUT_PATH]
#
# Default OUTPUT_PATH: bin/madaros
#
# The build has two explicit bootstrap modes:
#
#   madaros-seed: compile current main.sio with a declared, raw Madaros ELF.
#                 This is the operational path: M_n -> M_(n+1).
#   lean-audit:   derive a fresh lean_single seed from the legacy bootstrap ELF,
#                 then compile main.sio. This is a root-of-trust audit path.
#
# The C -> lean_single chain is currently bit-rotted (#725), so using it as the
# default development path makes current Madaros progress depend on a historical
# compiler bug. madaros-seed does not claim to repair that root chain; callers
# must record the seed digest and run lean-audit independently when auditing it.
# Because these compiles are CPU-heavy, the script serializes through
# scripts/dev/souc-build-lock.sh so multiple agents do not stampede the workspace pod.
#
# Environment:
#   SOUNIO_MADAROS_BOOTSTRAP_MODE — `madaros-seed`, `lean-audit`, or
#                                   `external-seed`. Defaults to `madaros-seed`.
#   SOUNIO_MADAROS_SEED           — raw Madaros ELF required by `madaros-seed`.
#   SOUC_BIN / SOUNIO_SOUC_BIN    — raw ELF required by `external-seed`; a legacy
#                                   explicit override remains accepted as an
#                                   explicit external-seed selection.
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

require_raw_elf() {
    local cand="$1"
    [[ -n "$cand" && -x "$cand" && -s "$cand" ]] || {
        echo "error: build_modular_madaros: seed is missing, empty, or not executable: $cand" >&2
        return 1
    }
    [[ "$(head -c4 "$cand" 2>/dev/null)" == $'\x7fELF' ]] || {
        echo "error: build_modular_madaros: seed is not a raw ELF: $cand" >&2
        return 1
    }
}

require_madaros_seed() {
    local cand="$1"
    local banner
    require_raw_elf "$cand"
    banner="$("$cand" --version 2>&1 || true)"
    [[ "$banner" == *"Madaros"* ]] || {
        echo "error: build_modular_madaros: operational seed does not identify as Madaros: $cand" >&2
        return 1
    }
}

LEAN_SRC="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"
BOOTSTRAP_MODE="${SOUNIO_MADAROS_BOOTSTRAP_MODE:-}"
MADAROS_SEED="${SOUNIO_MADAROS_SEED:-$ROOT_DIR/bin/madaros-linux-x86_64}"

# A caller that expressly supplies the legacy override is selecting an external
# seed. All other callers advance from the versioned operational Madaros seed.
if [[ -z "$BOOTSTRAP_MODE" ]]; then
    if [[ -n "${SOUC_BIN:-}" || -n "${SOUNIO_SOUC_BIN:-}" ]]; then
        BOOTSTRAP_MODE="external-seed"
    else
        BOOTSTRAP_MODE="madaros-seed"
    fi
fi

case "$BOOTSTRAP_MODE" in
    madaros-seed|lean-audit|external-seed) ;;
    *)
        echo "error: build_modular_madaros: unsupported SOUNIO_MADAROS_BOOTSTRAP_MODE=$BOOTSTRAP_MODE" >&2
        echo "  expected: madaros-seed, lean-audit, or external-seed" >&2
        exit 2
        ;;
esac

if [[ "$BOOTSTRAP_MODE" == "madaros-seed" ]]; then
    [[ -z "${SOUC_BIN:-}" && -z "${SOUNIO_SOUC_BIN:-}" ]] || {
        echo "error: build_modular_madaros: madaros-seed cannot be combined with SOUC_BIN/SOUNIO_SOUC_BIN" >&2
        exit 2
    }
    require_madaros_seed "$MADAROS_SEED"
fi

if [[ "$BOOTSTRAP_MODE" == "external-seed" && -z "${SOUC_BIN:-}" && -z "${SOUNIO_SOUC_BIN:-}" ]]; then
    echo "error: build_modular_madaros: external-seed requires SOUC_BIN or SOUNIO_SOUC_BIN" >&2
    exit 2
fi

if [[ ! -f "$SRC" ]]; then
    echo "error: modular compiler source not found: $SRC" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# Select a declared operational Madaros seed, an explicit external source-tracking
# seed, or a lean audit derivation. Every non-default path is named by the caller.
TMP_SEED_DIR=""
cleanup() {
    if [[ -n "$TMP_SEED_DIR" && -d "$TMP_SEED_DIR" ]]; then
        rm -rf "$TMP_SEED_DIR"
    fi
}
trap cleanup EXIT

if [[ "$BOOTSTRAP_MODE" == "madaros-seed" ]]; then
    SEED="$MADAROS_SEED"
    echo "→ bootstrap mode: madaros-seed (operational M_n -> M_(n+1))"
    echo "  Madaros seed: $SEED"
elif [[ "$BOOTSTRAP_MODE" == "external-seed" ]]; then
    BOOTSTRAP_ELF="$(resolve_bootstrap_elf)"
    SEED="$BOOTSTRAP_ELF"
    echo "→ bootstrap mode: explicit legacy seed (SOUC_BIN/SOUNIO_SOUC_BIN): $SEED"
else
    BOOTSTRAP_ELF="$(resolve_bootstrap_elf)"
    if [[ ! -f "$LEAN_SRC" ]]; then
        echo "error: lean_single source not found for seed derivation: $LEAN_SRC" >&2
        exit 1
    fi
    TMP_SEED_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-seed.XXXXXX")"
    SEED="$TMP_SEED_DIR/gen_seed.elf"
    echo "→ bootstrap mode: lean-audit (root-of-trust audit; see #725)"
    echo "  deriving source-tracking seed from lean_single.sio"
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

# A raw Madaros ELF does not accept the legacy positional source/output form:
# its second positional argument is parsed as another input file. Keep the
# native-v2 compile verb explicit on the operational M_n -> M_(n+1) path while
# preserving the legacy invocation contract for the root-audit seed paths.
if [[ "$BOOTSTRAP_MODE" == "madaros-seed" ]]; then
    scripts/dev/souc-build-lock.sh "$SEED" --native-v2-compile "$SRC" "$OUT"
else
    scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$OUT"
fi

if [[ ! -s "$OUT" ]]; then
    echo "error: modular compiler build produced no output: $OUT" >&2
    exit 1
fi

chmod +x "$OUT"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"
