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
#   SOUNIO_MADAROS_BUILD_RECEIPT — optional TSV receipt binding the output ELF
#                                  to the checked-out source and build inputs

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-$ROOT_DIR/artifacts/self-hosted/madaros}"
if [[ "$OUT" == -* ]]; then
    echo "error: output path must not start with '-': $OUT" >&2
    exit 2
fi

portable_sha256() {
    local output digest
    if command -v sha256sum >/dev/null 2>&1; then
        output="$(LC_ALL=C sha256sum "$1" 2>/dev/null)" || return 1
    elif command -v shasum >/dev/null 2>&1; then
        output="$(LC_ALL=C shasum -a 256 "$1" 2>/dev/null)" || return 1
    else
        return 1
    fi
    digest="${output%%[[:space:]]*}"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$digest"
}

is_elf_binary() {
    [[ "$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n')" == "7f454c46" ]]
}

run_compiler_build_checked() {
    local label="$1"
    local compiler="$2"
    local source="$3"
    local output="$4"
    local transcript
    transcript="$(mktemp "${TMPDIR:-/tmp}/madaros-build-transcript.XXXXXX")"
    if ! scripts/dev/souc-build-lock.sh "$compiler" "$source" "$output" 2>&1 | tee "$transcript"; then
        echo "error: $label compiler invocation failed" >&2
        rm -f "$transcript"
        return 1
    fi
    if LC_ALL=C grep -Eq '^(error(:|\[E[0-9]+\])|parse error:|fatal:|run_check_mode:)' "$transcript"; then
        echo "error: $label emitted a primary compiler diagnostic despite returning success" >&2
        rm -f "$transcript"
        return 1
    fi
    rm -f "$transcript"
}

# Resolve the bootstrap ELF used to derive the source-tracking seed. It MUST be an
# ELF — never the bin/souc wrapper (a #!-script that now routes to Madaros); the
# `#!` guard skips any wrapper. bin/souc-linux-x86_64 is preferred over the older
# bin/souc-lean-single-x86_64 seed because it tracks the current lean_single.sio
# source (it can compile it), whereas the committed lean_single seed lags (#725).
# An explicitly-provided SOUC_BIN/SOUNIO_SOUC_BIN wins (e.g. a fresh gen3.elf from
# `make build`).
resolve_bootstrap_elf() {
    local cand explicit
    if [[ -n "${SOUC_BIN:-}" && -n "${SOUNIO_SOUC_BIN:-}" && "$SOUC_BIN" != "$SOUNIO_SOUC_BIN" ]]; then
        echo "error: build_modular_madaros: conflicting SOUC_BIN and SOUNIO_SOUC_BIN overrides" >&2
        return 1
    fi
    explicit="${SOUC_BIN:-${SOUNIO_SOUC_BIN:-}}"
    if [[ -n "$explicit" ]]; then
        if [[ ! -x "$explicit" ]]; then
            echo "error: build_modular_madaros: explicit seed is missing or not executable: $explicit" >&2
            return 1
        fi
        if ! is_elf_binary "$explicit"; then
            echo "error: build_modular_madaros: explicit seed is not an ELF binary: $explicit" >&2
            return 1
        fi
        echo "$explicit"
        return 0
    fi
    for cand in "$ROOT_DIR/bin/souc-linux-x86_64" "$ROOT_DIR/bin/souc-lean-single-x86_64" \
                "$ROOT_DIR/bin/souc"; do
        if [[ -n "$cand" && -x "$cand" ]] && is_elf_binary "$cand"; then
            echo "$cand"
            return 0
        fi
    done
    echo "error: build_modular_madaros: no bootstrap ELF found" >&2
    echo "  tried (binary only): $ROOT_DIR/bin/souc-linux-x86_64, $ROOT_DIR/bin/souc-lean-single-x86_64, $ROOT_DIR/bin/souc" >&2
    return 1
}

BOOTSTRAP_ELF="$(resolve_bootstrap_elf)"
LEAN_SRC="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"

if [[ ! -f "$SRC" ]]; then
    echo "error: modular compiler source not found: $SRC" >&2
    exit 1
fi

SOURCE_GIT_SHA_BEFORE="$(git rev-parse HEAD)"
SOURCE_TREE_SHA_BEFORE="$(git rev-parse 'HEAD^{tree}')"
if [[ -z "$(git status --porcelain --untracked-files=all)" ]]; then
    WORKTREE_CLEAN_BEFORE=1
else
    WORKTREE_CLEAN_BEFORE=0
fi
BUILD_SCRIPT_SHA256_BEFORE="$(portable_sha256 "$ROOT_DIR/scripts/ci/build_modular_madaros.sh")"
BOOTSTRAP_SHA256_BEFORE="$(portable_sha256 "$BOOTSTRAP_ELF")"
LEAN_SOURCE_SHA256_BEFORE="$(portable_sha256 "$LEAN_SRC")"
MODULAR_SOURCE_SHA256_BEFORE="$(portable_sha256 "$SRC")"

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

if [[ -n "${SOUC_BIN:-}" || -n "${SOUNIO_SOUC_BIN:-}" ]]; then
    SEED="$BOOTSTRAP_ELF"
    BUILD_STRATEGY="provided-seed"
    echo "→ using provided seed directly (SOUC_BIN/SOUNIO_SOUC_BIN): $SEED"
else
    BUILD_STRATEGY="derived-current-lean-single"
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
    run_compiler_build_checked "source-tracking seed derivation" "$BOOTSTRAP_ELF" "$LEAN_SRC" "$SEED"
    if [[ ! -s "$SEED" ]]; then
        echo "error: seed derivation produced no output: $SEED" >&2
        exit 1
    fi
    chmod +x "$SEED"
    if ! is_elf_binary "$SEED"; then
        echo "error: derived source-tracking seed is not an ELF binary: $SEED" >&2
        exit 1
    fi
fi

echo "Building Madaros (modular compiler):"
echo "  seed:  $SEED"
echo "  src:   $SRC"
echo "  out:   $OUT"

# Serialize the heavy build and reject diagnostic-bearing zero exits.
run_compiler_build_checked "modular Madaros build" "$SEED" "$SRC" "$OUT"

if [[ ! -s "$OUT" ]]; then
    echo "error: modular compiler build produced no output: $OUT" >&2
    exit 1
fi

chmod +x "$OUT"
if ! is_elf_binary "$OUT"; then
    echo "error: modular compiler output is not an ELF binary: $OUT" >&2
    exit 1
fi
echo "Madaros ready: $OUT ($(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") bytes)"

if [[ -n "${SOUNIO_MADAROS_BUILD_RECEIPT:-}" ]]; then
    RECEIPT="$SOUNIO_MADAROS_BUILD_RECEIPT"
    RECEIPT_DIR="$(dirname "$RECEIPT")"
    mkdir -p "$RECEIPT_DIR"
    RECEIPT_TMP="${RECEIPT}.tmp.$$"
    SOURCE_GIT_SHA_AFTER="$(git rev-parse HEAD)"
    SOURCE_TREE_SHA_AFTER="$(git rev-parse 'HEAD^{tree}')"
    if [[ -z "$(git status --porcelain --untracked-files=all)" ]]; then
        WORKTREE_CLEAN_AFTER=1
    else
        WORKTREE_CLEAN_AFTER=0
    fi
    BUILD_SCRIPT_SHA256_AFTER="$(portable_sha256 "$ROOT_DIR/scripts/ci/build_modular_madaros.sh")"
    BOOTSTRAP_SHA256_AFTER="$(portable_sha256 "$BOOTSTRAP_ELF")"
    LEAN_SOURCE_SHA256_AFTER="$(portable_sha256 "$LEAN_SRC")"
    MODULAR_SOURCE_SHA256_AFTER="$(portable_sha256 "$SRC")"
    SOURCE_STABLE=0
    if [[ "$SOURCE_GIT_SHA_BEFORE" == "$SOURCE_GIT_SHA_AFTER" \
          && "$SOURCE_TREE_SHA_BEFORE" == "$SOURCE_TREE_SHA_AFTER" \
          && "$BUILD_SCRIPT_SHA256_BEFORE" == "$BUILD_SCRIPT_SHA256_AFTER" \
          && "$BOOTSTRAP_SHA256_BEFORE" == "$BOOTSTRAP_SHA256_AFTER" \
          && "$LEAN_SOURCE_SHA256_BEFORE" == "$LEAN_SOURCE_SHA256_AFTER" \
          && "$MODULAR_SOURCE_SHA256_BEFORE" == "$MODULAR_SOURCE_SHA256_AFTER" ]]; then
        SOURCE_STABLE=1
    fi
    {
        printf 'schema\tsounio.madaros.build-receipt.v2\n'
        printf 'source_git_sha_before\t%s\n' "$SOURCE_GIT_SHA_BEFORE"
        printf 'source_git_sha_after\t%s\n' "$SOURCE_GIT_SHA_AFTER"
        printf 'source_tree_sha_before\t%s\n' "$SOURCE_TREE_SHA_BEFORE"
        printf 'source_tree_sha_after\t%s\n' "$SOURCE_TREE_SHA_AFTER"
        printf 'worktree_clean_before\t%s\n' "$WORKTREE_CLEAN_BEFORE"
        printf 'worktree_clean_after\t%s\n' "$WORKTREE_CLEAN_AFTER"
        printf 'source_stable\t%s\n' "$SOURCE_STABLE"
        printf 'build_strategy\t%s\n' "$BUILD_STRATEGY"
        printf 'build_script_sha256_before\t%s\n' "$BUILD_SCRIPT_SHA256_BEFORE"
        printf 'build_script_sha256_after\t%s\n' "$BUILD_SCRIPT_SHA256_AFTER"
        printf 'bootstrap_path\t%s\n' "$(realpath "$BOOTSTRAP_ELF")"
        printf 'bootstrap_sha256_before\t%s\n' "$BOOTSTRAP_SHA256_BEFORE"
        printf 'bootstrap_sha256_after\t%s\n' "$BOOTSTRAP_SHA256_AFTER"
        printf 'lean_source_sha256_before\t%s\n' "$LEAN_SOURCE_SHA256_BEFORE"
        printf 'lean_source_sha256_after\t%s\n' "$LEAN_SOURCE_SHA256_AFTER"
        printf 'modular_source_sha256_before\t%s\n' "$MODULAR_SOURCE_SHA256_BEFORE"
        printf 'modular_source_sha256_after\t%s\n' "$MODULAR_SOURCE_SHA256_AFTER"
        printf 'seed_sha256\t%s\n' "$(portable_sha256 "$SEED")"
        printf 'output_path\t%s\n' "$(realpath "$OUT")"
        printf 'output_sha256\t%s\n' "$(portable_sha256 "$OUT")"
    } >"$RECEIPT_TMP"
    mv "$RECEIPT_TMP" "$RECEIPT"
    echo "Madaros build receipt: $RECEIPT"
fi
