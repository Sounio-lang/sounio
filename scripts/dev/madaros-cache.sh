#!/usr/bin/env bash
# scripts/dev/madaros-cache.sh — content-addressed cache for the two heavy
# Madaros build stages. Source this file; do not execute directly.
#
# WHY
# ---
# build_modular_madaros.sh runs two single-core compiles under one global
# exclusive lock, and redoes BOTH every time:
#   stage 1  bootstrap ELF  + lean_single.sio  -> gen_seed.elf     (minutes)
#   stage 2  gen_seed.elf   + main.sio (+tree) -> madaros          (~20 min)
# Stage 1's inputs almost never change. Stage 2's inputs change only when
# self-hosted/ or stdlib/ change. Measured 2026-09-25: an agent verifying a
# 5-file patch spent ~75 min of wall clock, most of it queued on the lock
# behind other agents rebuilding identical trees.
#
# WHAT
# ----
# Each stage output is stored under a key that is a sha256 of ALL its inputs
# (binary bytes + working-tree contents of the source files, so an uncommitted
# edit changes the key). A hit copies the artifact out in ~3 s and never touches
# the build lock. A miss builds as before and stores the result atomically.
#
# The cache is shared across worktrees and agents on the same pod. Default:
#   $SOUNIO_MADAROS_CACHE, else /workspace/.cache/madaros, else $ROOT/.cache/madaros
#
# Every artifact is stored with its own sha256 beside it and re-verified on
# read; a corrupt entry is deleted and treated as a miss. Nothing here can
# hand you a binary that is not byte-identical to what the build produced.
#
# Env:
#   SOUNIO_MADAROS_CACHE     cache directory
#   SOUNIO_MADAROS_NOCACHE=1 bypass reads (still writes), for A/B checks

if [[ -n "${_SOUNIO_MADAROS_CACHE_LOADED:-}" ]]; then return 0; fi
_SOUNIO_MADAROS_CACHE_LOADED=1

_mc_root() { cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd; }

madaros_cache_dir() {
    local d="${SOUNIO_MADAROS_CACHE:-}"
    if [[ -z "$d" ]]; then
        if [[ -d /workspace && -w /workspace ]]; then d=/workspace/.cache/madaros
        else d="$(_mc_root)/.cache/madaros"; fi
    fi
    mkdir -p "$d/seed" "$d/madaros" "$d/fixed-point"
    echo "$d"
}

_mc_sha() { sha256sum "$1" | cut -c1-64; }

# Key of the whole source tree the modular compiler is built from. Working-tree
# contents (not commit ids), tracked + untracked-but-not-ignored, so a dirty
# checkout keys differently from its HEAD. ~2.5 s on 2.5k files.
madaros_tree_key() {
    local root; root="$(_mc_root)"
    ( cd "$root" && git ls-files -z -co --exclude-standard self-hosted stdlib \
        | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}

# Stage 1 key: bootstrap ELF bytes + lean_single.sio bytes.
madaros_seed_key() {  # <bootstrap-elf> <lean-src>
    printf 'seed-v1\n%s\n%s\n' "$(_mc_sha "$1")" "$(_mc_sha "$2")" | sha256sum | cut -c1-64
}

# Stage 2 key: seed ELF bytes + whole tree. (main.sio is inside the tree.)
madaros_build_key() {  # <seed-elf>
    printf 'madaros-v1\n%s\n%s\n' "$(_mc_sha "$1")" "$(madaros_tree_key)" | sha256sum | cut -c1-64
}

# madaros_cache_get <stage> <key> <out>  -> 0 on verified hit (out written), 1 on miss
madaros_cache_get() {
    local stage="$1" key="$2" out="$3"
    [[ "${SOUNIO_MADAROS_NOCACHE:-0}" == "1" ]] && return 1
    madaros_cache_usable || return 1
    local dir; dir="$(madaros_cache_dir)/$stage/$key"
    [[ -s "$dir/artifact" && -s "$dir/artifact.sha256" ]] || return 1
    if [[ "$(_mc_sha "$dir/artifact")" != "$(cut -c1-64 "$dir/artifact.sha256")" ]]; then
        echo "[madaros-cache] corrupt entry $stage/$key — discarding" >&2
        rm -rf "$dir"
        return 1
    fi
    mkdir -p "$(dirname "$out")"
    cp --reflink=auto -f "$dir/artifact" "$out"
    chmod +x "$out"
    touch "$dir"   # LRU marker
    echo "[madaros-cache] HIT $stage/$key (built $(cat "$dir/built" 2>/dev/null || echo '?'))" >&2
    return 0
}

# madaros_cache_put <stage> <key> <artifact>   (atomic: build in tmp, rename)
madaros_cache_put() {
    local stage="$1" key="$2" art="$3"
    [[ -s "$art" ]] || return 0
    madaros_cache_usable || return 0
    local base; base="$(madaros_cache_dir)/$stage"
    local tmp; tmp="$(mktemp -d "$base/.tmp.XXXXXX")"
    cp --reflink=auto -f "$art" "$tmp/artifact"
    _mc_sha "$tmp/artifact" > "$tmp/artifact.sha256"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$tmp/built"
    ( cd "$(_mc_root)" && { git rev-parse --short HEAD 2>/dev/null || echo nogit; } ) > "$tmp/head"
    if mv -T "$tmp" "$base/$key" 2>/dev/null; then
        echo "[madaros-cache] stored $stage/$key" >&2
    else
        rm -rf "$tmp"   # someone else stored the same key first; fine
    fi
}

# Keep the newest N entries per stage. SOUNIO_MADAROS_CACHE_KEEP overrides the
# default for every stage (CI sets it low: the Actions cache is size-bounded).
madaros_cache_prune() {
    local base; base="$(madaros_cache_dir)"
    local stage keep
    for stage in seed:4 madaros:8 fixed-point:8; do
        keep="${SOUNIO_MADAROS_CACHE_KEEP:-${stage#*:}}"; stage="${stage%%:*}"
        ( cd "$base/$stage" 2>/dev/null || exit 0
          ls -1t | grep -v '^\.' | tail -n +$((keep+1)) | xargs -r rm -rf )
    done
}

# The tree key hashes <root>/stdlib. A build pointed at another stdlib would be
# keyed as if it used this one, so any such build bypasses the cache entirely.
madaros_cache_usable() {
    local root; root="$(_mc_root)"
    [[ -z "${SOUNIO_STDLIB_PATH:-}" ]] && return 0
    [[ "$(realpath -m "$SOUNIO_STDLIB_PATH")" == "$(realpath -m "$root/stdlib")" ]]
}
