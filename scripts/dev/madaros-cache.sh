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
    (
        cd "$root" || exit 1
        # Without git metadata (souc-build-remote.sh ships a tarball to SLURM
        # nodes) fall back to every regular file under the two trees. The key
        # is still a pure function of content; it just may include ignored files.
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git ls-files -z -co --exclude-standard self-hosted stdlib
        else
            find self-hosted stdlib -type f -print0 2>/dev/null
        fi | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64
    )
}

# Environment the compiler can read while it builds. Every env read in
# self-hosted/ is a SOUNIO_* variable, and many change the emitted ELF
# (SOUNIO_MADAROS_DEP_MERGE picks another lowering path, SOUNIO_DISABLE_MM_DCE
# turns off cross-module DCE, SOUNIO_OCP_SKIP_* drop optimiser passes, the
# *_SABOTAGE_* hooks miscompile on purpose). So every SOUNIO_* variable is part
# of the key -- fail-safe: an unknown variable costs a cache miss, never a wrong
# hit -- except pure plumbing that only names paths, pools or this cache.
# SOUNIO_STDLIB_PATH is excluded here because madaros_cache_usable refuses the
# cache outright when it points anywhere but this checkout.
madaros_env_fingerprint() {
    # `|| true`: grep exits 1 when nothing matches (the common case), which
    # under a caller's `set -o pipefail` would otherwise fail the pipeline.
    { env | LC_ALL=C sort | grep -E '^SOUNIO_[A-Z0-9_]*=' \
        | grep -vE '^SOUNIO_(MADAROS_CACHE[A-Z_]*|MADAROS_NOCACHE|STDLIB_PATH|BUILD_SLOTS|CI_RUNNER|TEST_JOBS|SLOW_TESTS_AVAILABLE|MADAROS_FP_[A-Z_]*|[A-Z0-9_]*_(BIN|DIR|KEEP|REPORT_DIR))=' \
        || true
      # The native backend picks its target from HOSTTYPE/OSTYPE
      # (self-hosted/native/codegen.sio, codegen_x86_linux.sio).
      echo "HOSTTYPE=${HOSTTYPE:-}"; echo "OSTYPE=${OSTYPE:-}"
    } | sha256sum | cut -c1-64
}

# Stage 1 key: bootstrap ELF bytes + lean_single.sio bytes + build env.
madaros_seed_key() {  # <bootstrap-elf> <lean-src>
    printf 'seed-v2\n%s\n%s\n%s\n' "$(_mc_sha "$1")" "$(_mc_sha "$2")" "$(madaros_env_fingerprint)" \
        | sha256sum | cut -c1-64
}

# Stage 2 key: seed ELF bytes + whole tree + build env. (main.sio is inside
# the tree.)
# Pass the tree key explicitly (and export it as MADAROS_CACHE_TREE_AT_KEY for
# madaros_cache_build_locked) so the key and the store check refer to the SAME
# tree snapshot; without it the tree is hashed here, once.
madaros_build_key() {  # <seed-elf> [tree-key]
    printf 'madaros-v2\n%s\n%s\n%s\n' "$(_mc_sha "$1")" "${2:-$(madaros_tree_key)}" "$(madaros_env_fingerprint)" \
        | sha256sum | cut -c1-64
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
    mkdir -p "$(dirname "$out")" || return 1
    if ! cp --reflink=auto -f "$dir/artifact" "$out" || ! chmod +x "$out" ||
       [[ "$(_mc_sha "$out")" != "$(cut -c1-64 "$dir/artifact.sha256")" ]]; then
        echo "[madaros-cache] copy of $stage/$key to $out failed -- treating as a miss" >&2
        rm -f "$out"
        return 1
    fi
    touch "$dir" 2>/dev/null || true   # LRU marker
    echo "[madaros-cache] HIT $stage/$key (built $(cat "$dir/built" 2>/dev/null || echo '?'))" >&2
    return 0
}

# madaros_cache_put <stage> <key> <artifact>   (atomic: build in tmp, rename)
madaros_cache_put() {
    local stage="$1" key="$2" art="$3"
    [[ -s "$art" ]] || return 0
    madaros_cache_usable || return 0
    if [[ "${SOUNIO_MADAROS_CACHE_READONLY:-0}" == "1" ]]; then
        echo "[madaros-cache] read-only: not storing $stage/$key" >&2
        return 0
    fi
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
        # Measured 2026-09-25 in CI: with an empty stage directory `grep -v`
        # matched nothing and exited 1, and under the caller's pipefail that
        # killed build_modular_madaros.sh right after a successful build.
        ( cd "$base/$stage" 2>/dev/null || exit 0
          { ls -1t | grep -v '^\.' || true; } | tail -n +$((keep+1)) | xargs -r rm -rf ) || true
    done
    return 0
}

# The tree key hashes <root>/stdlib. A build pointed at another stdlib would be
# keyed as if it used this one, so any such build bypasses the cache entirely.
madaros_cache_usable() {
    local root; root="$(_mc_root)"
    [[ -z "${SOUNIO_STDLIB_PATH:-}" ]] && return 0
    [[ "$(realpath -m "$SOUNIO_STDLIB_PATH")" == "$(realpath -m "$root/stdlib")" ]]
}

# madaros_cache_build_locked <stage> <key> <out> <build command...>
# Hit -> copy out, no lock. Miss -> take the global build lock, look again
# (another build may have stored the same key while this one waited), and only
# then build and store. Returns the build command's status.
madaros_cache_build_locked() {
    local stage="$1" key="$2" out="$3"; shift 3
    madaros_cache_get "$stage" "$key" "$out" && return 0
    local lib; lib="$(_mc_root)/scripts/dev/madaros-cache.sh"
    # Tree the key was derived from. Callers that derive the key from a tree
    # snapshot export it as MADAROS_CACHE_TREE_AT_KEY (see madaros_build_key);
    # otherwise fall back to hashing now, which can race a concurrent edit.
    MADAROS_CACHE_TREE_AT_KEY="${MADAROS_CACHE_TREE_AT_KEY:-$(madaros_tree_key)}" \
    "$(_mc_root)/scripts/dev/souc-build-lock.sh" bash -c '
        lib="$1"; stage="$2"; key="$3"; out="$4"; shift 4
        # shellcheck source=/dev/null
        source "$lib"
        if madaros_cache_get "$stage" "$key" "$out"; then
            echo "[madaros-cache] stored by a concurrent build while waiting for the lock" >&2
            # A hit is the artifact for $key, i.e. for the tree the key was
            # derived from; say so if this worktree has moved on since.
            [[ "$(madaros_tree_key)" == "$MADAROS_CACHE_TREE_AT_KEY" ]] \
                || echo "[madaros-cache] note: worktree changed since $stage/$key was keyed; artifact is for the keyed tree" >&2
            exit 0
        fi
        # The key was computed before waiting for the lock; if the tree moved
        # meanwhile (another agent editing this worktree) the output no longer
        # corresponds to it, so it is used but not stored.
        tree_before="$(madaros_tree_key)"
        "$@" || exit $?
        if [[ "$(madaros_tree_key)" == "$tree_before" && "$tree_before" == "$MADAROS_CACHE_TREE_AT_KEY" ]]; then
            madaros_cache_put "$stage" "$key" "$out"
        else
            echo "[madaros-cache] source tree changed since $stage/$key was computed -- not storing" >&2
        fi
    ' _ "$lib" "$stage" "$key" "$out" "$@"
}
