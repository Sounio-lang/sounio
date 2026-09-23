#!/usr/bin/env bash
# Write a content-addressed build manifest for a Madaros ELF just produced by
# scripts/ci/build_modular_madaros.sh.
#
# Usage:
#   bash scripts/ci/write_madaros_build_manifest.sh <elf> <manifest_out> [seed_elf]
#
# The manifest is what scripts/ci/verify_madaros_build_manifest.sh checks a
# consumer's downloaded/cached artifact against before trusting it, instead of
# every job independently rebuilding Madaros from source (see
# scripts/ci/ensure_madaros_build.sh, the one entry point both producers and
# consumers call).
#
# Environment:
#   SOUNIO_EXPECTED_SOURCE_SHA — the commit this build was made from (normally
#                                ${{ github.sha }} in CI). Falls back to the
#                                checked-out HEAD for local/manual use.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "write_madaros_build_manifest"

ELF="${1:?usage: write_madaros_build_manifest.sh <elf> <manifest_out> [seed_elf]}"
MANIFEST="${2:?usage: write_madaros_build_manifest.sh <elf> <manifest_out> [seed_elf]}"
SEED_ELF="${3:-}"

require_nonempty_file "$ELF" "no ELF to write a manifest for: $ELF"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

compiler_sha256="$(sha256_of "$ELF")"
require_nonempty "$compiler_sha256" "sha256sum produced nothing for $ELF"

source_sha="${SOUNIO_EXPECTED_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null || true)}"
require_nonempty "$source_sha" "could not determine source_sha (set SOUNIO_EXPECTED_SOURCE_SHA or run inside a git checkout)"

if [[ -n "$SEED_ELF" && -f "$SEED_ELF" ]]; then
    seed_sha256="$(sha256_of "$SEED_ELF")"
else
    seed_sha256="unknown"
fi

# Same bootstrap-ELF preference order as build_modular_madaros.sh's
# resolve_bootstrap_elf(), duplicated rather than plumbed through so this
# script stays a standalone, independently-checkable measurement of what is
# actually on disk right now -- not a trust-the-caller echo.
bootstrap_elf=""
for cand in "${SOUC_BIN:-}" "${SOUNIO_SOUC_BIN:-}" \
            "$ROOT_DIR/bin/souc-linux-x86_64" "$ROOT_DIR/bin/souc-lean-single-x86_64" \
            "$ROOT_DIR/bin/souc"; do
    if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
        bootstrap_elf="$cand"
        break
    fi
done
if [[ -n "$bootstrap_elf" ]]; then
    bootstrap_elf_sha256="$(sha256_of "$bootstrap_elf")"
    bootstrap_elf_rel="$(realpath --relative-to="$ROOT_DIR" "$bootstrap_elf" 2>/dev/null || printf '%s' "$bootstrap_elf")"
else
    bootstrap_elf_sha256="unknown"
    bootstrap_elf_rel="unknown"
fi

# Everything that changes the build RECIPE (not just the compiler source) must
# also change this digest, or a script edit could silently keep an old cache
# entry valid. Mirrors the actions/cache key composition in ci.yml -- an
# independent recomputation, not a read of the key GHA used, so a bug in that
# key expression itself is still caught by verify_madaros_build_manifest.sh.
toolchain_digest="$(cat "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$ROOT_DIR/scripts/dev/souc-build-lock.sh" | sha256sum | awk '{print $1}')"

built_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
builder_job="${GITHUB_JOB:-local}"

mkdir -p "$(dirname "$MANIFEST")"
python3 - "$MANIFEST" \
    "$compiler_sha256" "$source_sha" "$seed_sha256" \
    "$bootstrap_elf_rel" "$bootstrap_elf_sha256" "$toolchain_digest" \
    "$built_at_utc" "$builder_job" <<'PY'
import json
import sys

(manifest_out, compiler_sha256, source_sha, seed_sha256,
 bootstrap_elf_path, bootstrap_elf_sha256, toolchain_digest,
 built_at_utc, builder_job) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "compiler_sha256": compiler_sha256,
    "source_sha": source_sha,
    "seed_sha256": seed_sha256,
    "bootstrap_elf_path": bootstrap_elf_path,
    "bootstrap_elf_sha256": bootstrap_elf_sha256,
    "toolchain_digest": toolchain_digest,
    "built_at_utc": built_at_utc,
    "builder_job": builder_job,
}
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "  wrote $MANIFEST"
echo "    compiler_sha256=$compiler_sha256"
echo "    source_sha=$source_sha"
echo "    seed_sha256=$seed_sha256"

gate_pass "manifest written for $ELF"
