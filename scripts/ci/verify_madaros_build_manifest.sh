#!/usr/bin/env bash
# Verify a Madaros ELF against its build manifest before any job trusts it --
# whether the pair arrived via actions/upload-artifact+download-artifact
# (same ci.yml run) or actions/cache (a separate workflow file racing the
# same push event). A cache/artifact HIT is not itself trust: this script is
# what makes it trust, by recomputing every claim the manifest makes rather
# than reading them back.
#
# Usage:
#   bash scripts/ci/verify_madaros_build_manifest.sh <elf> <manifest>
#
# Environment:
#   SOUNIO_EXPECTED_SOURCE_SHA — if set, the manifest's source_sha must equal
#                                it exactly (normally ${{ github.sha }} in CI:
#                                every Phase-1 consumer runs against the same
#                                commit as the producer, so this costs nothing
#                                in practice and catches a stale/cross-commit
#                                artifact outright). Unset: source_sha is not
#                                checked (best-effort local use).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "verify_madaros_build_manifest"

ELF="${1:?usage: verify_madaros_build_manifest.sh <elf> <manifest>}"
MANIFEST="${2:?usage: verify_madaros_build_manifest.sh <elf> <manifest>}"

[[ -f "$ELF" && -f "$MANIFEST" ]] || gate_fail "nothing to verify: elf=$ELF manifest=$MANIFEST (cache/artifact miss -- caller should build instead)"
require_nonempty_file "$ELF" "$ELF exists but is empty"
require_nonempty_file "$MANIFEST" "$MANIFEST exists but is empty"

if head -c2 "$ELF" 2>/dev/null | grep -q '#!'; then
    gate_fail "$ELF is a wrapper script, not a raw ELF -- refusing to certify a wrapper"
fi

schema_version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("schema_version",""))' "$MANIFEST" 2>/dev/null)" \
    || gate_fail "$MANIFEST is not valid JSON"
[[ "$schema_version" == "1" ]] || gate_fail "unsupported manifest schema_version=$schema_version (expected 1) -- a schema change must bust the cache, not be silently trusted"

manifest_field() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$MANIFEST" "$1"
}

manifest_compiler_sha256="$(manifest_field compiler_sha256)"
manifest_source_sha="$(manifest_field source_sha)"
manifest_toolchain_digest="$(manifest_field toolchain_digest)"
manifest_bootstrap_elf_path="$(manifest_field bootstrap_elf_path)"
manifest_bootstrap_elf_sha256="$(manifest_field bootstrap_elf_sha256)"

require_nonempty "$manifest_compiler_sha256" "manifest has no compiler_sha256"
require_nonempty "$manifest_source_sha" "manifest has no source_sha"
require_nonempty "$manifest_toolchain_digest" "manifest has no toolchain_digest"

# 1. The ELF on disk must actually be the ELF the manifest describes -- catches
#    truncation/corruption in transit, or a manifest copied next to the wrong file.
actual_compiler_sha256="$(sha256sum "$ELF" | awk '{print $1}')"
if [[ "$actual_compiler_sha256" != "$manifest_compiler_sha256" ]]; then
    gate_fail "compiler_sha256 mismatch: manifest says $manifest_compiler_sha256, $ELF is actually $actual_compiler_sha256"
fi

# 2. The commit this was built from must be the commit the caller expects --
#    the check that makes a stale or cross-commit cache/artifact refuse rather
#    than silently pass as this PR's own build.
if [[ -n "${SOUNIO_EXPECTED_SOURCE_SHA:-}" && "$manifest_source_sha" != "${SOUNIO_EXPECTED_SOURCE_SHA}" ]]; then
    gate_fail "source_sha mismatch: manifest says $manifest_source_sha, expected ${SOUNIO_EXPECTED_SOURCE_SHA} (SOUNIO_EXPECTED_SOURCE_SHA) -- this artifact was built from a different commit"
fi

# 3. Recompute the toolchain digest independently from the build scripts
#    CURRENTLY on disk -- an independent check of the same inputs the
#    actions/cache key hashes, so a bug in that key expression (a missed
#    input) is still caught here rather than silently serving a stale build.
actual_toolchain_digest="$(cat "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$ROOT_DIR/scripts/dev/souc-build-lock.sh" | sha256sum | awk '{print $1}')"
if [[ "$actual_toolchain_digest" != "$manifest_toolchain_digest" ]]; then
    gate_fail "toolchain_digest mismatch: manifest says $manifest_toolchain_digest, current build scripts hash to $actual_toolchain_digest -- the build recipe changed since this artifact was built"
fi

# 4. The bootstrap ELF is best-effort informational -- environments legitimately
#    differ in which candidate resolve_bootstrap_elf() picks (a fresh gen3.elf
#    locally vs. bin/souc-linux-x86_64 in CI). Warn, do not fail, unless the
#    manifest never recorded one at all.
if [[ -n "$manifest_bootstrap_elf_path" && "$manifest_bootstrap_elf_path" != "unknown" && -f "$ROOT_DIR/$manifest_bootstrap_elf_path" ]]; then
    actual_bootstrap_elf_sha256="$(sha256sum "$ROOT_DIR/$manifest_bootstrap_elf_path" | awk '{print $1}')"
    if [[ "$actual_bootstrap_elf_sha256" != "$manifest_bootstrap_elf_sha256" ]]; then
        echo "  warning: bootstrap_elf_sha256 mismatch for $manifest_bootstrap_elf_path (manifest=$manifest_bootstrap_elf_sha256, actual=$actual_bootstrap_elf_sha256) -- informational only" >&2
    fi
fi

echo "  verify: OK"
echo "    elf=$ELF"
echo "    compiler_sha256=$actual_compiler_sha256"
echo "    source_sha=$manifest_source_sha"

gate_pass "$ELF matches $MANIFEST"
