#!/usr/bin/env bash
# CITATION <-> toolchain version parity gate (SIO-159, ADR-018 section 1).
#
# The toolchain carries one Semantic Versioning number. `souc --version`
# reports it with the engine build after it, as in
# `3.0.0-rc.1 (madaros 4f1c2e9a)`. CITATION.cff is the authority for the
# released version. At a stable release tag the two must be the same
# string, and the tag must name it too.
#
# Usage:
#   bash scripts/ci/citation_version_gate.sh            # mechanism check
#   bash scripts/ci/citation_version_gate.sh v3.0.0     # tag parity check
#
# Without a tag, the gate verifies that --version reports the toolchain
# version (the newest CHANGELOG section) in the ADR-018 format. With a
# tag, it additionally requires tag == CITATION.cff version == reported
# toolchain version, and fails otherwise.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "[citation-version] FAIL: $*" >&2
  exit 1
}

toolchain_version() {
  sed -n 's/^## \[\(.*\)\].*/\1/p' CHANGELOG.md | head -n 1
}

citation_version() {
  sed -n 's/^version: //p' CITATION.cff | head -n 1
}

TOOLCHAIN="$(toolchain_version)"
CITATION="$(citation_version)"
[[ -n "$TOOLCHAIN" ]] || fail "no version section found in CHANGELOG.md"
[[ -n "$CITATION" ]] || fail "no version field found in CITATION.cff"
[[ "$TOOLCHAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] \
  || fail "toolchain version '$TOOLCHAIN' is not MAJOR.MINOR.PATCH[-PRE]"

# The reported string must be the toolchain version with the engine build
# after it, never the engine banner alone. bin/souc derives it from the
# same CHANGELOG source; running the binary is CI's job (it needs the
# materialized Madaros ELF), so here we pin the source-level contract.
if ! grep -q 'printf '"'"'%s (madaros %s)' bin/souc; then
  fail "bin/souc --version does not print '<toolchain> (madaros <build>)'"
fi

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "[citation-version] PASS: toolchain=$TOOLCHAIN citation=$CITATION (no tag: parity applies at release tags)"
  exit 0
fi

TAG_VERSION="${TAG#v}"
[[ "$TAG_VERSION" == "$TOOLCHAIN" ]] \
  || fail "tag $TAG names $TAG_VERSION but the toolchain reports $TOOLCHAIN"
[[ "$TAG_VERSION" == "$CITATION" ]] \
  || fail "tag $TAG names $TAG_VERSION but CITATION.cff records $CITATION"
echo "[citation-version] PASS: tag=$TAG toolchain=$TOOLCHAIN citation=$CITATION"
