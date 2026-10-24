#!/usr/bin/env bash
# Full pinned-member corpus gate. Source snapshots identify fixture revisions.
# A failing member fails the gate; no engine skips.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILER="${1:?pass the native Madaros candidate}"
case "$COMPILER" in /*) ;; *) COMPILER="$PWD/$COMPILER" ;; esac
OUT="${2:-$(mktemp -d "${TMPDIR:-/tmp}/madaros-macos-corpus.XXXXXX")}"
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_METAL_RUNTIME="$ROOT/artifacts/self-hosted/libsounio_metal.dylib"
export SOUNIO_METAL_TRACE=1
export CORPUS_REQUIRE_METAL=1
export CORPUS_REQUIRE_200=1
bash scripts/dev/build_native_metal_runtime.sh "$SOUNIO_METAL_RUNTIME" > "$OUT/metal-runtime-build.log" 2>&1
"${CXX:-/usr/bin/clang++}" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror \
    scripts/ci/metal_binary64_conformance.cpp -o "$OUT/metal-conformance"
"$OUT/metal-conformance" "$SOUNIO_METAL_RUNTIME" > "$OUT/metal-conformance.log" 2>&1
"${CXX:-/usr/bin/clang++}" -std=c++23 -O2 -pthread -Wall -Wextra -Werror \
    scripts/ci/madaros_macos_corpus.cpp -o "$OUT/corpus"
bash scripts/ci/madaros_macos_corpus_diagnostics_gate.sh "$OUT/corpus" "$OUT/diagnostic-negative"
shasum -a 256 "$COMPILER" "$SOUNIO_METAL_RUNTIME" tests/manifests/madaros_macos_200.txt \
    vendor/metal-softfloat/softfloat64.metal > "$OUT/identity-with-runtime.txt"
"$OUT/corpus" "$COMPILER" tests/manifests/madaros_macos_200.txt "$OUT/results" "${CORPUS_JOBS:-4}" "${CORPUS_BUILD_TIMEOUT:-180}"
cat "$OUT/results/summary.txt"
echo "PASS: all 200 manifest members compiled and executed, including required real Metal launches; see source snapshots for fixture revisions"
