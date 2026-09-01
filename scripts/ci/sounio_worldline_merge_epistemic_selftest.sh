#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
crate="$repo_root/tools/uneqself-worldline-merge-ffi/Cargo.toml"
fixture="$repo_root/tests/run-pass/worldline_merge_advice.sio"
artifact_root="${SOUNIO_SYNC004_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-sync004.XXXXXX")}"
cargo_bin="${SOUNIO_SYNC004_CARGO:-cargo}"
mkdir -p "$artifact_root"

cd "$repo_root"
bin/souc check "$fixture" > "$artifact_root/sounio-check.log" 2>&1
"$cargo_bin" fmt --check --manifest-path "$crate" > "$artifact_root/cargo-fmt.log" 2>&1
"$cargo_bin" clippy --all-targets --manifest-path "$crate" -- -D warnings > "$artifact_root/cargo-clippy.log" 2>&1
"$cargo_bin" test --manifest-path "$crate" > "$artifact_root/cargo-test.log" 2>&1
"$cargo_bin" build --release --manifest-path "$crate" > "$artifact_root/cargo-build.log" 2>&1
cc -fsyntax-only -x c tools/uneqself-worldline-merge-ffi/include/uneqself_worldline_merge.h

library="$repo_root/tools/uneqself-worldline-merge-ffi/target/release/libuneqself_worldline_merge.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  library="$repo_root/tools/uneqself-worldline-merge-ffi/target/release/libuneqself_worldline_merge.dylib"
fi
test -f "$library"
exported_analysis_symbols="$(nm -g "$library" | awk '{print $NF}' | grep -Ec '^_?uneqself_worldline_merge_analyze_v1$' || true)"
if [[ "$exported_analysis_symbols" != "1" ]]; then
  echo "SYNC004_SOUNIO_ABI status=FAIL reason=unexpected_analysis_symbol_count count=$exported_analysis_symbols" >&2
  exit 1
fi

if nm -g "$library" | grep -Eq 'merge_(decide|commit|select|mutate)|authorize_merge'; then
  echo "SYNC004_SOUNIO_ABI status=FAIL reason=authority_symbol_exposed" >&2
  exit 1
fi

printf '%s\n' \
  "SYNC004_SOUNIO_ABI status=PASS wire_version=1 vector_byte_exact=true" \
  "digest_only=true evidence_categories=6 human_review_required=true" \
  "decision_authority=false ledger_mutation=false branch_selection=false exported_analysis_symbols=1" \
  "artifact_root=$artifact_root"
