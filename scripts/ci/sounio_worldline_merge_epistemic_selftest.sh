#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
crate="$repo_root/tools/uneqself-worldline-merge-ffi/Cargo.toml"
fixture="$repo_root/tests/run-pass/worldline_merge_advice.sio"
artifact_root="${SOUNIO_SYNC004_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-sync004.XXXXXX")}"
cargo_bin="${SOUNIO_SYNC004_CARGO:-cargo}"
mkdir -p "$artifact_root"

current_stage="bootstrap"
dump_failure() {
  local status="$1"
  set +e
  printf 'SYNC004_SOUNIO_ABI status=FAIL stage=%s exit=%s artifact_root=%s\n' \
    "$current_stage" "$status" "$artifact_root" >&2
  for log in "$artifact_root"/*.log; do
    [[ -f "$log" ]] || continue
    printf '%s\n' "--- $(basename "$log") ---" >&2
    sed -n '1,240p' "$log" >&2
  done
}
trap 'status=$?; dump_failure "$status"; exit "$status"' ERR

export CARGO_TARGET_DIR="$artifact_root/cargo-target"

cd "$repo_root"
current_stage="sounio-check"
bin/souc check "$fixture" > "$artifact_root/sounio-check.log" 2>&1
current_stage="cargo-fmt"
"$cargo_bin" fmt --check --manifest-path "$crate" > "$artifact_root/cargo-fmt.log" 2>&1
current_stage="cargo-clippy"
"$cargo_bin" clippy --locked --all-targets --manifest-path "$crate" -- -D warnings > "$artifact_root/cargo-clippy.log" 2>&1
current_stage="cargo-test"
"$cargo_bin" test --locked --manifest-path "$crate" > "$artifact_root/cargo-test.log" 2>&1
current_stage="cargo-build"
"$cargo_bin" build --locked --release --manifest-path "$crate" > "$artifact_root/cargo-build.log" 2>&1
current_stage="c-header"
cc -fsyntax-only -x c tools/uneqself-worldline-merge-ffi/include/uneqself_worldline_merge.h

library="$CARGO_TARGET_DIR/release/libuneqself_worldline_merge.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  library="$CARGO_TARGET_DIR/release/libuneqself_worldline_merge.dylib"
fi
current_stage="library-present"
test -f "$library"
current_stage="symbol-audit"
exported_analysis_symbols="$(nm -g "$library" | awk '{print $NF}' | grep -Ec '^_?uneqself_worldline_merge_analyze_v1$' || true)"
if [[ "$exported_analysis_symbols" != "1" ]]; then
  echo "SYNC004_SOUNIO_ABI status=FAIL reason=unexpected_analysis_symbol_count count=$exported_analysis_symbols" >&2
  exit 1
fi

if nm -g "$library" | grep -Eq 'merge_(decide|commit|select|mutate)|authorize_merge'; then
  echo "SYNC004_SOUNIO_ABI status=FAIL reason=authority_symbol_exposed" >&2
  exit 1
fi

trap - ERR
printf '%s\n' \
  "SYNC004_SOUNIO_ABI status=PASS wire_version=1 vector_byte_exact=true" \
  "digest_only=true evidence_categories=6 human_review_required=true" \
  "decision_authority=false ledger_mutation=false branch_selection=false exported_analysis_symbols=1" \
  "artifact_root=$artifact_root"
