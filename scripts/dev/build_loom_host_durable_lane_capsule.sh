#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BASE_BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_capsule.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_exec_quorum_capsule.sh"

fail() {
  printf 'build-loom-host-durable-lane-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --output ABSOLUTE_PATH\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate field $key in $path"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "missing field $key in $path"
  printf '%s\n' "$found"
}

replace_field() {
  local path="$1" key="$2" value="$3" stage count
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in $path"
  stage="${path}.next"
  sed "s|^${key}=.*|${key}=${value}|" "$path" > "$stage"
  mv "$stage" "$path"
}

OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$OUTPUT" == /* && ! -L "$OUTPUT" ]] || usage
[[ -x "$BASE_BUILDER" && -x "$PROMOTER" ]] || fail 'base capsule tools are absent'
for tool in sha256sum tar strip objcopy strings grep sed mv install mktemp chmod; do
  command -v "$tool" >/dev/null 2>&1 || fail "required canonicalization tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-durable-capsule.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
BASE_ARCHIVE="$WORK/base.tar"
base_output="$($BASE_BUILDER --output "$BASE_ARCHIVE")"
[[ "$base_output" == 'LOOM_HOST_EXEC_QUORUM_CAPSULE_BUILD PASS '* ]] ||
  fail "base capsule build failed: $base_output"
BASE_SHA256="$(sha256_file "$BASE_ARCHIVE")"
"$PROMOTER" --archive "$BASE_ARCHIVE" --expected-sha256 "$BASE_SHA256" \
  --mode verify >/dev/null

EXTRACTED="$WORK/extracted"
mkdir -m 0700 "$EXTRACTED"
tar --no-same-owner --same-permissions -xf "$BASE_ARCHIVE" -C "$EXTRACTED"
CAPSULE="$EXTRACTED/capsule-v1"
RELEASE="$CAPSULE/release"
RELEASE_MANIFEST="$RELEASE/release.manifest.v1"
CAPSULE_MANIFEST="$CAPSULE/meta/capsule.manifest.v1"
RUNTIME="$RELEASE/bin/sounio-loom-runtime"
for input in "$RELEASE_MANIFEST" "$CAPSULE_MANIFEST" "$RUNTIME"; do
  [[ -f "$input" && ! -L "$input" ]] || fail "canonicalization input is absent or linked: $input"
done
[[ -z "$(find "$CAPSULE" -type l -print -quit)" ]] || fail 'base capsule contains a symlink'

chmod u+w "$(dirname "$RUNTIME")" "$RUNTIME" "$RELEASE_MANIFEST" "$CAPSULE_MANIFEST"
strip --strip-debug "$RUNTIME"
CANONICAL_RUNTIME="$WORK/sounio-loom-runtime.canonical"
objcopy --remove-section=.note.gnu.build-id "$RUNTIME" "$CANONICAL_RUNTIME"
install -m 0555 "$CANONICAL_RUNTIME" "$RUNTIME"
chmod 0555 "$(dirname "$RUNTIME")"
if strings "$RUNTIME" | grep -F "$ROOT_DIR/tools/loom/_build" >/dev/null; then
  fail 'canonical OCaml runtime retained its absolute build root'
fi
printf 'LOOM_DURABLE_LANE_EXIT\n' | SOUNIO_LOOM_DURABLE_LANE_CANARY=1 \
  "$RUNTIME" _durable-lane-canary >/dev/null ||
  fail 'canonical OCaml runtime did not execute the material canary'

SOURCE_COMMIT="$(record_value "$RELEASE_MANIFEST" source_commit)"
RUNTIME_SHA256="$(sha256_file "$RUNTIME")"
RELEASE_DIGEST="$(printf '%s\n%s\n%s\n' "$SOURCE_COMMIT" "$RUNTIME_SHA256" \
  'strip-debug+remove-.note.gnu.build-id' | sha256sum | cut -d ' ' -f 1)"
RELEASE_ID="9030-hostq-${RELEASE_DIGEST:0:32}"
replace_field "$RELEASE_MANIFEST" release_id "$RELEASE_ID"
replace_field "$RELEASE_MANIFEST" product_exec_ingress_runtime_sha256 "$RUNTIME_SHA256"
printf 'product_exec_ingress_runtime_canonicalization=strip-debug+remove-.note.gnu.build-id\n' \
  >> "$RELEASE_MANIFEST"
chmod 0444 "$RELEASE_MANIFEST"
RELEASE_MANIFEST_SHA256="$(sha256_file "$RELEASE_MANIFEST")"
replace_field "$CAPSULE_MANIFEST" release_id "$RELEASE_ID"
replace_field "$CAPSULE_MANIFEST" release_manifest_sha256 "$RELEASE_MANIFEST_SHA256"
printf 'product_exec_ingress_runtime_canonicalization=strip-debug+remove-.note.gnu.build-id\n' \
  >> "$CAPSULE_MANIFEST"
chmod 0444 "$CAPSULE_MANIFEST"

ARCHIVE_STAGE="$WORK/capsule.tar"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --format=posix --pax-option=delete=atime,delete=ctime \
  -C "$EXTRACTED" -cf "$ARCHIVE_STAGE" capsule-v1
ARCHIVE_SHA256="$(sha256_file "$ARCHIVE_STAGE")"
"$PROMOTER" --archive "$ARCHIVE_STAGE" --expected-sha256 "$ARCHIVE_SHA256" \
  --mode verify >/dev/null
mkdir -p "$(dirname "$OUTPUT")"
output_stage="$(mktemp "$(dirname "$OUTPUT")/.loom-durable-capsule.XXXXXX")"
install -m 0600 "$ARCHIVE_STAGE" "$output_stage"
mv -fT "$output_stage" "$OUTPUT"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
chmod 0600 "$OUTPUT.sha256"

printf 'LOOM_HOST_DURABLE_LANE_CAPSULE_BUILD PASS archive=%s archive_sha256=%s release_id=%s release_manifest_sha256=%s source_commit=%s base_archive_sha256=%s ocaml_runtime_sha256=%s canonicalization=strip-debug+remove-.note.gnu.build-id semantic_authority=Sounio operational_language=OCaml shell_oracle_authority=false production_activation=false parity_open=false claim_ready=false\n' \
  "$OUTPUT" "$ARCHIVE_SHA256" "$RELEASE_ID" "$RELEASE_MANIFEST_SHA256" \
  "$SOURCE_COMMIT" "$BASE_SHA256" "$RUNTIME_SHA256"
