#!/usr/bin/env bash
# Partition only the canonical harness inventory. No compiler runs or test policy here.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $# == 2 ]] || { echo 'usage: full_suite_partition.sh SHARD OUTPUT_DIR' >&2; exit 64; }
case "$1" in 0) suffix=00 ;; 1) suffix=01 ;; 2) suffix=02 ;; 3) suffix=03 ;; *) echo 'invalid full-suite shard' >&2; exit 64 ;; esac
out=$2
: "${SOUNIO_TEST_SOUC_BIN:?fresh native compiler is required}"
mkdir -p "$out"
work=$(mktemp -d "$out/.partition.XXXXXX")
trap 'rm -rf "$work"' EXIT
LC_ALL=C bash "$ROOT/scripts/run_sio_test_suite.sh" --list-tests > "$work/canonical.list"
[[ -s "$work/canonical.list" ]] || { echo 'canonical suite inventory is empty' >&2; exit 1; }
while IFS= read -r file; do
    [[ -n "$file" && -f "$ROOT/$file" ]] || { echo "invalid canonical entry: $file" >&2; exit 1; }
done < "$work/canonical.list"
LC_ALL=C sort "$work/canonical.list" > "$work/expected.list"
[[ -z "$(uniq -d "$work/expected.list")" ]] || { echo 'duplicate canonical entries' >&2; exit 1; }
split -n r/4 -d "$work/canonical.list" "$work/part-"
cat "$work"/part-0[0-3] | LC_ALL=C sort > "$work/actual.list"
cmp -s "$work/expected.list" "$work/actual.list" || { echo 'suite partition omitted or duplicated a case' >&2; exit 1; }
[[ -s "$work/part-$suffix" ]] || { echo 'selected suite shard is empty' >&2; exit 1; }
cp "$work/canonical.list" "$out/canonical.list"
cp "$work/part-$suffix" "$out/selected.list"
printf '%s\n' "$1" > "$out/GROUP"
git -C "$ROOT" rev-parse HEAD > "$out/SOURCE_GIT_SHA"
sha256sum "$SOUNIO_TEST_SOUC_BIN" > "$out/COMPILER.sha256"
printf 'FULL_SUITE_CANONICAL_PARTITION_PASS shard=%s\n' "$1"
