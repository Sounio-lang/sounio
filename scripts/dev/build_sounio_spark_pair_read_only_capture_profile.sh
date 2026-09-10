#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
RESTORE_AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_restore_capsule.sio"
PROFILE_AUTHORITY="$ROOT_DIR/stdlib/coordination/spark_pair_read_only_capture_profile.sio"
ENTRYPOINT="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile_main.sio"
OUTPUT="${SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_OUTPUT:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-read-only-capture-profile}"

fail() {
  printf 'build-sounio-spark-pair-read-only-capture-profile: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$RESTORE_AUTHORITY" ]] || fail "restore authority is missing: $RESTORE_AUTHORITY"
[[ -f "$PROFILE_AUTHORITY" ]] || fail "capture profile is missing: $PROFILE_AUTHORITY"
[[ -f "$ENTRYPOINT" ]] || fail "capture profile entrypoint is missing: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-read-only-capture-profile-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_read_only_capture_profile.sio"
compiled="$work/sounio-spark-pair-read-only-capture-profile"

mkdir -p "$(dirname "$OUTPUT")"
sed -n '1,$p' "$RESTORE_AUTHORITY" "$PROFILE_AUTHORITY" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -x "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

zero=0000000000000000000000000000000000000000000000000000000000000000
snapshot=1111111111111111111111111111111111111111111111111111111111111111
node0=2222222222222222222222222222222222222222222222222222222222222222
node1=3333333333333333333333333333333333333333333333333333333333333333
probe="$($OUTPUT "$snapshot" "$node0" "$node1" "$zero" 0 131071 127 0 0)"
case "$probe" in
  SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_PASS*' reason=PREINSTALL_PROVENANCE code=315 effect=NONE material_dispatch=false') ;;
  *) fail "capture profile failed its exact DENY315 probe: $probe" ;;
esac

printf 'BUILT_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE path=%s language=Sounio engine=%s expected_restore=DENY315 restorable=false\n' \
  "$OUTPUT" "$ENGINE"
