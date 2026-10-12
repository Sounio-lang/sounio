#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTH="$ROOT_DIR/scripts/ci/madaros_refresh_auth.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

check() {
  local required="$1" app="$2" key="$3" pat="$4" expected_rc="$5" expected_mode="$6" rc=0
  PUBLISH_REQUIRED="$required" APP_ID="$app" HAS_APP_KEY="$key" HAS_PAT="$pat" \
    bash "$AUTH" >"$WORK/out" 2>"$WORK/err" || rc=$?
  [[ "$rc" == "$expected_rc" ]] || { cat "$WORK/err" >&2; exit 1; }
  [[ "$(cat "$WORK/out")" == "mode=$expected_mode" ]] || exit 1
  if [[ "$rc" != 0 ]]; then
    grep -Fq 'maintainer approval' "$WORK/err"
    grep -Fq 'GITHUB_TOKEN is not a fallback' "$WORK/err"
  fi
}

check true '' false false 1 none
check true 123 false false 1 none
check true '' true false 1 none
check true 123 true false 0 app
check true 123 true true 0 app
check true '' false true 0 pat
check true 123 false true 0 pat
check false '' false false 0 none
check false 123 true true 0 none
echo MADAROS_REFRESH_AUTH_SELFTEST_OK
