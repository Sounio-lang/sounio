#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$ROOT_DIR/formal/tla/SounioFleet.tla"
CONFIG="$ROOT_DIR/formal/tla/SounioFleet.cfg"
GENERATOR="$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py"
RUNTIME_TEST="$ROOT_DIR/scripts/ci/sounio_coord_fleetd_selftest.sh"
FLEET_TEST="$ROOT_DIR/scripts/ci/sounio_coord_fleet_selftest.sh"
TLA_JAR_SHA256='936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88'
TLA_JAR_URL='https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar'
JRE_SHA256='2413149700df0f7d440500a84a8f764c535f21e5a5e87d38328b64eec2c5b500'
JRE_URL='https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jre_x64_linux_hotspot_21.0.12.1_1.tar.gz'
CACHE_ROOT="${SOUNIO_TLA_CACHE:-$HOME/.cache/sounio/tla}"
TLA_JAR="${TLA2TOOLS_JAR:-$CACHE_ROOT/tla2tools-936a262061c9.jar}"
JAVA="${SOUNIO_TLA_JAVA:-}"
WORKSPACE_CACHE='/workspace/.home/openvscode-server/.cache/sounio/tla'
ALLOW_DOWNLOAD="${SOUNIO_TLA_ALLOW_DOWNLOAD:-1}"

fail() {
  printf 'sounio-coord-fleet-model-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -n "$JAVA" ]] || JAVA="$(find "$CACHE_ROOT" -type f -path '*/bin/java' -print -quit 2>/dev/null || true)"
[[ -n "$JAVA" ]] || JAVA="$(find "$WORKSPACE_CACHE" -type f -path '*/bin/java' -print -quit 2>/dev/null || true)"
[[ -n "$JAVA" && -x "$JAVA" ]] || JAVA="$(command -v java || true)"
if [[ -z "$JAVA" && "$ALLOW_DOWNLOAD" == 1 ]]; then
  mkdir -p "$CACHE_ROOT"
  curl -fL --retry 3 -o "$CACHE_ROOT/temurin-jre21.tar.gz.download" "$JRE_URL"
  [[ "$(sha256sum "$CACHE_ROOT/temurin-jre21.tar.gz.download" | awk '{print $1}')" == "$JRE_SHA256" ]] || \
    fail 'portable Temurin JRE digest does not match the reviewed bundle'
  mv "$CACHE_ROOT/temurin-jre21.tar.gz.download" "$CACHE_ROOT/temurin-jre21.tar.gz"
  tar -xzf "$CACHE_ROOT/temurin-jre21.tar.gz" -C "$CACHE_ROOT"
  JAVA="$(find "$CACHE_ROOT" -type f -path '*/bin/java' -print -quit 2>/dev/null || true)"
fi
[[ -n "$JAVA" && -x "$JAVA" ]] || \
  fail 'Java 11+ is required; set SOUNIO_TLA_JAVA to a portable JRE'

if [[ ! -f "$TLA_JAR" && -f "$WORKSPACE_CACHE/tla2tools-936a262061c9.jar" ]]; then
  TLA_JAR="$WORKSPACE_CACHE/tla2tools-936a262061c9.jar"
fi
if [[ ! -f "$TLA_JAR" ]]; then
  [[ "$ALLOW_DOWNLOAD" == 1 ]] || \
    fail "tla2tools.jar is missing: $TLA_JAR"
  mkdir -p "$CACHE_ROOT"
  curl -fL --retry 3 -o "$TLA_JAR.download" "$TLA_JAR_URL"
  mv "$TLA_JAR.download" "$TLA_JAR"
fi
[[ "$(sha256sum "$TLA_JAR" | awk '{print $1}')" == "$TLA_JAR_SHA256" ]] || \
  fail 'tla2tools.jar digest does not match the reviewed tool bundle'

catalog="$($GENERATOR --model "$MODEL" --config "$CONFIG" \
  --check-test "$RUNTIME_TEST" --check-test "$FLEET_TEST")"
[[ "$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)["sabotages"]))' <<< "$catalog")" == 8 ]] || \
  fail 'model-derived sabotage catalog does not contain eight controls'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-fleet-tlc.XXXXXX")"
trap 'rm -rf "$work"' EXIT
cp "$MODEL" "$CONFIG" "$work/"
(
  cd "$work"
  "$JAVA" -cp "$TLA_JAR" tlc2.TLC -cleanup -workers 1 -config SounioFleet.cfg \
    SounioFleet.tla >tlc.log 2>&1
) || {
  cat "$work/tlc.log" >&2
  fail 'TLC rejected the fleet transition system'
}
grep -q 'Model checking completed. No error has been found.' "$work/tlc.log" || {
  cat "$work/tlc.log" >&2
  fail 'TLC did not establish all configured invariants'
}

echo 'sounio-coord-fleet-model-selftest: PASS tlc=exhaustive invariants=8 model_controls=8 runtime_witnesses=bound'
