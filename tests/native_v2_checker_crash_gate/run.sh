#!/usr/bin/env bash
set -u

BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"

pass=0
tot=0

run_valid() {
  local src="$1"
  local name
  name="$(basename "$src" .sio)"
  tot=$((tot + 1))
  local out rc
  out="$("$BIN" --check "$src" 2>&1)"
  rc=$?
  if [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q "check: OK"; then
    echo "PASS valid   $name"
    pass=$((pass + 1))
  else
    echo "FAIL valid   $name (rc=$rc)"
    printf '%s\n' "$out" | sed 's/^/  | /'
  fi
}

run_invalid() {
  local src="$1"
  local name
  name="$(basename "$src" .sio)"
  tot=$((tot + 1))
  local out rc
  out="$("$BIN" --check "$src" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && ! printf '%s\n' "$out" | grep -q "check: OK"; then
    echo "PASS invalid $name"
    pass=$((pass + 1))
  else
    echo "FAIL invalid $name (rc=$rc)"
    printf '%s\n' "$out" | sed 's/^/  | /'
  fi
}

for src in "$DIR"/valid/*.sio; do
  run_valid "$src"
done

for src in "$DIR"/invalid/*.sio; do
  run_invalid "$src"
done

echo "---- $pass/$tot ----"
test "$pass" -eq "$tot"
