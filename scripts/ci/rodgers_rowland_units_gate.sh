#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-rr-units.XXXXXX")"
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

cxx="${CXX:-g++}"
"$cxx" --version | head -n 1
"$cxx" -std=c++23 -O2 -Wall -Wextra -Werror \
  tests/oracles/rodgers_rowland_units.cpp \
  -o "$tmp_dir/rodgers_rowland_units"

"$tmp_dir/rodgers_rowland_units" | tee "$tmp_dir/cpp23.log"
grep -Fxq "RR_CPP23_UNITS_OK" "$tmp_dir/cpp23.log"
grep -Fxq "closed_form=14/14" "$tmp_dir/cpp23.log"
grep -Fxq "fu_derivative=14/14" "$tmp_dir/cpp23.log"
grep -Fxq "logP_derivative=14/14" "$tmp_dir/cpp23.log"
grep -Fxq "uncertainty_roundtrip=14/14" "$tmp_dir/cpp23.log"
grep -Fxq "definition_triangle=PASS" "$tmp_dir/cpp23.log"
grep -Fxq "pow10_anchor=PASS" "$tmp_dir/cpp23.log"

git rev-parse HEAD
./bin/souc --version
./bin/souc check stdlib/darwin_pbpk/core/rodgers_rowland.sio
./bin/souc run stdlib/darwin_pbpk/core/rodgers_rowland.sio \
  | tee "$tmp_dir/sounio.log"

grep -Fxq "T5 PASS: Kp = fu_plasma * Kpu (14/14)" "$tmp_dir/sounio.log"
grep -Fxq "T6 PASS: dKp/dfu = Kpu > 0 (14/14; h=1e-6)" "$tmp_dir/sounio.log"
grep -Fxq "T7 PASS: dKp/dlogP analytical agreement (14/14; h=1e-6)" "$tmp_dir/sounio.log"
grep -Fxq "T8 PASS: raw GUM sqrt recovers variance (14/14)" "$tmp_dir/sounio.log"
grep -Fxq "T9 PASS: rr_pow10 absolute anchors (-3, 7)" "$tmp_dir/sounio.log"
grep -Fxq "PASS" "$tmp_dir/sounio.log"

echo "RODGERS_ROWLAND_UNITS_GATE_OK"
