#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-/usr/bin/g++}"
SOURCE="${ROOT}/tools/pireus/xor_lowering_material_parity.cpp"

cleanup=0
if [[ -n "${PIREUS_MATERIAL_BUILD_DIR:-}" ]]; then
  BUILD_DIR="${PIREUS_MATERIAL_BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
else
  BUILD_DIR="$(mktemp -d)"
  cleanup=1
fi

if [[ "${cleanup}" == 1 ]]; then
  trap 'rm -rf "${BUILD_DIR}"' EXIT
fi

BIN="${BUILD_DIR}/pireus_xor_lowering_material_parity"
ASM="${BUILD_DIR}/pireus_xor_lowering_material_parity.s"
OBJDUMP="${BUILD_DIR}/pireus_xor_lowering_material_parity.objdump"

FLAGS=(
  -std=c++20
  -O3
  -fno-fast-math
  -fno-associative-math
  -ffp-contract=off
  -fno-tree-vectorize
  -fno-tree-slp-vectorize
  -Wall
  -Wextra
  -Werror
)

"${CXX}" "${FLAGS[@]}" "${SOURCE}" -o "${BIN}"
"${CXX}" "${FLAGS[@]}" -S -masm=intel "${SOURCE}" -o "${ASM}"
(
  cd "${BUILD_DIR}"
  objdump -d -Mintel ./pireus_xor_lowering_material_parity
) > "${OBJDUMP}"

"${BIN}"
printf 'compiler=%s\n' "${CXX}"
printf 'binary=%s\n' "${BIN}"
printf 'assembly=%s\n' "${ASM}"
printf 'objdump=%s\n' "${OBJDUMP}"
