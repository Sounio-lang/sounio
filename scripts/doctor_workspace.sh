#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "==> doctor_workspace"
echo "pwd: $(pwd)"
echo "user: $(whoami)"
echo "shell: ${SHELL:-unknown}"

echo
echo "==> toolchain"
command -v rustc && rustc --version || true
command -v cargo && cargo --version || true
command -v clang && clang --version | head -n 1 || true
command -v gcc && gcc --version | head -n 1 || true
command -v cmake && cmake --version | head -n 1 || true
command -v ninja && ninja --version || true
command -v python3 && python3 --version || true

echo
echo "==> sounio env"
echo "SOUNIO_STDLIB_PATH=${SOUNIO_STDLIB_PATH:-<unset>}"
echo "SOUC_BIN=${SOUC_BIN:-<unset>}"

if [[ -n "${SOUC_BIN:-}" && -x "${SOUC_BIN}" ]]; then
  echo
  echo "==> checked-in compiler"
  "${SOUC_BIN}" --version || true
  "${SOUC_BIN}" info || true
else
  echo "WARN: checked-in SOUC_BIN not executable or not found"
fi

if [[ -d stdlib ]]; then
  echo "==> stdlib present"
else
  echo "ERROR: stdlib directory missing"
  exit 1
fi

echo
echo "==> doctor_workspace: OK"
