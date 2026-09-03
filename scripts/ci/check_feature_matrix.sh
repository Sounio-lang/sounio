#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [[ "$SKIP_BUILD" = "1" ]]; then
  echo "[feature-matrix] skipped (SKIP_BUILD=1, feature checks require cargo)"
  exit 0
fi

has_llvm15() {
  if [[ -n "${LLVM_SYS_150_PREFIX:-}" && -d "${LLVM_SYS_150_PREFIX}" ]]; then
    return 0
  fi

  if command -v llvm-config-15 >/dev/null 2>&1; then
    return 0
  fi

  if command -v llvm-config >/dev/null 2>&1; then
    local version
    version="$(llvm-config --version 2>/dev/null || true)"
    [[ "$version" == 15* ]]
    return
  fi

  return 1
}

echo "[feature-matrix] baseline"
(cd "$ROOT_DIR" && cargo check -p souc)

echo "[feature-matrix] cuda"
(cd "$ROOT_DIR" && cargo check -p souc --features cuda)

echo "[feature-matrix] gpu"
(cd "$ROOT_DIR" && cargo check -p souc --features gpu)

if has_llvm15; then
  echo "[feature-matrix] llvm"
  (cd "$ROOT_DIR" && cargo check -p souc --features llvm --bin souc)
else
  echo "[feature-matrix] llvm skipped (LLVM 15 not detected)"
fi

echo "[feature-matrix] ok"
