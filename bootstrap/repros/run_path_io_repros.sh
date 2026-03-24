#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPILER="${1:-$ROOT/bootstrap/boot4_a1.elf}"
TMP_DIR="${TMPDIR:-/tmp}/boot4-path-repros-$$"

mkdir -p "$TMP_DIR"

cases=(
  "path_file_size_literal_ok"
  "path_file_size_global_buf"
  "path_file_size_local_small_buf"
  "path_file_size_local_large_buf"
  "path_read_file_global_buf"
  "path_read_file_local_small_buf"
)

pass=0

for name in "${cases[@]}"; do
  src="$ROOT/bootstrap/repros/$name.sio"
  elf="$TMP_DIR/$name.elf"
  "$COMPILER" "$src" "$elf" >/dev/null
  chmod +x "$elf"
  set +e
  "$elf" >/dev/null
  status=$?
  set -e
  if [ "$status" -ne 42 ]; then
    echo "FAIL $name exit=$status"
    exit 1
  fi
  echo "PASS $name"
  pass=$((pass + 1))
done

echo "PASS=$pass"
