#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./runner kernel.ptx \
  --mode basic --threads 8 --mem-words 437 \
  --init-file init_mem.bin --type i64 --print-count 437 \
  > gpu.log 2>&1
grep -q 'sounio_kaxi_runtime status=pass' gpu.log
