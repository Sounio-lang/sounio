#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_DIAG_ISOLATED:-0}" == "1" ]]; then
  echo "[preflight] isolated diagnostic mode detected; skipping workspace cargo conflict scan"
  exit 0
fi

declare -a conflicts=()

while IFS= read -r pid; do
  [[ -z "${pid}" ]] && continue

  # Ignore this script process chain.
  if [[ "${pid}" == "$$" || "${pid}" == "$PPID" ]]; then
    continue
  fi

  cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"

  [[ -z "${cwd}" || -z "${cmd}" ]] && continue

  if [[ "${cwd}" == "${ROOT_DIR}" || "${cwd}" == "${ROOT_DIR}/"* ]]; then
    conflicts+=("${pid}|${cwd}|${cmd}")
  fi
done < <(pgrep -x cargo || true)

if (( ${#conflicts[@]} > 0 )); then
  echo "[preflight] active cargo process(es) detected in workspace:"
  for entry in "${conflicts[@]}"; do
    IFS='|' read -r pid cwd cmd <<< "${entry}"
    echo "  pid=${pid} cwd=${cwd} cmd=${cmd}"
  done
  echo "[preflight] stop the conflicting cargo jobs and rerun fast_gate."
  exit 1
fi

echo "[preflight] no active workspace cargo jobs detected"
