#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "==> bootstrap_workspace: starting"
echo "==> repo root: $(pwd)"
echo "==> user: $(whoami)"

if [[ -d .git ]]; then
  echo "==> git repo detected"
  git status --short || true
else
  echo "ERROR: /workspace is not a git repository"
  exit 1
fi

mkdir -p /workspace/artifacts
mkdir -p /workspace/.sounio-state

chmod +x scripts/*.sh 2>/dev/null || true
chmod +x tools/*.sh 2>/dev/null || true
chmod +x dev/*.sh 2>/dev/null || true
chmod +x dist/*.sh 2>/dev/null || true

echo "==> bootstrap_workspace: done"
