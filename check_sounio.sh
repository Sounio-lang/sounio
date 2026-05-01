#!/usr/bin/env bash
# check_sounio.sh — top-level shim
#
# The canonical implementation lives at scripts/dev/check_sounio.sh after the
# v2.0.0 worktree cleanup (commit 6eedd8fe). This shim preserves the legacy
# `bash check_sounio.sh ...` invocation used by .github/workflows/ci.yml
# (Sounio Lint job) and any external tooling that still references the old
# top-level path.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT_DIR/scripts/dev/check_sounio.sh" "$@"
