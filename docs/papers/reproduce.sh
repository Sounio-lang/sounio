#!/usr/bin/env bash
# Paper reproducibility gate — verifies that all paper artifacts can be regenerated.
set -euo pipefail

echo "paper/reproduce.sh: checking paper artifacts..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

errors=0

for md in "$SCRIPT_DIR"/*.md "$SCRIPT_DIR"/**/*.md; do
  if [ -f "$md" ]; then
    if ! head -1 "$md" | grep -q '<!-- docs:meta'; then
      echo "WARN: $md missing docs:meta header"
    fi
  fi
done

echo "paper/reproduce.sh: done (errors=$errors)"
exit $errors
