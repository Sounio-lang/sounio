#!/usr/bin/env bash
# Resolve the souc compiler binary. Prints the path and exits 0, or exits 1 with instructions.
set -euo pipefail

# 1. Explicit env vars
if [[ -n "${SOUC_BIN:-}" ]] && [[ -x "$SOUC_BIN" ]]; then
    echo "$SOUC_BIN"; exit 0
fi
if [[ -n "${SOUC:-}" ]] && [[ -x "$SOUC" ]]; then
    echo "$SOUC"; exit 0
fi

# 2. On PATH
if command -v souc >/dev/null 2>&1; then
    command -v souc; exit 0
fi

# 3. Common repo-relative location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATES=(
    "$SCRIPT_DIR/../bin/souc"
    "./bin/souc"
    "$SCRIPT_DIR/../../../artifacts/omega/souc-bin/souc-linux-x86_64"
    "./artifacts/omega/souc-bin/souc-linux-x86_64"
)
for c in "${CANDIDATES[@]}"; do
    if [[ -x "$c" ]]; then
        echo "$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"; exit 0
    fi
done

# 4. Not found
echo >&2 "ERROR: souc compiler not found."
echo >&2 "Set SOUC_BIN=/path/to/souc or add souc to PATH."
echo >&2 "Download from: https://github.com/sounio-lang/sounio/releases"
exit 1
