#!/usr/bin/env bash
# Compatibility entrypoint. Arrow acceptance is mandatory official C++26.
set -euo pipefail
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$dir/arrow_cpp_gate.sh" "$@"
