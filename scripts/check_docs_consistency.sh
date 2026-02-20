#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail=0

check_no_match() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if rg -n "$pattern" "$file" >/tmp/sounio_doc_check_match.txt 2>/dev/null; then
    echo "$message"
    cat /tmp/sounio_doc_check_match.txt
    fail=1
  fi
}

check_no_match 'cd[[:space:]]+compiler' "tests/README.md" \
  "tests/README.md contains stale 'cd compiler' commands:"

check_no_match '0\.100\.0' "INSTALL.md" \
  "INSTALL.md contains stale version expectation 0.100.0:"

check_no_match 'Darwin RAG\+\+|FastAPI|api/routers|services/settings\.py' "docs/codebase_overview.md" \
  "docs/codebase_overview.md appears to describe a different project:"

check_no_match 'Astro Starter Kit: Minimal|npm create astro@latest -- --template minimal|Seasoned astronaut\?' "website/README.md" \
  "website/README.md still contains starter template content:"

check_no_match 'cd[[:space:]]+compiler' "docs/qnn/PERFORMANCE_HANDBOOK.md" \
  "docs/qnn/PERFORMANCE_HANDBOOK.md contains stale 'cd compiler' commands:"

check_no_match '0\.100\.0' "docs/compiler/COMPILER_ARCHITECTURE_OVERVIEW.md" \
  "docs/compiler/COMPILER_ARCHITECTURE_OVERVIEW.md contains stale version expectation 0.100.0:"

check_no_match 'cd[[:space:]]+compiler' "docs/QNN_PERFORMANCE_GUIDE.md" \
  "docs/QNN_PERFORMANCE_GUIDE.md contains stale 'cd compiler' commands:"

check_no_match 'cd[[:space:]]+compiler' "docs/GLM_4.7_INTEGRATION.md" \
  "docs/GLM_4.7_INTEGRATION.md contains stale 'cd compiler' commands:"

check_no_match 'GLM_API_KEY="[^"]{20,}"' "docs/GLM_4.7_INTEGRATION.md" \
  "docs/GLM_4.7_INTEGRATION.md appears to include a non-placeholder API key example:"

check_no_match '\(~400 tests\)|\(3800\+ tests' "INSTALL.md" \
  "INSTALL.md contains stale hard-coded test suite counts:"

workspace_version="$(awk '
  /^\[workspace\.package\]/ { in_pkg=1; next }
  /^\[/ && in_pkg { in_pkg=0 }
  in_pkg && /version[[:space:]]*=/ {
    gsub(/.*"/, "", $0); gsub(/".*/, "", $0); print; exit
  }' Cargo.toml)"
readme_version="$(sed -n 's/^\*\*Current Version\*\*:[[:space:]]*//p' README.md | head -n1)"

if [[ -n "$workspace_version" && -n "$readme_version" && "$workspace_version" != "$readme_version" ]]; then
  echo "README version mismatch: README='$readme_version' Cargo.workspace='$workspace_version'"
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  rm -f /tmp/sounio_doc_check_match.txt
  exit 1
fi

rm -f /tmp/sounio_doc_check_match.txt
echo "Documentation consistency check passed."
