#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORKFLOWS=()
while IFS= read -r path; do
  WORKFLOWS+=("$path")
done < <(find .github/workflows -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) | sort)

if [[ ${#WORKFLOWS[@]} -eq 0 ]]; then
  echo "No workflows found under .github/workflows"
  exit 0
fi

has_real_ripgrep() {
  command -v rg >/dev/null 2>&1 && rg --version 2>/dev/null | grep -qi '^ripgrep '
}

workflow_script_refs() {
  if has_real_ripgrep; then
    rg -No "(?:\\./)?scripts/[A-Za-z0-9_./*?-]+" "${WORKFLOWS[@]}"
  else
    grep -H -o -E "(\\./)?scripts/[A-Za-z0-9_./*?-]+" "${WORKFLOWS[@]}"
  fi
}

mapfile -t REFS < <(
  workflow_script_refs \
    | sed -E 's#^.+:##' \
    | sed -E 's#^[[:space:]]+##' \
    | sed -E 's#[\"\047),:;]+$##' \
    | sort -u
)

if [[ ${#REFS[@]} -eq 0 ]]; then
  echo "No scripts/* references found in workflows."
  exit 0
fi

missing=0
for ref in "${REFS[@]}"; do
  path="${ref#./}"
  # Workflow path filters contain globs, not executable path prefixes.
  # Expand patterns as data (never eval) and still reject an empty match set.
  matches=()
  if [[ "$path" == *'*'* || "$path" == *'?'* ]]; then
    mapfile -t matches < <(compgen -G "$path" || true)
  elif [[ -e "$path" ]]; then
    matches=("$path")
  fi
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Missing workflow script reference: $ref"
    missing=1
    continue
  fi

  # Check every matched shell target; globs cannot hide missing execute bits.
  for target in "${matches[@]}"; do
    if [[ "$target" == *.sh && ! -x "$target" ]]; then
      echo "Non-executable workflow shell target: $target (reference: $ref)"
      missing=1
    fi
  done
done

if [[ $missing -ne 0 ]]; then
  exit 1
fi

echo "Workflow script reference check passed (${#REFS[@]} references)."
