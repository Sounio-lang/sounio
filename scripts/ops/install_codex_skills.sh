#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
DEST_DIR="$CODEX_HOME/skills"

mkdir -p "$DEST_DIR"

shopt -s nullglob
for skill_dir in "$ROOT_DIR"/skills/*; do
  if [[ ! -d "$skill_dir" ]]; then
    continue
  fi

  skill_name="$(basename "$skill_dir")"
  dest="$DEST_DIR/$skill_name"

  if [[ -e "$dest" && ! -L "$dest" ]]; then
    echo "[install-codex-skills] skip: $dest exists and is not a symlink"
    continue
  fi

  ln -sfn "$skill_dir" "$dest"
  echo "[install-codex-skills] linked: $skill_name -> $dest"
done

echo "[install-codex-skills] done"
echo "[install-codex-skills] Restart Codex to pick up new skills."

