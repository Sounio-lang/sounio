#!/usr/bin/env bash
# Install Sounio language support for Claude Code into a target project.
# Usage: bash install.sh [target-directory]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-.}"

if [[ ! -d "$TARGET" ]]; then
    echo "Target directory '$TARGET' does not exist." >&2
    exit 1
fi

echo "Installing Sounio Claude Code extension into: $TARGET"

# 1. CLAUDE.md — append if exists, create if not
if [[ -f "$TARGET/CLAUDE.md" ]]; then
    echo ""
    echo "Found existing CLAUDE.md — appending Sounio section."
    echo "" >> "$TARGET/CLAUDE.md"
    echo "---" >> "$TARGET/CLAUDE.md"
    echo "" >> "$TARGET/CLAUDE.md"
    cat "$SRC/CLAUDE.md" >> "$TARGET/CLAUDE.md"
else
    cp "$SRC/CLAUDE.md" "$TARGET/CLAUDE.md"
fi

# 2. Skills
mkdir -p "$TARGET/skills"
cp -r "$SRC/skills/sounio-check" "$TARGET/skills/"
cp -r "$SRC/skills/sounio-run" "$TARGET/skills/"
cp -r "$SRC/skills/sounio-lint" "$TARGET/skills/"

# 3. Scripts
mkdir -p "$TARGET/scripts"
cp "$SRC/scripts/resolve-souc.sh" "$TARGET/scripts/"
cp "$SRC/scripts/sounio-lint.sh" "$TARGET/scripts/"
cp "$SRC/scripts/post-tool-use.sh" "$TARGET/scripts/"
chmod +x "$TARGET/scripts/resolve-souc.sh" "$TARGET/scripts/sounio-lint.sh" "$TARGET/scripts/post-tool-use.sh"

# 4. Settings (merge if exists)
if [[ -f "$TARGET/.claude/settings.json" ]]; then
    echo "Existing .claude/settings.json found — please merge manually:"
    echo "  Source: $SRC/.claude/settings.json"
else
    mkdir -p "$TARGET/.claude"
    cp "$SRC/.claude/settings.json" "$TARGET/.claude/settings.json"
fi

echo ""
echo "Done. Sounio Claude Code extension installed."
echo ""
echo "Next steps:"
echo "  1. Set your compiler path:  export SOUC_BIN=/path/to/souc"
echo "  2. Start Claude Code in your project directory"
echo "  3. Try: /sounio-check myfile.sio"
echo ""
echo "If you don't have souc, the syntax rules and lint still work — Claude"
echo "will write correct Sounio without the compiler. Type-checking is optional."
