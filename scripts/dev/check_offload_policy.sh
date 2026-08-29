#!/usr/bin/env bash
# scripts/dev/check_offload_policy.sh — opt-in pre-commit / CI gate for the
# LLM-offload review policy in .claude/AGENT_OFFLOAD_POLICY.md.
#
# Verifies that any staged changes touching protected paths have a matching
# evidence row in .claude/llm_offload_log.md (date == today, target name
# substring-matched against staged file basenames).
#
# Usage:
#   scripts/dev/check_offload_policy.sh                 # check staged changes
#   scripts/dev/check_offload_policy.sh --files a.sio   # check explicit files
#   scripts/dev/check_offload_policy.sh --install       # install as git pre-commit hook
#   scripts/dev/check_offload_policy.sh --uninstall     # remove the hook
#
# Exit codes:
#   0  policy satisfied (or no protected files touched)
#   1  protected files touched without offload evidence
#   2  invocation error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$ROOT/.claude/llm_offload_log.md"
HOOK_DEST="$ROOT/.git/hooks/pre-commit"

# --- Protected path patterns (extended regex against `git diff --name-only`) ---
MATH_REVIEW_PATHS='^(stdlib/epistemic/(knightian|composed_effects|knowledge|gum|covariance|multivariate|sobol|aleatoric)\.sio|stdlib/clinical/.*\.sio|formal/lean4/Sounio.*\.lean)$'
CLINICAL_REVIEW_PATHS='^(stdlib/clinical/.*\.sio|tests/run-pass/vancomycin.*\.sio|tests/stdlib/clinical/.*\.sio|formal/lean4/SounioVancomycin.*\.lean)$'
EXTERNAL_FACING_PATHS='^(docs/papers/.*|docs/dissertation/.*|docs/research/irb_protocol_draft\.md)$'

cmd="${1:-}"
case "$cmd" in
    --install)
        cat > "$HOOK_DEST" <<EOF
#!/usr/bin/env bash
# Auto-installed by scripts/dev/check_offload_policy.sh --install
exec "$ROOT/scripts/dev/check_offload_policy.sh"
EOF
        chmod +x "$HOOK_DEST"
        echo "Installed: $HOOK_DEST -> $0"
        exit 0
        ;;
    --uninstall)
        rm -f "$HOOK_DEST"
        echo "Removed: $HOOK_DEST"
        exit 0
        ;;
    --files)
        shift
        files=("$@")
        ;;
    "")
        mapfile -t files < <(git -C "$ROOT" diff --cached --name-only --diff-filter=ACMR)
        ;;
    -h|--help)
        sed -n '2,20p' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: unknown arg '$cmd'" >&2
        exit 2
        ;;
esac

if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

# Today's date in the log format the policy uses
TODAY="$(date +%Y-%m-%d)"

violations=()
for f in "${files[@]}"; do
    [[ -z "$f" ]] && continue

    needs=""
    if [[ "$f" =~ $MATH_REVIEW_PATHS ]]; then needs="${needs}math-review,"; fi
    if [[ "$f" =~ $CLINICAL_REVIEW_PATHS ]]; then needs="${needs}review,"; fi
    if [[ "$f" =~ $EXTERNAL_FACING_PATHS ]]; then needs="${needs}fan-out,"; fi
    [[ -z "$needs" ]] && continue

    base="$(basename "$f")"
    # Look for a today-dated row whose Target column mentions the basename.
    if grep -qE "^\| *$TODAY *\|.*$base" "$LOG" 2>/dev/null; then
        continue
    fi
    violations+=("$f (needs: ${needs%,})")
done

if [[ ${#violations[@]} -gt 0 ]]; then
    echo "OFFLOAD POLICY VIOLATION (.claude/AGENT_OFFLOAD_POLICY.md):"
    echo ""
    for v in "${violations[@]}"; do
        echo "  - $v"
    done
    echo ""
    echo "Fix:"
    echo "  1. Run the appropriate offload (math-review / review / fan-out)"
    echo "  2. Append a row to .claude/llm_offload_log.md (date $TODAY, Target column"
    echo "     contains the file basename)"
    echo "  3. Re-stage and commit"
    echo ""
    echo "  See .claude/AGENT_OFFLOAD_POLICY.md for the per-trigger commands."
    echo ""
    echo "To bypass once with rationale: append to .claude/llm_offload_log.md"
    echo "  with Outcome 'WAIVED' and Note column explaining why."
    exit 1
fi

exit 0
