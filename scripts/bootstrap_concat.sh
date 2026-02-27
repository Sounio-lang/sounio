#!/usr/bin/env bash
#
# bootstrap_concat.sh
#
# Concatenates the self-hosted compiler's core pipeline .sio files in
# leaf-first dependency order into a single file, then runs the pinned
# binary checker against it.
#
# Sounio has NO forward references -- helpers must precede callers.
# The ordering below respects that invariant.

set -euo pipefail

# ── Paths (relative to repo root) ──────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PINNED_BIN="artifacts/omega/souc-bin/souc-linux-x86_64"
OUTPUT_DIR="build"
OUTPUT_FILE="$OUTPUT_DIR/bootstrap_stage1.sio"

# ── Dependency-ordered file list (leaf-first) ──────────────────────
FILES=(
  # 1. Leaf modules (no cross-file deps)
  self-hosted/intern.sio
  self-hosted/lexer/token.sio
  self-hosted/lexer/span.sio

  # 2. Lexer layer
  self-hosted/lexer/cursor.sio
  self-hosted/lexer/errors.sio
  self-hosted/lexer/tables.sio
  self-hosted/lexer/numparse.sio
  self-hosted/lexer/reader.sio
  self-hosted/lexer/mod.sio

  # 3. Parser layer
  self-hosted/parser/ast.sio
  self-hosted/parser/parser.sio
  self-hosted/parser/exprs.sio
  self-hosted/parser/items.sio
  self-hosted/parser/stmts.sio
  self-hosted/parser/types.sio
  self-hosted/parser/patterns.sio
  self-hosted/parser/mod.sio

  # 4. Resolve layer
  self-hosted/resolve/scope.sio
  self-hosted/resolve/symbol.sio
  self-hosted/resolve/imports.sio
  self-hosted/resolve/resolve.sio
  self-hosted/resolve/mod.sio

  # 5. Check layer
  self-hosted/check/types.sio
  self-hosted/check/defs.sio
  self-hosted/check/env.sio
  self-hosted/check/borrow.sio
  self-hosted/check/units.sio
  self-hosted/check/epistemic.sio
  self-hosted/check/hyper.sio
  self-hosted/check/effects.sio
  self-hosted/check/refinement.sio
  self-hosted/check/compat.sio
  self-hosted/check/check.sio
  self-hosted/check/mod.sio

  # 6. Entry point (bootstrap-trimmed: lex/parse/resolve/check only)
  self-hosted/main_bootstrap.sio
)

# ── Verify all source files exist ──────────────────────────────────
echo "=== Bootstrap Concat: verifying source files ==="
MISSING=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  MISSING: $f"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo ""
  echo "ABORT: $MISSING file(s) missing. Cannot proceed."
  exit 1
fi
echo "  All ${#FILES[@]} files present."

# ── Verify pinned binary exists and is executable ──────────────────
if [ ! -f "$PINNED_BIN" ]; then
  echo "ABORT: Pinned binary not found at $PINNED_BIN"
  exit 1
fi
if [ ! -x "$PINNED_BIN" ]; then
  echo "WARN: Pinned binary not executable, setting +x"
  chmod +x "$PINNED_BIN"
fi

# ── Create output directory ────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Concatenate in dependency order ────────────────────────────────
echo ""
echo "=== Concatenating ${#FILES[@]} files -> $OUTPUT_FILE ==="

TOTAL_LINES=0

# Start fresh
> "$OUTPUT_FILE"

for f in "${FILES[@]}"; do
  LINES=$(wc -l < "$f")
  TOTAL_LINES=$((TOTAL_LINES + LINES))

  {
    echo ""
    echo "// ════════════════════════════════════════════════════════════════"
    echo "// SOURCE: $f  ($LINES lines)"
    echo "// ════════════════════════════════════════════════════════════════"
    echo ""
    cat "$f"
  } >> "$OUTPUT_FILE"

  printf "  %-45s %6d lines\n" "$f" "$LINES"
done

echo ""
echo "  Total: $TOTAL_LINES source lines -> $OUTPUT_FILE"
OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
echo "  Output file: $OUTPUT_LINES lines (includes separator comments)"

# ── Run the pinned binary checker ──────────────────────────────────
echo ""
echo "=== Running pinned checker: $PINNED_BIN check $OUTPUT_FILE ==="
echo ""

CHECKER_EXIT=0
"$PINNED_BIN" check "$OUTPUT_FILE" 2>&1 || CHECKER_EXIT=$?

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ "$CHECKER_EXIT" -eq 0 ]; then
  echo "  RESULT: PASS  (exit code 0)"
else
  echo "  RESULT: FAIL  (exit code $CHECKER_EXIT)"
fi
echo "  Files concatenated: ${#FILES[@]}"
echo "  Source lines:        $TOTAL_LINES"
echo "  Output:              $OUTPUT_FILE"
echo "════════════════════════════════════════════════════════════════════"

exit "$CHECKER_EXIT"
