#!/usr/bin/env bash
#
# Build and run a narrow bootstrap-safe epistemic test bundle.
#
# This keeps the flat bundle small enough for the pinned seed compiler while
# still exercising the Policy/Contest/Robust checker boundary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PINNED_BIN="${PINNED_BIN:-bin/souc}"
OUTPUT_DIR="build"
OUTPUT_FILE="$OUTPUT_DIR/bootstrap_knowledge_tests.sio"
ARTIFACT_DIR="artifacts/omega"
CHECK_LOG="$ARTIFACT_DIR/bootstrap_knowledge_check.log"
RUN_LOG="$ARTIFACT_DIR/bootstrap_knowledge_run.log"
SUMMARY_FILE="$ARTIFACT_DIR/bootstrap_knowledge_tests.v1.json"
STACK_KB="${STACK_KB:-65536}"

FILES=(
  self-hosted/intern.sio
  self-hosted/lexer/token.sio
  self-hosted/bootstrap_knowledge_shims.sio
  self-hosted/lexer/span.sio
  self-hosted/lexer/cursor.sio
  self-hosted/lexer/errors.sio
  self-hosted/lexer/tables.sio
  self-hosted/lexer/numparse.sio
  self-hosted/lexer/reader.sio
  self-hosted/lexer/mod.sio
  self-hosted/parser/ast.sio
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
  self-hosted/check/ontology_side_table_cache.sio
  self-hosted/check/check.sio
  self-hosted/ir/ir.sio
  self-hosted/check/mod.sio
  self-hosted/test_knowledge_bootstrap.sio
)

mkdir -p "$OUTPUT_DIR" "$ARTIFACT_DIR"

for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ABORT: missing $f"
    exit 1
  fi
done

if [ ! -x "$PINNED_BIN" ]; then
  JIT_FALLBACK="bin/souc"
  if [ -x "$JIT_FALLBACK" ]; then
    echo "knowledge_bootstrap: pinned binary not found at $PINNED_BIN, falling back to $JIT_FALLBACK"
    PINNED_BIN="$JIT_FALLBACK"
  else
    echo "ABORT: no usable souc binary found (checked $PINNED_BIN and $JIT_FALLBACK)"
    exit 1
  fi
fi

TOTAL_LINES=0
STRIPPED_DECL_LINES=0
> "$OUTPUT_FILE"

echo "=== Bootstrap knowledge bundle ==="
for f in "${FILES[@]}"; do
  LINES=$(wc -l < "$f")
  TOTAL_LINES=$((TOTAL_LINES + LINES))
  DECL_LINES=$(grep -Ec '^[[:space:]]*(module[[:space:]]+[A-Za-z0-9_:]+[[:space:]]*;?|use[[:space:]].*)[[:space:]]*$' "$f" || true)
  STRIPPED_DECL_LINES=$((STRIPPED_DECL_LINES + DECL_LINES))

  {
    echo ""
    echo "// SOURCE: $f"
    echo ""
    sed \
      -e '/^[[:space:]]*module[[:space:]]\+[A-Za-z0-9_:]\+[[:space:]]*;*[[:space:]]*$/d' \
      -e '/^[[:space:]]*use[[:space:]].*$/d' \
      "$f" |
      if [ "$f" = "self-hosted/check/mod.sio" ]; then
        # Keep this focused Knowledge bootstrap gate off unrelated Wave 10
        # hypercomplex law-profile probes. Those probes depend on a broader
        # IR metadata surface than this narrow bundle is meant to validate.
        awk '
          /^fn check_hyper_expr_law_profile_probe\(/ { skip = 1 }
          /^fn check_program\(/ { skip = 0 }
          skip == 0 { print }
        '
      else
        cat
      fi
  } >> "$OUTPUT_FILE"

  if [ "$DECL_LINES" -gt 0 ]; then
    printf "  %-40s %6d lines  (stripped %d decls)\n" "$f" "$LINES" "$DECL_LINES"
  else
    printf "  %-40s %6d lines\n" "$f" "$LINES"
  fi
done

cat >> "$OUTPUT_FILE" <<'EOF'

fn main() with IO, Mut, Panic, Div, Alloc {
    let code = run_knowledge_bootstrap_tests()
    if code != 0 {
        panic("knowledge bootstrap tests failed")
    }
}
EOF

CHECK_EXIT=0
RUN_EXIT=125

echo ""
echo "=== Check: $PINNED_BIN check $OUTPUT_FILE ==="
set +e
( ulimit -s "$STACK_KB" 2>/dev/null || true
  "$PINNED_BIN" check "$OUTPUT_FILE" ) 2>&1 | tee "$CHECK_LOG"
CHECK_EXIT="${PIPESTATUS[0]}"
set -e

if [ "$CHECK_EXIT" -eq 0 ]; then
  echo ""
  echo "=== Run: $PINNED_BIN run $OUTPUT_FILE ==="
  set +e
  ( ulimit -s "$STACK_KB" 2>/dev/null || true
    "$PINNED_BIN" run "$OUTPUT_FILE" ) 2>&1 | tee "$RUN_LOG"
  RUN_EXIT="${PIPESTATUS[0]}"
  set -e
else
  echo "=== Run skipped: check failed ===" | tee "$RUN_LOG"
fi

cat > "$SUMMARY_FILE" <<EOF
{
  "schema": "sounio.bootstrap.knowledge-tests.v1",
  "generated_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "output_file": "$OUTPUT_FILE",
  "file_count": ${#FILES[@]},
  "source_lines": $TOTAL_LINES,
  "stripped_decl_lines": $STRIPPED_DECL_LINES,
  "check": {
    "status": "$(if [ "$CHECK_EXIT" -eq 0 ]; then echo pass; else echo fail; fi)",
    "exit_code": $CHECK_EXIT,
    "log": "$CHECK_LOG"
  },
  "run": {
    "status": "$(if [ "$CHECK_EXIT" -ne 0 ]; then echo not_run; elif [ "$RUN_EXIT" -eq 0 ]; then echo pass; else echo fail; fi)",
    "exit_code": $RUN_EXIT,
    "log": "$RUN_LOG"
  }
}
EOF

echo ""
echo "=== Summary ==="
echo "  output:                 $OUTPUT_FILE"
echo "  files:                  ${#FILES[@]}"
echo "  source lines:           $TOTAL_LINES"
echo "  stripped module/use:    $STRIPPED_DECL_LINES"
echo "  check exit:             $CHECK_EXIT"
echo "  run exit:               $RUN_EXIT"
echo "  summary:                $SUMMARY_FILE"

if [ "$CHECK_EXIT" -ne 0 ]; then
  exit "$CHECK_EXIT"
fi

exit "$RUN_EXIT"
