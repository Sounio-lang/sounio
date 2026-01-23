#!/bin/bash

# Analyze and classify all stdlib files with main() functions
# Outputs classification lists to /tmp/stdlib_main_analysis/

set -e

STDLIB_DIR="stdlib"
OUTPUT_DIR="/tmp/stdlib_main_analysis"

echo "Analyzing stdlib files with main() functions..."
mkdir -p "$OUTPUT_DIR"

# Find all files with main() at the start of a line
echo "Finding all stdlib files with main()..."
find "$STDLIB_DIR" -name "*.sio" -type f -exec grep -l "^fn main()" {} \; > "$OUTPUT_DIR/all_mains.txt"

TOTAL=$(wc -l < "$OUTPUT_DIR/all_mains.txt")
echo "Found $TOTAL files with main()"

# Classify into categories
echo "Classifying files..."

# Explicit test files (test_*.sio)
grep "test_" "$OUTPUT_DIR/all_mains.txt" > "$OUTPUT_DIR/explicit_tests.txt" || true
TESTS=$(wc -l < "$OUTPUT_DIR/explicit_tests.txt")

# Named examples (*_example.sio)
grep "_example\.sio" "$OUTPUT_DIR/all_mains.txt" > "$OUTPUT_DIR/named_examples.txt" || true
EXAMPLES=$(wc -l < "$OUTPUT_DIR/named_examples.txt")

# Module demos (everything else)
grep -v "test_" "$OUTPUT_DIR/all_mains.txt" | grep -v "_example\.sio" > "$OUTPUT_DIR/module_demos.txt" || true
DEMOS=$(wc -l < "$OUTPUT_DIR/module_demos.txt")

# Create a summary
cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
Stdlib main() Function Analysis
================================

Total files with main(): $TOTAL
  - Explicit tests (test_*.sio): $TESTS
  - Named examples (*_example.sio): $EXAMPLES
  - Module demos (rest): $DEMOS

Files by category:
  $OUTPUT_DIR/explicit_tests.txt
  $OUTPUT_DIR/named_examples.txt
  $OUTPUT_DIR/module_demos.txt

To view the transformation plan:
  python3 scripts/generate_transformation_manifest.py

Then execute:
  python3 scripts/extract_stdlib_mains.py
SUMMARY

cat "$OUTPUT_DIR/summary.txt"

echo ""
echo "✓ Analysis complete. Results in: $OUTPUT_DIR"
