#!/usr/bin/env bash
# prepare_corpus.sh — Collect all .sio files into a single corpus for LLM fine-tuning.
#
# Usage: bash docs/training/finetune/prepare_corpus.sh
# Output: docs/training/finetune/sounio_corpus.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT="$REPO_ROOT/docs/training/finetune/sounio_corpus.txt"

echo "=== Sounio Corpus Builder ==="
echo "Repo root: $REPO_ROOT"
echo ""

# Directories to scan (relative to repo root)
DIRS=(stdlib tests examples benchmarks)
INCLUDE_ONTOLOGY="${LORA_CORPUS_INCLUDE_ONTOLOGY:-0}"

# Files deliberately left out of the corpus. Each carries a literal-suffix form
# (0_i64, 10.0_f64, .sum::<f64>()) that Madaros rejects; keeping them would teach
# the very syntax the rest of the corpus avoids. validate_lora_assets.py reads this
# list and fails if any of these files has a section in the corpus.
EXCLUDE_FILES=(examples/test_i64_bug_fix.sio stdlib/pbpk/regulatory.sio)

> "$OUTPUT"  # truncate

FILE_COUNT=0
TOTAL_LINES=0

for dir in "${DIRS[@]}"; do
    SEARCH_DIR="$REPO_ROOT/$dir"
    if [ ! -d "$SEARCH_DIR" ]; then
        echo "  [skip] $dir/ (not found)"
        continue
    fi

    while IFS= read -r -d '' sio_file; do
        REL_PATH="${sio_file#$REPO_ROOT/}"
        if [[ "$INCLUDE_ONTOLOGY" != "1" && "$REL_PATH" == stdlib/ontology/* ]]; then
            continue
        fi
        skip=0
        for excluded in "${EXCLUDE_FILES[@]}"; do
            [[ "$REL_PATH" == "$excluded" ]] && skip=1
        done
        if [[ "$skip" == 1 ]]; then
            continue
        fi
        LINE_COUNT=$(wc -l < "$sio_file")
        TOTAL_LINES=$((TOTAL_LINES + LINE_COUNT))
        FILE_COUNT=$((FILE_COUNT + 1))

        echo "// === FILE: $REL_PATH ===" >> "$OUTPUT"
        sed 's/[[:space:]]\+$//' "$sio_file" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
    done < <(find "$SEARCH_DIR" -name '*.sio' -type f -print0 | sort -z)
done

CORPUS_SIZE=$(du -h "$OUTPUT" | cut -f1)

echo ""
echo "=== Corpus Summary ==="
echo "  Files collected: $FILE_COUNT"
echo "  Total lines:     $TOTAL_LINES"
echo "  Corpus size:     $CORPUS_SIZE"
echo "  Output:          $OUTPUT"
echo ""
echo "Done."
