#!/bin/bash
# List files with parse errors (first error line contains "Parse error" or "Expected")
SOUC=/home/demetrios/work/sounio/target/debug/souc
STDLIB=/home/demetrios/work/sounio/stdlib

for f in $(find "$STDLIB" -name "*.sio" | sort); do
    result=$("$SOUC" check "$f" 2>&1)
    if echo "$result" | grep -q "^error\|^Error"; then
        first=$(echo "$result" | grep "^error\|^Error" | head -1)
        detail=$(echo "$result" | grep "× Expected\|× Parse error\|× Unexpected" | head -1)
        if [ -n "$detail" ]; then
            echo "$f: $detail"
        fi
    fi
done | head -50

