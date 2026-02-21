#!/bin/bash
SOUC=/home/demetrios/work/sounio/target/debug/souc
STDLIB=/home/demetrios/work/sounio/stdlib

for f in $(find "$STDLIB" -name "*.sio" | sort); do
    result=$("$SOUC" check "$f" 2>&1)
    if echo "$result" | grep -q "^error\|^Error"; then
        echo "$result" | grep "^error\|^Error" | head -1
    fi
done | sort | uniq -c | sort -rn | head -30

