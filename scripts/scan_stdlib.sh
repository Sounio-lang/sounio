#!/bin/bash
SOUC=/home/demetrios/work/sounio/target/debug/souc
STDLIB=/home/demetrios/work/sounio/stdlib
pass=0
fail=0

for f in $(find "$STDLIB" -name "*.sio" | sort); do
    result=$("$SOUC" check "$f" 2>&1)
    if echo "$result" | grep -q "^error\|^Error"; then
        fail=$((fail+1))
    else
        pass=$((pass+1))
    fi
done

echo "PASS:$pass FAIL:$fail"

