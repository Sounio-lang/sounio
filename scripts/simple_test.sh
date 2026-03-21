#!/bin/bash

# Simple test of grep patterns

TEST_FILE="test_patterns.sio"
cat > "$TEST_FILE" << 'EOF'
// Test patterns
fn test() {
    // Should match: .first(), .last(), .iter().sum()
    let x = arr.first();
    let y = arr.last();
    let z = arr.iter().sum();
    
    // Should NOT match: .ε, .value, println(), print()
    let conf = knowledge.ε;
    let val = knowledge.value;
    println("test");
    print("test");
    
    // Should match: &mut
    fn bad(&mut x: i32) {}
    
    // Should match: Vec
    let v: Vec<i32> = Vec::new();
    
    // Should match: semicolon (not in array literal)
    let a = 10;
    
    // Should NOT match: semicolon in array literal
    let arr = [0; 10];
}
EOF

echo "Testing grep patterns..."
echo "======================="

echo "\n1. Testing method call pattern:"
METHOD_PATTERN='\.\w+\([^)]*\)'
echo "Pattern: $METHOD_PATTERN"
grep -n -E "$METHOD_PATTERN" "$TEST_FILE"

echo "\n2. Testing &mut pattern:"
grep -n "&mut" "$TEST_FILE"

echo "\n3. Testing Vec pattern:"
grep -n "\\bVec\\b" "$TEST_FILE"

echo "\n4. Testing semicolon pattern:"
grep -n ";" "$TEST_FILE"

echo "\n5. Testing array literal semicolon exclusion:"
while IFS= read -r line; do
    LINE_NUM=$(echo "$line" | cut -d: -f1)
    LINE_CONTENT=$(echo "$line" | cut -d: -f2-)
    if ! echo "$LINE_CONTENT" | grep -q "\\[[^]]*;[^]]*\\]"; then
        if echo "$LINE_CONTENT" | grep -q ";"; then
            echo "Line $LINE_NUM has non-array-literal semicolon: $LINE_CONTENT"
        fi
    else
        echo "Line $LINE_NUM has array literal semicolon (OK): $LINE_CONTENT"
    fi
done < <(grep -n ";" "$TEST_FILE")

rm "$TEST_FILE"