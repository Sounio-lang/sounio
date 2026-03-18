#!/bin/bash
# check_sounio.sh - Validate Sounio code for Rust-isms and common errors
# Based on analysis of REAL Sounio code in tests/run-pass/

set -e

echo "🔍 Sounio Code Validator"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if files provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <file.sio> [file2.sio ...]"
    echo "       $0 *.sio (check all .sio files)"
    exit 1
fi

TOTAL_ERRORS=0
TOTAL_FILES=0
TOTAL_WARNINGS=0

for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}⚠  File not found: $file${NC}"
        continue
    fi
    
    echo -e "\n${BLUE}📄 Checking: $file${NC}"
    echo "----------------------------------------"
    
    FILE_ERRORS=0
    FILE_WARNINGS=0
    TOTAL_FILES=$((TOTAL_FILES + 1))
    
    # 1. Check for Rust methods (should use manual iteration)
    echo "1. Checking for Rust methods (should use manual iteration):"
    # Find method calls but exclude known Sounio patterns
    METHOD_ERRORS=0
    while IFS= read -r line; do
        LINE_NUM=$(echo "$line" | cut -d: -f1)
        LINE_CONTENT=$(echo "$line" | cut -d: -f2-)
        
        # Check if it's a method call (word.word())
        if echo "$LINE_CONTENT" | grep -q -E '\.\w+\('; then
            # Exclude allowed patterns
            if ! echo "$LINE_CONTENT" | grep -q -E '\.(println|print|assert|as\s+|ε|value|epsilon|validity|provenance)\('; then
                echo -e "  ${RED}Line $LINE_NUM: $LINE_CONTENT${NC}"
                METHOD_ERRORS=$((METHOD_ERRORS + 1))
            fi
        fi
    done < <(grep -n -E '\.\w+\(' "$file" 2>/dev/null || true)
    
    if [ $METHOD_ERRORS -gt 0 ]; then
        echo -e "${RED}❌ Found $METHOD_ERRORS Rust-like method call(s)${NC}"
        FILE_ERRORS=$((FILE_ERRORS + METHOD_ERRORS))
    else
        echo -e "${GREEN}✅ No Rust-like method calls found${NC}"
    fi
    
    # 2. Check for &mut (should be &!)
    echo -e "\n2. Checking for &mut (should be &!):"
    MUT_COUNT=$(grep -c "&mut" "$file" 2>/dev/null || true)
    if [ $MUT_COUNT -gt 0 ]; then
        echo -e "${RED}❌ Found &mut (should be &!):${NC}"
        grep -n "&mut" "$file" | while read -r line; do
            echo -e "  ${RED}$line${NC}"
        done
        FILE_ERRORS=$((FILE_ERRORS + MUT_COUNT))
    else
        echo -e "${GREEN}✅ No &mut found${NC}"
    fi
    
    # 3. Check for Vec (should be fixed arrays)
    echo -e "\n3. Checking for Vec (should be fixed arrays):"
    VEC_COUNT=$(grep -c "\\bVec\\b" "$file" 2>/dev/null || true)
    if [ $VEC_COUNT -gt 0 ]; then
        echo -e "${RED}❌ Found Vec (should be [Type; Size]):${NC}"
        grep -n "\\bVec\\b" "$file" | while read -r line; do
            echo -e "  ${RED}$line${NC}"
        done
        FILE_ERRORS=$((FILE_ERRORS + VEC_COUNT))
    else
        echo -e "${GREEN}✅ No Vec found${NC}"
    fi
    
    # 4. Check for semicolons (except in array literals and extern "C" blocks)
    echo -e "\n4. Checking for semicolons (should be none):"
    SEMI_COUNT=0
    IN_EXTERN_C=0
    LINE_NUM=0
    while IFS= read -r LINE_CONTENT; do
        LINE_NUM=$((LINE_NUM + 1))

        # Remove comments for checking
        CLEAN_CONTENT=$(echo "$LINE_CONTENT" | sed 's#//.*##')

        # Track extern "C" { } blocks — semicolons inside are valid FFI declarations
        if echo "$CLEAN_CONTENT" | grep -q 'extern[[:space:]]*"C"'; then
            IN_EXTERN_C=1
        fi
        if [ $IN_EXTERN_C -eq 1 ] && echo "$CLEAN_CONTENT" | grep -q '^[[:space:]]*}'; then
            IN_EXTERN_C=0
            continue
        fi
        if [ $IN_EXTERN_C -eq 1 ]; then
            continue
        fi

        # Only process lines that actually contain a semicolon
        if ! echo "$CLEAN_CONTENT" | grep -q ';'; then
            continue
        fi

        # Check if it's an array literal [x; y] pattern
        # Match [ followed by anything, then ;, then anything, then ]
        if echo "$CLEAN_CONTENT" | grep -q '\[[^]]*;[^]]*\]'; then
            # Array literal with semicolon - this is OK
            :
        else
            echo -e "  ${RED}Line $LINE_NUM: $LINE_CONTENT${NC}"
            SEMI_COUNT=$((SEMI_COUNT + 1))
        fi
    done < "$file"
    
    if [ $SEMI_COUNT -gt 0 ]; then
        echo -e "${RED}❌ Found $SEMI_COUNT unnecessary semicolon(s)${NC}"
        FILE_ERRORS=$((FILE_ERRORS + SEMI_COUNT))
    else
        echo -e "${GREEN}✅ No unnecessary semicolons${NC}"
    fi
    
    # 5. Check for missing effects on functions
    echo -e "\n5. Checking function effects:"
    # Get all function signatures
    FUNCTIONS=$(grep -n "^fn " "$file" 2>/dev/null || true)
    if [ -n "$FUNCTIONS" ]; then
        echo "Found functions:"
        echo "$FUNCTIONS" | sed 's/^/  /'
        
        while IFS= read -r func_line; do
            LINE_NUM=$(echo "$func_line" | awk -F: '{print $1}')
            FUNC_SIG=$(echo "$func_line" | cut -d: -f2-)
            
            # Get function name
            FUNC_NAME=$(echo "$FUNC_SIG" | sed 's/^fn //' | awk '{print $1}' | sed 's/(.*//')
            
            # Check if function has &! parameter
            if echo "$FUNC_SIG" | grep -q "&!"; then
                # Check if it has Mut effect
                if ! echo "$FUNC_SIG" | grep -q "with.*Mut"; then
                    echo -e "${YELLOW}⚠  Function '$FUNC_NAME' has &! parameter but may need 'with Mut' effect${NC}"
                    FILE_WARNINGS=$((FILE_WARNINGS + 1))
                fi
            fi
            
        done < <(echo "$FUNCTIONS")

        # Scan function bodies for division without 'with Div' effect
        while IFS= read -r div_fn; do
            DIV_FUNC_NAME=$(echo "$div_fn" | sed 's/^.*fn //' | sed 's/(.*//' | xargs)
            echo -e "${YELLOW}⚠  Function '$DIV_FUNC_NAME' divides in body but may need 'with Div' effect${NC}"
            FILE_WARNINGS=$((FILE_WARNINGS + 1))
        done < <(awk '{
            if ($0 ~ /^fn / || $0 ~ /^pub fn /) {
                fn_sig=$0; body=""; in_fn=1
            } else if (in_fn) {
                body = body " " $0
            }
            if ($0 ~ /^}/ && in_fn) {
                if (body ~ /[[:alnum:]_][[:space:]]*\/[[:space:]]*[[:alnum:]_]/ && fn_sig !~ /with.*Div/)
                    print fn_sig
                in_fn=0
            }
        }' "$file" 2>/dev/null || true)
    else
        echo -e "${GREEN}✅ No functions found${NC}"
    fi
    
    # 6. Check for explicit returns (Sounio requires them)
    echo -e "\n6. Checking for return statements:"
    RETURN_COUNT=$(grep -c "return " "$file" 2>/dev/null || true)
    FUNC_COUNT=$(grep -c "^fn " "$file" 2>/dev/null || true)
    
    if [ $FUNC_COUNT -gt 0 ]; then
        if [ $RETURN_COUNT -eq 0 ]; then
            echo -e "${YELLOW}⚠  No 'return' statements found in $FUNC_COUNT functions${NC}"
            echo "   (Sounio requires explicit returns for non-void functions)"
            FILE_WARNINGS=$((FILE_WARNINGS + 1))
        else
            echo -e "${GREEN}✅ Found $RETURN_COUNT return statements${NC}"
        fi
    else
        echo -e "${GREEN}✅ No functions to check${NC}"
    fi
    
    # 7. Check for proper Knowledge type usage
    echo -e "\n7. Checking Knowledge type patterns:"
    KNOWLEDGE_COUNT=$(grep -c "Knowledge\[" "$file" 2>/dev/null || true)
    if [ $KNOWLEDGE_COUNT -gt 0 ]; then
        echo -e "${BLUE}ℹ  Found $KNOWLEDGE_COUNT Knowledge type references${NC}"
        
        # Check for Knowledge constructor patterns
        if grep -q "Knowledge(" "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ Using Knowledge constructor syntax${NC}"
        fi
        
        # Check for .ε field access
        if grep -q "\.ε" "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ Using .ε field access for confidence${NC}"
        fi
    else
        echo -e "${GREEN}✅ No Knowledge types found (optional)${NC}"
    fi
    
    # Summary for this file
    echo -e "\n📊 File Summary: $file"
    if [ $FILE_ERRORS -eq 0 ]; then
        echo -e "${GREEN}✅ PASS: No critical errors found${NC}"
    else
        echo -e "${RED}❌ FAIL: $FILE_ERRORS critical error(s) found${NC}"
        TOTAL_ERRORS=$((TOTAL_ERRORS + FILE_ERRORS))
    fi
    
    if [ $FILE_WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠  WARNINGS: $FILE_WARNINGS warning(s)${NC}"
        TOTAL_WARNINGS=$((TOTAL_WARNINGS + FILE_WARNINGS))
    fi
    
    echo "----------------------------------------"

done

# Final summary
echo -e "\n📈 VALIDATION COMPLETE"
echo "======================="
echo "Files checked: $TOTAL_FILES"
echo "Critical errors: $TOTAL_ERRORS"
echo "Warnings: $TOTAL_WARNINGS"

if [ $TOTAL_ERRORS -eq 0 ]; then
    if [ $TOTAL_WARNINGS -eq 0 ]; then
        echo -e "${GREEN}🎉 All files passed validation with no warnings!${NC}"
    else
        echo -e "${YELLOW}⚠  All files passed but have $TOTAL_WARNINGS warning(s) to review${NC}"
    fi
    echo -e "\n${GREEN}Common fixes for warnings:${NC}"
    echo "  • Add 'with Mut' effect for functions with &! parameters"
    echo "  • Add 'with Div' effect for functions with division"
    echo "  • Ensure all non-void functions have explicit return statements"
    exit 0
else
    echo -e "${RED}⚠  $TOTAL_ERRORS critical error(s) need fixing${NC}"
    echo -e "\n${RED}Common fixes for errors:${NC}"
    echo "  • Change &mut to &!"
    echo "  • Remove .len(), .push(), etc. (use manual iteration)"
    echo "  • Use fixed arrays [Type; Size] instead of Vec"
    echo "  • Remove unnecessary semicolons (except in array literals [x; y])"
    echo "  • Add 'with Mut' effect for functions with &! parameters"
    exit 1
fi
