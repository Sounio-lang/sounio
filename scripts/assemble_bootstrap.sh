#!/bin/bash
# Assembles the single-file zero-import bootstrap compiler
# Output: self-hosted/bootstrap/bootstrap_v0.sio

set -e
SOUNIO=self-hosted
OUT="self-hosted/bootstrap/bootstrap_v0.sio"
mkdir -p self-hosted/bootstrap

# Helper: strip module/use declarations from a file
strip_file() {
    local file="$1"
    local label="$2"
    echo ""
    echo "// ============================================================"
    echo "// Inlined from: $file"
    echo "// ============================================================"
    grep -v "^module \|^use " "$file" | grep -v "^$" | head -9999
}

{
    cat << 'EOF'
// Bootstrap compiler v0 — zero-import single-file Sounio compiler
// Assembles: lexer + parser + AST + IR types + CodeBuffer + lean driver
// Usage: souc run self-hosted/bootstrap/bootstrap_v0.sio -- <source.sio> <output.elf>
// JIT memory: ~3MB (zero imports = no Rust module loading overhead)
EOF

    strip_file "$SOUNIO/lexer/span.sio"
    strip_file "$SOUNIO/lexer/token.sio"
    strip_file "$SOUNIO/lexer/errors.sio"
    strip_file "$SOUNIO/lexer/reader.sio"
    strip_file "$SOUNIO/lexer/numparse.sio"
    strip_file "$SOUNIO/lexer/tables.sio"
    strip_file "$SOUNIO/lexer/cursor.sio"
    strip_file "$SOUNIO/lexer/mod.sio"
    strip_file "$SOUNIO/parser/ast.sio"
    strip_file "$SOUNIO/parser/parser.sio"
    strip_file "$SOUNIO/parser/types.sio"
    strip_file "$SOUNIO/parser/patterns.sio"
    strip_file "$SOUNIO/parser/stmts.sio"
    strip_file "$SOUNIO/parser/exprs.sio"
    strip_file "$SOUNIO/parser/items.sio"
    strip_file "$SOUNIO/parser/recovery.sio"
    strip_file "$SOUNIO/parser/mod.sio"

} > "$OUT"

echo "Assembled bootstrap_v0.sio: $(wc -l < $OUT) lines"
