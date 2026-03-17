#!/usr/bin/env bash
# Appends the IR types, CodeBuffer, file I/O helpers, and lean driver
# to bootstrap_v0.sio to create a fully self-contained zero-import bootstrap.
set -e

OUT="self-hosted/bootstrap/bootstrap_v0.sio"
IR="self-hosted/ir/ir.sio"
ENC="self-hosted/native/encode.sio"
DRIVER="self-hosted/compiler/render_native_compile_driver_lean.sio"

echo "--- Appending IR types to ${OUT}"

cat >> "$OUT" <<'SECTION_SEPARATOR'

// ============================================================
// Minimal IR types (inlined from ir::ir — zero imports)
// ============================================================

let IR_INVALID_REG: i64 = -1
let IR_INVALID_LABEL: i64 = -1
let IR_INVALID_FN: i64 = -1
let IR_MAX_INSTRS: i64 = 128
let IR_MAX_PARAMS: i64 = 64

SECTION_SEPARATOR

# IrOpcode enum: lines 59-349
sed -n '59,349p' "$IR" >> "$OUT"
echo "" >> "$OUT"

# IrRegList struct (line 351-354), IrInstr struct (356-372), IrFunction struct (404-414)
sed -n '351,372p' "$IR" >> "$OUT"
echo "" >> "$OUT"
sed -n '404,414p' "$IR" >> "$OUT"
echo "" >> "$OUT"

# ir_empty_name, ir_name_eq, ir_name_from_bytes, make_name, ir_name_from_string_list,
# ir_name_is_some, ir_name_is_box, ir_name_is_none, ir_empty_reg_list, ir_reg_list_single
sed -n '1592,1594p' "$IR" >> "$OUT"   # ir_empty_name
echo "" >> "$OUT"
sed -n '1600,1612p' "$IR" >> "$OUT"   # ir_name_eq
echo "" >> "$OUT"
sed -n '1614,1621p' "$IR" >> "$OUT"   # ir_name_from_bytes
echo "" >> "$OUT"
sed -n '1623,1632p' "$IR" >> "$OUT"   # make_name
echo "" >> "$OUT"
sed -n '1634,1642p' "$IR" >> "$OUT"   # ir_name_from_string_list
echo "" >> "$OUT"
sed -n '1671,1683p' "$IR" >> "$OUT"   # ir_name_is_some + ir_name_is_none
echo "" >> "$OUT"
sed -n '1685,1691p' "$IR" >> "$OUT"   # ir_empty_reg_list + ir_reg_list_single
echo "" >> "$OUT"

echo "--- Appending CodeBuffer to ${OUT}"

cat >> "$OUT" <<'SECTION_SEPARATOR'
// ============================================================
// CodeBuffer (inlined from native::encode — zero imports)
// ============================================================

SECTION_SEPARATOR

sed -n '8,63p' "$ENC" >> "$OUT"
echo "" >> "$OUT"

echo "--- Appending file I/O helpers"

cat >> "$OUT" <<'SECTION_SEPARATOR'
// ============================================================
// File I/O helpers (inlined — write_file is a stdlib built-in)
// ============================================================

fn io_write_bytes_to_file(path: string, data: [i8; 65536], len: i64) -> bool with IO {
    let result = write_file(path, data, len)
    result == 0
}

fn io_write_native_binary(path: string, bytes: [i8; 65536], len: i64) -> bool with IO {
    io_write_bytes_to_file(path, bytes, len)
}

fn io_write_wide_binary(path: string, bytes: [i8; 262144], len: i64) -> bool with IO {
    let result = write_file(path, bytes, len)
    result == 0
}

fn io_write_huge_binary(path: string, bytes: [i8; 16777216], len: i64) -> bool with IO {
    let result = write_file(path, bytes, len)
    result == 0
}

SECTION_SEPARATOR

echo "--- Appending lean driver (stripped of use/module lines)"

cat >> "$OUT" <<'SECTION_SEPARATOR'
// ============================================================
// Lean compile driver (inlined from render_native_compile_driver_lean)
// ============================================================

SECTION_SEPARATOR

# Strip module/use declarations from lean driver
grep -v '^use \|^module ' "$DRIVER" >> "$OUT"

echo "--- Done. Lines: $(wc -l < "$OUT")"
