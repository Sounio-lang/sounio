#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IR="$ROOT/self-hosted/ir/ir.sio"
LOWER="$ROOT/self-hosted/ir/lower.sio"
INLINE="$ROOT/self-hosted/ir/inline.sio"
OPT_STRATEGY="$ROOT/self-hosted/ir/opt_strategy.sio"
NORMALIZE="$ROOT/self-hosted/ir/normalize.sio"
CHECKER="$ROOT/self-hosted/check/check.sio"
TYPES="$ROOT/self-hosted/check/types.sio"
MIR="$ROOT/self-hosted/native/machine_ir.sio"
X86="$ROOT/self-hosted/native/codegen_x86_linux.sio"
X86_MIRROR="$ROOT/self-hosted/native/codegen.sio"
CONTRACT="$ROOT/docs/internal/compiler/RAW_FIELD_TERMINAL_CONTRACT.md"
WITNESS="$ROOT/tests/native-v2/raw_ptr_field_semantics_witness.sio"
NEGATIVE="$ROOT/tests/compile-fail/raw_const_field_store.sio"
ARRAY_NEGATIVE="$ROOT/tests/compile-fail/raw_array_field_projection.sio"
NONIDENT_READ="$ROOT/tests/compile-fail/raw_nonident_field_read.sio"
NONIDENT_STORE="$ROOT/tests/compile-fail/raw_nonident_field_store.sio"
ERROR_CATALOG="$ROOT/docs/llm-guide/error-catalog.md"
ERROR_EXPLANATION="$ROOT/docs/llm-guide/explanations/E229.md"
ERROR_EXPLANATION_NONIDENT="$ROOT/docs/llm-guide/explanations/E230.md"

require_fixed() {
  local file="$1"
  local text="$2"
  if ! grep -Fq -- "$text" "$file"; then
    printf 'raw-field terminal gate: missing `%s` in %s\n' "$text" "${file#$ROOT/}" >&2
    exit 1
  fi
}

require_fixed "$IR" 'pub let IR_FIELD_MODE_MANAGED: i64 = 0'
require_fixed "$IR" 'pub let IR_FIELD_MODE_REF: i64 = 1'
require_fixed "$IR" 'pub let IR_FIELD_MODE_RAW: i64 = 2'
require_fixed "$IR" 'pub fn ir_opcode_has_cfg_label(op: IrOpcode) -> bool'
require_fixed "$IR" 'IrOpcode::IrLabel => true'
require_fixed "$IR" 'IrOpcode::IrJump => true'
require_fixed "$IR" 'IrOpcode::IrBranchTrue => true'
require_fixed "$IR" 'IrOpcode::IrBranchFalse => true'
cfg_label_predicate="$(sed -n '/pub fn ir_opcode_has_cfg_label(/,/^}/p' "$IR")"
if [[ "$(grep -c -- '=> true' <<<"$cfg_label_predicate")" != 4 ]] \
  || grep -Eq 'IrField(Get|Set)' <<<"$cfg_label_predicate"; then
  printf 'raw-field terminal gate: CFG label predicate must contain only Label/Jump/BranchTrue/BranchFalse\n' >&2
  exit 1
fi
require_fixed "$IR" 'offset_words: i64'
require_fixed "$IR" 'storage_words: i64'
require_fixed "$IR" 'raw_projectable: i64'
require_fixed "$LOWER" 'fn lower_typeexpr_storage_words_ref(ty: &TypeExpr) -> i64'
require_fixed "$LOWER" 'fn lower_typeexpr_raw_projectable_ref(ty: &TypeExpr) -> i64'
require_fixed "$LOWER" 'fn raw_field_offset_words(self, struct_name: Name, field_name: Name) -> i64'
require_fixed "$LOWER" 'if field.raw_projectable != 1 || field.storage_words != 1 { return -1 }'
require_fixed "$LOWER" 'let explicit_raw_ptr_kind = self.field_base_explicit_local_raw_ptr_kind(&e.left)'
require_fixed "$LOWER" 'set_instr.label_id = IR_FIELD_MODE_RAW'
require_fixed "$LOWER" 'instr.label_id = IR_FIELD_MODE_RAW'
require_fixed "$LOWER" 'fn field_base_raw_ptr_operand_kind(self, base_opt: &Option<Box<Expr>>) -> i64'
if [[ "$(rg -c 'raw_ptr_operand_kind > 0 && !explicit_raw_deref' "$LOWER")" != 2 ]]; then
  printf 'raw-field terminal gate: lowerer must reject non-identifier raw reads and writes\n' >&2
  exit 1
fi

require_fixed "$INLINE" 'if ir_opcode_has_cfg_label(out.op) && out.label_id >= 0 {'
require_fixed "$OPT_STRATEGY" 'if ir_opcode_has_cfg_label(instr.op) && instr.label_id > max_id {'
require_fixed "$OPT_STRATEGY" 'label_id:  if ir_opcode_has_cfg_label(instr.op) { ir_opt_remap_label(instr.label_id, label_offset) } else { instr.label_id },'
require_fixed "$NORMALIZE" 'use ir::ir::{ir_opcode_has_cfg_label}'
require_fixed "$NORMALIZE" 'let new_label = if ir_opcode_has_cfg_label(instr.op) && old_label >= 0 && old_label < 256 {'

raw_lookup="$(sed -n '/fn raw_field_offset_words(/,/^    }/p' "$LOWER")"
if grep -Eq 'field_idx_from_name|field_idx_from_name_simple|buf\[0\].*% 64' <<<"$raw_lookup"; then
  printf 'raw-field terminal gate: raw offset lookup contains a legacy fallback\n' >&2
  exit 1
fi

raw_read="$(sed -n '/fn lower_field_access_expr_ref(/,/fn lower_index_expr_ref(/p' "$LOWER")"
raw_store="$(sed -n '/fn lower_assign_stmt_ref(/,/fn lower_assign_stmt(/p' "$LOWER")"
require_fixed "$LOWER" 'if explicit_ref_deref || explicit_raw_deref {'
if ! grep -Fq 'lo.lower_opt_expr_ref(&(*base_expr).left)' <<<"$raw_store" \
  || ! grep -Fq 'self.lower_opt_expr_ref(&(*base_expr_ref).left)' <<<"$raw_read"; then
  printf 'raw-field terminal gate: raw field projection no longer lowers its identifier directly\n' >&2
  exit 1
fi
if ! grep -Fq 'if explicit_raw_ptr_kind == 1' <<<"$raw_store"; then
  printf 'raw-field terminal gate: lowerer no longer fails closed for *const stores\n' >&2
  exit 1
fi

require_fixed "$CHECKER" 'ty_terminal_storage_kind(raw_ty) == 1'
require_fixed "$CHECKER" 'else if code == 229 { print("raw inline aggregate field projection requires Place IR") }'
require_fixed "$CHECKER" 'else if code == 230 { print("raw field projection requires an identifier pointer operand") }'
require_fixed "$CHECKER" 'explicit_raw_operand_kind > 0 && explicit_raw_operand_kind < 3 && ty_terminal_storage_kind(field_ty) == 3'
if [[ "$(rg -c 'if raw_operand_storage_kind == 1 \|\| raw_operand_storage_kind == 2' "$CHECKER")" != 2 ]]; then
  printf 'raw-field terminal gate: E230 must accept only raw const/mut categories, never inline-array category 3\n' >&2
  exit 1
fi
require_fixed "$TYPES" 'pub fn ty_terminal_storage_kind(ty: TypeEntry) -> i64'
require_fixed "$TYPES" '0=other, 1=raw const, 2=raw mut, 3=inline array'
require_fixed "$NEGATIVE" '(*p).value = 1.0'
require_fixed "$ARRAY_NEGATIVE" '// expected: error[E229]'
require_fixed "$ARRAY_NEGATIVE" 'raw inline aggregate field projection requires Place IR'
require_fixed "$ARRAY_NEGATIVE" '(*p).singleton'
require_fixed "$ARRAY_NEGATIVE" '(*p).wide'
require_fixed "$ARRAY_NEGATIVE" 'singleton: [i64; 1]'
require_fixed "$ARRAY_NEGATIVE" 'wide: [f64; 2]'
require_fixed "$ERROR_CATALOG" '| E229 | type-checker/raw-field | error | raw inline aggregate field projection requires Place IR |'
require_fixed "$ERROR_EXPLANATION" '# E229'
require_fixed "$NONIDENT_READ" '// expected: error[E230]'
require_fixed "$NONIDENT_READ" '(*(addr as *const RawNonIdentFieldRead)).value'
require_fixed "$NONIDENT_STORE" '// expected: error[E230]'
require_fixed "$NONIDENT_STORE" '(*(addr as *mut RawNonIdentFieldStore)).value = 1'
require_fixed "$ERROR_CATALOG" '| E230 | type-checker/raw-field | error | raw field projection requires an identifier pointer operand |'
require_fixed "$ERROR_EXPLANATION_NONIDENT" '# E230'

checker_e229_calls="$(rg -n 'report_error_at(_inplace)?\([^\n]*, 229,' "$CHECKER" || true)"
if [[ "$(wc -l <<<"$checker_e229_calls" | tr -d ' ')" != 2 ]]; then
  printf 'raw-field terminal gate: expected exactly two E229 checker emission sites\n%s\n' "$checker_e229_calls" >&2
  exit 1
fi

e229_owners="$(rg -l --hidden --glob '!.git/**' --glob '!training/**' --glob '!archive/**' \
  'E229|code == 229|report_error_at(_inplace)?\([^\n]*, 229,' "$ROOT" | sed "s#^$ROOT/##" | sort)"
expected_e229_owners="$(printf '%s\n' \
  'docs/governance/DOCS_AUTHORITY_MATRIX.md' \
  'docs/governance/topic-registry.v1.json' \
  'docs/internal/compiler/RAW_FIELD_TERMINAL_CONTRACT.md' \
  'docs/llm-guide/error-catalog.md' \
  'docs/llm-guide/explanations/E229.md' \
  'scripts/ci/raw_ptr_field_semantics_gate.sh' \
  'self-hosted/check/check.sio' \
  'tests/compile-fail/raw_array_field_projection.sio')"
if [[ "$e229_owners" != "$expected_e229_owners" ]]; then
  printf 'raw-field terminal gate: E229 ownership collision\nexpected:\n%s\nactual:\n%s\n' \
    "$expected_e229_owners" "$e229_owners" >&2
  exit 1
fi

checker_e230_calls="$(rg -n 'report_error_at(_inplace)?\([^\n]*, 230,' "$CHECKER" || true)"
if [[ "$(wc -l <<<"$checker_e230_calls" | tr -d ' ')" != 2 ]]; then
  printf 'raw-field terminal gate: expected exactly two E230 checker emission sites\n%s\n' "$checker_e230_calls" >&2
  exit 1
fi

e230_owners="$(rg -l --hidden --glob '!.git/**' --glob '!training/**' --glob '!archive/**' \
  'E230|code == 230|report_error_at(_inplace)?\([^\n]*, 230,' "$ROOT" | sed "s#^$ROOT/##" | sort)"
expected_e230_owners="$(printf '%s\n' \
  'docs/governance/DOCS_AUTHORITY_MATRIX.md' \
  'docs/governance/topic-registry.v1.json' \
  'docs/internal/compiler/RAW_FIELD_TERMINAL_CONTRACT.md' \
  'docs/llm-guide/error-catalog.md' \
  'docs/llm-guide/explanations/E230.md' \
  'scripts/ci/raw_ptr_field_semantics_gate.sh' \
  'self-hosted/check/check.sio' \
  'tests/compile-fail/raw_nonident_field_read.sio' \
  'tests/compile-fail/raw_nonident_field_store.sio')"
if [[ "$e230_owners" != "$expected_e230_owners" ]]; then
  printf 'raw-field terminal gate: E230 ownership collision\nexpected:\n%s\nactual:\n%s\n' \
    "$expected_e230_owners" "$e230_owners" >&2
  exit 1
fi

require_fixed "$MIR" 'mi.arg_index = instr.label_id'
require_fixed "$MIR" 'mi.arg_index = field_mode'
require_fixed "$MIR" 'machine_instr_field_mode(instr)'
require_fixed "$MIR" 'instr.arg_index == IR_FIELD_MODE_RAW'

for backend in "$X86" "$X86_MIRROR"; do
  require_fixed "$backend" 'instr.arg_index == IR_FIELD_MODE_RAW'
  machine_raw="$(sed -n '/if instr.opcode == MIR_OP_FIELD_LOAD/,/if instr.opcode == MIR_OP_INDEX_LOAD/p' "$backend")"
  if ! grep -Fq 'emit_mov_reg_imm(c.code, 3, instr.aux.value)' <<<"$machine_raw"; then
    printf 'raw-field terminal gate: missing direct Machine-IR raw offset in %s\n' "${backend#$ROOT/}" >&2
    exit 1
  fi
done

require_fixed "$X86" 'label_id == IR_FIELD_MODE_RAW'
core_raw_get="$(sed -n '/label_id == IR_FIELD_MODE_RAW/,/label_id == IR_FIELD_MODE_REF/p' "$X86" | head -n 8)"
if grep -Eq 'resolve_handle|object_header|load_rax_mem_rax\(' <<<"$core_raw_get"; then
  printf 'raw-field terminal gate: core raw load reintroduced handle/header/deref semantics\n' >&2
  exit 1
fi

require_fixed "$WITNESS" 'prefix: [f64; 2]'
require_fixed "$WITNESS" 'nested: RawFieldNested'
require_fixed "$WITNESS" 'fn read_const_value(p: *const RawFieldWide) -> f64'
require_fixed "$WITNESS" 'fn write_from_callee('
require_fixed "$WITNESS" 'struct RawFieldCollision'
require_fixed "$WITNESS" 'assert((*wide).nested.marker == 88)'

require_fixed "$CONTRACT" 'final extension of legacy field-address lowering'
require_fixed "$CONTRACT" '0 managed handle'
require_fixed "$CONTRACT" '1 typed reference to a managed handle slot'
require_fixed "$CONTRACT" '2 raw address'
require_fixed "$CONTRACT" 'does not claim general pointer arithmetic'
require_fixed "$CONTRACT" 'storage width is exactly one word'
require_fixed "$CONTRACT" 'arrays are never projectable, including a one-element array.'
require_fixed "$CONTRACT" 'named-aggregate handle slots are projectable'
require_fixed "$CONTRACT" 'E229'
require_fixed "$CONTRACT" 'E230'
require_fixed "$CONTRACT" 'ir_opcode_has_cfg_label'

if rg -n 'IR_FIELD_MODE_[A-Z_]+: i64 = ([3-9]|[1-9][0-9]+)' "$IR" >/dev/null; then
  printf 'raw-field terminal gate: a field mode outside 0..2 was introduced\n' >&2
  exit 1
fi

bash "$ROOT/scripts/ci/ref_field_autoderef_static_gate.sh"
printf 'raw-field terminal static gate passed.\n'
