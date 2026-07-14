#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOWER="$ROOT/self-hosted/ir/lower.sio"
WITNESS="$ROOT/tests/native-v2/ref_field_autoderef_semantics_witness.sio"

require_fixed() {
  local file="$1"
  local text="$2"
  if ! grep -Fq -- "$text" "$file"; then
    printf 'ref-field static gate: missing `%s` in %s\n' "$text" "${file#$ROOT/}" >&2
    exit 1
  fi
}

# The encoded local type is the semantic boundary: reads accept both reference
# kinds, while explicit field stores normalize only mutable references.
require_fixed "$LOWER" 'if (*ty).kind == TypeExprKind::TypeRefMut { return 2 }'
require_fixed "$LOWER" 'if (*ty).kind == TypeExprKind::TypeReference { return 1 }'
store_block="$(sed -n '/fn lower_assign_stmt_ref(/,/fn lower_assign_stmt(/p' "$LOWER")"
read_block="$(sed -n '/fn lower_field_access_expr_ref(/,/fn lower_index_expr_ref(/p' "$LOWER")"
if ! grep -Fq 'let explicit_ref_deref = explicit_ref_kind == 2' <<<"$store_block" \
  || grep -Fq 'let explicit_ref_deref = explicit_ref_kind > 0' <<<"$store_block"; then
  printf 'ref-field static gate: FieldSet must normalize only mutable references\n' >&2
  exit 1
fi
if ! grep -Fq 'let explicit_ref_deref = explicit_ref_kind > 0' <<<"$read_block" \
  || grep -Fq 'let explicit_ref_deref = explicit_ref_kind == 2' <<<"$read_block"; then
  printf 'ref-field static gate: FieldGet must normalize shared and mutable references\n' >&2
  exit 1
fi

# Keep the witness broad enough to cover the metadata paths changed by the
# normalization, plus a let-bound mutable-reference alias.
require_fixed "$WITNESS" 'second: f64'
require_fixed "$WITNESS" 'samples: [f64; 2]'
require_fixed "$WITNESS" 'let alias = pair'
require_fixed "$WITNESS" '(*alias).second = value'
require_fixed "$WITNESS" '(*pair).samples[1] = value'
require_fixed "$WITNESS" 'let alias: &!RefFieldCollision = pair'
require_fixed "$WITNESS" 'let alias = &!local'
require_fixed "$WITNESS" '(*alias).guard == 99.5'

printf 'ref-field autoderef static gate passed.\n'
