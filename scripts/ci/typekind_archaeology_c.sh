#!/usr/bin/env bash
# Derive family-C TypeKind positions from tests/typekind/c/index.tsv (PROTOCOLO v3).
# Position is NOT stored. Empty pass+refuse => Garden.
#
# Ghost identity/inner files under tests/typekind/c/ are attempts, not fixtures.
# They must stay behaviour-identical to NoSuchType. Divergence means the kind
# is no longer a label — fill index pass/refuse then, not before.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INDEX="${1:-$ROOT_DIR/tests/typekind/c/index.tsv}"
COMPILER="${SOUNIO_TYPEKIND_ARCHAEOLOGY_BIN:-$ROOT_DIR/bin/souc}"
C_DIR="$ROOT_DIR/tests/typekind/c"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
ulimit -s 1048576 2>/dev/null || true

fail() {
  echo "TYPEKIND_ARCHAEOLOGY_C_FAIL reason=$1" >&2
  exit 1
}

[[ -f "$INDEX" ]] || fail "missing_index path=$INDEX"
[[ -x "$COMPILER" ]] || fail "missing_compiler path=$COMPILER"
[[ -d "$C_DIR" ]] || fail "missing_c_dir path=$C_DIR"

check_rc() {
  local src="$1" log="$2"
  set +e
  "$COMPILER" check "$src" >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-typekind-c.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# --- control: NoSuchType ghost pair must both check ---
ghost_id="$C_DIR/NoSuchType.ghost_identity.sio"
ghost_inner="$C_DIR/NoSuchType.ghost_inner.sio"
[[ -f "$ghost_id" && -f "$ghost_inner" ]] || fail "missing_ghost_control"
gid_rc="$(check_rc "$ghost_id" "$work/ghost_id.log")"
gin_rc="$(check_rc "$ghost_inner" "$work/ghost_inner.log")"
[[ "$gid_rc" == "0" ]] || fail "ghost_identity_control_failed rc=$gid_rc"
[[ "$gin_rc" == "0" ]] || fail "ghost_inner_control_failed rc=$gin_rc"

layer_of() {
  local kind="$1"
  local ty="Ty${kind}"
  local texpr="Type${kind}"
  local hlir="HlirType${kind}"
  local deepest="none"
  if grep -qE "TypeKind::${ty}\b" self-hosted/check/types.sio 2>/dev/null; then
    deepest="checker"
  fi
  if grep -qE "TypeExprKind::${texpr}\b" self-hosted/parser/ast.sio 2>/dev/null; then
    deepest="parser"
  fi
  # parser names TypeExpr; if both parser and checker, parser is shallower
  # deepest = last layer that still names it. Recompute in depth order.
  deepest="none"
  grep -qE "TypeExprKind::${texpr}\b" self-hosted/parser/ast.sio 2>/dev/null && deepest="parser"
  grep -qE "TypeKind::${ty}\b" self-hosted/check/types.sio 2>/dev/null && deepest="checker"
  grep -qE "${hlir}\b" self-hosted/hlir/*.sio 2>/dev/null && deepest="hlir"
  grep -qE "TypeKind::${ty}\b|${hlir}\b" self-hosted/ir/*.sio 2>/dev/null && deepest="ir"
  grep -qE "${hlir}\b|TypeKind::${ty}\b" self-hosted/native/*.sio self-hosted/llvm/*.sio 2>/dev/null && deepest="codegen"
  printf '%s' "$deepest"
}

echo "TYPEKIND_ARCHAEOLOGY_C sha_main=$(git rev-parse HEAD) compiler=$COMPILER"
printf 'kind\tposition\tdeepest_named_layer\tghost_id_rc\tghost_inner_rc\tpass_fixture\trefuse_fixture\n'

failures=0
rows=0

while IFS=$'\t' read -r kind pass_fixture refuse_fixture expected_diag deepest_layer || [[ -n "${kind:-}" ]]; do
  [[ -z "${kind:-}" ]] && continue
  [[ "$kind" == \#* ]] && continue
  [[ "$kind" == "kind" ]] && continue

  rows=$((rows + 1))
  pass_fixture="${pass_fixture:-}"
  refuse_fixture="${refuse_fixture:-}"
  expected_diag="${expected_diag:-}"
  deepest_layer="${deepest_layer:-}"
  [[ "$pass_fixture" == "-" ]] && pass_fixture=""
  [[ "$refuse_fixture" == "-" ]] && refuse_fixture=""
  [[ "$expected_diag" == "-" ]] && expected_diag=""

  computed="$(layer_of "$kind")"
  if [[ -n "$deepest_layer" && "$computed" != "$deepest_layer" ]]; then
    echo "TYPEKIND_ARCHAEOLOGY_C_FAIL reason=layer_drift kind=$kind indexed=$deepest_layer computed=$computed" >&2
    failures=$((failures + 1))
  fi

  id_sio="$C_DIR/${kind}.ghost_identity.sio"
  inner_sio="$C_DIR/${kind}.ghost_inner.sio"
  id_rc="NA"
  inner_rc="NA"
  if [[ -f "$id_sio" ]]; then
    id_rc="$(check_rc "$id_sio" "$work/${kind}.id.log")"
  fi
  if [[ -f "$inner_sio" ]]; then
    inner_rc="$(check_rc "$inner_sio" "$work/${kind}.inner.log")"
  fi

  # Divergence from NoSuchType: the name is no longer a label.
  if [[ "$id_rc" != "NA" && "$id_rc" != "$gid_rc" ]]; then
    echo "TYPEKIND_ARCHAEOLOGY_C_FAIL reason=ghost_identity_diverged kind=$kind rc=$id_rc (NoSuchType rc=$gid_rc) — fill index pass/refuse; this kind is no longer a label" >&2
    failures=$((failures + 1))
  fi
  if [[ "$inner_rc" != "NA" && "$inner_rc" != "$gin_rc" ]]; then
    echo "TYPEKIND_ARCHAEOLOGY_C_FAIL reason=ghost_inner_diverged kind=$kind rc=$inner_rc (NoSuchType rc=$gin_rc) — fill index pass/refuse; this kind started discriminating" >&2
    failures=$((failures + 1))
  fi

  position="Garden"
  if [[ -n "$pass_fixture" && -n "$refuse_fixture" ]]; then
    echo "TYPEKIND_ARCHAEOLOGY_C_FAIL reason=unexpected_indexed_pair kind=$kind — family C has no constructing program; do not index the ghost pair" >&2
    failures=$((failures + 1))
    position="INVALID"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$position" "${deepest_layer:-$computed}" "$id_rc" "$inner_rc" \
    "$pass_fixture" "$refuse_fixture"
done < "$INDEX"

[[ "$rows" -ge 12 ]] || fail "too_few_rows n=$rows (family C is 12 kinds)"

if [[ "$failures" -gt 0 ]]; then
  fail "failures=$failures"
fi

echo "TYPEKIND_ARCHAEOLOGY_C_OK rows=$rows ghost_control=NoSuchType"
