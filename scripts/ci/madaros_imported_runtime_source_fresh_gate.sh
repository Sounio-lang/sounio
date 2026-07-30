#!/usr/bin/env bash
# #901 source-fresh fixed point: M1 bootstrap, M2 self-build, M3 self-build,
# then execute the nominal-layout and 256/257-capacity acceptance gates on M3.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_DIR:-}"
KEEP="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_KEEP:-0}"
STRUCTURAL_ONLY="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_STRUCTURAL_ONLY:-0}"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"

fail() { echo "[madaros-imported-runtime-source-fresh] FAIL: $*" >&2; exit 1; }
sha256() { sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'; }
assert_no_fallback() {
  local log="$1"
  if grep -Eiq 'native_prebundle:|falling back to full IR path|specialized lower failed|multi-mod fallback|compact modular IR table path|legacy compact IR differential enabled' "$log"; then
    cat "$log" >&2
    fail "fallback marker observed in self-build log: $log"
  fi
}

for path in \
  scripts/ci/build_modular_madaros.sh \
  scripts/dev/souc-build-lock.sh \
  scripts/ci/madaros_imported_runtime_acceptance_gate.sh \
  scripts/ci/madaros_struct_layout_capacity_gate.sh \
  self-hosted/compiler/main.sio \
  self-hosted/ir/ir.sio \
  self-hosted/ir/lower.sio; do
  [[ -f "$ROOT_DIR/$path" ]] || fail "required source missing: $path"
done

if [[ "$STRUCTURAL_ONLY" == 1 ]]; then
  grep -Fq 'M2_SHA' "$0" || fail 'M2 receipt contract missing'
  grep -Fq 'M3_SHA' "$0" || fail 'M3 receipt contract missing'
  grep -Fq 'M2_SHA" != "$M3_SHA' "$0" || fail 'fixed-point comparison missing'
  echo '[madaros-imported-runtime-source-fresh] PASS structural_only=1'
  exit 0
fi

ARCHIVE_SHA="git-checkout"
if [[ -d "$ROOT_DIR/.git" ]] || git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  HEAD_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
  TREE_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD^{tree})"
  [[ -z "$(git -C "$ROOT_DIR" status --porcelain)" ]] || fail 'source tree must be clean and committed'
else
  HEAD_SHA="${SOUNIO_SOURCE_HEAD:-}"
  TREE_SHA="${SOUNIO_SOURCE_TREE:-}"
  ARCHIVE_SHA="${SOUNIO_SOURCE_ARCHIVE_SHA256:-}"
  [[ "$HEAD_SHA" =~ ^[0-9a-f]{40}$ ]] || fail 'archive mode requires SOUNIO_SOURCE_HEAD'
  [[ "$TREE_SHA" =~ ^[0-9a-f]{40}$ ]] || fail 'archive mode requires SOUNIO_SOURCE_TREE'
  [[ "$ARCHIVE_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'archive mode requires SOUNIO_SOURCE_ARCHIVE_SHA256'
fi

if [[ -n "$WORK" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-imported-runtime-source-fresh.XXXXXX)"
fi
[[ "$KEEP" == 1 ]] || trap 'rm -rf "$WORK"' EXIT

M1="$WORK/madaros-m1.elf"
M2="$WORK/madaros-m2.elf"
M3="$WORK/madaros-m3.elf"

(
  cd "$ROOT_DIR"
  unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_MADAROS_SEED
  bash scripts/ci/build_modular_madaros.sh "$M1"
) >"$WORK/m1-build.log" 2>&1 || { cat "$WORK/m1-build.log" >&2; fail 'M1 build failed'; }

(
  cd "$ROOT_DIR"
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" scripts/dev/souc-build-lock.sh \
    "$M1" --native-v2-compile "$SRC" -o "$M2"
) >"$WORK/m2-build.log" 2>&1 || { cat "$WORK/m2-build.log" >&2; fail 'M2 self-build failed'; }

(
  cd "$ROOT_DIR"
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" scripts/dev/souc-build-lock.sh \
    "$M2" --native-v2-compile "$SRC" -o "$M3"
) >"$WORK/m3-build.log" 2>&1 || { cat "$WORK/m3-build.log" >&2; fail 'M3 self-build failed'; }
assert_no_fallback "$WORK/m2-build.log"
assert_no_fallback "$WORK/m3-build.log"

for compiler in "$M1" "$M2" "$M3"; do
  [[ -s "$compiler" ]] || fail "compiler generation missing: $compiler"
  chmod +x "$compiler"
done
M1_SHA="$(sha256 "$M1")"
M2_SHA="$(sha256 "$M2")"
M3_SHA="$(sha256 "$M3")"
if [[ "$M2_SHA" != "$M3_SHA" ]]; then
  fail "self-host fixed point not reached M2=$M2_SHA M3=$M3_SHA"
fi

MADAROS_RAW_BIN="$M3" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256="$M3_SHA" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR="$WORK/imported-runtime" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_runtime_acceptance_gate.sh" \
  >"$WORK/imported-runtime.log" 2>&1 || { cat "$WORK/imported-runtime.log" >&2; fail 'imported runtime acceptance failed'; }

MADAROS_RAW_BIN="$M3" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECTED_SHA256="$M3_SHA" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR="$WORK/layout-capacity" \
SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_struct_layout_capacity_gate.sh" \
  >"$WORK/layout-capacity.log" 2>&1 || { cat "$WORK/layout-capacity.log" >&2; fail 'layout capacity acceptance failed'; }

cat >"$WORK/receipt.tsv" <<EOF
source_head\t$HEAD_SHA
source_tree\t$TREE_SHA
source_archive_sha256\t$ARCHIVE_SHA
main_sha256\t$(sha256 "$SRC")
ir_sha256\t$(sha256 "$ROOT_DIR/self-hosted/ir/ir.sio")
lower_sha256\t$(sha256 "$ROOT_DIR/self-hosted/ir/lower.sio")
build_script_sha256\t$(sha256 "$ROOT_DIR/scripts/ci/build_modular_madaros.sh")
m1_sha256\t$M1_SHA
m2_sha256\t$M2_SHA
m3_sha256\t$M3_SHA
fixed_point\tM2_equals_M3
imported_runtime\tpass
catalog_layouts\t256,257
known_layout_miss\trefused_no_elf
fallback\t0
EOF

cat "$WORK/receipt.tsv"
echo "[madaros-imported-runtime-source-fresh] PASS source=$HEAD_SHA fixed_point=M2_equals_M3 fallback=0"
