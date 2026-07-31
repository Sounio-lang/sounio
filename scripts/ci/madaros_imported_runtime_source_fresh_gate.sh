#!/usr/bin/env bash
# #901 source-fresh fixed point: M1 bootstrap, M2 self-build, M3 self-build,
# then execute the nominal-layout and 256/257-capacity acceptance gates on M3.

set -euo pipefail
export LC_ALL=C
export SOUNIO_TARGET_OVERRIDE=x86_64-linux

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_DIR:-}"
KEEP="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_KEEP:-0}"
STRUCTURAL_ONLY="${SOUNIO_MADAROS_IMPORTED_RUNTIME_SOURCE_FRESH_STRUCTURAL_ONLY:-0}"
SRC="$ROOT_DIR/self-hosted/compiler/main.sio"
FACADE="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob_facade.sio"
D6="$ROOT_DIR/tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio"
D11="$ROOT_DIR/tests/run-pass/clinical_shift_robust_risk_transport_witness.sio"
D12="$ROOT_DIR/tests/run-pass/clinical_linear_target_monitor_witness.sio"

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
  self-hosted/ir/lower.sio \
  tests/run-pass/madaros_native_multimodule_scale_prob_facade.sio \
  tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio \
  tests/run-pass/clinical_shift_robust_risk_transport_witness.sio \
  tests/run-pass/clinical_linear_target_monitor_witness.sio; do
  [[ -f "$ROOT_DIR/$path" ]] || fail "required source missing: $path"
done

if [[ "$STRUCTURAL_ONLY" == 1 ]]; then
  grep -Fq 'M2_SHA' "$0" || fail 'M2 receipt contract missing'
  grep -Fq 'M3_SHA' "$0" || fail 'M3 receipt contract missing'
  grep -Fq 'M2_SHA" != "$M3_SHA' "$0" || fail 'fixed-point comparison missing'
  echo '[madaros-imported-runtime-source-fresh] CHECK structural_only=1 acceptance=not_run'
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

SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$M3" --science-boundary-closure "$FACADE" \
  >"$WORK/facade-closure.log" 2>&1 || { cat "$WORK/facade-closure.log" >&2; fail 'facade closure enumeration failed'; }
assert_no_fallback "$WORK/facade-closure.log"
grep -Fxq $'status\tcomplete' "$WORK/facade-closure.log" || fail 'facade closure is incomplete'
grep -Fxq $'saturated\tfalse' "$WORK/facade-closure.log" || fail 'facade closure saturated'
grep -Fxq $'parse_failed\tfalse' "$WORK/facade-closure.log" || fail 'facade closure parse failed'
for node in \
  "$ROOT_DIR/stdlib/prob/lib.sio" \
  "$ROOT_DIR/stdlib/prob/distributions.sio" \
  "$ROOT_DIR/stdlib/special/gamma.sio" \
  "$ROOT_DIR/stdlib/special/igamma.sio" \
  "$ROOT_DIR/stdlib/special/erf.sio"; do
  grep -Fxq $'node\t'"$node" "$WORK/facade-closure.log" || fail "facade closure missed physical node: $node"
done

run_imported_elf() {
  local label="$1" source="$2" expected="$3"
  local case_dir="$WORK/$label" elf="$WORK/$label/witness.elf"
  mkdir -p "$case_dir"
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$M3" --check "$source" >"$case_dir/check.log" 2>&1 \
    || { cat "$case_dir/check.log" >&2; fail "$label checker failed"; }
  SOUNIO_DUMP_MERGED_CALLS=1 SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$M3" --native-v2-compile "$source" -o "$elf" >"$case_dir/compile.log" 2>&1 \
    || { cat "$case_dir/compile.log" >&2; fail "$label compile failed"; }
  assert_no_fallback "$case_dir/compile.log"
  grep -Fq 'imported_compile: typecheck ok' "$case_dir/compile.log" || fail "$label missed modular checker receipt"
  grep -Fq 'Merged IR:' "$case_dir/compile.log" || fail "$label missed merged IR receipt"
  local main_ic
  main_ic="$(awk '
    /^MERGE_DUMP: body user_main / {
      inline = $0
      sub(/^.* name=main ic=/, "", inline)
      if (inline ~ /^[0-9]+$/) {
        print inline
        exit
      }
      in_user_main = 1
      next
    }
    in_user_main && /^ name=main ic=[0-9]+$/ {
      sub(/^ name=main ic=/, "")
      print
      exit
    }
    in_user_main && /^MERGE_DUMP:/ { exit }
  ' "$case_dir/compile.log")"
  [[ "$main_ic" =~ ^[0-9]+$ && "$main_ic" -gt 10 ]] || { cat "$case_dir/compile.log" >&2; fail "$label user_main body is absent or trivial ic=${main_ic:-missing}"; }
  [[ -s "$elf" ]] || fail "$label emitted no ELF"
  [[ "$(od -An -tx1 -N4 "$elf" | tr -d ' \n')" == 7f454c46 ]] || fail "$label output is not ELF"
  chmod +x "$elf"
  set +e
  "$elf" >"$case_dir/runtime.log" 2>&1
  local runtime_rc=$?
  set -e
  [[ "$runtime_rc" -eq 0 ]] || { cat "$case_dir/runtime.log" >&2; fail "$label ELF rc=$runtime_rc"; }
  if [[ -n "$expected" ]]; then
    grep -Fxq "$expected" "$case_dir/runtime.log" || { cat "$case_dir/runtime.log" >&2; fail "$label exact marker absent"; }
  fi
}

run_imported_elf facade-elf-42 "$FACADE" '42'
run_imported_elf imported-d6 "$D6" ''
run_imported_elf imported-d11 "$D11" ''
run_imported_elf imported-d12 "$D12" 'PROOF-CARRYING LINEAR TARGET MONITOR D12 PASS'

DEFAULT_DIR="$WORK/default-souc"
DEFAULT_ELF="$DEFAULT_DIR/facade-default.elf"
mkdir -p "$DEFAULT_DIR"
env -u SOUNIO_SOUC_BIN \
  SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$M3" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  "$ROOT_DIR/bin/souc" info >"$DEFAULT_DIR/info.log" 2>&1 \
  || { cat "$DEFAULT_DIR/info.log" >&2; fail 'default bin/souc info failed'; }
grep -Fxq "raw_elf:      $M3" "$DEFAULT_DIR/info.log" \
  || { cat "$DEFAULT_DIR/info.log" >&2; fail 'default bin/souc did not resolve the source-fresh M3'; }
env -u SOUNIO_SOUC_BIN \
  SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$M3" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  "$ROOT_DIR/bin/souc" compile "$FACADE" -o "$DEFAULT_ELF" >"$DEFAULT_DIR/compile.log" 2>&1 \
  || { cat "$DEFAULT_DIR/compile.log" >&2; fail 'default bin/souc facade compile failed'; }
assert_no_fallback "$DEFAULT_DIR/compile.log"
[[ -x "$DEFAULT_ELF" ]] || fail 'default bin/souc did not produce an executable artifact'
[[ "$(od -An -tx1 -N4 "$DEFAULT_ELF" | tr -d ' \n')" == 7f454c46 ]] || fail 'default bin/souc output is not ELF'
"$DEFAULT_ELF" >"$DEFAULT_DIR/runtime.log" 2>&1 || { cat "$DEFAULT_DIR/runtime.log" >&2; fail 'default bin/souc ELF failed'; }
grep -Fxq '42' "$DEFAULT_DIR/runtime.log" || fail 'default bin/souc ELF missed 42'

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
bootstrap_seed\tlean_single_source_tracking
self_host_chain\tM1_to_M2_to_M3_madaros
target_authority\tx86_64-linux_explicit
imported_runtime\tpass
declared_layouts\t256,257_external_and_own
known_layout_miss\trefused_no_elf
facade_vertical\tprob::lib::{uniform_mean}->closure->ELF->42
default_souc\tcompile->executable_ELF->42
default_souc_authority\tM3_sha256=$M3_SHA
imported_d6\tpass
imported_d11\tpass
imported_d12\tpass
witness_fallback\t0
EOF

cat "$WORK/receipt.tsv"
echo "[madaros-imported-runtime-source-fresh] PASS source=$HEAD_SHA fixed_point=M2_equals_M3 fallback=0"
