#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { printf 'PIREUS_TYPED_XOR_GPU_GATE_FAIL: %s\n' "$*" >&2; exit 1; }

grep -q 'callee == "pireus_sed_xor_convolution' self-hosted/hlir/lower.sio && \
  fail 'nominal selector remains in HLIR lowering'

for src in tests/gpu/sedenion_mul_source_level.sio tests/gpu/sedenion_mul_source_level_renamed.sio; do
  tag="$(basename "$src" .sio)"
  for target in dgx-sm121 metal; do
    suffix=ptx
    [[ "$target" == metal ]] && suffix=metal
    out="$work/$tag.$suffix"
    log="$work/$tag-$target.log"
    SOUNIO_PIREUS_OPERATOR_TRACE=1 "$SOUC" compile "$src" -o "$out" --backend gpu --gpu-target "$target" >"$log" 2>&1 || \
      fail "$tag did not compile for $target"
    [[ -s "$out" ]] || fail "$tag produced no $target artifact"
    grep -q '^PIREUS_HLIR_TYPED operator_kind=1 bits=4 twist=1 candidate=0 argc=3 callee_len=0$' "$log" || \
      fail "$tag lacks checker-owned empty-callee HLIR identity"
    if [[ "$target" == metal ]]; then
      grep -q 'pireus.recipe=xor-shuffle+sign-xor+twofold-mul+compensated-reduce' "$out" || \
        fail "$tag lacks Metal Pireus recipe"
      grep -q 'fma(' "$out" || fail "$tag lacks Metal twofold multiply"
    else
      grep -q 'shfl.sync.bfly.b32' "$out" || fail "$tag lacks DGX XOR shuffle"
      grep -q 'st.global.f64' "$out" || fail "$tag lacks DGX f64 store"
    fi
    printf 'PIREUS_TYPED_XOR_ARTIFACT source=%s target=%s sha256=%s\n' \
      "$tag" "$target" "$(sha256sum "$out" | cut -d' ' -f1)"
  done
done

host_log="$work/host-mut.log"
if "$SOUC" check tests/compile-fail/gpu_exclusive_ref_write_requires_mut_host.sio >"$host_log" 2>&1; then
  fail 'host exclusive-reference store escaped Mut'
fi
grep -q 'missing: Mut' "$host_log" || fail 'host rejection did not identify Mut'

for negative in pireus_magic_name_poison sedenion_f32_no_f64_lowering octonion_mul_source_level; do
  log="$work/$negative.log"
  out="$work/$negative.ptx"
  rc=0
  SOUNIO_PIREUS_OPERATOR_TRACE=1 "$SOUC" compile "tests/gpu/$negative.sio" \
    -o "$out" --backend gpu --gpu-target dgx-sm121 >"$log" 2>&1 || rc=$?
  grep -q '^PIREUS_HLIR_TYPED ' "$log" && fail "$negative selected the f64x16 typed operator"
  if [[ "$rc" -eq 0 && -s "$out" ]]; then
    grep -q 'shfl.sync.bfly.b32' "$out" && fail "$negative materialized the XOR shuffle"
  fi
  printf 'PIREUS_TYPED_XOR_NEGATIVE source=%s rc=%s\n' "$negative" "$rc"
done

echo 'PIREUS_TYPED_XOR_GPU_GATE_PASS positives=4 negatives=4 authority=checker-metadata'

# The terminal float2 recipe cannot silently discard additional stores or
# reinterpret a different pointer mapping as its positional three-buffer ABI.
for negative in pireus_metal_duplicate_operator pireus_metal_reordered_operands pireus_metal_extra_store; do
  log="$work/$negative.log"
  out="$work/$negative.metal"
  if "$SOUC" compile "tests/gpu/$negative.sio" -o "$out" --backend gpu --gpu-target metal >"$log" 2>&1; then
    fail "$negative silently materialized an unsupported Metal shape"
  fi
  grep -q 'PIREUS_METAL_COMPOUND_SHAPE_REFUSED' "$log" || fail "$negative failed for an unrelated reason"
done
printf 'PIREUS_METAL_COMPOUND_REFUSAL_GATE_PASS\n'

for negative in pireus_metal_duplicate_operator pireus_metal_reordered_operands pireus_metal_extra_store; do
  log="$work/$negative-ptx.log"
  if "$SOUC" compile "tests/gpu/$negative.sio" -o "$work/$negative.ptx" --backend gpu --gpu-target dgx-sm121 >"$log" 2>&1; then
    fail "$negative silently materialized an unsupported PTX shape"
  fi
  grep -q 'PIREUS_PTX_COMPOUND_SHAPE_REFUSED' "$log" || fail "$negative failed for an unrelated reason"
done
log="$work/f32-refusal.log"
if "$SOUC" compile tests/gpu/sedenion_f32_no_f64_lowering.sio -o "$work/f32.ptx" --backend gpu --gpu-target dgx-sm121 >"$log" 2>&1; then
  fail 'typed f32 aggregate silently fell through to scalar lowering'
fi
grep -q 'HLIR_LOWERING_REFUSED' "$log" || fail 'f32 failed for an unrelated reason'
printf 'PIREUS_PTX_COMPOUND_AND_PRECISION_REFUSAL_GATE_PASS\n'
