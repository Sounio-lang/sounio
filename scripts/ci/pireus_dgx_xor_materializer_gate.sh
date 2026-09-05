#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE=tests/gpu/sedenion_mul_source_level.sio
SELECTED=tools/pireus/cross_arch_candidates.values.v1
RECEIPT=tools/cluster/evidence/pireus_dgx_typed_xor.receipt.v1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { printf 'PIREUS_DGX_XOR_MATERIALIZER_GATE_FAIL: %s\n' "$*" >&2; exit 1; }

grep -q '^selected=dgx-sm121/xor-shuffle+sign-xor+mul+tree-reduce ' "$SELECTED" || fail "ontology-selected recipe is absent"
grep -qx 'hardware=2x-NVIDIA-DGX-Spark-GB10-sm121' "$RECEIPT" || fail "receipt is not bound to both Sparks"
grep -qx 'result=PASS' "$RECEIPT" || fail "material receipt did not pass"
grep -qx 'semantic_authority_role=SEMANTIC_AUTHORITY' "$RECEIPT" || fail "Sounio semantic authority binding is absent"
grep -qx 'producer_role=MATERIAL_PARITY' "$RECEIPT" || fail "DGX material role drifted"
SOUNIO_PIREUS_OPERATOR_TRACE=1 "$SOUC" "$SOURCE" --gpu-target dgx-sm121 -o "$work/xor.ptx" >"$work/build.log" 2>&1 || fail "public Sounio GPU build failed"
grep -q '^PIREUS_HLIR_TYPED operator_kind=1 bits=4 twist=1 candidate=0 argc=3 callee_len=0$' "$work/build.log" || fail "typed empty-callee HLIR identity is absent"
ptx_sha="$(sha256sum "$work/xor.ptx" | cut -d' ' -f1)"
receipt_sha="$(sed -n 's/^ptx_sha256=//p' "$RECEIPT")"
[[ "$ptx_sha" = "$receipt_sha" ]] || fail "current public PTX is not the two-node materialized artifact"
grep -q '^\.target sm_121$' "$work/xor.ptx" || fail "PTX target drifted"
grep -q '^\.visible \.entry step(' "$work/xor.ptx" || fail "entry ABI is absent"
grep -q 'shfl.sync.bfly.b32' "$work/xor.ptx" || fail "selected shuffle primitive is absent"
grep -q 'st.global.f64' "$work/xor.ptx" || fail "result store is absent"
printf 'PIREUS_DGX_XOR_MATERIALIZER_GATE_PASS ptx_sha256=%s material_job=%s nodes=2\n' "$ptx_sha" "$(sed -n 's/^slurm_job_id=//p' "$RECEIPT")"
