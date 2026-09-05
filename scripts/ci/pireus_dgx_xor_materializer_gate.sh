#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE=tests/gpu/pireus_sed_xor_convolution_f64.sio
SELECTED=tools/pireus/cross_arch_candidates.values.v1
RECEIPT=tools/cluster/evidence/pireus_dgx_operator_foundry.receipt
BINDING=tools/pireus/evidence/pireus_dgx_xor_material.receipt.v1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { printf 'PIREUS_DGX_XOR_MATERIALIZER_GATE_FAIL: %s\n' "$*" >&2; exit 1; }

grep -q '^selected=dgx-sm121/xor-shuffle+sign-xor+mul+tree-reduce ' "$SELECTED" || fail "ontology-selected recipe is absent"
grep -qx 'selected_lowering=warp-shuffle' "$RECEIPT" || fail "material receipt selected a different lowering"
grep -qx 'hardware=2x-NVIDIA-DGX-Spark-GB10-sm121' "$RECEIPT" || fail "receipt is not bound to both Sparks"
grep -qx 'result=PASS' "$RECEIPT" || fail "material receipt did not pass"
grep -qx 'semantic_authority_role=SEMANTIC_AUTHORITY' "$BINDING" || fail "Sounio semantic authority binding is absent"
grep -qx 'material_role=MATERIAL_PARITY' "$BINDING" || fail "DGX material role drifted"
grep -qx 'ontology_selection_sha256=1e67f342902ba987323b7952d23e9a9252c7826532b8d4ec334dabf42dcfbb4f' "$BINDING" || fail "ontology selection hash drifted"
grep -qx 'frozen_basis_semantics_sha256=100404ef5ea29c6d7fb945bfca3fb2433eb2f88aece42d6f5ef8e6b9067c326e' "$BINDING" || fail "basis semantics hash drifted"
"$SOUC" build "$SOURCE" --backend gpu --gpu-target dgx-sm121 -o "$work/xor.ptx" >"$work/build.log" 2>&1 || fail "public Sounio GPU build failed"
ptx_sha="$(sha256sum "$work/xor.ptx" | cut -d' ' -f1)"
receipt_sha="$(sed -n 's/^shuffle_ptx_sha256=//p' "$RECEIPT")"
[[ "$ptx_sha" = "$receipt_sha" ]] || fail "current public PTX is not the two-node materialized artifact"
grep -qx "ptx_sha256=$ptx_sha" "$BINDING" || fail "binding receipt names a different PTX"
grep -q '^\.target sm_121$' "$work/xor.ptx" || fail "PTX target drifted"
grep -q '^\.visible \.entry sedenion_xor_product(' "$work/xor.ptx" || fail "entry ABI is absent"
grep -q 'shfl.sync.bfly.b32' "$work/xor.ptx" || fail "selected shuffle primitive is absent"
grep -q 'st.global.f64' "$work/xor.ptx" || fail "result store is absent"
printf 'PIREUS_DGX_XOR_MATERIALIZER_GATE_PASS ptx_sha256=%s material_job=11571 nodes=2\n' "$ptx_sha"
