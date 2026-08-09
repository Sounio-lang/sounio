<!-- docs:meta
topic_id: repo.docs.audit.madaros-d3-openslice-len-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-d3-openslice-len-2026-08-06
-->

# Madaros D3 closeout — open-slice / array `.len()` (2026-08-06)

**Status:** CLOSED (open-slice + local array `.len()` under shipped Madaros)  
**Gate:** `scripts/ci/madaros_d3_openslice_len_gate.sh` → `MADAROS_D3_OPENSLICE_LEN_GATE_OK`  
**Witnesses:**
- `tests/epistemic_trust/madaros_d3_openslice_len_local.sio`
- `tests/run-pass/d3_openslice_len/{lib,main}.sio`

## Symptom

`data.len()` on `&[f64]` (and bare local arrays) type-checked, but Madaros native
lowering treated `.len()` as an ordinary method, minting a body-less `len`
function → SIGSEGV at `lower_array: seed_begin`. `lean_single` printed the
correct length. After intrinsic lowering, `println(n)` still SEGVd until
`expr_result_scalar_kind_ref` classified intrinsic `.len()` as kind 1 (i64).

## Fix

1. `self-hosted/ir/lower.sio` — lower zero-arg `.len()` on non-string locals to
   `IrArrayLen` (set `label_id=1` when the receiver is a ref binding).
2. `self-hosted/native/codegen_x86_linux.sio` — IrArrayLen with `label_id=1`
   uses `native_v2_ref_array_resolve_base_rax_into`.
3. `expr_result_scalar_kind_ref` — intrinsic `.len()` → scalar kind 1.
4. Promote tip Madaros → `bin/madaros-linux-x86_64`.

## Non-claims

Does not close exclusive-ref fragile chains or the broader IrModule memory-wall.
Open-slice `.push()` / growing buffers remain out of scope.
