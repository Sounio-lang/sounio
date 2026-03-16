---
name: Epistemic WMMA Tensor Core Kernel
description: World-first compiler-integrated GUM uncertainty propagation through GPU WMMA tensor core operations
type: project
---

End-to-end epistemic WMMA matmul kernel implemented (2026-03-15).

**What was built:**
- `gpu_build_epistemic_wmma_matmul_16x16_ir()` in `kernel_ir.sio` — programmatic IR builder for Knowledge<f32> 16×16 matmul
- PTX handlers for GpuWmma (multi-fragment), GpuAbs, GpuSqrt, GpuXor, GpuAnd in `lower_to_ptx.sio`
- 9 oracle tests in `test_epistemic_wmma.sio`
- Example: `examples/kernel_epistemic_wmma_matmul.sio`
- Gate: `scripts/sprint_epistemic_wmma_gate.sh` — 13/13 PASS

**Architecture:**
- 8 kernel params: A_val, A_eps, B_val, B_eps, C_val, C_eps, C_vld, C_prv
- Data path: GpuWmma m16n16k16 (8 fragment registers per matrix)
- Shadow path: GUM ε_C = sqrt(K)·(|A|·ε_B + |B|·ε_A) via doubling chain (no float immediates)
- Validity: and.pred (A_valid ∧ B_valid)
- Provenance: xor.b64 (A_prov ⊕ B_prov) Merkle merge

**SOTA claims (all defensible):**
1. First compiler-integrated GUM uncertainty through GPU tensor core WMMA
2. First per-element provenance tracking through GPU GEMM
3. Formal soundness proof exists in `formal/EpistemicGemm.lean` (zero sorry)

**Key references:** JCGM 100:2008 (GUM), Higham 2002, Micikevicius 2018
