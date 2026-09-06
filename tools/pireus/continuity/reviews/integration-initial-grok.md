**Verdict: BLOCKER** (not CLEAR). Review is of this diff only vs claimed bases `6e2f041` / `6b2b7ff9`. No execution, no expected-value tables, no formal-closure or hardware-performance claim. Files and effect-id tables not in the diff are noted as unresolved deps, not assumed present.

---

### BLOCKER

**1. `GpuOpcode` numeric order (enum tags)**
`IrXorConvolutionS` is appended after `IrStorePtr` with an explicit “preserve every existing opcode discriminant” comment. `GpuXorConvolutionF64` is **inserted before** `GpuOctMatvec`, shifting every later `GpuOpcode` tag. Same class of invariant, opposite edit. Any numeric snapshot, debug dump, or cross-crate `i64` opcode is now wrong.

**2. PTX target ABI hijack**
`gpu_lower_to_ptx_for_target` emits `ptx_emit_dgx_sm121_header` (`.version 9.0` / `.target sm_121`) when `dgx_sm121 || gpu_has_xor_convolution_f64(kernel)`. Default `gpu_lower_to_ptx(..., false)` therefore produces Blackwell-only PTX for every xor-convolution kernel, including the non-`dgx-sm121` path. That is a target ABI break, not an opt-in profile.

**3. HLIR lowering not preserved without checker sidecar**
Sedenion/octonion `*` is no longer selected by `hlir_oct_is` alone. It requires `hlir_checked_hyper_mul_algebra(span)` (`op_kind == 1` + matching span). On miss, `checked_algebra < 0` and the node **falls through to ordinary binary mul** (not `hlir_lower_report_error`). `hlir_lower_module` still installs an empty metadata section. GPU was switched to `hlir_lower_module_with_epistemic`; any remaining `hlir_lower_module` caller silently drops the typed operator.

**4. Core-IR virtual form has no successful native lowering**
`ir_xor_convolution_s_virtual` (`label_id=0`, `candidate=AUTO`) is emitted for expression `*` with `algebra_tag==4`. Native v2 only accepts `candidate=AUTO` **and** `label_id==1` (ptr3). `lower_xor_convolution_s` / `emit_xor_convolution_s` require `IR_XOR_CANDIDATE_TWO_ZMM_INDEXED_PERMUTE` and physical ZMM ids; failure is `ud2`. Expression-level sedenion mul therefore cannot complete native codegen. Only the `*dst = (*lhs) * (*rhs)` ptr3 shape is live.

**5. Missing dependency in this integration**
`self-hosted/hlir/mod.sio` adds `pub use native_bridge::*` with **no** `native_bridge` source in the diff. If that module is not already on current main, this does not build (and if it pulls native into HLIR, it is also a layering/cycle risk). Unshown but required by the new call sites, and not defined here: `check_program_epistemic_into`, `IrEpistemicSection.hyper_exprs` / `algebra_tag` / `op_kind`, `lowerer_find_hyper_expr_id_for_span_ref`. Absence of any of those is the same class of break.

---

### MAJOR

**6. Candidate numeric order is not a single ABI**
HLIR: `AUTO=0`, `WARP_SHUFFLE=1`, `GLOBAL_LOAD=2`, `TWO_ZMM=3`.
IR: `AUTO=0`, `TWO_ZMM=1`.
GPU contract accepts `xor_candidate` in `[AUTO, GLOBAL_LOAD]` only; HLIR `TWO_ZMM=3` is out of range and **falls through with no GPU opcode** (comment says deliberate). Cross-layer copies of the integer are mis-decoded.

**7. Metal vs PTX material ABI**
Same `GpuXorConvolutionF64`: PTX is `f64×16` via `lhs/rhs/addr` regs; Metal is `float2×16` twofold, **ignores those regs**, and uses `kernel.params[0..2]` by position. `metal_lower_ops` then returns after the first xor op and **drops every other op** in the kernel. Host buffer layout cannot be shared Metal/PTX; extra kernel work is discarded.

**8. `Hyper` type lowering narrowed**
`hlir_type_from_ast` now requires a tail scalar named `f64` (`Hyper<Octonion, f64>` / `Hyper<Sedenion, f64>`). `Hyper<Octonion>` / `Hyper<Sedenion>` with no scalar arg no longer map to the HLIR octonion/sedenion kinds, so pointer provenance / xor deferral never starts.

**9. Octonion scalar expansion order changed**
Non-`bits==4` path was `r[i⊕j] += …` over `(i,j)`; it is now `r[d] += …` over `(d,i)` with `j=i⊕d`. Same terms, different f64 reduction order. That is a lowering-preservation change for existing octonion HLIR.

**10. `GpuShflBfly` no longer refused**
It now emits `shfl.sync.bfly.b32` with **hardcoded** clamp `15` (16-lane). Any prior 32-lane use of this opcode is now a different ABI.

**11. Checker GPU store exception uses magic `5`**
`has_effect_id(..., 5)` is not tied to a named `effect_name_to_id` in this diff (Mut is documented as `1`). If GPU is not id `5` on current main, exclusive-ref stores are either still rejected in kernels or incorrectly allowed on the host.

---

### What looks locally consistent (not clearance)

- `IrXorConvolutionS` append, `hlir_instr_new` defaults, ptr3 v2 register plan (`dst=0,src1=6,src2=8,xor=10,bcast=18,gpr0`) and overlap checks are internally consistent.
- HLIR store deferral `args=[pa,pb,pout]` matches GPU `lhs/rhs/addr` for the PTX path.
- PTX butterfly vs global-load split preserves the stated serial `j` loop; Metal/PTX sign **index shapes** agree with the comments (`σ(d⊕j,j)` vs `σ(i,i⊕d)`). Convention equality with `ossm_Lsgn` is **not** re-proven here.

---

**Scope limits:** no repo walk of `6b2b7ff9` beyond this patch; no check that `GpuOpcode` is only matched by name; no effect-id table; no `native_bridge` body; no claim that Metal twofold ≈ PTX f64; historical ontology receipts left historical.
