<!-- docs:meta
topic_id: repo.docs.decisions.adr-012-hyper-native-v2-bridge
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-012-hyper-native-v2-bridge
-->

# ADR 012: Bridge IrHyperMulO to native-v2 Backend Pipeline

- **Status:** experimental
- **Date:** 2026-09-24
- **Context:**
  The Sounio front-end and typechecker parse, type-check, and lower hypercomplex expressions (e.g. `pa: &Hyper<Octonion, f64>`, `a * b` with effect `with NonAssoc`) into high-level IR nodes (`IrOpcode::IrHyperMulO`). Furthermore, the mathematical lowering logic for IrHyperMulO exists in `self-hosted/native/lower_ir.sio:1774` (`lower_hyper_mul_o_fano`) producing an optimal 31-instruction / 186-byte single-ZMM AVX-512 Fano sequence that has been proven mathematically sound and validated bit-exact in hardware silicon.
  
  However, the compiler driver that executes programs via `souc run` and `souc build` (`native-v2`, `self-hosted/native/codegen_x86_linux.sio`) currently has no handler for `IrOpcode::IrHyperMulO` (returning `rc=12` on unhandled opcode). Consequently, hypercomplex vector multiplication is not yet directly callable from ordinary source-level Sounio code in native executable ELFs.

- **Decision:**
  Implement the full architectural bridge connecting `IrHyperMulO` (and siblings `IrHyperMulQ`, `IrHyperMulS`, `IrHyperNormSqO`) to `native-v2`:
  1. **ZMM Vector Register Allocation:**
     Support consecutive vector register constraints in `native-v2` register allocator for `[dst, dst+6]` working registers without clobbering caller state.
  2. **Control Table Materialization:**
     Generate dynamic preamble in functions using `IrHyperMulO` that preloads the 21 control ZMMs (`ctrl_base+0..20`) directly from `formal/generated/octonion_fano_table.sio`.
  3. **Hardware Capability Gate (CPUID):**
     Condition the emission of EVEX ZMM instructions on `cpuid_has_avx512f()`. For targets lacking AVX-512 (including Grace ARM64 nodes and legacy x86), automatically legalize `IrHyperMulO` into scalar loops or YMM/NEON sequences.
  4. **Contract A Precision Flag (`hyper_fma`):**
     Implement target flag `hyper_fma` (default `off` emitting `VMULPD` + `VADDPD` for exact scalar compatibility; `on` emitting `VFMADD231PD`).

- **Acceptance Criteria:**
  A source-level test in `tests/run-pass/hyper_octonion_source_e2e.sio` containing:
  ```sio
  let a: Hyper<Octonion, f64> = ...;
  let b: Hyper<Octonion, f64> = ...;
  let c = a * b;
  ```
  compiles cleanly with `souc build` into an ELF executable whose disassembly exhibits the 31-instruction EVEX sequence (`vbroadcastsd`, `vpermpd`, `vxorpd`, `vfmadd231pd`), and executes bit-exact against Lean 4 `#eval` vectors.
