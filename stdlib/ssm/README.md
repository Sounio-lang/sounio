# stdlib/ssm — Sedenion State-Space Model

Algebra-level primitives for H-SSM (quaternion) and S-SSM (sedenion)
state-space models.  Distilled from the S-SSM zero-divisor gating
experiments on branch `claude/s-ssm-zero-divisor-gating-KbKQe`.

## Key theorem

```
(e3 + e10) * (e6 − e15) = 0   in 𝕊 (sedenions)
```

Proved formally in `formal/lean4/SounioZeroDivisorBridge.lean`.
This is the constructive reset mechanism: A_sed ⊗ h = 0 exactly when
h lies in the ZD null space of A.  No division algebra (ℝ, ℂ, ℍ, 𝕆) can
do this — ZD gating is structurally unique to sedenions and higher CD
algebras.

## Usage

```sio
use ssm::*

fn demo() with IO, Mut, Div, Panic {
    // 1. H-SSM: state decays by 0.95 each step, never zero
    var h_hssm: [f64; 16] = [0.0; 16]
    ssm_zd_init(&!h_hssm)            // h = e6 - e15
    ssm_hssm_step(&!h_hssm)          // h' = 0.95 * h  (norm preserved)

    // 2. S-SSM: ZD annihilation — exact reset
    var h_sssm: [f64; 16] = [0.0; 16]
    ssm_zd_init(&!h_sssm)            // h = e6 - e15
    ssm_sssm_step(&!h_sssm)          // h' = A_sed ⊗ h = 0  EXACTLY

    // 3. Selective gate (Mamba-style)
    let g = ssm_gate(2.0)             // ≈ 0.88
    ssm_apply_gate(&!h_hssm, g)       // h *= g

    // 4. Erase input token
    var h4: [f64; 16] = [0.0; 16]
    ssm_erase_input(&!h4, 0.01)       // inject B · embed_ERASE
}
```

## API

| Function | Description |
|---|---|
| `ssm_exp(x)` | exp(x) via Taylor series |
| `ssm_sigmoid(x)` | logistic sigmoid, clamped [0,1] |
| `ssm_sqrt(x)` | Newton–Raphson sqrt |
| `ssm_norm16(h)` | Euclidean norm of 16D state |
| `ssm_dot16(a, b)` | inner product of two 16D vectors |
| `ssm_sed_mul(a, b, out)` | sedenion product (Cayley-Dickson k=4) |
| `ssm_hssm_step(h)` | H-SSM: `h ← 0.95 · h` (quaternion block scaling) |
| `ssm_sssm_step(h)` | S-SSM: `h ← (e3+e10) ⊗ h`  (ZD transition) |
| `ssm_zd_init(h)` | set h = e6 − e15 (canonical ZD erase direction) |
| `ssm_erase_input(h, scale)` | inject erase signal: h[6]+=scale, h[15]-=scale |
| `ssm_gate(x)` | sigmoid gate value ∈ (0,1) |
| `ssm_apply_gate(h, g)` | scale all components: `h[i] *= g` |
| `ssm_kernel_basis(k, v)` | k-th basis of ker(A_sed), k ∈ {0..3} |
| `ssm_cokernel_basis(k, v)` | k-th permanent basis vector, k ∈ {0..3} |
| `ssm_rng_seed(a, b)` | seed xorshift64 RNG |
| `ssm_rng_next()` | next pseudorandom i64 |

## Design

- **Array API**: all state vectors are `[f64; 16]` passed by ref (`&![f64;16]`).
  Avoids SRET limits; consistent with `stdlib/algebra/sedenion.sio`.
- **Self-contained**: no stdlib imports.  The sedenion product is implemented
  directly from the Cayley-Dickson decomposition matching
  `stdlib/math/sedenion.sio`.
- **No training loop**: `step_*`, `update_*`, `run_*` functions stay in
  `examples/`.  This module provides the algebra only.

## Verified properties

| Test | Result |
|---|---|
| ZD exact reset: ‖(e3+e10) ⊗ (e6-e15)‖ = 0 | PASS |
| H-SSM no reset: ‖A_quat ⊗ (e6-e15)‖ ≈ 0.95 | PASS |
| Gate ∈ (0, 1) | PASS |
| Gate does not amplify | PASS |
| ker(A_sed) basis k=0 annihilated | PASS |
| ker(A_sed) basis k=1 annihilated | PASS |
| coker direction survives A_sed | PASS |
| ssm_norm16 matches manual Σxᵢ² | PASS |
| ssm_dot16 correct | PASS |
| RNG nonzero stream | PASS |
| Sedenion identity: 1 * e3 = e3 | PASS |
| Sedenion ZD pair: (e3+e10)*(e6-e15) = 0 | PASS |
| Erase input injects correct components | PASS |

## Run tests

```bash
./bin/souc run stdlib/ssm/lib.sio
# expect: 13/13 PASS, ALL PASS
```
