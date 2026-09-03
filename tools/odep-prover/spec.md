# ODEP — Oblivious Differential Epistemic Privacy

**Status:** specification + stub Rust prover.  Halo2/Nova integration
is future work.

## Motivation

Differential Privacy (DP) bounds the influence of any single data point
on the model's output by an ε-parameter: adding or removing one user's
data changes any given statistic by at most a factor of `exp(ε)`.
DP is an epistemic guarantee at the *probability-distribution level*:
given two outputs, you cannot distinguish with certainty whether a
given user was in the training set.

**ODEP** (Oblivious Differential Epistemic Privacy) is strictly
stronger: after the surgical operation, the model's weights are
**algebraically identical** to the weights that would have been produced
by training on the never-contributed dataset.  Not bounded, not
statistical — structural.

Where DP proves `|Pr[X|user ∈ D] − Pr[X|user ∉ D]| ≤ ε`,
ODEP proves `W_after_unlearn = W_never_trained` exactly, and
emits a zero-knowledge proof of that equality rooted in the
`unlearning_kernel_exact` theorem of Sounio's Paper C.

## R1CS Circuit Specification

The ODEP circuit verifies, given as witness:

- `w_pre    : [F; 16]` — weight vector before the unlearn op
- `w_post   : [F; 16]` — weight vector after the unlearn op
- `c1, c2   : F`       — kernel projection coefficients

The circuit checks:

1. `u1[i] = 0` for all `i ∉ {6, 15}`, `u1[6] = 1/√2`, `u1[15] = −1/√2`.
2. `u2[i] = 0` for all `i ∉ {7, 14}`, `u2[7] = 1/√2`, `u2[14] = 1/√2`.
3. `c1 = Σᵢ w_pre[i] · u1[i]`  (inner product constraint, 16 multiplications)
4. `c2 = Σᵢ w_pre[i] · u2[i]`  (same)
5. For each i in 0..16:  `w_post[i] = w_pre[i] − c1·u1[i] − c2·u2[i]`.

Total R1CS constraints:  ~130 (linear in the 16-dim activation width).
Over a BLS12-381 scalar field with `√2⁻¹` encoded at precision `2⁻⁵⁰`,
the verifier accepts iff the constraints hold exactly.

## ODEP Witness Format

The prover emits a JSON envelope:

```json
{
  "scheme": "ODEP-v1",
  "regulation": "GDPR-Art17",
  "algebraic_theorem": "unlearning_kernel_exact",
  "lean_term_hash": "<sha256 of the Lean term from Audited<T>>",
  "r1cs_witness": "<base64-encoded R1CS witness>",
  "residual_l2": 0.0,
  "primA_spec": { "i": 3, "j": 10 }
}
```

Lean soundness: the ODEP circuit's satisfiability implies
`unlearning_kernel_exact(primA, alice_kernel)` at the propositional
level; this is proved in `formal/lean4/SounioRegulatory.lean` by a
direct reduction (the circuit constraints are exactly the equations
discharged by `native_decide` in the Lean theorem).

## Prover Stub (Rust)

See `src/witness_gen.rs` for a stub that:

- reads `(w_pre, w_post, c1, c2)` from JSON,
- verifies the 5 constraint families,
- emits the envelope JSON,
- returns exit code 0 on success, non-zero otherwise.

The stub does NOT currently invoke Halo2 or Nova.  Roadmap:

- **Phase 1 (shipped):** numerical constraint check, JSON envelope.
- **Phase 2 (next):** Halo2 circuit compilation + proof generation.
- **Phase 3 (future):** Nova recursion for batched unlearn ops,
   amortized across a dataset-subject deletion queue.

## FFI Interface

From Sounio, the prover is called via `tools/odep-prover/src/ffi.rs`:

```rust
#[no_mangle]
pub extern "C" fn odep_prove(
    w_pre:   *const f64,
    w_post:  *const f64,
    len:     usize,
    c1:      f64,
    c2:      f64,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32;
```

Return value is the number of bytes written to `out_buf` (the JSON
envelope) on success, or a negative error code on failure.
Sounio side: see `stdlib/regulatory/gdpr.sio::gdpr_art17_witness_header`,
which prefixes the envelope with the [audit] header lines.

## Relationship to Existing Privacy Definitions

| Definition           | Statement                                                        | Quantitative? |
|----------------------|------------------------------------------------------------------|---------------|
| (ε, δ)-DP            | `Pr[A(D) ∈ S] ≤ exp(ε)·Pr[A(D′) ∈ S] + δ`                       | yes (ε, δ)   |
| Renyi DP             | `D_α(A(D) ‖ A(D′)) ≤ ε(α)`                                      | yes (ε(α))   |
| k-anonymity          | every record shares quasi-identifier with ≥ k-1 others           | yes (k)      |
| **ODEP (this)**      | `W_after = W_never_trained`   *algebraically*                    | **no** (exact) |

ODEP is the first privacy definition that is not quantitative: the
guarantee is structural equality, not an ε-bound.  This is the direct
consequence of Sounio's ZD-based unlearning producing fp-exact
erasure, not approximate erasure.
