<!-- docs:meta
topic_id: repo.docs.gpu.batched-hyper-syntax
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.batched-hyper-syntax
-->

# Elegant batched hypercomplex GPU syntax (array-of-Hyper)

The batched tensor-core intrinsics accept **first-class `Hyper<Algebra, f64>` values and arrays**
instead of raw `&[f64;N]` layouts. Because `Hyper<Octonion,f64>` is 8 f64 (and `Hyper<Sedenion,f64>`
is 16), an array `[Hyper<Octonion,f64>;16]` has the same memory layout as `&[f64;128]`, so the
compiler lowers it to the **byte-identical** wmma `L(a)·H` tile — the types are purely a source-level
nicety, no codegen change.

```sio
// Batched octonion multiply D = L(a)·H over a 16-state batch:
kernel fn step(a:  &Hyper<Octonion,f64>,
               H:  &[Hyper<Octonion,f64>;16],
               out:&![f64;256]) with GPU {
    oct_batch_mul(a, H, out)
}

// Full O-SSM forward cell y = Re(C⊗sigmoid(A⊗H + B·x)):
kernel fn step(A:&Hyper<Octonion,f64>, B:&Hyper<Octonion,f64>, x:&[f64;16],
               C:&Hyper<Octonion,f64>, H:&[Hyper<Octonion,f64>;16], y:&![f64;16]) with GPU {
    ossm_oct_cell(A, B, x, C, H, y)
}
```

Hypercomplex-valued parameters (A, B, C, and the state batch H) use the `Hyper<…>` types; genuinely
scalar parameters (the input `x`, the real output `y`) stay `&[f64;N]`. Sedenion variants use
`Hyper<Sedenion,f64>`. Recognized intrinsics: `oct_batch_mul`, `sed_batch_mul`, `ossm_oct_step`,
`ossm_oct_cell`, `ossm_oct_recur`, `ossm_sed_cell`, `ossm_sed_recur`. All HW-validated on the DGX
Spark GB10 (sm_121a) — identical numerics to the raw `&[f64;N]` form.
