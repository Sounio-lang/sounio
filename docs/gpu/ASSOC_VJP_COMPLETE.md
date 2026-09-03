<!-- docs:meta
topic_id: repo.docs.gpu.assoc-vjp-complete
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.assoc-vjp-complete
-->

# The associator VJP is complete — dH (kernel) + da, db (decomposition)

The associator `[a,b,H] = (a⊗b)⊗H − a⊗(b⊗H)` is now **fully differentiable** on tensor cores. Its
reverse-mode VJP has three gradients, all validated on the DGX Spark GB10:

## dH — gradient w.r.t. the batched input
A dedicated kernel (`oct_assoc_bwd_dh` / `sed_assoc_bwd_dh`, merged): `dH = L(a⊗b)ᵀ·dD − L(b)ᵀ·(L(a)ᵀ·dD)`,
three transposed m16n16k16 tiles + subtract, mirroring the forward associator. GB10: octonion 0/128
(maxerr 1e-4), sedenion 0/256 (3e-4).

## da, db — gradients w.r.t. the weights a, b
These **decompose entirely into already-merged tensor-core kernels** (`ossm_oct_bwd` — which yields both
`L(a)ᵀ·dD` and the f64 `da_accum`; and `oct_batch_mul` — which yields `q = L(b)·H`) plus host glue (the
`a⊗b` product VJP, which is a single-octonion contraction):

```
dP        = da_accum(H, dD)              = ossm_oct_bwd(a, H, dD).dA          (exact f64)
dq        = −L(a)ᵀ·dD                    = −ossm_oct_bwd(a, H, dD).dHprev
q         = L(b)·H                       = oct_batch_mul(b, H)
da_from_r = −da_accum(q, dD)             = −ossm_oct_bwd(·, qᵀ, dD).dA
db_from_q =  da_accum(H, dq)             =  ossm_oct_bwd(·, H, dq).dA
da_from_P[i] = Σ_j σ(i,j)·b[j]·dP[i⊕j]   ;   db_from_P[j] = Σ_i σ(i,j)·a[i]·dP[i⊕j]     (host)
da = da_from_P + da_from_r   ;   db = db_from_P + db_from_q
```

GB10 vs the analytic VJP: `dP` maxerr 1.1e-16 (exact f64); `da` maxerr 1e-4; `db` maxerr 2e-4. That the
weight gradients fall out of the merged primitives is itself a small result: the associator's full
backward is expressible in the existing tensor-core op set.

Harness: `run_assoc_dadb.cu`. With this the associator is a fully trainable operation — the remaining
step to an end-to-end model that *uses* the associator (whose signal was shown, ablatably, in
`NONASSOC_BENCHMARK.md`) is wiring these gradients through a training loop.
