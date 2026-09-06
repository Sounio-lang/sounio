<!-- docs:meta
topic_id: repo.docs.audit.pireus-typed-xor-gpu-2026-09-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pireus-typed-xor-gpu-2026-09-05
-->

# Pireus typed XOR GPU lane

```text
Semantic-Lane-ID: pireus-typed-xor-gpu-20260905
Owner: codex
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: ordinary Hyper<Sedenion,f64> multiplication is the
  semantic source; targets select only a material implementation and preserve
  the explicit Cayley-Dickson twist and reduction order
Transformation: carry the checker's span-indexed Hyper multiplication identity
  into HLIR and materialize it on DGX PTX or Apple Metal without recognizing a
  function spelling
Types-Changed: Hyper<Sedenion,f64> gains no new source syntax; f32 is kept out
  of the f64-only material contract
Effects-Changed: GPU authorizes stores through exclusive device references;
  host stores through &!T still require Mut
IR-Changed: checker IrHyperExprInfo is reduced into HLIR lowering state;
  XorConvolution retains bits, twist, candidate, arity, and empty callee
Claims-Introduced: a checked ordinary Sedenion multiply can select the Pireus
  f64x16 material operator independently of source identifiers
Claims-Forbidden: sub-quadratic complexity; exact Apple f64 equivalence;
  f32 support; U250 material parity; arbitrary twisted-convolution support
Assumptions: Cayley-Dickson Convention X; algebra_tag 4 is Sedenion;
  op_kind 1 is multiplication; current GPU material ABI is three pointers
Write-Set: self-hosted/check/check.sio, self-hosted/compiler/main.sio,
  self-hosted/hlir/lower.sio, self-hosted/gpu/hlir_to_gpu.sio, focused tests,
  gates, and this audit
Read-Set: self-hosted/ir/lower.sio, self-hosted/ir/ir.sio,
  self-hosted/hlir/ir.sio, target materializers, existing Pireus receipts
Positive-Witness: two identifier-distinct ordinary Sedenion kernels each emit
  checker-owned empty-callee HLIR and material PTX plus Metal recipes
Negative-Witness: host &! store without Mut; old magic name with scalar args;
  Octonion; Sedenion f32; malformed bits, twist, arity, and candidate
Acceptance-Gate: scripts/ci/pireus_typed_xor_gpu_gate.sh plus DGX and Apple
  hardware execution of an artifact descended from the ordinary typed source
Integration-Target: origin/main
Authoritative-Only-If: Sounio creates the first executable and expected
  semantics; C++/CUDA and Metal runners remain MATERIAL_PARITY only
```

## Boundary

The old explicit functions named `pireus_sed_xor_convolution_*` remain useful
historical foundry fixtures, but their spelling no longer selects an HLIR
operator. The public path is the ordinary expression `left * right` after the
checker has resolved both operands as the same `Hyper` algebra.

`GPU` absorbs mutation of device memory because kernel signatures reject the
host `Mut` effect. This does not weaken the host rule: the negative witness
still requires `Mut` for the same `&!T` store outside a GPU body.

The Apple materializer remains explicitly approximate (`float2` twofold
storage); compile success is not runtime parity. Hardware receipts are required
before this lane can claim material closure.

## Material closure

The current-source compiler (`sha256=26f435138dd4964331b6ee35116e0a60df576d5d1f58df016af821e41229f25f`)
lowered the ordinary `Hyper<Sedenion,f64>` expression to PTX and MSL while
reporting an empty callee and the checker-owned contract `(bits=4, twist=1,
candidate=0, argc=3)`. The identifier-renamed witness produced the same
contract; the magic-name, f32, Octonion, and malformed-contract witnesses did
not materialize the operator.

Slurm job 11709 executed the PTX entry `step` on both DGX Spark GB10 nodes and
matched all 16 frozen Sounio lanes on both ranks. Apple Metal 32023.921
compiled the MSL entry `step`, and the Swift 6.4 runner on an Apple M5 Max
matched 256 basis pairs and all 16 twofold lanes. These runs establish material
parity only; Sounio remains the semantic authority. The Apple path retains its
documented approximate `float2` storage boundary.
