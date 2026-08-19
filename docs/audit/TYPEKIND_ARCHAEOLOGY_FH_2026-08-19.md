<!-- docs:meta
topic_id: repo.docs.audit.typekind-archaeology-fh-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.typekind-archaeology-fh-2026-08-19
-->

# TypeKind archaeology — families F + H (+ ladder control)

**Date:** 2026-08-19  
**Engine (canonical):** Madaros v0.80.0 (`./bin/souc`, default)  
**sha_main:** `98eb2b4f41a3cecfa1eccae2f635ade3c62f653f`  
**Protocol:** bus ARQUEOLOGIA (claude-1 fleet-orchestrator 2026-08-19)  
**Rule:** Claim-ready only if a program is **REFUSED** because of that kind. Accept-only = label, not a type.

Ladder:

| position | meaning |
|---|---|
| **Garden** | name in `TypeKind` enum and nothing more that fires |
| **Hypothesis** | constructor and/or checker rule exists; **no user program** constructs the kind |
| **Executable** | a program constructs the kind and the checker imposes something (run, not read) |
| **Claim-ready** | a **wrong** program is **rejected** with that kind in the expect/found story |

**Control positive for the ladder itself:** TyI64 / TyBool / TyArray must reach Claim-ready. If they do not, the criterion is too tight.

Probes: `/tmp/arch_fh/probe_*.sio` this session; command `timeout 45 ./bin/souc check <file>`.

---

## Family F (shape / gradient / complexity)

| kind | position | evidencia | ficheiro:linha | sha_main | notas |
|---|---|---|---|---|---|
| **VecShaped** | Hypothesis | enum+`ty_vec_shaped`+compat; **zero callers** of `ty_vec_shaped(` outside `types.sio`. Surface `let v: Vec<i64,3>=…` → E001 expected **Vec** (TyNamed label), not TyVecShaped. `var a:Vec; var b:Vec; a=b` OK as names. | `self-hosted/check/types.sio:93`, `:2617`; probe_vecshaped rc=1 E001 | 98eb2b4f41a3 | **sem doc**; semente de forma, não tipo vivo |
| **MatrixShaped** | Hypothesis | `ty_matrix_shaped` **called** only from `epistemic.sio` (internal). Surface `Matrix<…>` → E001 expected **Matrix** (named), not kind. No user construct of TyMatrixShaped. | `types.sio:94`, `:2641`; `epistemic.sio:1715+`; probe_matrixshaped rc=1 E001 | 98eb2b4f41a3 | **sem doc** |
| **Broadcastable** | Hypothesis | `ty_broadcastable` defined, **never called**. Surface name → TyNamed. | `types.sio:95`, `:2665`; probe_broadcastable rc=1 E001 | 98eb2b4f41a3 | |
| **Differentiable** | Hypothesis | `ty_differentiable` only from `epistemic.sio`. Surface `Differentiable<…>` → TyNamed E001; bare `var a:Differentiable` OK as name. | `types.sio:100`, `:2758`; `epistemic.sio:1920`; probe_diff rc=1 E001 | 98eb2b4f41a3 | |
| **Gradient** | Hypothesis | `ty_gradient` only `epistemic.sio:2238`. Surface → TyNamed. | `types.sio:101`, `:2783`; probe_grad rc=1 E001 | 98eb2b4f41a3 | |
| **Jacobian** | Hypothesis | `ty_jacobian` only `epistemic.sio:1931`. Surface → TyNamed. | `types.sio:102`, `:2808`; probe_jac rc=1 E001 | 98eb2b4f41a3 | |
| **BigO** | Hypothesis | `ty_bigO` **is** called from `epistemic.sio` (complexity bridges). **No user program** reaches TyBigO: surface `BigO` / `BigO<i64,2>` is **TyNamed** (“expected BigO”). `is_bigO_type` + compat exist. **Not empty enum** — internal seed of complexity types; **not** article-ready surface. | `types.sio:111`, `ty_bigO` `:2941`; `epistemic.sio:2097,2102,2107`; probe_bigo rc=1 E001 expected BigO found i64 | 98eb2b4f41a3 | **semente grande de complexidade**, não claim |
| **Amortized** | Hypothesis | `ty_amortized` defined, **never called**. Surface TyNamed. | `types.sio:112`, `:2966`; probe_amortized rc=1 E001 | 98eb2b4f41a3 | semente irmã do BigO |

---

## Family H (wide float / wide int / memory)

| kind | position | evidencia | ficheiro:linha | sha_main | notas |
|---|---|---|---|---|---|
| **F128** | **Claim-ready** | Madaros **refuses** any surface use with **E218** (parser). Probes: bind, arith, signature-only — all rc=1 E218. Not constructible under V0-A Madaros. | `parser/types.sio:24-41` E218; `check.sio:13378`; probe_f128_bind/arith/sig_only | 98eb2b4f41a3 | Canon=Madaros. lean_single history: may accept arith (known dual-engine); **not** measured as authority here |
| **F256** | **Claim-ready** | Same E218 refuse on Madaros for bind+arith. | same; probe_f256_* | 98eb2b4f41a3 | same |
| **I128** | **Claim-ready** | Construct: `let x:i128=1; x+x` → check OK. Refuse: `let x:i128=true` → E001 expected **i128** found bool. | `types.sio:123`, `ty_i128` `:411`; probe_i128 / probe_i128_bad | 98eb2b4f41a3 | |
| **U128** | **Claim-ready** | Construct OK; refuse true→u128 E001 expected **u128**. | `types.sio:124`, `:432`; probe_u128 / _bad | 98eb2b4f41a3 | |
| **RawPtr** | **Claim-ready** | Construct: `*mut i64` / `*const f64` + cast 0 OK. Refuse: `let p:*mut i64=1` E001 expected **\*mut i64** found i64. | `types.sio:127`, `ty_raw_ptr` `:817`; `check.sio:16517`; probe_rawptr / _bad | 98eb2b4f41a3 | **sem doc** no inventário de 22; superfície `*mut`/`*const` real |
| **SliceMut** | **Claim-ready** | Construct: `fn take(s:&![i64])` + `take(&!a)` OK. Refuse: `take(&a)` E009 expected **&![i64]** found `&[i64;2]`. | `types.sio:28`, `ty_slice_mut` `:884`; probe_slice_mut / _bad | 98eb2b4f41a3 | **sem doc**; recusa de exclusividade |

---

## Ladder control (ordinary types)

| kind | position | evidencia | ficheiro:linha | sha_main | notas |
|---|---|---|---|---|---|
| **TyI64** | **Claim-ready** | construct `let x:i64=1` OK; `let x:i64=true` E001 expected i64 | `types.sio:17` | 98eb2b4f41a3 | controlo: critério **não** demasiado apertado |
| **TyBool** | **Claim-ready** | construct OK; `let b:bool=1` E001 expected bool | `types.sio:19` | 98eb2b4f41a3 | controlo OK |
| **TyArray** | **Claim-ready** | `[i64;3]=[1,2,3]` OK; `[i64;3]=[1,2]` E001 expected [i64;3] found [i64;2] | `types.sio:24` | 98eb2b4f41a3 | controlo OK — length is part of the type |

**Conclusion on criterion:** three ordinary types all Claim-ready via refuse. The ladder is usable for judging the 99; F-family Hypothesis mass is real structure, not an over-tight rule.

---

## Summary counts (F+H+control)

| position | n | kinds |
|---|---:|---|
| Garden | 0 | — |
| Hypothesis | 8 | VecShaped MatrixShaped Broadcastable Differentiable Gradient Jacobian BigO Amortized |
| Executable | 0 | (none stuck here: refuse witnesses pulled ordinary/H to Claim-ready) |
| Claim-ready | 9 | F128 F256 I128 U128 RawPtr SliceMut + TyI64 TyBool TyArray |

---

## Notes (founder-facing)

1. **BigO / Amortized:** not empty enum stubs only — there is `ty_bigO` / `ty_amortized`, compat, and epistemic **bridges that mint TyBigO** internally. There is **no** user-facing complexity type that rejects a wrong program *as TyBigO*. Surface `BigO` is a **name tag** (TyNamed). Treat as a **large seed** for a complexity article, not as a shipped type. Same for Amortized with even less wiring (zero `ty_amortized` callers).

2. **F128/F256:** under **Madaros** (canonical), refuse is hard and early (**E218** parser). That is Claim-ready as a *reserved* surface, not as a numeric type you can compute with. Dual-engine history (lean_single may still emit ELFs for f128 `+`) is real but **out of authority** for this census — Madaros is the clock.

3. **VecShaped / MatrixShaped / SliceMut / RawPtr “sem doc”:** RawPtr and SliceMut are **Claim-ready** despite no concept-registry blurb — the compiler already refuses. Vec/Matrix shaped kinds are **Hypothesis** (and surface names are labels).

4. **No promotion by analogy:** MatrixShaped internal use does not lift VecShaped; BigO internal mint does not lift Amortized to Executable.

---

## Machine table

```
kind	position	evidence	file:line	sha_main	notes
VecShaped	Hypothesis	ty_vec_shaped never called; surface Vec is TyNamed E001	types.sio:93,2617	98eb2b4f41a3	no doc; label not kind
MatrixShaped	Hypothesis	ty_matrix_shaped only epistemic internal; surface Matrix TyNamed	types.sio:94; epistemic.sio:1715	98eb2b4f41a3	no doc
Broadcastable	Hypothesis	ty_broadcastable never called; surface TyNamed	types.sio:95,2665	98eb2b4f41a3	
Differentiable	Hypothesis	ty_differentiable only epistemic; surface TyNamed	types.sio:100; epistemic.sio:1920	98eb2b4f41a3	
Gradient	Hypothesis	ty_gradient only epistemic:2238; surface TyNamed	types.sio:101	98eb2b4f41a3	
Jacobian	Hypothesis	ty_jacobian only epistemic:1931; surface TyNamed	types.sio:102	98eb2b4f41a3	
BigO	Hypothesis	ty_bigO internal only; surface BigO is TyNamed not TyBigO	types.sio:111,2941; epistemic.sio:2097	98eb2b4f41a3	complexity seed not claim
Amortized	Hypothesis	ty_amortized never called; surface TyNamed	types.sio:112,2966	98eb2b4f41a3	complexity seed
F128	Claim-ready	Madaros E218 refuse bind/arith/sig	parser/types.sio:24-41; probe_f128_*	98eb2b4f41a3	Madaros canon; reserved not numeric
F256	Claim-ready	Madaros E218 refuse	same; probe_f256_*	98eb2b4f41a3	
I128	Claim-ready	construct OK; E001 expected i128 vs bool	types.sio:123; probe_i128*	98eb2b4f41a3	
U128	Claim-ready	construct OK; E001 expected u128 vs bool	types.sio:124; probe_u128*	98eb2b4f41a3	
RawPtr	Claim-ready	*mut/*const OK; E001 *mut vs i64	types.sio:127; probe_rawptr*	98eb2b4f41a3	no doc but real
SliceMut	Claim-ready	&! slice OK; E009 &! vs &	types.sio:28; probe_slice_mut*	98eb2b4f41a3	no doc but real
TyI64	Claim-ready	control OK+E001	types.sio:17	98eb2b4f41a3	ladder control
TyBool	Claim-ready	control OK+E001	types.sio:19	98eb2b4f41a3	ladder control
TyArray	Claim-ready	control OK+E001 length	types.sio:24	98eb2b4f41a3	ladder control
```

*Censo produz a tabela; a decisão de registo de conceitos é do founder.*
