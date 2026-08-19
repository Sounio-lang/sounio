<!-- docs:meta
topic_id: repo.docs.audit.typekind-archaeology-fh-v2-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.typekind-archaeology-fh-v2-2026-08-19
-->

# TypeKind archaeology F+H — reavaliação PROTOCOLO v2

**Date:** 2026-08-19  
**Supersedes positions in:** `TYPEKIND_ARCHAEOLOGY_FH_2026-08-19.md` (v1 letter allowed Claim-ready on refuse-only; **wrong under monotone ladder**)  
**Engine:** Madaros v0.80.0 (`./bin/souc check`)  
**sha_main:** `a4d44ec22c11b11c81186bb364839e33c731bdcf` (tip at re-eval; probes same as v1 session)

## Protocol v2 (applied)

1. **Monotone:** Claim-ready ⇒ Executable ⇒ Hypothesis ⇒ Garden. No pass-construct ⇒ **max Hypothesis** (unless Reserva).  
2. **Reserva (off-ladder):** active refuse of **all** use with **named** diagnostic; no program passes. Honest fail-closed slot.  
3. **Two-program test:** must-pass + must-fail. Both fail ⇒ Reserva. Pass+fail ⇒ Claim-ready. Without both ⇒ ≤ Hypothesis.  
4. **Layer debt:** deepest layer that still **names** the kind: parser | checker | HLIR | IR | codegen.

---

## Corrections from v1

| kind | v1 (errado se aplicável) | v2 |
|---|---|---|
| **F128** | Claim-ready (só recusa) | **Reserva** — certo e errado falham E218; nenhum passa |
| **F256** | Claim-ready | **Reserva** — idem |
| VecShaped…Amortized | Hypothesis | **Hypothesis** (mantém; sem prog que construa o *kind* e passe) |
| I128/U128/RawPtr/SliceMut | Claim-ready | **Claim-ready** (mantém; par passa+falha) |
| TyI64/TyBool/TyArray | Claim-ready | **Claim-ready** (controlo da escada OK) |

---

## Tabela v2

Formato: `kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main | notas`

### Família F

| kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main | notas |
|---|---|---|---|---|---|---|
| VecShaped | **Hypothesis** | checker (`types.sio:93`, `ty_vec_shaped:2617`; compat) | *nenhum* — superfície `Vec` é TyNamed, não TyVecShaped | E001 expected Vec found […] (TyNamed) | a4d44ec22c11 | ctor **nunca chamado**; sem HLIR `VecShaped` |
| MatrixShaped | **Hypothesis** | checker (+ `epistemic.sio` minta interno) | *nenhum* user TyMatrixShaped | E001 expected Matrix (TyNamed) | a4d44ec22c11 | HLIR tem `HlirTypeMat*` **outro** sistema (f32 mats), não este kind |
| Broadcastable | **Hypothesis** | checker only | *nenhum* | E001 TyNamed | a4d44ec22c11 | ctor zero callers |
| Differentiable | **Hypothesis** | checker (+ epistemic interno) | *nenhum* user | E001 TyNamed | a4d44ec22c11 | |
| Gradient | **Hypothesis** | checker (+ epistemic:2238) | *nenhum* user | E001 TyNamed | a4d44ec22c11 | |
| Jacobian | **Hypothesis** | checker (+ epistemic:1931) | *nenhum* user | E001 TyNamed | a4d44ec22c11 | |
| BigO | **Hypothesis** | checker (+ `ty_bigO` em epistemic:2097+) | *nenhum* user TyBigO; `var a:BigO; b:BigO; a=b` passa como **TyNamed** | `let c:BigO&lt;i64,2&gt;=1` E001 expected BigO found i64 | a4d44ec22c11 | **semente de complexidade** interna; superfície é etiqueta; **não** Reserva (sem E-nomeado de complexidade) |
| Amortized | **Hypothesis** | checker only | *nenhum* | E001 TyNamed | a4d44ec22c11 | ctor zero callers; semente irmã |

### Família H

| kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main | notas |
|---|---|---|---|---|---|---|
| F128 | **Reserva** | **parser** (E218) + checker `TyF128` (morto para source) | *deve* falhar: bind/arith/sig — todos **E218** rc=1 | *deve* falhar: mesmo E218 | a4d44ec22c11 | **ambos falham** = Reserva. Sem HLIR F128. Canon=Madaros |
| F256 | **Reserva** | **parser** (E218) + checker `TyF256` | idem E218 | idem E218 | a4d44ec22c11 | Reserva |
| I128 | **Claim-ready** | **HLIR** `HlirTypeI128` (`hlir/ir.sio:113`) | `let x:i128=1; x+x` **OK** | `let x:i128=true` **E001** expected i128 | a4d44ec22c11 | par correto |
| U128 | **Claim-ready** | **HLIR** `HlirTypeU128` | construct OK | true→u128 E001 | a4d44ec22c11 | |
| RawPtr | **Claim-ready** | **IR** (`lower.sio` TypeRawPtr) / HLIR `HlirTypePtr` | `*mut i64` / cast 0 **OK** | `*mut i64 = 1` **E001** | a4d44ec22c11 | parser `TypeRawPtr`; sem doc no inventário 22 |
| SliceMut | **Claim-ready** | **checker** (`TySliceMut`) | `fn f(s:&![i64])` + `&!a` **OK** | `take(&a)` **E009** expected &![i64] | a4d44ec22c11 | não visto como kind no HLIR enum; dívida checker→HLIR |

### Controlo da escada

| kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main | notas |
|---|---|---|---|---|---|---|
| TyI64 | **Claim-ready** | **codegen** (native) | i64 arith OK | true→i64 E001 | a4d44ec22c11 | controlo: critério monótono OK |
| TyBool | **Claim-ready** | **codegen** | bool OK | 1→bool E001 | a4d44ec22c11 | |
| TyArray | **Claim-ready** | **codegen** | `[i64;3]` OK | len 2 vs 3 E001 | a4d44ec22c11 | |

---

## Contagens v2 (F+H+controlo)

| posição | n | kinds |
|---|---:|---|
| Garden | 0 | |
| Hypothesis | **8** | toda a família F |
| Executable | 0 | |
| **Reserva** | **2** | F128, F256 |
| Claim-ready | **7** | I128 U128 RawPtr SliceMut + 3 controlos |

---

## Dívida de camadas (Regra 3) — extracto F/H

| kind | checker | HLIR same-name? | sentido da dívida |
|---|---|---|---|
| VecShaped, Broadcastable, Differentiable, Gradient, Jacobian, BigO, Amortized | sim | **não** | checker→HLIR **apagamento** (kind some no backend nomeado) |
| MatrixShaped | sim | não (há Mat2/3/4 f32 **outros**) | homónimos perigosos — não é o mesmo kind |
| F128/F256 | sim (+ parser Reserva) | **não** | Reserva no parser; HLIR ainda sem wide float kind |
| I128/U128 | sim | **sim** HlirTypeI128/U128 | OK até HLIR; codegen wide-int residual fora deste censo |
| RawPtr | sim | Ptr (nome diferente) | alinhamento de nomes checker↔HLIR |
| SliceMut | sim | **não** enum HLIR | checker→HLIR apagamento de exclusividade de slice |
| HLIR-only Vec*/Mat*/Octonion/… | não no TypeKind user | **sim só HLIR** | sentido inverso: backend conhece, linguagem não (fora F/H mas citado no protocolo) |

---

## Notas

1. **BigO/Amortized:** continuam semente de complexidade (ctors + bridges internos para BigO). v2 não os promove: nenhum programa de utilizador constrói **TyBigO** e passa. Superfície `BigO` que unifica consigo mesma é **etiqueta TyNamed**, não tipagem de complexidade.

2. **F128/F256 Reserva** é a correcção central do censo F+H: recusar tudo com E218 é **honesto e superior a Hypothesis**, e **não** é Claim-ready.

3. **Controlo da escada:** TyI64/Bool/Array continuam Claim-ready com par passa/falha → regra monótona + dois programas **não** está demasiado apertada para tipos reais.

---

## Machine lines

```
kind	posicao	camada_mais_profunda	prog_certo	prog_errado	sha_main	notas
VecShaped	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	no_ctor_call
MatrixShaped	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	internal_only
Broadcastable	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	
Differentiable	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	
Gradient	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	
Jacobian	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	
BigO	Hypothesis	checker	none_user_TyBigO	E001_TyNamed	a4d44ec22c11	complexity_seed
Amortized	Hypothesis	checker	none	E001_TyNamed	a4d44ec22c11	complexity_seed
F128	Reserva	parser+checker	FAIL_E218	FAIL_E218	a4d44ec22c11	v1_was_wrong_ClaimReady
F256	Reserva	parser+checker	FAIL_E218	FAIL_E218	a4d44ec22c11	v1_was_wrong_ClaimReady
I128	Claim-ready	HLIR	PASS	E001	a4d44ec22c11	
U128	Claim-ready	HLIR	PASS	E001	a4d44ec22c11	
RawPtr	Claim-ready	IR/HLIR_Ptr	PASS	E001	a4d44ec22c11	
SliceMut	Claim-ready	checker	PASS	E009	a4d44ec22c11	no_HLIR_kind
TyI64	Claim-ready	codegen	PASS	E001	a4d44ec22c11	ladder_control
TyBool	Claim-ready	codegen	PASS	E001	a4d44ec22c11	ladder_control
TyArray	Claim-ready	codegen	PASS	E001	a4d44ec22c11	ladder_control
```
