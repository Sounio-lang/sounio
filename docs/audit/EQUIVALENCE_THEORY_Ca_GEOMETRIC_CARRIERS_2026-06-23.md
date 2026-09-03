<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-ca-geometric-carriers-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-ca-geometric-carriers-2026-06-23
-->

# Equivalence Theory — Feature C-a: geometric carriers (`Tensor<Hyperbolic<κ>>`)
## Teoria da Equivalência — Recurso C-a: portadores geométricos (`Tensor<Hyperbolic<κ>>`)

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis

---

## EN — Summary

The third mirror of the thesis: a carrier that **knows its manifold and curvature**, so the Euclidean operations that are invalid on the manifold are type errors and the admissible operations are the manifold ones (geodesic distance, exp/log maps). New module `stdlib/math/hyperbolic.sio`.

### What was built

- **`HypPoint`** — a point on the 2-dimensional hyperboloid model H² ⊂ ℝ^{2,1} (`⟨x,x⟩_M = -1`, `c0 > 0`), with the Minkowski form `⟨x,y⟩_M = -x0·y0 + x1·y1 + x2·y2`. Generalises to n dimensions by widening the coordinate array; 2D is shipped as the concrete, runnable instance.
- **The type carries its theory — for free, by nominal typing.** `HypPoint` is a struct, so `a + b` on two hyperboloid points is rejected by the checker with **E004 "these types cannot be combined with this operator"** — you cannot add two manifold points as if they were ℝⁿ vectors. This is the *right* mechanism: Sounio's nominal struct typing already refuses Euclidean ops on a manifold carrier; no bespoke checker change is needed (contrast Feature B, where group-indexed comparability is not expressible nominally and did need checker wiring).
- **Admissible operations (functions, not operators):** `hyp_inner` (Minkowski form), `hyp_distance` = `arccosh(-⟨x,y⟩_M)` (the closed form, K = -1), `hyp_distance_kappa` (curvature K = -κ² rescaling), `hyp_exp` / `hyp_log` (exp/log maps through the tangent space), `hyp_tangent_norm`.
- **Curvature κ** is an isometry-invariant of the manifold; in the wider design it is exposed as `Invariant<f64, Diff1>` (smooth-conjugacy invariant), tying C-a to Feature B. That tie is type-level only here (B has no value constructor), and is documented in the module header.

### Verification (acceptance criteria, all met)

| Criterion | Result |
|---|---|
| Euclidean `+` of two `HypPoint` is a type error | **E004** via `madaros --check` (`tests/compile-fail/hyperbolic_euclidean_add.sio`) |
| Geodesic distance matches `d = arccosh(-⟨x,y⟩_M)` | dist(o, [cosh r, sinh r, 0]) = r within 1e-4 |
| `exp`/`log` round-trip within tolerance | exp_o(log_o(q)) = q within 1e-4 (all 3 coords) |
| (extra) symmetry, identity, \|log_o(q)\|_M = dist, κ-scaling | all within 1e-4 |

`tests/run-pass/hyperbolic_geodesic.sio` returns 0 (all 8 assertions) when compiled by the **seed** (`bin/souc-lean-single-x86_64`, the canonical fixed-point compiler). `stdlib/math/hyperbolic.sio` passes standalone `souc check`.

### Honest accounting — "no claims X, delivers Y"

- **Verified under the seed, not the prebuilt madaros.** The run-pass test SIGSEGVs under `madaros run` because the prebuilt madaros's native codegen has a **pre-existing by-value-large-struct-return bug** (the same class that crashes the compiler self-test, documented in the A-1 note) — `HypPoint` is a 24-byte struct returned by value. The seed compiles by-value structs correctly, so verification is via the seed. This is a toolchain limitation, not a defect in the geometry.
- **Elementary transcendentals are inlined (private to the module).** `math::pure`'s `sqrt`/`ln`/`exp`/… are not `pub`, and the multimodule thin-link path enforces privacy (a pre-existing limitation that breaks even existing math-importing tests). Inlining keeps the module dependency-free and runnable through the multimodule path with only public cross-module calls. The implementations mirror `math::pure` (range-reduced ln/exp, Newton sqrt).
- **2D hyperboloid shipped; n-D is a straightforward array widening** (noted, not yet generalised). Parallel transport is provided by the existing `stdlib/math/diffgeo.sio` (RK4) and was not re-implemented here.
- **No ORC dataset** with the values cited in the prompt (±0.258/±0.270) exists in the repo — external/planned; the `HypPoint` type is the carrier it would live in.

### Files

- `stdlib/math/hyperbolic.sio` — new module (self-contained hyperboloid geometry).
- `tests/run-pass/hyperbolic_geodesic.sio`, `tests/compile-fail/hyperbolic_euclidean_add.sio`.

---

## PT — Resumo

O terceiro espelho da tese: um portador que **conhece sua variedade e curvatura**, de modo que as operações euclidianas inválidas na variedade são erros de tipo e as operações admissíveis são as da variedade (distância geodésica, mapas exp/log). Novo módulo `stdlib/math/hyperbolic.sio`.

### O que foi construído

- **`HypPoint`** — ponto no modelo do hiperboloide 2D H² ⊂ ℝ^{2,1} (`⟨x,x⟩_M = -1`, `c0 > 0`), com a forma de Minkowski `⟨x,y⟩_M = -x0·y0 + x1·y1 + x2·y2`. Generaliza para n dimensões ampliando o vetor de coordenadas; o caso 2D é entregue como instância concreta e executável.
- **O tipo carrega sua teoria — de graça, por tipagem nominal.** `HypPoint` é uma struct, então `a + b` entre dois pontos do hiperboloide é rejeitado pelo verificador com **E004 "these types cannot be combined with this operator"** — não se pode somar dois pontos da variedade como se fossem vetores de ℝⁿ. Este é o mecanismo *correto*: a tipagem nominal de structs do Sounio já recusa operações euclidianas sobre um portador-variedade; nenhuma mudança no verificador é necessária (ao contrário do Recurso B, cuja comparabilidade indexada por grupo não é expressável nominalmente e exigiu integração no verificador).
- **Operações admissíveis (funções, não operadores):** `hyp_inner`, `hyp_distance` = `arccosh(-⟨x,y⟩_M)`, `hyp_distance_kappa` (curvatura K = -κ²), `hyp_exp`/`hyp_log`, `hyp_tangent_norm`.
- **A curvatura κ** é um invariante de isometria; no desenho amplo é exposta como `Invariant<f64, Diff1>`, ligando C-a ao Recurso B (laço em nível de tipo apenas, pois B não tem construtor de valor).

### Verificação (critérios de aceitação, todos atendidos)

`+` euclidiano de dois `HypPoint` → erro de tipo **E004** (`madaros --check`); distância geodésica = `arccosh(-⟨x,y⟩_M)` (dist(o,[cosh r,sinh r,0]) = r com 1e-4); ida-e-volta exp/log com 1e-4; além de simetria, identidade, |log|=dist e escala de curvatura. `tests/run-pass/hyperbolic_geodesic.sio` retorna 0 (8 asserções) compilado pelo **seed** (`bin/souc-lean-single-x86_64`). O módulo passa no `souc check` isolado.

### Prestação de contas honesta — "não prometer X e entregar Y"

- **Verificado sob o seed, não o madaros pré-compilado.** O teste SIGSEGVa sob `madaros run` por causa de um bug **pré-existente** de retorno de struct grande por valor na geração de código nativa do madaros (mesma classe que derruba o self-test); `HypPoint` tem 24 bytes retornados por valor. O seed compila corretamente, então a verificação é via seed. Limitação de cadeia de ferramentas, não defeito da geometria.
- **Transcendentais elementares embutidas (privadas ao módulo).** `sqrt`/`ln`/`exp` de `math::pure` não são `pub`, e o caminho thin-link multimódulo impõe privacidade (limitação pré-existente que quebra até testes existentes). Embutir mantém o módulo sem dependências e executável com apenas chamadas públicas entre módulos.
- **Hiperboloide 2D entregue; n-D é ampliação direta do vetor** (anotado). Transporte paralelo já existe em `stdlib/math/diffgeo.sio` (RK4).
- **Nenhum dataset ORC** com os valores citados (±0.258/±0.270) existe no repositório — externo/planejado.

### Arquivos

- `stdlib/math/hyperbolic.sio`; `tests/run-pass/hyperbolic_geodesic.sio`; `tests/compile-fail/hyperbolic_euclidean_add.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the module, tests, and this bilingual note were AI-drafted and human-reviewed. Geometry behaviour is backed by a re-runnable seed-compiled test (exit 0) and the type-error guarantee by `madaros --check`. / Desenvolvido com assistência de IA sob direção humana; o comportamento geométrico tem respaldo em teste reexecutável compilado pelo seed (saída 0) e a garantia de erro de tipo via `madaros --check`.
