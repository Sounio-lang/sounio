<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-b-invariant-scaffold-2026-06-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-b-invariant-scaffold-2026-06-22
-->

# Equivalence Theory — Feature B scaffold: `Invariant<T, G>` decidable core
## Teoria da Equivalência — Recurso B (scaffold): núcleo decidível de `Invariant<T, G>`

*Date / Data:* 2026-06-22
*Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record / Autor responsável:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis

---

## EN — Summary

`Invariant<T, G>` is the second mirror of the same abstraction as Feature A: an observable carries the **group** under which it is invariant, so the compiler can refuse to compare two numbers that do not live in the same space ("depth of the depressive attractor — invariant under WHICH group?"). It is generalised dimensional analysis: **units → group equivariance**.

This commit delivers the **decidable core** — the load-bearing, prompt-emphasised part ("the type system stays decidable; the heavy content goes to the prover") — and a precise wiring spec. New module `self-hosted/check/invariance.sio`:

- `enum Group { DiffInf, Diff1, BiLip, Homeo }` — a **finite total-order lattice**, `Homeo ⊃ BiLip ⊃ Diff1 ⊃ DiffInf` (larger group ⇒ stronger invariant). Invariance assignments are classical (topological entropy → Homeo; Hausdorff/Kaplan–Yorke dim → BiLip; Lyapunov exponent → Diff/smooth conjugacy) and cited in the module header from the canon, not web-searched.
- `group_intersect`, `group_subset`, `transform_in_group`, and the **comparability gate** `frames_comparable(ga, gb, c)`: two observables of groups `ga, gb` whose measurement frames are related by a transform of finest class `c` may be added/compared/ratioed **iff `c ∈ (ga ∩ gb)`**. In a total order the group intersection is never empty, so rejection is driven by the **frame-relation class** — the load-bearing gate.
- `FrameRegistry` — the discharged-obligation table mapping a frame pair to the finest known transform class; identical frames → `DiffInf` (identity, always comparable), unknown distinct frames → `Homeo` (conservative: most likely to reject an unproven cross-frame comparison).

### Decidability boundary (respected)

Group membership is undecidable in general. This module decides **only** lattice membership + intersection (rank comparison). The claim "this concrete measurement transform belongs to class C" is a proof obligation destined for Lean 4 — **not** decided here. The type system stays decidable; the prover carries the rest.

### Verification evidence

The core was verified **independently of the heavy/degraded build toolchain** via a standalone seed `SRC OUT` harness (small types only, no large-struct returns), exit 0, exercising both prompt acceptance cases and four more:

- A ratio of two **Lyapunov-based** depths (`Group::Diff1`) in frames related only by a `Homeo` → **NOT comparable** (a type error once wired). ✓
- A ratio of two **entropy-based** depths (`Group::Homeo`) across a `Homeo`-related frame → **comparable** (type-checks). ✓
- Same frame (identity = `DiffInf`) → always comparable; `Diff1 ∩ Homeo = Diff1`; BiLip-over-BiLip comparable / BiLip-over-Homeo not. ✓

The committed module additionally passes standalone `souc check`.

### Honest scope — "no claims X, delivers Y"

- **Delivered:** the decidable Group lattice + comparability rule + FrameRegistry, verified.
- **Deferred (NOT in this commit), with reason:** the type-checker wiring — parser recognition of `Invariant<T, G>`, type representation, and the binary-op compatibility hook. The mechanism is fully specified in the module's `Integration / deferred` section: represent `Invariant<T,G>` via the **existing name-mangling path** (mirror `mangle_generic_name`, encoding G in the `TyNamed` name) to **avoid the 131-site `TypeEntry` field churn** (the escape-analysis precedent avoided exactly this), then decode `ga, gb` and consult `FrameRegistry` at the binop site (`compat.sio` / `checker_finish_binary_units_inplace`). This wiring needs a full Madaros rebuild per iteration, and runtime self-test verification on this WIP branch is degraded by a pre-existing crash (documented in the A-1 note), so it is held as a clean next increment rather than landed half-verified.
- **No Lean obligation emitted/discharged** (the "transform ∈ class" claim) — the Lean export surface is orphaned; held pending the per-gap decision.

### Files

- `self-hosted/check/invariance.sio` — new module (decidable core + FrameRegistry + wiring spec).

---

## PT — Resumo

`Invariant<T, G>` é o segundo espelho da mesma abstração do Recurso A: um observável carrega o **grupo** sob o qual é invariante, para que o compilador recuse comparar dois números que não vivem no mesmo espaço. É análise dimensional generalizada: **unidades → equivariância de grupo**.

Este commit entrega o **núcleo decidível** — a parte central enfatizada pelo enunciado ("o sistema de tipos permanece decidível; o conteúdo pesado vai para o provador") — e uma especificação precisa da integração. Novo módulo `self-hosted/check/invariance.sio`:

- `enum Group { DiffInf, Diff1, BiLip, Homeo }` — **reticulado de ordem total finita**, `Homeo ⊃ BiLip ⊃ Diff1 ⊃ DiffInf` (grupo maior ⇒ invariante mais forte). Atribuições clássicas (entropia topológica → Homeo; dimensão de Hausdorff/Kaplan–Yorke → BiLip; expoente de Lyapunov → conjugação suave) citadas do cânone no cabeçalho, não pesquisadas na web.
- `group_intersect`, `group_subset`, `transform_in_group` e a **porta de comparabilidade** `frames_comparable(ga, gb, c)`: dois observáveis dos grupos `ga, gb` cujos referenciais de medição se relacionam por uma transformação de classe mais fina `c` podem ser somados/comparados/divididos **se e somente se `c ∈ (ga ∩ gb)`**. Em ordem total a interseção nunca é vazia, então a rejeição é conduzida pela **classe da relação entre referenciais**.
- `FrameRegistry` — tabela de obrigações descarregadas; referenciais idênticos → `DiffInf` (identidade, sempre comparável), referenciais distintos desconhecidos → `Homeo` (conservador).

### Fronteira de decidibilidade (respeitada)

Pertinência a grupo é indecidível em geral. Este módulo decide **apenas** pertinência ao reticulado + interseção (comparação de posto). A afirmação "esta transformação concreta pertence à classe C" é uma obrigação para Lean 4 — **não** decidida aqui.

### Evidência de verificação

Núcleo verificado **independentemente da cadeia de build pesada/degradada** via harness autônomo (forma `SRC OUT` do seed; só tipos pequenos), saída 0, exercitando os dois casos de aceitação do enunciado e mais quatro. O módulo também passa no `souc check` isolado.

### Escopo honesto — "não prometer X e entregar Y"

- **Entregue:** o reticulado de grupos decidível + regra de comparabilidade + FrameRegistry, verificados.
- **Adiado (NÃO neste commit), com motivo:** a integração no verificador de tipos (reconhecimento de `Invariant<T,G>` no parser, representação de tipo e gancho de compatibilidade do operador binário). O mecanismo está especificado na seção `Integration / deferred` do módulo: representar `Invariant<T,G>` pela **via de mangling de nomes já existente** para **evitar a alteração de 131 literais de `TypeEntry`**. Essa integração exige rebuild completo do Madaros por iteração e a verificação por self-test neste ramo WIP está degradada por um crash pré-existente; por isso é mantida como incremento limpo seguinte.
- **Nenhuma obrigação Lean emitida/descarregada** — superfície órfã; adiado.

### Arquivos

- `self-hosted/check/invariance.sio` — novo módulo.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the lattice design, comparability rule, verification harness, and this bilingual note were AI-drafted and human-reviewed. The decidable core's behaviour is backed by a re-runnable seed-compiled harness (exit 0). / Desenvolvido com assistência de IA sob direção humana; o comportamento do núcleo decidível tem como respaldo um harness reexecutável compilado pelo seed (saída 0).
