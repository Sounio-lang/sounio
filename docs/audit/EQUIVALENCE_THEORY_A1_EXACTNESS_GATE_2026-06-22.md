<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-a1-exactness-gate-2026-06-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-a1-exactness-gate-2026-06-22
-->

# Equivalence Theory — Feature A-1: carrier-exactness gate on e-graph float reassociation
## Teoria da Equivalência — Recurso A-1: porta de exatidão do portador na reassociação de ponto flutuante do e-graph

*Date / Data:* 2026-06-22
*Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record / Autor responsável:* Demetrios C. Agourakis · *Standing co-author / Coautor permanente:* Dionisio Chiuratto Agourakis

---

## EN — Summary

The architectural thesis is **"types carry their theory of equivalence"**: equivalence is not absolute, it is indexed by a structure the compiler must carry. The Sounio e-graph already carried one such index — the **value-algebra** axis (`EgSmallContext.reassoc_strategy`: Cayley-Dickson non-associativity, with the octonion Fano/168-theorem predicate `ir_can_reassociate_triple`). Feature A-1 adds the **orthogonal second axis it was missing: representation exactness.**

IEEE-754 `f64`/`f32` addition and multiplication are commutative but **not associative** — reassociation changes the bits. This is a property of the floating-point *representation*, independent of the value algebra (ℝ, ℂ, octonions…). The pre-existing code reassociated `FADD`/`FSUB` *unconditionally*, justified by a comment reasoning only about the value algebra ("addition is associative in all Cayley-Dickson algebras"). A-1 makes the representation-exactness axis explicit and gates reassociation on it.

### What changed

- New field `EgSmallContext.allow_inexact_reassoc: bool`, default `false` (inexact reassociation forbidden).
- In `eg_small_saturate_float`, both reassociation blocks are now gated: reassociate **iff `(value-algebra permits) AND (allow_inexact_reassoc)`**. `FADD`/`FSUB` (associative in every Cayley-Dickson algebra) is bound by the exactness gate alone; `FMUL` ANDs the exactness gate with the existing `reassoc_strategy`/Fano gate.
- The two **PRECISION_PRESERVING** epistemic-pass sites (`opt_cleanup.sio`) explicitly set `allow_inexact_reassoc = true`. That GUM-guided pass (Sprint 225) deliberately trades bits for a lower-uncertainty evaluation order; it remains the *only* path that reassociates `f64`, now via a named, structural opt-in rather than an implicit one.
- The misleading ℝ-associativity comment is corrected to name the IEEE-754 / value-algebra distinction.

### Honest accounting — "no claims X, delivers Y"

1. **This is NOT a fix for a live default-path miscompile.** Discovery established that the default compile path (`ocp_egraph_mini_pass` → `eg_small_saturate`) maps every operator to integer opcodes and applies only identity/annihilation — it never reassociated `f64`. f64 reassociation existed *only* on the opt-in PRECISION_PRESERVING pass. A-1 therefore **makes the exactness boundary explicit and structurally enforced** (no future caller of `eg_small_saturate_float` can silently reassociate `f64`) and **corrects a mathematically wrong justification** — it does not change shipped codegen behaviour.
2. **The discrimination test is `f64`-FADD-blocked vs integer-ADD-identity-allowed**, NOT the prompt's "`f64` vs `Rational`". `Rational` is a `{num, den}` struct whose arithmetic is *function calls* (`rat_add`, …), never `EG_OP_FADD` e-nodes — it cannot enter the float-reassociation path, so it cannot be the discriminator. The realizable, shipped discriminator is the float/integer opcode split.
3. **The determinism proof is the `gen2 == gen3` MD5 fixed-point** (`Makefile`), NOT x86_64↔PTX cross-backend bit-identity — no cross-backend bit-identity harness exists in this repository (the CUDA gate does oracle functional comparison only). A-1 must *preserve* the fixed point; it does not establish cross-backend identity.
4. **No Lean obligations are emitted.** The Lean export surface (`epistemic_lean_export.sio`) is orphaned (wired into no driver); `formal/*.lean` proves the *Rust* compiler, not the self-hosted checker. Lean emission/discharge is deferred (held for per-gap decision).

### Verification evidence

- Two unit tests added to `run_compiler_main_self_tests` (compiler/main.sio), built into Madaros and executed in **real production codegen** (`artifacts/self-hosted/madaros-a1v2`):
  - **T143b** — default `EgSmallContext`: float chain `(a+b)+c` yields **0** reassociation merges (gate blocks). PASS.
  - **T143c** — with `allow_inexact_reassoc = true`: the same chain yields **> 0** merges (opt-in still reassociates; epistemic pass not regressed). PASS.
  - **T1201** ("algebra law enforcement — reassoc_strategy for CD tower") still passes → the Cayley-Dickson gating, and the deliberate octonion opt-in below, are intact.
- The tests are placed first in the runner because this WIP branch has a **pre-existing** SIGSEGV in `run_compiler_main_self_tests` (downstream of T1201) and **pre-existing** failures T08/T09 (`ir_opt` dispatch / cloned-branch-target — an unrelated subsystem). A pristine build (no A-1 edits) reproduces T08/T09 and the SIGSEGV identically, confirming they are not introduced by A-1. Runtime execution of the full self-test to completion is blocked by that pre-existing crash; the A-1 tests themselves pass.
- All three edited modules pass standalone `souc check`; the full Madaros bundle builds successfully with the edits (94.7 MB ELF).

### Blind-spot resolution

`eg_small_init_algebra` (the explicit Cayley-Dickson constructor, used only by the octonion Fano tests T81/T82) is set `allow_inexact_reassoc: true` as a **deliberate, commented** decision: that constructor exists only for callers that have declared intent to reassociate within value-algebra limits, so the Fano/168 predicate remains the deciding gate and the tests' semantics are preserved. The default-safe path is `eg_small_init()` (false). Without this, T82 (Fano triple must reassociate) would silently regress.

### Files changed

- `self-hosted/ir/egraph.sio` — exactness field, gate, corrected comment, octonion-constructor opt-in.
- `self-hosted/ir/opt_cleanup.sio` — opt-in at the two PRECISION_PRESERVING epistemic sites.
- `self-hosted/compiler/main.sio` — T143b / T143c discrimination tests.

### Deferred (out of scope)

A-layer-2 (Lyapunov/error-budget gating — no manifest substrate: the dynamical manifests have no chaotic/vector-field/Lyapunov column), Feature B (`Invariant<T,G>`), Features C-a/C-b, Lean obligation pipeline, cross-backend PTX bit-identity harness.

---

## PT — Resumo

A tese arquitetural é **"os tipos carregam sua teoria da equivalência"**: a equivalência não é absoluta, é indexada por uma estrutura que o compilador precisa carregar. O e-graph do Sounio já carregava um desses índices — o eixo da **álgebra de valores** (`EgSmallContext.reassoc_strategy`: não-associatividade de Cayley-Dickson, com o predicado de Fano/teorema 168 `ir_can_reassociate_triple` para octônios). O recurso A-1 acrescenta o **segundo eixo, ortogonal, que faltava: a exatidão da representação.**

A adição e a multiplicação em `f64`/`f32` (IEEE-754) são comutativas mas **não associativas** — reassociar altera os bits. Isso é propriedade da *representação* em ponto flutuante, independente da álgebra de valores (ℝ, ℂ, octônios…). O código pré-existente reassociava `FADD`/`FSUB` *incondicionalmente*, justificado por um comentário que raciocinava apenas sobre a álgebra de valores. O A-1 torna o eixo de exatidão explícito e condiciona a reassociação a ele.

### O que mudou

- Novo campo `EgSmallContext.allow_inexact_reassoc: bool`, padrão `false` (reassociação inexata proibida).
- Em `eg_small_saturate_float`, os dois blocos de reassociação passam a ser condicionados: reassociar **se e somente se `(a álgebra de valores permite) E (allow_inexact_reassoc)`**. `FADD`/`FSUB` depende só da porta de exatidão; `FMUL` combina a porta de exatidão com a porta `reassoc_strategy`/Fano já existente.
- Os dois pontos do passo epistêmico **PRECISION_PRESERVING** (`opt_cleanup.sio`) definem explicitamente `allow_inexact_reassoc = true`. Esse passo guiado por GUM (Sprint 225) troca bits deliberadamente por uma ordem de avaliação de menor incerteza; continua sendo o *único* caminho que reassocia `f64`, agora por adesão explícita e estrutural, não implícita.
- O comentário enganoso sobre associatividade em ℝ é corrigido.

### Prestação de contas honesta — "não prometer X e entregar Y"

1. **NÃO é correção de um erro de compilação ativo no caminho padrão.** A investigação estabeleceu que o caminho de compilação padrão nunca reassociava `f64` (mapeia tudo para opcodes inteiros e aplica só identidade/aniquilação). A reassociação de `f64` existia *apenas* no passo opcional PRECISION_PRESERVING. O A-1, portanto, **torna a fronteira de exatidão explícita e estruturalmente garantida** e **corrige uma justificativa matematicamente incorreta** — não altera o comportamento de geração de código entregue.
2. **O teste de discriminação é `f64`-FADD-bloqueado vs ADD-inteiro-identidade-permitido**, e NÃO "`f64` vs `Rational`". `Rational` é uma struct `{num, den}` cuja aritmética são *chamadas de função*, nunca nós `EG_OP_FADD` — não entra no caminho de reassociação de ponto flutuante.
3. **A prova de determinismo é o ponto fixo `gen2 == gen3` (MD5)**, e NÃO a identidade de bits entre back-ends x86_64↔PTX — não existe nenhum mecanismo de identidade de bits entre back-ends neste repositório.
4. **Nenhuma obrigação Lean é emitida.** A superfície de exportação Lean está órfã; `formal/*.lean` prova o compilador *Rust*, não o verificador auto-hospedado. Adiado.

### Evidência de verificação

- Dois testes unitários (T143b/T143c) adicionados a `run_compiler_main_self_tests`, compilados no Madaros e executados em **geração de código de produção real** (`artifacts/self-hosted/madaros-a1v2`): T143b (padrão → 0 fusões, porta bloqueia) PASSA; T143c (com adesão → > 0 fusões, passo epistêmico preservado) PASSA; T1201 (regras de álgebra do tronco Cayley-Dickson) continua passando.
- Os testes ficam no início do executor porque este ramo WIP tem um SIGSEGV **pré-existente** (a jusante de T1201) e falhas **pré-existentes** T08/T09 (`ir_opt`, subsistema não relacionado). Uma compilação do tronco sem as edições A-1 reproduz T08/T09 e o SIGSEGV de forma idêntica, confirmando que não foram introduzidos pelo A-1.
- Os três módulos editados passam no `souc check` isolado; o pacote completo do Madaros compila com as edições (ELF de 94,7 MB).

### Resolução do ponto cego

`eg_small_init_algebra` (construtor Cayley-Dickson explícito, usado só pelos testes de Fano T81/T82) recebe `allow_inexact_reassoc: true` como decisão **deliberada e comentada**, preservando a semântica dos testes (o predicado de Fano/168 permanece a porta decisória). O caminho seguro por padrão é `eg_small_init()` (false).

### Arquivos alterados

- `self-hosted/ir/egraph.sio`, `self-hosted/ir/opt_cleanup.sio`, `self-hosted/compiler/main.sio`.

### Adiado (fora de escopo)

A-camada-2 (porta de Lyapunov — sem substrato nos manifestos), Recurso B (`Invariant<T,G>`), Recursos C-a/C-b, pipeline de obrigações Lean, identidade de bits PTX entre back-ends.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

This change was developed with AI assistance (Anthropic Claude, "Opus 4.8", via the Claude Code agent harness) under human direction: repository discovery, the exactness-gate design, the implementation edits, the verification harness, and this bilingual note were AI-drafted and human-reviewed. All compiler behaviour claims are backed by re-runnable commands (`build_modular_madaros.sh`, `--self-test`). / Esta alteração foi desenvolvida com assistência de IA (Anthropic Claude, "Opus 4.8") sob direção humana; todas as afirmações sobre o comportamento do compilador têm comandos reexecutáveis como respaldo.
