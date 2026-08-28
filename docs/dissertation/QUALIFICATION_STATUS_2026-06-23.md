<!-- docs:meta
topic_id: repo.docs.dissertation.qualification-status-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.qualification-status-2026-06-23
-->

# Status de Qualificação — Dissertação de Mestrado (PUC-SP)

**Título:** *GUM-Native Pharmacokinetic Simulation via Epistemic Gradual Compilation (Rapamycin PBPK)*
**Instituição:** PUC-SP · **Defesa:** 22 de setembro de 2026
**Snapshot:** branch `codex/website-living-language-plan`, 2026-06-23.

## 1. Prontidão — 6/6 gates VERDES (todos os ativos PASS)

| Gate | Estado | Sustenta |
|---|---|---|
| `dissertation_pbpk_suite_gate` | ✅ PASS (51/53 ativos + 2 PEND declarados) | Contribuições 1–3 em 4 classes de fármaco |
| `dissertation_pbpk28_parity_gate` | ✅ PASS (9/9, <1% RMSE) | PBPK28 permeabilidade-limitada (Node↔Sounio) |
| `dissertation_pbpk_hessian_gate` | ✅ PASS (5/5) | GUM de 2ª ordem (Hessiana), Contribuição 3 |
| `dissertation_frontend_parity_gate` | ✅ PASS (14/14 compartimentos) | PBPK14 well-stirred |
| `dissertation_confidence_gate_gate` | ✅ PASS | **Contribuição 2** (gates de confiança em compilação) |
| `dissertation_dossier_gate` | ✅ PASS (5/5) | Geração de artefatos (CSV/markdown) |

Os 2 PEND da `pbpk_suite` (`pbpk28_rapamycin_clinical`, `pbpk28_semaglutide_clinical`) aguardam dados observados de literatura — não são falhas, são lacunas conhecidas declaradas no output.

**`rapamycin_kaxi_fuse_prior` (K-AXI fusion witness):** agora **PASS** — restauração do subsistema `Seq<T>` concluída (branch `feat/seq-restore`, landed em `84aa8a583`). A testemunha `cl_post=10.982609 sd_post=1.641761 sd_expected=1.641761` valida a fusão K-AXI com fixed-point gen2==gen3.

## 2. Contexto — regressão do engine e correção (2026-06-14/23)

Em 2026-06-14, `bin/souc` foi promovido a wrapper de Madaros (compilador self-hosted modular). O Madaros carrega bugs no backend nativo (`println(var)` → segfault; arrays em structs → `native_driver_function_codegen_failed`) que quebram os gates da dissertação.

**Causa raiz identificada:**
- `self-hosted/ir/lower.sio:6468`: só `ExprIntLit` roteia para `print_int`; variáveis f64/i64 roteiam para `print` (string printer) → SIGSEGV
- O ambiente do pod K8s define `SOUC_BIN=/workspace/sounio/bin/souc` → sobrepõe o `SOUC_BIN:=souc-seq-leansingle.sh` dos gate scripts

**Correção aplicada (sem rebuild de Madaros):**
1. `bin/souc`: verbo-translation para `lean_single` (`run`/`check`/`compile` → SRC OUT format)
2. `scripts/ci/souc-seq-leansingle.sh`: adicionado case `compile` com `chmod +x` do output

Gates correm com `SOUC=scripts/ci/souc-seq-leansingle.sh SOUC_BIN=scripts/ci/souc-seq-leansingle.sh` que usa `bin/souc-linux-x86_64` (lean_single com Seq<T>, 2.3 MB, compilado 2026-06-16).

## 3. Três contribuições — estado repo-backed

- **Contribuição 1 — GUM-through-ODE:** `repo-backed`. Propagação de 1ª e 2ª ordem (Hessiana) no integrador RK4/BS32, validada em rapamicina, vancomicina, tacrolimo, tirzepatida.
- **Contribuição 2 — Gates de confiança em compilação:** `repo-backed`. `with Epistemic(N)` rejeita over-claim em compilação; aceita o teto honesto. Verificado contra falso-positivo e laundering.
- **Contribuição 3 — Orçamentos ISO de incerteza:** `repo-backed`. Hessiana + cumulantes (4ª ordem). Dominância de fonte por classe: CL (rapamicina), F_oral (tacrolimo), MIC (vancomicina).

## 4. Escopo honesto — trabalho futuro (não bloqueia a qualificação)

- **Bug de println em Madaros** (variáveis → segfault): requer threading de informação de tipo do checker para o lowering ou, alternativamente, commit + rebuild via CI (slurm/SLURM-pilot). Rastreado em `slurm-jobs/madaros-frame-fix/`.
- **Descarga das provas Lean** (tacrolimo/DDI/vancomicina): statements existem; descarga algébrica é trabalho futuro.
- **GPU PBPK14:** K-AXI Phase Y valida testemunha de 2 compartimentos; PBPK14 Tsit5 GPU é `future-work`.
- **Validação clínica de coorte:** casos ancorados em literatura/caso único; coorte prospectiva é `future-work`.

## 5. Veredito

A qualificação está **defensável**: 6/6 gates verdes (todos os ativos PASS), três contribuições `repo-backed`, caso-líder vancomicina + segundo caso tacrolimo com validação de profundidade, escopo futuro declarado com honestidade calibrada. Defesa: 22 de setembro de 2026.
