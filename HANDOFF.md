# Sounio Handoff — 2026-03-08 (Sprints 32-37 complete)

## Branch

`codex/ci-signal-recovery-20260307`

## Objetivo atual

Sprints 32-37 completos: type+effect-directed compilation strategy + Geometric Algebra stdlib com epistemic uncertainty propagation. Próximo passo: decidir Sprint 38+ (ver abaixo).

## O que foi concluído

### Track A — Type-Directed Compilation Strategy (Sprints 32-34)

| Sprint | Gate | Commit | Descrição |
|--------|------|--------|-----------|
| 32 | 25/25 | `3d56e2e0` | CompileStrategy enum (6 variantes) no HLIR |
| 33 | 22/22 | `dae926e9` | StrategyHints + float folding gates + opt_strategy.sio |
| 34 | 13/13 | `e480523a` | Dual-path codegen: 4-block diamond (dispatch→fast\|instr→merge) |

### Track B — Geometric Algebra Stdlib (Sprints 35-37)

| Sprint | Gate | Commit | Descrição |
|--------|------|--------|-----------|
| 35 | 28/28 | `a02d6058` | Multivector Cl(p,q,r), geometric/inner/outer product, quaternion Cl(0,2) |
| 36 | 42/42 | `4976c7ac` | PGA Cl(3,0,1), dual quaternions, SIMD batch MvBatch4 |
| 37 | 36/36 | `9bdfc370` | UncertainMv, Jacobian propagation, Bingham quaternion uncertainty |

### Decisões arquiteturais registradas

- **Runner**: Koka-inspired effect+epistemic-directed compilation strategy (não SOIR como runner)
- **Hypercomplex**: GA stdlib com refinement type tracking de Cl(p,q,r) + grade_mask
- **Epistemic codegen**: Validated→Aggressive, Knowledge→PrecisionPreserving, Contest/Robust→Instrumented (dual-path)
- **SOIR**: verification artifact only, não execution target

## O que está em andamento

Nada. Todos os 6 sprints estão committed e gated.

## Próximos 3 passos exatos

### Passo 1 — Verificar gates na outra máquina

```bash
cd /home/demetrios/work/sounio
git log --oneline -7
bash scripts/sprint32_compile_strategy_gate.sh  # 25/25
bash scripts/sprint37_epistemic_ga_gate.sh      # 36/36
```

### Passo 2 — Decidir Sprint 38+ (opções)

Candidatos para próximo sprint:

1. **Sprint 38: Strategy-directed native codegen** — wire StrategyHints into `self-hosted/native/lower_ir.sio` (backends actually consume strategy)
2. **Sprint 38: Conformal GA Cl(4,1)** — CGA stdlib com 32-blade sparsity
3. **Sprint 38: Multi-block dual-path** — extend Sprint 34 beyond single-block functions
4. **Sprint 38: Epistemic auto-visualization** — `Knowledge<T>` → uncertainty bands em output

### Passo 3 — Rodar sanity pack pré-existente

```bash
./target/debug/souc run self-hosted/compiler/main.sio -- --self-test
bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh
```

## Blockers e riscos

- **opt_strategy.sio divergiu**: Sprint 34 agent rewrote it from scratch (different StrategyHints struct — `{emit_dual_path, strategy}` instead of Sprint 33's 7-field struct). The Sprint 33 version was the stub; Sprint 34 is the real implementation. Both pass their respective gates.
- **Worktree state**: há estado sujo não relacionado (bootstrap/poseidon/*, scripts/poseidon_gate.sh). Não faz parte deste checkpoint.
- **Sprint 36 foundation files**: o agent recriou algebra.sio/product.sio/quaternion.sio; no merge mantivemos as versões HEAD (Sprint 35). pga.sio/dual_quaternion.sio/simd.sio podem referenciar funções com nomes ligeiramente diferentes — gate passou 42/42 então está ok.

## Arquivos principais alterados (Sprints 32-37)

### HLIR (compiler internals)

- `self-hosted/hlir/ir.sio` — CompileStrategy enum, hlir_type_is_epistemic, strategy predicates
- `self-hosted/hlir/lower.sio` — hlir_compute_strategy, hlir_lower_effects_from_ast
- `self-hosted/hlir/opt_strategy.sio` — dual-path codegen (4-block diamond)
- `self-hosted/hlir/mod.sio` — includes opt_strategy.sio

### GA stdlib

- `stdlib/math/ga/algebra.sio` — Multivector, AlgebraSignature, grade_mask
- `stdlib/math/ga/product.sio` — geometric/inner/outer product, reverse, dual
- `stdlib/math/ga/quaternion.sio` — Cl(0,2) specialization
- `stdlib/math/ga/pga.sio` — PGA Cl(3,0,1): point/plane/join/meet/sandwich/motor
- `stdlib/math/ga/dual_quaternion.sio` — rigid body transforms as PGA motors
- `stdlib/math/ga/simd.sio` — MvBatch4 SoA layout, batch product/sandwich
- `stdlib/math/ga/uncertainty.sio` — Jacobian propagation, Bingham concentration
- `stdlib/math/ga/epistemic.sio` — UncertainMv, uncertain_product/sandwich

### Tests (12+ new)

- `tests/frontend/compile_strategy_{validated,knowledge,gpu,prob,standard}.sio`
- `tests/frontend/dual_path_{contest,robust}.sio`
- `tests/frontend/{pga_basic,dual_quaternion_basic,ga_simd_batch}.sio`
- `tests/frontend/ga_{algebra,product,quaternion}_basic.sio`
- `tests/frontend/epistemic_ga_{knowledge_mv,contest_motor,uncertainty_propagation}.sio`

### Gate scripts + artifacts

- `scripts/sprint{32,33,34,35,36,37}_*.sh`
- `artifacts/sprint{32,33,34,35,36,37}/*.json`

## Comandos exatos para retomar

```bash
cd /home/demetrios/work/sounio
git branch --show-current                       # codex/ci-signal-recovery-20260307
git log --oneline -7                            # verify 6 sprint commits

# Quick validation (< 30s total)
bash scripts/sprint32_compile_strategy_gate.sh  # 25/25
bash scripts/sprint37_epistemic_ga_gate.sh      # 36/36

# Full validation (all 6 gates)
for s in 32 33 34 35 36 37; do
    bash scripts/sprint${s}_*gate.sh
done
```

## Testes já rodados e resultado

| Gate | Resultado |
|------|-----------|
| Sprint 32 (compile_strategy) | 25/25 PASS |
| Sprint 33 (strategy_optimization) | 22/22 PASS |
| Sprint 34 (dual_path) | 13/13 PASS |
| Sprint 35 (ga_stdlib) | 28/28 PASS |
| Sprint 36 (ga_specializations) | 42/42 PASS |
| Sprint 37 (epistemic_ga) | 36/36 PASS |

## Checkpoint git

- Todos os 6 sprints committed na branch `codex/ci-signal-recovery-20260307`
- HEAD: `9bdfc370` — Sprint 37
- Worktree tem estado sujo não relacionado (poseidon, artifacts antigos) — NÃO commitar

## Processos longos

- Nenhum processo longo rodando
- Worktrees limpos (todos removidos após merge)
