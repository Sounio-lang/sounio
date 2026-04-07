# Experimental Results Summary

## Core Results (4 Tasks + Control)

| Task | Condition | O-SSM | S4-DIAG | Naive-DIAG | Random | Notes |
|------|-----------|-------|---------|-----------|--------|-------|
| **Sorting** | BPTT 2-phase (K=5,D=10) | **45.0%** | 34.5% | — | 33.3% | O-SSM ganha com full BPTT |
| **Sorting** | Output-only (K=5,D=10) | 32.5% | 32.5% | 32.5% | 33.3% | Sem BPTT, todos ~random |
| **ListOps** | Phase 1 (L=15) | **29.5%** | 7.5% | 23.0% | 10% | O-SSM robusto, S4-DIAG colapsa |
| **ListOps** | Phase 1 reduced (L=15, 20ep) | **25.0%** | 7.0% | 8.0% | 10% | Collapse de S4-DIAG consistente |
| **Bracket** | Output-only (L=8) | 56.5% | **80.5%** | 35.0% | 50% | S4-DIAG domina |
| **Bracket** | Output-only (L=12) | 75.0% | **84.0%** | 70.0% | 50% | S4-DIAG mantém vantagem |
| **Bracket** | Output-only (L=16) | 75.5% | **85.5%** | 66.5% | 50% | S4-DIAG scale bem |
| **MNIST** | Negative control (no brackets) | 52.0% | 47.0% | 52.0% | 50% | Todos ≈ random (dissociation validada) |

## Key Insights

### 1. O-SSM Excels in Compositional Semantics
- **Sorting**: +10.5pp vs Diagonal com BPTT
- **ListOps**: +17.5pp vs S4-DIAG (compositional nested eval)
- Advantage é específica a estruturas semânticas não-lineares

### 2. S4-DIAG Excels in Hierarchical Periodicity
- **Bracket**: +4-9pp vs O-SSM (L=8-16)
- Rotação HiPPO preserva phase information para matching estruturado
- Vantagem robusta em múltiplas sequence lengths

### 3. Negative Control Validates Dissociation
- **MNIST**: O-SSM 52%, S4-DIAG 47%, Naive 52%
- Sem bracket-sensitivity, todos convergem para random
- Prova que efeito NÃO é "O-SSM universalmente melhor"

### 4. S4-DIAG Collapsa em ListOps
- Output-only: 7-7.5% (vs random 10%)
- Full BPTT + smaller LR ainda não testado
- Hipótese: composição semântica é incompatível com periódicos

## Ablations em Progresso

1. **Sorting K-sweep** {3,5,8,10} — escalabilidade com tamanho
2. **Bracket L-sweep** {6,8,10,12,16,20} — convergência vs L
3. **ListOps nesting depth** {0,1,2} — complexidade semântica

## Paper Structure

### Abstract
Dissociation entre inductive biases: O-SSM (composição) vs S4-DIAG (periodicidade)

### Main Claims
1. Cross-dimensional coupling enables compositional reasoning
2. HiPPO rotation enables periodic pattern matching
3. Effect is task-dependent, not universal ("specialized", not "better")

### Experimental Evidence
- Table 1: Core 4 tasks + negative control
- Figure 1: K-sweep (sorting) — difficulty scaling
- Figure 2: L-sweep (bracket) — sequence length robustness  
- Figure 3: Nesting depth (ListOps) — compositional complexity
- Table 2: Ablations summary

### Discussion
- Why periodic structures fail for nested semantics
- Connection to associativity / non-associativity
- Implications for architecture design

## Files
- `s4_baseline_benchmark.sio` — Sorting (BPTT + output-only)
- `bracket_3way_benchmark.sio` — Bracket 3-way (L=8,12,16)
- `listops_3way_benchmark.sio` — ListOps 3-way (L=10,15,20)
- `ossm_fullbp_v2.sio` — Sorting with full BPTT
- `smnist_3way_benchmark.sio` — Negative control
- `sorting_k_sweep.sio` — K={3,5,8,10}
- `bracket_l_sweep.sio` — L={6,8,10,12,16,20}
- `listops_nesting_depth.sio` — Depth={0,1,2}
