# Experimental Results Summary

> ⚠️ **CORRECTION NOTICE (2026-07-16).** Every O-SSM number in this document that came from an
> octonion or sedenion multiplication table was computed on a table with a sign error in the
> `e2·e5` product (`-a2*b5+a5*b2` instead of `+a2*b5-a5*b2`). That table **fails alternativity
> and composition** — it is *not* the octonions (and the 8-dim `zd_*` "zero-divisor" table is not
> a division algebra), so any *positive* non-associativity result below is a table artifact, not
> an octonion result. This document's **conclusions were already the honest negative** ("non-
> associativity is decorative / a LIABILITY; the mechanism is cross-dim coupling, not octonion
> algebra"), and those stand. What does **not** survive the corrected algebra are the residual
> positive claims, individually flagged inline below:
> - §"★★★★★ Trajectory Divergence … slight advantage" (the `triple_product` benchmark — BROKEN table).
> - §"Fractal-G2 v2/v3 + Fano curriculum" (`+3.6pp`, `7× advantage`, `+2.1pp` — BROKEN table).
> - §"ZD Deep Dive … ZD advantage +24.45pp" (the `zd_*` / `gate_vs_zd` suite — BROKEN 8-dim table;
>   `zd_bptt` re-runs to +0.00pp on the corrected sedenion).
>
> Only two paper-cited benchmarks were **re-measured** on the corrected algebra (lean_single,
> seeded; PR #1024): `multihead_unit_oct` — sorting-like octonion **32.8%** vs diagonal **54.0%**
> (reversed), octonion only leads the associator probe (17.7% vs 7.6%); and `listops` (already a
> valid isomorphic table) — octonion **13.0%** = H-SSM 13.0% vs diagonal 20.5%. Every other broken-
> table number below is **not re-validated** — no corrected figure is asserted for it. Consistent
> with A/B re-audit NEGATIVE, ABIDE associator null, and PR #907's representational-capacity reframe.

## Core Results (4 Tasks + Control)

| Task | Condition | O-SSM | S4-DIAG | Naive-DIAG | H-SSM | Random | Notes |
|------|-----------|-------|---------|-----------|-------|--------|-------|
| **Sorting** | BPTT 2-phase (K=5,D=10) | **45.0%** | 34.5% | — | — | 33.3% | O-SSM ganha com full BPTT |
| **Sorting** | Output-only (K=5,D=10) | 32.5% | 32.5% | 32.5% | — | 33.3% | Sem BPTT, todos ~random |
| **ListOps** | Phase 1 (L=15) | 29.5% | 7.5% | 23.0% | — | 10% | O-SSM robusto, S4-DIAG colapsa |
| **ListOps** | Phase 1 reduced (L=15, 20ep) | 25.0% | 7.0% | 8.0% | — | 10% | Collapse de S4-DIAG consistente |
| **ListOps** | **H-SSM ablation (L=15)** | 29.5% | 7.5% | 23.0% | **31.5%** | 10% | **H-SSM ≥ O-SSM: coupling suffices, non-assoc. decorative** |
| **Bracket** | Output-only (L=8) | 56.5% | **80.5%** | 35.0% | — | 50% | S4-DIAG domina |
| **Bracket** | Output-only (L=12) | 75.0% | **84.0%** | 70.0% | — | 50% | S4-DIAG mantém vantagem |
| **Bracket** | Output-only (L=16) | 75.5% | **85.5%** | 66.5% | — | 50% | S4-DIAG scale bem |
| **Bracket** | **H-SSM ablation (L=12)** | 70.0% | 62.5% | 60.5% | 66.0% | 50% | O-SSM leads structural tasks, H-SSM middle ground |
| **MNIST** | Negative control (no brackets) | 52.0% | 47.0% | 52.0% | — | 50% | Todos ≈ random (dissociation validada) |

## Key Insights

### 1. Cross-Dimensional Coupling Drives Composition (NOT Non-Associativity)
- **H-SSM ablation**: Quaternion (associative, non-commutative) = 31.5% on ListOps
- **O-SSM comparison**: Octonion (non-associative, non-commutative) = 29.5% on ListOps
- **H-SSM ≥ O-SSM** → non-associativity is decorative, coupling is essential
- **vs Naive-DIAG**: 23% (diagonal, commutative) — commutativity kills performance
- **Verdict**: Mechanism is cross-dim coupling + non-commutativity, not octonion algebra

### 2. S4-DIAG Excels in Hierarchical Periodicity
- **Bracket L=12**: S4-DIAG 62.5%, O-SSM 70%, H-SSM 66%
- **Bracket L=8,12,16**: S4-DIAG consistently above O-SSM (80.5%, 84%, 85.5%)
- Rotação HiPPO preserva phase information para matching estruturado
- Vantagem robusta, não depende de composição

### 3. Negative Control Validates Dissociation
- **MNIST**: O-SSM 52%, S4-DIAG 47%, Naive 52%
- Sem bracket-sensitivity, todos convergem para random
- Prova que efeito NÃO é "O-SSM universalmente melhor"

### 4. S4-DIAG Collapsa em ListOps
- Output-only: 7-7.5% (vs random 10%)
- **Hypothesis**: periodicidade é incompatível com composição semântica não-linear
- H-SSM 31.5% confirma que composição é proprietário de cross-coupling, não algebra específica

## H-SSM Ablation v1: A @ h (64-param O-SSM vs 8-param H-SSM) — FLAWED

**Methodological Issue**: O-SSM used dense `A @ h` (64-param linear matrix), NOT true Cayley product.
H-SSM used true Hamilton quaternion product (8 params). This was a capacity comparison, not an algebra comparison.

| Task | O-SSM (64p) | H-SSM (8p) | Naive (8p) | Verdict |
|------|------------|-----------|-----------|---------|
| ListOps L=15 | 29.5% | 31.5% | 23.0% | Not meaningful — parameter mismatch |
| Bracket L=12 | 70.0% | 66.0% | 60.5% | Not meaningful — parameter mismatch |

## H-SSM Ablation v2: BPTT — Training Dynamics Differ

BPTT through A reveals gradient conditioning differences:

| Model | Phase 1 (output-only) | Phase 2 (BPTT) | Delta |
|-------|----------------------|----------------|-------|
| O-SSM (64p) | 24.4% | **10.6%** (collapsed!) | -13.8pp |
| H-SSM (8p) | 31.8% | **15.8%** | -16.0pp |

O-SSM BPTT collapses to random. H-SSM retains signal. Different gradient conditioning.

## ★ DEFINITIVE: Native Algebra (Parameter-Matched, 8p each)

**THE CLEAN ABLATION**: True Cayley product vs Hamilton product vs diagonal, all 8 A-params.

| Task | O-SSM (Cayley, 8p) | H-SSM (Hamilton, 8p) | Naive (Diag, 8p) | Random |
|------|---------------------|----------------------|-------------------|--------|
| **ListOps L=15** | 28.7% | **32.9%** | 27.3% | 10% |
| **Bracket L=12** | 56.5% | **67.5%** | 41.6% | 50% |

**1000 eval samples, SE ≈ ±1.6%. Differences >3pp are statistically significant.**

### Key Findings:
1. **H-SSM > O-SSM on BOTH tasks** — non-associativity is a LIABILITY, not an advantage
   - ListOps: +4.2pp (Hamilton > Cayley, ~2.6σ)
   - Bracket: +11.0pp (Hamilton > Cayley, ~6.9σ)
2. **H-SSM > Naive on both tasks** — cross-coupling matters
   - ListOps: +5.6pp
   - Bracket: +25.9pp (massive advantage from structured coupling)
3. **O-SSM barely > Naive** — Cayley product coupling is partially cancelled by non-associativity instability
   - ListOps: +1.4pp (within noise)
   - Bracket: +14.9pp (Cayley coupling does help, but Hamilton coupling helps MORE)
4. **The "O-SSM advantage" in prior results was a parameter capacity artifact**
   - Old O-SSM: 64-param dense `A @ h` = generic linear map with 8× more capacity
   - True O-SSM: 8-param `oct_mul(A, h)` = constrained Cayley product

### Why Hamilton > Cayley:
- **Associativity provides better gradient conditioning**: (AB)C = A(BC) → chain rule is well-defined
- **Non-associativity creates gradient noise**: (A⊗h1)⊗h2 ≠ A⊗(h1⊗h2) → accumulated compositions drift
- **Hamilton's two-block structure**: independent 4D halves can specialize for different features
- **Cayley's full 8D coupling**: over-coupling may reduce specialization despite increased interaction

## ★★ Associativity Probe: THE Ground-Truth Test

**Design**: Ground truth IS algebraically non-associative. Input: [a, b, c, mode] where basis elements are multiplied LEFT=(e_a⊗e_b)⊗e_c or RIGHT=e_a⊗(e_b⊗e_c). For ~33.5% of triples, LEFT ≠ RIGHT (non-zero associator via Fano plane).

**If non-associativity matters as inductive bias**: O-SSM should separate from H-SSM specifically on non-associative triples.

| Model | Overall | Assoc-triples (665) | Non-assoc-triples (335) | Delta |
|-------|---------|---------------------|------------------------|-------|
| O-SSM (Cayley) | 10.1% | 12.5% | **5.4%** | -7.1% |
| H-SSM (Hamilton) | 10.8% | 13.5% | **5.4%** | -8.2% |
| Naive | 9.4% | 12.9% | 2.4% | -10.5% |
| S4-DIAG | 9.4% | 12.9% | 2.4% | -10.5% |
| Random | 6.25% | 6.25% | 6.25% | 0% |

**Result**: O-SSM and H-SSM have **IDENTICAL** accuracy on non-associative triples (5.373%).
Zero separation. The Cayley algebra provides ZERO inductive bias for learning non-associative patterns,
even when the ground truth requires non-associative computation.

**Coupling helps**: Both O-SSM/H-SSM (5.4%) beat Naive/S4-DIAG (2.4%) on non-assoc triples.
**Non-associativity doesn't help**: O-SSM = H-SSM exactly.

**Verdict**: Non-associativity is not an inductive bias that neural SSMs can leverage. The octonionic
Cayley product's non-associative structure is invisible to gradient-based learning at this scale.

## ★★★ Triple-Product Test: Artin's Theorem and Architectural Activation

### The Critical Insight

All previous O-SSM benchmarks used `h' = A⊗h + B·x`. By **Artin's theorem**, any two elements
in an alternative algebra generate an associative subalgebra. Therefore `A⊗h` is ALWAYS locally
associative — **non-associativity was architecturally DORMANT** in every previous experiment.

The fix: **Triple product** `h' = (A⊗h)⊗x` — three independent octonions interact multiplicatively.
`(A⊗h)⊗x ≠ A⊗(h⊗x)` for octonions (non-assoc ACTIVE), but `=` for quaternions (still assoc).

### Results

| Model | Overall | Assoc-triples (665) | Non-assoc-triples (335) |
|-------|---------|---------------------|------------------------|
| O-SSM-Triple (non-assoc ACTIVE) | 4.5% | 6.8% | **0.0%** |
| H-SSM-Triple (assoc, blind) | 4.5% | 6.8% | **0.0%** |
| O-SSM-Additive (non-assoc DORMANT) | 8.4% | 11.9% | 1.5% |
| Naive-DIAG | 8.4% | 12.6% | 0.0% |
| Random | 6.25% | 6.25% | 6.25% |

### Key Findings

1. **O-SSM-Triple = H-SSM-Triple exactly** (0% on non-assoc triples both) — even with
   architecturally activated non-associativity, no separation
2. **Triple product hurts training** — both triple models (4.5%) perform WORSE than additive (8.4%)
   → multiplicative input injection creates optimization difficulty
3. **Artin dormancy partially confirmed** — additive O-SSM (1.5%) slightly above triple (0%) on
   non-assoc triples, but both negligible
4. **Non-associativity is unlearnable at H=8** — the Cayley product's non-associative structure
   cannot be exploited by gradient descent with 8-dim state and output-only training

### Interpretation

The triple product architecture successfully ACTIVATES non-associativity (verified: e1⊗e2⊗e5
gives different LEFT/RIGHT results). But the model cannot LEARN to exploit it.

## ★★★★ Multi-Head Unit-Octonion + Associator-as-Feature

### The Composition Algebra Hypothesis (CONFIRMED)

By Hurwitz's theorem, octonions are the MAXIMAL composition algebra: ||ab|| = ||a||·||b||.
Multiplication by a unit octonion is an ISOMETRY of R^8 — the exact property that makes
unitary/orthogonal RNNs work (Arjovsky 2016, AUSSM 2025). This prevents gradient vanishing/explosion.

**Architecture**: 4 heads × 8D, each head parameterized as A = decay × unit_octonion.
Associator ||[A,h,x]|| = ||(A⊗h)⊗x - A⊗(h⊗x)|| computed per head, fed as 4 extra features.

### Results: Associativity Probe (16 classes, random=6.25%)

| Model | Overall | Assoc-triples | Non-assoc-triples | A-params |
|-------|---------|---------------|-------------------|----------|
| MH-Oct (unit + assoc) | **15.5%** | **22.3%** | 2.1% | 36 |
| MH-Quat (unit) | **15.3%** | 20.5% | 5.1% | 36 |
| MH-Dense | 8.5% | 11.6% | 2.4% | 256 |
| MH-Diag | 7.6% | 11.1% | 0.6% | 32 |

### Results: ListOps L=15 (10 classes, random=10%)

| Model | Accuracy | A-params |
|-------|----------|----------|
| MH-Oct | 27.1% | 36 |
| MH-Quat | 33.8% | 36 |
| MH-Dense | 34.4% | 256 |
| MH-Diag | **54.0%** | 32 |

### Key Findings

1. **COMPOSITION ALGEBRA IS THE MECHANISM**: Unit-algebra models (Oct 15.5%, Quat 15.3%) are
   2.5× above random on probe, while dense (8.5%) and diagonal (7.6%) stay near random.
   Norm preservation (||Ah|| = ||h|| for unit A) prevents gradient pathology.

2. **Non-associativity still doesn't separate**: Oct (2.1%) vs Quat (5.1%) on non-assoc triples.
   The associator feature provides information but doesn't translate to better classification.

3. **Unit parameterization >> dense**: MH-Quat with 36 params (33.8%) matches MH-Dense with
   256 params (34.4%) on ListOps. Structured norm-preserving is as good as 7× more free params.

4. **Task specificity confirmed**: MH-Diag wins ListOps (54%) because MIN/MAX/MED doesn't need
   coupling — independent channels that each track one statistic work best.

## ★★★★★ Trajectory Divergence: First Signal of Non-Associativity Value

> ⚠️ **RETRACTED (2026-07-16):** this "first signal" ran on the broken `triple_product` octonion
> table (non-alternative, not a composition algebra — see top notice). The 0.1096 < 0.1157
> high-divergence MSE gap and the "slight advantage" conclusion are table artifacts and are **not
> re-validated** on the corrected algebra. No corrected figure is asserted here.

### Domain Shift: Dynamical Systems, Not Classification

**Key insight**: Non-associativity lives in trajectory divergence, not discrete classification.
For octonionic agents: `(A⊗B)⊗C ≠ A⊗(B⊗C)` → LEFT and RIGHT trajectories DIVERGE.
For quaternions: they're IDENTICAL (associativity).

### Architecture Fix: Identity Init + Residual

Previous triple product tests used h₀=0 (zero-state trap: `(A⊗0)⊗x = 0`).
Fix: h₀ = (1,0,...,0) (identity octonion) + 0.5·B·embed residual connection.

### Results (2 agents, 8 triples, MSE regression, 80ep × 200 samples)

| Model | MSE (all) | MSE (low-div) | MSE (high-div) | Ratio hi/lo |
|-------|-----------|---------------|-----------------|-------------|
| **O-SSM-Triple** | **0.1159** | 0.1284 | **0.1096** | 0.854 |
| H-SSM-Triple | 0.1185 | 0.1240 | 0.1157 | 0.933 |
| Naive-DIAG | 0.1164 | 0.1554 | 0.0969 | 0.624 |

### The Signal

**O-SSM-Triple has the lowest MSE on high-divergence triples among coupled models** (0.1096 vs 0.1157).
The gap is 0.006 (small) but consistent with the hypothesis: octonionic dynamics track
bracketing-dependent trajectory divergence slightly better than quaternionic dynamics.

**Caveats**:
- Output-only training (A not updated) — BPTT would likely amplify the effect
- Only 2 agents / 8 triples — need 4+ agents for statistical power
- The 0.006 gap is small relative to the 0.125 baseline — needs GPU scale to confirm
- Naive-DIAG actually has the lowest high-div MSE (0.097) because additive gradient flow is stronger

### Interpretation

This is the FIRST experiment where O-SSM outperforms H-SSM on any metric.
The difference is small but directionally correct: on a task where the ground truth
IS non-associative dynamics, the octonionic model shows a slight advantage.

The result suggests non-associativity provides value when:
1. **The domain is dynamical** (continuous trajectories, not discrete classification)
2. **The loss is regression** (MSE, not cross-entropy)
3. **The state is properly initialized** (identity octonion, not zero)
4. **Residual connections preserve signal** (additive bypass for gradient flow)
5. **The divergence is genuine** (ground truth computed from actual Cayley products)

This is the seed that needs GPU-scale compute to grow into a conclusive result.

## Completed Ablations

1. **Sorting K-sweep** {3,5,8,10} — escalabilidade com tamanho ✓
2. **Bracket L-sweep** {6,8,10,12,16,20} — convergência vs L ✓
3. **ListOps nesting depth** {0,1,2} — complexidade semântica ✓
4. **H-SSM v1 ablation** (flawed: 64p vs 8p) ✓
5. **BPTT ablation** (O-SSM collapses, H-SSM retains signal) ✓
6. **Native algebra parameter-matched** (H-SSM > O-SSM on both tasks) ✓
7. **Associativity probe** (O-SSM = H-SSM on non-assoc triples) ✓
8. **Triple-product probe** (Artin activation, O-SSM-Triple = H-SSM-Triple = 0% on non-assoc) ✓
9. **Multi-head unit-oct + associator** (composition algebra norm preservation IS the mechanism) ✓
10. **O-FSSM L-BFGS** (second-order doesn't unlock non-assoc; associative attractor phenomenon) ✓
11. **Trajectory divergence** (O-SSM-Triple 0.1096 < H-SSM-Triple 0.1157 on high-div triples — first signal, 2 agents) ✓
12. **Analytical BPTT** (exact Jacobian verified, but DESTROYS learning: 0.125 baseline. Non-assoc gradient landscape is adversarial) ✓
13. **Riemannian BPTT** (manifold projection doesn't help — gradient direction is the problem, not constraint) ✓
14. **Fractal-G2 v1** (G2-manifold A update → MeanAssocNorm=0.075 ALIVE, first signal) ✓ ⚠️ **RETRACTED 2026-07-16 — broken octonion table (see top notice); not re-validated.**
15. **Fractal-G2 v2** (full Jacobian + analytical associator grad → ~~+3.6pp over H-SSM, 3 seeds~~) ✓ ⚠️ **RETRACTED 2026-07-16 — the `+3.6pp` came from the broken octonion table; not re-validated.**
16. **Fractal-G2 v3 annealed** (λ=0.15→0.03 + 10 seeds → ~~O-SSM NA=4.0±0.7%, best seed 8.7% > 6.25% random~~) ✓ ⚠️ **RETRACTED 2026-07-16 — broken octonion table; not re-validated.**
17. **~~★★★★★ Fractal-G2 v3 + Fano curriculum~~** (~~compiler knows Fano triples → H-SSM collapses to 0.2% NA while O-SSM retains 1.5% → 7× advantage ratio, +2.1pp overall gap~~) ⚠️ **RETRACTED 2026-07-16 — the `7× advantage` and `+2.1pp` gap are artifacts of the broken octonion table (non-alternative, not a composition algebra). Not re-validated on the corrected algebra; no corrected figure asserted. This was the document's strongest positive non-associativity claim and it does not survive.**

## Paper Structure

### Abstract
Dissociation entre inductive biases: O-SSM (composição) vs S4-DIAG (periodicidade)

### Main Claims (Revised after NATIVE ALGEBRA ablation)
1. **Non-associativity is a LIABILITY**: True Cayley O-SSM (8p) loses to Hamilton H-SSM (8p) on both ListOps (-4.2pp) and Bracket (-11pp)
2. **Coupling + associativity is the optimal combination**: H-SSM combines cross-dimensional coupling (non-commutativity) with gradient-stable associativity
3. **The "O-SSM advantage" in prior literature is a capacity artifact**: dense `A @ h` (64 params) is not true octonion multiplication — it's a general linear map with 8× more parameters
4. **Non-commutativity is essential for composition**: H-SSM (non-commutative, coupled) >> Naive (commutative, diagonal) on both tasks
5. **Coupling matters more than algebraic richness**: Hamilton (simpler, 32 FLOPs) > Cayley (richer, 120 FLOPs) because gradient conditioning trumps expressiveness

### Experimental Evidence
- Table 1: Core 4 tasks + negative control
- Figure 1: K-sweep (sorting) — difficulty scaling
- Figure 2: L-sweep (bracket) — sequence length robustness  
- Figure 3: Nesting depth (ListOps) — compositional complexity
- Table 2: Ablations summary

### Discussion

#### Native Algebra Ablation: THE DEFINITIVE MECHANISM ISOLATION

With parameter-matched models (8 A-params each), using TRUE algebraic products:

| Property | O-SSM (Cayley) | H-SSM (Hamilton) | Naive (Diagonal) |
|----------|---------------|-----------------|------------------|
| Non-commutative | ✓ (full 8D) | ✓ (two 4D blocks) | ✗ |
| Non-associative | ✓ | ✗ | ✗ |
| Cross-dim coupling | ✓ (all 8↔8) | ✓ (4↔4 × 2) | ✗ |
| A parameters | 8 | 8 | 8 |
| FLOPs per step | 120 | 32 | 8 |
| **ListOps L=15** | 28.7% | **32.9%** | 27.3% |
| **Bracket L=12** | 56.5% | **67.5%** | 41.6% |

**Interpretation**: Non-associativity is a **LIABILITY**:
1. Cayley product creates **gradient noise** via (A⊗h₁)⊗h₂ ≠ A⊗(h₁⊗h₂)
2. Hamilton product provides **stable chain rule** via (AB)C = A(BC)
3. Full 8D coupling (Cayley) is weaker than block-diagonal coupling (Hamilton) because over-coupling reduces feature specialization
4. The previous "O-SSM advantage" was a **capacity artifact**: dense `A @ h` had 64 free params vs 8

**The correct hierarchy is**: Coupling + Associativity > Coupling + Non-Associativity >> No Coupling

#### Why Periodicity Fails for Composition
S4-DIAG (HiPPO diagonal rotation) excels at **pattern matching with preserved phase** (Bracket: 62-85%) but collapses on **nested semantics** (ListOps: 7.5%).

- **Periodicity is structure-agnostic**: phase rotation preserves distance to landmarks
- **Composition is structure-aware**: nested evaluation requires tracking intermediate states
- **H-SSM succeeds**: non-commutative coupling allows intermediate reasoning

#### Implications for Architecture Design
1. **Coupling + associativity is optimal**: Hamilton quaternion product provides the best trade-off
2. **Non-associativity harms gradient flow**: avoid Cayley/octonion products in SSM state transitions
3. **Non-commutativity is essential**: H-SSM (+25.9pp) >> Naive on Bracket
4. **Block-diagonal coupling preferred**: two independent 4D blocks beat one fully-coupled 8D block
5. **Computational efficiency**: H-SSM uses 32 FLOPs vs Cayley's 120, while performing better

## Files
- `s4_baseline_benchmark.sio` — Sorting (BPTT + output-only)
- `bracket_3way_benchmark.sio` — Bracket 3-way (L=8,12,16)
- `listops_3way_benchmark.sio` — ListOps 3-way (L=10,15,20)
- `ossm_fullbp_v2.sio` — Sorting with full BPTT
- `smnist_3way_benchmark.sio` — Negative control
- `sorting_k_sweep.sio` — K={3,5,8,10}
- `bracket_l_sweep.sio` — L={6,8,10,12,16,20}
- `listops_nesting_depth.sio` — Depth={0,1,2}
- `hssm_4way_benchmark.sio` — v1 ablation (flawed: 64p vs 8p mismatch)
- `hssm_deep_listops.sio` — Depth-3 ListOps (all models ≈ random → too hard for output-only H=8)
- `hssm_bptt_ablation.sio` — BPTT through A (O-SSM collapses, H-SSM retains signal)
- **`hssm_native_algebra.sio`** — True Cayley vs Hamilton vs Diagonal, 8p each, 1000 eval. H-SSM wins both tasks.
- `associativity_probe_benchmark.sio` — Direct octonionic composition (additive arch). O-SSM = H-SSM on non-assoc triples.
- `triple_product_ossm_benchmark.sio` — Triple product per Artin's theorem. Non-assoc unlearnable at single-head H=8.
- **`multihead_unit_oct_benchmark.sio`** — **★★★★ THE ANSWER**: Multi-head unit-octonion with associator-as-feature. Composition algebra norm preservation IS the mechanism (2.5× random). Non-associativity does not separate. Unit parameterization matches dense with 7× fewer params.

---

## ★★★ ZD Deep Dive: Gate vs Capacity vs Zero-Divisors (2026-04-29)

> ⚠️ **RETRACTED (2026-07-16).** This entire section rests on an 8-dim `zd_*` / `gate_vs_zd`
> "zero-divisor" table that carries the same broken `e2·e5` sign error. Two compounding problems:
> (1) the table is non-alternative (not a composition algebra), and (2) **real octonions are a
> division algebra with no zero divisors at all** — a broken non-alternative 8-dim table can
> *manufacture* spurious zero-divisor behavior. So every "ZD advantage" figure here (including the
> headline `+24.45pp`) is a table artifact. On the corrected sedenion, `zd_bptt` re-runs to a
> +0.00pp ZD advantage. These results need re-derivation, and genuine zero divisors live in the
> 16-dim sedenion algebra, not this 8-dim table. **Not re-validated; no corrected figure asserted.**

### Motivation

After the definitive H-SSM > O-SSM result, the S-SSM (Selective Sedenion SSM) was introduced
as a candidate that exploits zero-divisors (ZD) — impossible in division algebras (R,C,H,O).
The 168 ZD projective classes in 𝕊 are formally bridged to the 168 non-Fano triples of 𝕆
(Bridge Theorem in `formal/lean4/SounioZeroDivisorBridge.lean`).

### Ablation Series

#### 4-Way (native_algebra_4way_benchmark.sio)

| Model | Architecture | ListOps L=15 | Bracket L=12 | Notes |
|-------|-------------|-------------|-------------|-------|
| O-SSM | Cayley, 8D, 8A | 28.7% | 56.5% | anchor |
| H-SSM | Hamilton, 8D, 8A | 32.9% | 67.5% | prev winner |
| S-SSM | Sedenion+gate, 16D, 16A+13gate | **41.3%** | **79.6%** | 2× A-params |
| Naive | diagonal, 8D, 8A | 27.3% | 41.6% | anchor |

#### 6-Way Gate vs ZD Ablation (gate_vs_zd_ablation.sio)

| Model | Description | ListOps | Bracket |
|-------|-------------|---------|---------|
| O-SSM | 8D, no gate | 28.7% | 56.5% |
| H-SSM | 8D, assoc, no gate | 32.9% | 67.5% |
| H-SSM-sel | H-SSM + Mamba gate | 28.9% | 55.8% |
| S-nogate | Sedenion 16D, no gate | 44.7% | 76.2% |
| S-SSM | Sedenion 16D + gate | 41.3% | 79.6% |
| Naive | diagonal, no gate | 27.3% | 41.6% |

**Decomposition:**
| Factor | ListOps | Bracket |
|--------|---------|---------|
| Gate alone (H-sel − H) | **−4.0pp** | **−11.7pp** |
| ZD+16D alone (Sng − H) | +11.8pp | +8.7pp |
| Gate+ZD+16D (S − H) | +8.4pp | +12.1pp |

→ Gate alone **hurts** when applied to H-SSM. ZD+16D is the dominant gain factor.

#### ZD vs Capacity Killer Experiment (zd_vs_capacity_ablation.sio)

| Model | Description | ListOps | Bracket |
|-------|-------------|---------|---------|
| H-SSM-8D | 2×quat, 8D, 8A, NO ZD | 32.9% | 67.5% |
| H-SSM-16D | 4×quat, 16D, 16A, NO ZD | **50.8%** | **79.1%** |
| S-nogate | Sedenion, 16D, 16A, ZD, no gate | 44.7% | 76.2% |

**Capacity gain (H16−H8):** +17.9pp ListOps, +11.6pp Bracket
**ZD net contribution (Sng−H16):** −6.1pp ListOps, −2.9pp Bracket

→ At matched dimensionality/params: H-SSM-16D > S-nogate. **ZD costs performance on retention tasks.**

#### ZD-Favoring Benchmarks: Token-Erase and Mode-Switch

**Token-Erase** (`token_erase_benchmark.sio`):
Sequence: [noise×4][ERASE][signal×3], label from signal only.

| Model | Accuracy | vs Random |
|-------|----------|-----------|
| H-SSM-16D | 39.3% | +29.3pp |
| S-SSM-rand | 39.8% | +29.8pp |
| S-SSM-ZDinit (A=e3+e10) | 23.1% | +13.1pp |

**Mode-Switch** (`mode_switch_benchmark.sio`):
Sequence: [mode-A×5][SWITCH][mode-B×5], label from mode-B only.

| Model | Accuracy | vs Random (50%) |
|-------|----------|-----------------|
| H-SSM-16D | 88.3% | +38.3pp |
| S-SSM-rand | 89.3% | +39.3pp |
| S-SSM-ZDinit (A=e3+e10) | 69.7% | +19.7pp |

→ ZD-init **does not help** on designed reset tasks. H-SSM and random S-SSM perform equivalently.

### Formal Contribution: Bridge Theorem

**`formal/lean4/SounioZeroDivisorBridge.lean`** — compiled, no sorry:

```
theorem nonfano_zd_bridge :
    nonFanoCount = unorderedZDPairs.length
-- Both = 168 = |PSL(2,7)|

theorem zd_census_matches_report :
    validPrims.length = 84 ∧
    orderedZDPairs.length = 336 ∧
    unorderedZDPairs.length = 168 ∧
    activeLabels.length = 7 ∧
    nonFanoCount = 168

theorem zd_labels_mirror_fano_indices :
    (activeLabels.map (fun L => L ^^^ 8)).mergeSort = [1, 2, 3, 4, 5, 6, 7]
-- 7 ZD xor-fibers ↔ 7 Fano line indices
```

### Summary of ZD Findings

| Question | Finding |
|----------|---------|
| Are ZD a formal algebraic feature? | **YES** — Bridge Theorem proven in Lean4 |
| Do ZD help on retention tasks? | **NO** — ZD costs −6.1pp vs H-SSM-16D on ListOps |
| Do ZD help on designed reset tasks? | **NO** — H-SSM-16D matches or exceeds S-SSM-ZDinit |
| Why did S-SSM win 4-way? | Primarily 2× A-params capacity (16D vs 8D) |
| What about the S-SSM gate? | Gate helps WITH ZD (+3.4pp Bracket) but hurts alone |

### Interpretation

Zero-divisors in 𝕊 are the algebraic image of non-associative triples in 𝕆 under Cayley-Dickson
(Bridge Theorem, 168 = 168). This is a deep structural correspondence. However:

1. **ZD as learned mechanism requires BPTT**: output-only training cannot align ERASE-token
   embeddings with ZD complement directions. The gradient does not flow back through A.

2. **ZD-init disrupts state dynamics**: starting near the ZD locus (A = e3+e10) causes excessive
   state annihilation at every token, not selectively at reset tokens. The model can't unlearn this.

3. **Capacity dominates at output-only scale**: H-SSM-16D wins because 16D ≥ 8D with better
   gradient conditioning from associativity, not because ZD is absent.

4. **Gate and ZD interact**: the Mamba gate compensates for ZD-induced instability (+3.4pp).
   Without gate, ZD is a net liability. The S-SSM advantage in the 4-way is capacity + gate,
   not the ZD property per se.

**Open question**: would ZD prove advantageous under BPTT training where A can learn to
align with ZD locus specifically at reset tokens? This requires gradient flow through sed_product.


---

## Direction 2+3: Full 4D Kernel Geometry + Capacity-Constrained Task (2026-04-29)

### Direction 2 — Lean4: 4D Right-Annihilator of (e3+e10)

File: `formal/lean4/SounioZeroDivisorBridge.lean` (§9-11 added)

**Proved theorems** (all by `native_decide`):

| Theorem | Statement |
|---------|-----------|
| `annihilators_valid` | All 4 right-annihilators are valid primitives |
| `annihilators_in_fiber9` | All 4 are in fiber 9 (xor-label = 3⊕10 = 9) |
| `annihilators_are_zero_pairs` | Each is annihilated by A = e3+e10 |
| `annihilators_complete` | A has EXACTLY 4 right-annihilators (degree-4 graph) |
| `annihilators_distinct` | The 4 are pairwise distinct (Nodup) |
| `fiber9_is_4regular` | The 12-vertex annihilation graph on fiber 9 is 4-regular |
| `fiber9_annihilation_is_local` | All annihilation in fiber 9 is intra-fiber |
| `every_primitive_has_4_annihilators` | EVERY primitive has exactly 4 right-annihilators |
| `total_forgettable_dimension` | 84 × 4 = 336 = total ordered pairs (structural) |
| `zd_selective_forgetting_summary` | Full structural summary theorem |

**Key insight (§9-11)**: In sedenion 𝕊 with 16D state:
- Every primitive ZD element A defines a **4D forgettable subspace** (right-annihilator)
- h → A⊗h kills the 4D kernel component EXACTLY, preserves the 12D co-kernel
- This 4:12 capacity split is **structurally impossible** in division algebras (ℝ, ℂ, ℍ, 𝕆)
- 168 projective ZD classes = 168 independent forgettable-4D configurations
- 7 fibers = 7 independent "forgetting channels" in 𝕊

**Note on conventions**: The Lean4 proof uses the `cdSigma` octonion table (Cayley-Dickson
derivation). The Sounio benchmarks use an `oct_p` table (explicit Fano-plane formula). Both
are valid isomorphic sedenion algebras with 168 ZD classes. The pair (e3+e10, e6-e15) is
shared by both conventions (verified by stdlib + all benchmarks). The other 3 kernel vectors
of A differ by convention.

---

### Direction 3 — Capacity-Constrained Forgetting Task

File: `examples/zd_capacity_constrained.sio`

**Task design**:
```
Phase 1: [PERM tokens]  — encode bits into 12D permanent subspace (co-kernel of A)
Phase 2: [TEMP token]   — encode pattern into ZD kernel direction u1 = (e6-e15)/√2
         [ERASE step]   — apply A = e3+e10
Phase 3: [FRESH token]  — encode new pattern into the freed u1 slot
Output:  predict perm_bit[0] XOR fresh_pattern
```

**Part A — Constructive proof of contamination gap** (all analytical, no training):

| Test | S-SSM-ZD | H-SSM |
|------|----------|-------|
| ‖u1=(e6-e15)/√2 after ERASE‖ | **0.000000** (exact annihilation) | 0.950000 |
| ‖u2=(e7+e14)/√2 after ERASE‖ | **0.000000** (exact annihilation) | 0.950000 |
| ‖e0 after ERASE‖ | 1.414214 (preserved, remapped) | 0.950000 |
| ⟨h_after, u1⟩ (after ERASE+fresh) | **2.000 = fresh only** | 3.900 = 0.95×temp + fresh |
| Contamination gap | 0 (no contamination) | **+1.9 = 95% of fresh signal** |

**Part B — Trained benchmark** (output-only SGD, 100 epochs × 400 samples):

| Model | Accuracy (2000 test) |
|-------|----------------------|
| S-SSM-ZD (A=e3+e10, ZD-init) | ~~**76.15%**~~ |
| H-SSM (A=0.95×I, no ZD) | 51.70% |
| **ZD advantage** | ~~**+24.45pp**~~ ⚠️ **RETRACTED — broken table artifact (see section notice above); `zd_bptt` → +0.00pp on corrected sedenion** |

**Structural ceiling**:
- S-SSM-ZD: temp erased EXACTLY → fresh slot clean → 100% achievable
- H-SSM: residual contamination = 95% of temp signal → ceiling < 100%

**Interpretation**:

ZD selective forgetting IS a learnable advantage under the capacity-constrained task design.
The key conditions that make ZD advantageous:
1. **Aligned initialization**: perm tokens → co-kernel (e0..e3), temp/fresh → kernel (u1)
2. **Structural separation**: the task enforces that temp and fresh share the SAME 4D slot
3. **Exact annihilation**: the ZD mechanism cleans the slot in ONE step (not exponential decay)

This is distinct from previous benchmarks (Token-Erase, Mode-Switch) where:
- The training objective did not specifically exploit the structural separation
- Output-only gradients could not align ERASE-token embeddings with the ZD complement

**Conclusion**: ~~ZD's hard-reset property creates a structural advantage that IS exploitable with
ZD-aligned embedding initialization + output-only training, as long as the task forces reuse
of the same 4D subspace for both temp and fresh information. The 24.45pp gap is the cleanest
empirical demonstration of ZD advantage to date.~~ ⚠️ **RETRACTED 2026-07-16 — this conclusion is
built on the broken 8-dim `zd_*` table (see section notice); the "zero divisors" it uses are
spurious. Not re-validated; `zd_bptt` → +0.00pp on the corrected sedenion.**


---

## Real Applications: ZD as Solution to Open Engineering Gaps (2026-04-29)

### Application 1 — Epistemic Collapse: Ghost Beliefs in Filtering

File: `examples/zd_epistemic_collapse.sio`

**The gap**: Bayesian filters (Kalman, particle filter) cannot eliminate probability mass from
"impossible zones" exactly. Hard constraints (physical walls, logical impossibilities) produce
ghost beliefs — numerically small but never exactly zero. This contaminates downstream decisions.

**Formal impossibility** (connects to SounioZeroDivisorBridge.lean §9):
- Division algebras: ker(A⊗·) = {0} → ghost beliefs are unavoidable
- Sedenion 𝕊: ker(A⊗·) = 4D subspace → exact algebraic collapse

**Results**:

| Measure | ZD-filter | H-filter (α=0.01) | H-filter (α=0.99) |
|---------|-----------|-------------------|-------------------|
| Ghost mass after 1 ERASE | **0.000000** | 0.000100 | 0.990000 |
| H-filter ghost accumulation (200 steps) | 1× | 1.16× | **22×** |
| Decision contamination (100 trials) | **0.000** | 0.100 | ~1.98 |

**lim(α→1) ghost mass = ∞**: any division-algebra filter degrades arbitrarily as decay
becomes less aggressive. ZD is independent of tuning and structurally exact.

---

### Application 2 — Anti-Windup PID: Exact I-Term Reset

File: `examples/zd_antiwindup_pid.sio`

**The gap**: Integrator windup causes overshoot and instability when PID output saturates.
All standard anti-windup methods (back-calculation, conditional integration, decay) are
approximate: they reduce but never eliminate the integral residual in a single step.

**Formal claim**: for any α ∈ (0,1) in a division-algebra state space, the I-term residual
after saturation is α^k × I_0 > 0 for all finite k. Exact reset is structurally impossible.

**ZD solution**: encode I-term in kernel direction (e6-e15)/√2. At saturation:
`I_new = (e3+e10) ⊗ I_encoded = 0` exactly, for ANY initial I value.

**Analytical results** (I_init = 3, 6, 9, 12, 15):
```
ZD → 0.000000 (exact, regardless of magnitude)
α=0.9 → I_init × 0.9 (always > 0)
α=0.5 → I_init × 0.5 (always > 0)
```

**PID simulation** (plant y[t+1]=0.8y+u, Kp=1.0, Ki=0.5, saturation=3.0):

| Strategy | Overshoot | Settling time |
|----------|-----------|---------------|
| No anti-windup | 1.420 | 4 steps |
| **ZD (exact reset)** | **0.082** | **2 steps** |
| Decay α=0.5 | 0.082 | 2 steps |
| Clamp (standard) | 0.620 | 4 steps |

ZD matches optimal decay without requiring any tuning parameter α. The advantage over
decay is structural: ZD is parameter-free and exact for arbitrarily large I values,
while any decay strategy requires tuning and degrades for large initial conditions.

---

### Unified Structural Argument

Both applications reduce to the same algebraic theorem:

> **In any division algebra (ℝ, ℂ, ℍ, 𝕆)**: ∀ linear maps A, ∀ h ≠ 0: A⊗h ≠ 0.
> Exact state collapse in a subspace is STRUCTURALLY IMPOSSIBLE.
>
> **In sedenion 𝕊**: ∃ A = e3+e10 with ker(A⊗·) = 4D subspace.
> Exact collapse of any component in that subspace is achievable in 1 step.

This theorem (proved in SounioZeroDivisorBridge.lean) directly motivates ZD-SSM as a
modeling substrate for systems requiring hard impossibility constraints — not soft decay.
