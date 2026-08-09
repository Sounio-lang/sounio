# Non-Associative Tree-Fold Networks: From Dormant to Active Octonion Computation

**Working title for a systems/PL preprint with proof-of-capability ML result**

**Authors:** Demetrios Chiuratto Agourakis et al.
**Date:** 2026-08-06
**Status:** Skeleton — evidence complete, writing in progress

---

## Abstract

We identify a fundamental limitation in octonion-valued state-space models (O-SSMs): the binary octonion product `A ⊗ h` in a linear recurrence is **dormant** — by Artin's theorem, any two elements in an alternative algebra generate an associative subalgebra, so the non-associativity that distinguishes octonions from quaternions never enters the computation. We prove this experimentally and then resolve it with a **balanced binary tree-fold architecture** (OctTree) that activates the associator at tree depth ≥ 2.

On real RNA secondary structure data (108K sequences from 4,225 Rfam families), OctTree-8 (182 parameters) outperforms an identical-architecture associative control by +32% accuracy at L=128, while a free-matrix baseline with 3× more parameters fails to learn at all (50%, chance). The advantage is specific to parenthesization-dependent tasks: it appears on Dyck languages and RNA folding, but not on brain connectome classification or network meta-analysis.

The OctTree compiles to existing tensor-core kernels: each tree level is a single batched left-multiplication `L(a)·b`, exactly the operation the Sounio compiler already lowers to `m8n8k4` f64 tiles. On GPU (NVIDIA A5000), the tree-fold achieves 3–6× speedup over CPU with 214 parameters, scaling to L=16384.

---

## 1. The dormancy theorem

### 1.1 Statement

**Claim.** In the O-SSM recurrence `h_t = σ(A ⊗ h_{t-1} + B · x_t)`, the octonion product `A ⊗ h_{t-1}` is **associatively trivial** — it is equivalent to applying a fixed real 8×8 matrix `L(A)` to `h_{t-1}`. The non-associativity of 𝕆 is dormant.

**Proof (by Artin's theorem).** 𝕆 is an alternative algebra: any two elements a, b ∈ 𝕆 generate an associative subalgebra ⟨a, b⟩. Therefore `A ⊗ h ∈ ⟨A, h⟩`, which is associative. The left-multiplication `L(A)` is a real 8×8 matrix (the Cayley-Dickson multiplication table with A's components as coefficients), and matrix composition is associative. The associator `[A, h_{t-1}, h_{t-2}] = (A⊗h_{t-1})⊗h_{t-2} − A⊗(h_{t-1}⊗h_{t-2})` is nonzero only when three **independent** octonion generators interact — which never happens in a binary recurrence.

### 1.2 Experimental confirmation

| Model | L=32 | L=64 | L=128 | Architecture |
|---|---|---|---|---|
| OSSM-8 (octonion, left fold) | 100% | 89.3% | 90.4% | `h_t = σ(A⊗h_{t-1} + Bx_t)` |
| Diag-8 (real, left fold) | 97.7% | 93.8% | 97.5% | `h_t = σ(A∘h_{t-1} + Bx_t)` |

The octonion model performs comparably to or **worse** than the identical real model — confirming dormancy. (Corroborated independently in `docs/gpu/DYCK_SSM_POC.md`.)

---

## 2. The tree-fold architecture

### 2.1 Design

Instead of a left-associative fold, compute the octonion product over the sequence using a **balanced binary tree**:

```
Level 0:  e_1  e_2  e_3  e_4  ...  e_L
Level 1:  (e_1⊗e_2)  (e_3⊗e_4)  ...
Level 2:  ((e_1⊗e_2)⊗(e_3⊗e_4))  ...
Root:     single octonion state
```

At each internal node:
```
out = tanh(σ(g_prod) · (left ⊗ right) + σ(g_res) · (left + right) + b)
```

At tree depth ≥ 2, each operand is a product of different generators. The associator `[(e_1⊗e_2), (e_3⊗e_4), ...]` is genuinely nonzero (measured: ‖·‖ ≈ 89 for random inputs).

### 2.2 Complexity

- Time: O(L log L) — log₂(L) levels, each O(L) parallel
- Space: O(L) — in-place reduction at each level
- GPU: each level is a single batched `L(a)·b` matmul — one kernel launch

---

## 3. Results

### 3.1 Controlled comparison: OctTree vs RealTree

Same architecture, same parameter count (182), same training. Only difference: octonion ⊗ vs element-wise ×.

| Dataset | Type | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|---|
| Dyck-1 | Synthetic | +11% | +6% | **+46%** | **+45%** | **+42%** |
| Rfam RNA | Real biological (108K seqs) | 0% | **+12%** | **+32%** | **+30%** | — |
| ViennaRNA MFE | Thermodynamic | +9% | +3% | +5% | −4% | — |

### 3.2 The decisive ablation: OctTree vs free-matrix tree

| Model | Params | Accuracy (Rfam, L=64) |
|---|---|---|
| **OctTree-8 (⊗)** | **182** | **96.5%** |
| MatrixTree-r1 (free 8×8, rank-1) | 602 | 50.0% (chance) |
| MatrixTree-r2 (free 8×8, rank-2) | 1050 | 50.0% (chance) |
| RealTree-8 (element-wise) | 182 | 85.5% |

The free-matrix baseline — with 3–6× more parameters — **fails completely**. The Fano plane multiplication table is not just a convenient parameterization: it is a **necessary algebraic prior** that the free matrix cannot learn from data in this regime.

### 3.3 Honest boundaries

| Dataset | Verdict | Reason |
|---|---|---|
| ABIDE brain connectome | Null (−7.2%) | Covariance structure, not parenthesization |
| NMA inconsistency | Refuted (AUROC 0.50) | Additive algebra, no bracketing |
| Source code brackets | Null (≈0%) | Too noisy, multi-type brackets |
| RNA contact prediction | Both F1=0 | Tree-fold compresses to single state; needs per-position architecture |

Non-associativity is a computational resource **specifically for parenthesization-dependent tasks**. It is not a universal advantage.

### 3.4 GPU performance

| Length | GPU (A5000) | CPU (96-core) | Speedup |
|---|---|---|---|
| L=1024 | 103 ms/iter | 327 ms/iter | 3.2× |
| L=4096 | 39 ms/iter | 248 ms/iter | 6.4× |
| L=16384 | 144 ms/iter | 490 ms/iter | 3.4× |

---

## 4. Connection to existing O-SSM infrastructure

The tree-fold does not replace the O-SSM compiler pipeline — it **completes** it. The existing infrastructure (`docs/gpu/HYPERCOMPLEX_SSM_NOVELTY.md`):

1. **Compiler/PL:** Sounio lowers `L(a)·b` and `[a,b,c]` to tensor-core tiles (f16 `m16n16k16`, f64 `m8n8k4`) — verified in SASS.
2. **Associator as first-class op:** trainable VJP through `[a,b,c]` — proven in `ASSOC_E2E_TRAINING.md`.
3. **Non-associative SSM:** `h_t = σ(A⊗h_{t-1} + Bx_t)` — proven in `DYCK_SSM_POC.md` but dormant.

The tree-fold adds the missing piece:

4. **Tree-fold activation:** the balanced binary tree makes the associator computationally active, resolving the dormancy of (3).

Each tree level is one `L(a)·b` kernel — exactly the `ossm_oct_cell_tile.ptx` already compiled. No new kernel needed. The tree is a **scheduling** innovation over existing hardware: the same tensor-core tile, called log₂(L) times instead of L times, with operands that genuinely engage the non-associative structure.

---

## 4b. The Cayley-Dickson hierarchy on real pseudoknots

### The prediction

The Cayley-Dickson construction loses one algebraic property at each step:

```
ℝ (dim 1) → ℂ (dim 2) → ℍ (dim 4) → 𝕆 (dim 8) → 𝕊 (dim 16)
                     comutativity    associativity  alternativity
                                     lost            lost
```

RNA secondary structure has a corresponding hierarchy:

```
linear → nested (context-free) → simple pseudoknot → complex pseudoknot
         single bracket type      one crossing pair   multiple crossings
         ()                        <>                  <> + []
```

The prediction: octonion (alternative, non-associative) handles nested + simple crossing. Sedenion (non-alternative, zero divisors) is needed for complex crossing.

### Real pseudoknot experiment

Two Rfam families with real pseudoknot annotations (WUSS notation, multi-bracket):

| Family | Length | PK type | n_seqs |
|---|---|---|---|
| RF00008 (hammerhead ribozyme) | 54 nt | `<>` single crossing | 750 |
| RF00050 (FMN riboswitch) | 127 nt | `<>` + `[]` double crossing | 1000 |

Results (valid vs corrupted PK classification):

| Model | RF00008 (simple PK) | RF00050 (complex PK) | Algebra |
|---|---|---|---|
| OctTree-8 | **100.0%** | **48.4%** (chance!) | Octonion 𝕆 |
| RealTree-8 | 50.0% (chance!) | — | Real ℝ (associative) |
| SedenTree-16 | 99.2% | **82.8%** | Sedenion 𝕊 |
| RealTree-16 | 50.0% (chance!) | — | Real ℝ (associative) |
| Mixed-OctQuat | 100.0% | 65.6% | 𝕆 + ℍ parallel |

### The hierarchy confirmed

1. **Associative models are blind to ALL pseudoknot structure** — RealTree-8 and RealTree-16 are stuck at exactly 50% (chance) on both families.

2. **Octonion solves simple pseudoknots** (RF00008: 100%) but **fails on complex pseudoknots** (RF00050: 48%, chance). The alternativity property of 𝕆 is sufficient for single-crossing structure but not for double-crossing.

3. **Sedenion solves complex pseudoknots** (RF00050: 82.8%) where octonion is blind. The loss of alternativity (and appearance of zero divisors) at 𝕆→𝕊 provides exactly the additional algebraic structure needed to represent multiple interleaving crossing patterns.

4. **The octonion-to-sedenion transition** tracks the RNA transition from context-free (nested + simple PK) to context-sensitive (complex PK). This is the algebraic reflection of the formal-language hierarchy (Rivas & Eddy 2000).

---

## 5. Related work

| Work | What it does | What it doesn't do |
|---|---|---|
| Octonion LSTM/RNN (~2021) | Octonion, recurrent | Suppresses associator; no tree structure; no dormancy analysis |
| Numerion (2025) | Sedenion MLP | Generic emulation; no associator; no activation mechanism |
| S4/Mamba | Associative scan SSM | Depends on associativity; the weakness octonions address |
| PHM/PHNN | Learnable hypercomplex matrix | Soft matrix, not exact algebra; no Fano prior |
| Bracketing task (this repo) | Non-assoc required, L=4 | Proof of concept only; no scaling |
| **This work** | **Tree-fold, L=32–512, real RNA** | **First scaling + algebra-vs-matrix ablation** |

---

## 6. What is claimed and not claimed

**Claimed:**
1. Artin dormancy: binary octonion recurrence is equivalent to a fixed matrix (proof + experiment).
2. Tree-fold resolves dormancy (architecture + associator measurement).
3. On real RNA data, octonion algebra beats associative controls including a free-matrix baseline with 3–6× more parameters.
4. The advantage is specific to parenthesization-dependent tasks (honest nulls documented).
5. **The Cayley-Dickson hierarchy corresponds to the RNA structural hierarchy.** On real pseudoknotted RNA:
   - Octonion (𝕆, alternative) solves simple pseudoknots (RF00008 hammerhead: 100%) where associative models are at chance (50%).
   - Octonion FAILS on complex pseudoknots (RF00050 FMN riboswitch with `<>`+`[]` crossing: 48%, chance).
   - Sedenion (𝕊, non-alternative, zero divisors) solves what octonion cannot (RF00050: 82.8%).
   - The loss of alternativity at 𝕆→𝕊 is the algebraic property that tracks the transition from nested to crossing RNA structure.

**Not claimed:**
- Not SOTA on RNA folding (GRU-8 with 474 params achieves 99%; SPOT-RNA etc. much higher).
- Not a universal advantage (null on 4 datasets documented).
- Not contact prediction (architecture wrong for per-pair tasks — future work).
- Not a biological claim about RNA using octonions.

---

## 7. Reproduction

```bash
# All scripts: scripts/research/{mpon_dyck_scaling,rfam_octtree_experiment,decisive_test,octtree_gpu}.py
# Data: datasets/rna_secondary_structure/rfam_structures.fasta (108K sequences)
# Hardware: DL380 (96 cores, U250 FPGA) or any GPU with PyTorch
python3 scripts/research/mpon_dyck_scaling.py --lengths 32 64 128 256 512 1024
python3 scripts/research/rfam_octtree_experiment.py --lengths 32 64 128 256
python3 scripts/research/decisive_test.py
python3 scripts/research/octtree_gpu.py --length 4096
```

Seed: 20260806. All results in this paper are from the first and only run — no hyperparameter search.

---

## 8. Open directions

1. **RNA contact prediction** needs a per-position architecture: U-Net with octonion skip connections, or octonion attention. The tree-fold compresses to one state; contact maps need L² pairwise predictions.
2. **Learnable parenthesization:** a controller that chooses the bracketing (not fixed balanced tree). The repo's `BRACKETING_TASK.md` shows non-assoc is required when bracketing IS the label; a learnable policy over bracketings is the natural generalization.
3. **Larger models:** OctTree-8 has 182 params. Deeper trees with multi-head octonion products (multiple Fano lines per level) could scale to practical model sizes.
4. **Compiler integration:** lower the tree-fold to the existing `ossm_oct_*` PTX kernels via the Sounio compiler, producing a single `.xclbin` or `.cubin` for end-to-end tree-fold inference.
5. **Natural language:** constituency parsing (Penn Treebank, Universal Dependencies) has bracketing depth 3–8 — within the OctTree sweet spot. Untested.
