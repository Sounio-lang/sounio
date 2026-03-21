<!-- docs:meta
topic_id: repo.paper.preprint
authority: repo_only
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.paper.preprint
-->

# Epistemic Shadow Lanes: Compiler-Generated Uncertainty Propagation for GPU Kernels

---

## Abstract

We present a GPU compiler transformation that generates first-order GUM-style (JCGM 100:2008) uncertainty propagation alongside every arithmetic instruction in GPU kernel code, with optional correlated-input and second-order extensions. For each value register in emitted PTX, the compiler produces three shadow registers — uncertainty variance, validity predicate, and provenance bitset — that implement the law of propagation of uncertainty without programmer intervention. We study whether epistemic metadata can guide five GPU-specific execution-time optimizations without violating a specified uncertainty budget: (1) warp-level ballot voting on validity predicates to gate budget-bounded dual-path execution, (2) Shannon entropy of the uncertainty distribution for kernel variant selection, (3) provenance-aware topological DAG scheduling with automatic stream parallelism, (4) speculative guards for refinement kernels below an uncertainty threshold, and (5) provenance-weighted kernel fusion scoring. Across a pharmacokinetic case study involving four noisy sensors, the technique eliminates approximately 190 lines of manual variance mathematics per CUDA kernel. Overhead is characterized by an instruction count model (Section 3.3); hardware benchmarks are in progress. We situate this work against Enzyme's shadow registers for automatic differentiation and Uncertain\<T\>'s first-order uncertain types, arguing that GUM variance propagation through GPU tensor cores occupies an unaddressed point in the design space.

---

## 1. Introduction

GPU-accelerated scientific computing routinely discards a critical property of its inputs: measurement uncertainty. A sensor reading of 50.0 ng/mL with standard uncertainty 2.0 ng/mL enters a GPU kernel as a bare `float`, and the output — however many multiply-accumulate operations later — is equally bare. The uncertainty is gone. The provenance is gone. The scientist must reconstruct both manually, if at all.

The Guide to the Expression of Uncertainty in Measurement (GUM, JCGM 100:2008) [1] standardizes how to propagate measurement uncertainty through computation. For a measurand $y = f(x_1, \ldots, x_N)$ with uncorrelated inputs, the combined standard uncertainty is:

$$u_c(y) = \sqrt{\sum_{i=1}^{N} \left(\frac{\partial f}{\partial x_i}\right)^2 u^2(x_i)}$$

This is a *mechanical transformation* of the same arithmetic the programmer already writes. The partial derivatives (sensitivity coefficients) for elementary operations are known at compile time: for addition, $\partial(a+b)/\partial a = 1$; for multiplication, $\partial(ab)/\partial a = b$. This observation is the basis of our technique.

### The gap

Three bodies of work surround this space without occupying it:

1. **Automatic differentiation on GPU.** Enzyme [2] inserts shadow registers alongside value registers in LLVM IR to propagate partial derivatives through GPU kernels. The structural mechanism — compiler-generated shadow computation in GPU IR — is analogous to our approach. However, Enzyme propagates *Jacobian entries* ($\partial f/\partial x_i$), not *combined variance* ($u_c^2(y)$). These are different mathematical objects with different composition rules (derivatives chain-multiply; variances quadrature-sum).

2. **Uncertain types.** Bornholt et al. [3] introduced Uncertain\<T\> as a first-order type for uncertain data in managed languages. The type-level concept is valuable, but Uncertain\<T\> uses Monte Carlo sampling at runtime (not analytical propagation) and has no GPU backend.

3. **CPU uncertainty libraries.** Python's `uncertainties` [4] and Julia's `Measurements.jl` [5] propagate uncertainty at the library level. Both are CPU-only, with measured overheads of 1,400x and 50-1,500x respectively [6,7], and neither can be used inside GPU kernels.

No prior work generates GUM-compliant variance propagation at the compiler level targeting GPU instruction sets. This paper fills that gap.

### Contributions

We present five techniques, implemented in the Sounio compiler, that are — to our knowledge — novel:

1. **Epistemic shadow lanes** (Section 3): a compiler pass that, for each arithmetic instruction on a value register `%r_val`, emits GUM variance propagation on a shadow register `%r_eps`, validity conjunction on `%p_valid`, and provenance OR-merge on `%r_prov`. Provenance uses bitwise OR (monotonic union) — not XOR — ensuring idempotent accumulation ($\text{prov}(a) \mathbin{|} \text{prov}(a) = \text{prov}(a)$, unlike XOR which self-cancels).

2. **Warp-vote epistemic fast-path** (Section 4.1): dual-path PTX where `vote.sync.ballot` on the validity predicate gates whether the warp executes full shadow propagation or a *budget-bounded* fast path that computes conservative uncertainty upper bounds. No prior system uses warp-level ballot voting on epistemic predicates. Critically, the fast path still propagates uncertainty — it does not skip shadow computation entirely, because small uncertainty $\neq$ zero uncertainty.

3. **Entropy-gated kernel dispatch** (Section 4.2): the host computes Shannon entropy $H(\epsilon)$ of the input uncertainty distribution and selects between kernel variants. No prior GPU compiler uses information-theoretic measures of data uncertainty for dispatch decisions. We note that entropy thresholds are currently calibrated for raw epsilon in $[0, 1]$; normalization to relative uncertainty ($\epsilon / |x|$) for unit-independence is straightforward but not yet implemented.

4. **Provenance-aware DAG scheduling** (Section 4.3): kernels with disjoint provenance tags (determined by bitset intersection) are assigned to separate CUDA streams. The compiler emits structured metadata; the host glue parses it and generates multi-stream launch code.

5. **Speculative epistemic execution** (Section 4.4): uncertainty-threshold guards that conditionally defer refinement kernels when aggregate uncertainty is below a configurable $\epsilon$. The threshold must be tied to a formal uncertainty budget to be scientifically meaningful — an ad hoc runtime convention is insufficient.

---

## 2. Background: GUM Propagation Rules

The GUM [1] specifies variance propagation for elementary operations assuming uncorrelated inputs. We implement the following rules, applied by the compiler at each arithmetic instruction:

| Source operation | Shadow emission: $u^2(\text{result})$ | GUM reference |
|-----------------|---------------------------------------|---------------|
| `add.f64 %c, %a, %b` | $u^2(a) + u^2(b)$ | Eq. (10), uncorrelated |
| `sub.f64 %c, %a, %b` | $u^2(a) + u^2(b)$ | Eq. (10) |
| `mul.f64 %c, %a, %b` | $b^2 u^2(a) + a^2 u^2(b)$ | Eq. (10) |
| `div.f64 %c, %a, %b` | $u^2(a)/b^2 + a^2 u^2(b)/b^4$ | Eq. (10) |
| `sqrt.f64 %c, %a` | $u^2(a) / (4a)$ | Eq. (13) |

For tensor core operations (`mma.sync.aligned`), the variance of the output tile $C = AB$ is computed element-wise:

$$u^2(C_{ij}) = \sum_k \left[ B_{kj}^2 \cdot u^2(A_{ik}) + A_{ik}^2 \cdot u^2(B_{kj}) \right]$$

which requires a shadow WMMA tile operating on uncertainty fragment operands alongside the data tile.

---

## 3. Epistemic Shadow Lanes

### 3.1 Register Layout

For each value register `%r_val` in the GpuKernelIr, the compiler allocates three shadow registers:

```
%r_val   — point estimate (the measurand)
%r_eps   — variance u²(val)
%p_valid — validity predicate (true if uncertainty is well-defined)
%r_prov  — 64-bit OR provenance bitset (monotonic union)
```

The shadow registers are allocated during the HLIR-to-GpuKernelIr lowering pass and propagated through all subsequent optimization passes (constant folding, dead code elimination, register allocation).

### 3.2 PTX Emission Example

For a source-level multiplication `let c = a * b`, the compiler emits:

```ptx
// Value lane
mul.f64  %r_c_val, %r_a_val, %r_b_val;

// Variance lane: u²(c) = b²·u²(a) + a²·u²(b)
mul.f64  %r_t1, %r_b_val, %r_b_val;
mul.f64  %r_t1, %r_t1, %r_a_eps;
mul.f64  %r_t2, %r_a_val, %r_a_val;
mul.f64  %r_t2, %r_t2, %r_b_eps;
add.f64  %r_c_eps, %r_t1, %r_t2;

// Validity conjunction
and.pred  %p_c_valid, %p_a_valid, %p_b_valid;

// Provenance merge (OR is associative, commutative, idempotent)
or.b64   %r_c_prov, %r_a_prov, %r_b_prov;
```

A single source multiplication becomes 8 PTX instructions (1 value + 5 variance + 1 validity + 1 provenance). The shadow computation adds 7 instructions per multiply, 3 per add, and 9 per divide — a 4–10x instruction count overhead depending on the operation mix. Wall-clock overhead is expected to be lower than the instruction ratio due to instruction-level parallelism and operand reuse, but this must be verified with hardware benchmarks (Section 7.1).

### 3.3 Overhead Model

For a kernel with $A$ additions, $M$ multiplications, and $D$ divisions, the instruction count overhead is:

$$\text{overhead} = \frac{A \cdot 4 + M \cdot 8 + D \cdot 10}{A + M + D}$$

giving a range of 4x (addition-dominated) to 10x (division-dominated) in instruction count. This may yield substantially lower wall-clock overhead than the raw instruction ratio — shadow operations reuse operands already in registers, and GPU architectures hide latency through warp-level parallelism — but the magnitude of that reduction is architecture- and occupancy-dependent and requires hardware measurement (Section 7.2).

### 3.4 Multi-Backend Emission

The same shadow lane logic targets three backends:

| Backend | Shadow precision | Tensor core support | Provenance |
|---------|-----------------|--------------------|-----------|
| PTX (NVIDIA CUDA) | f64 native | WMMA m16n8k16 | 64-bit OR |
| Metal (Apple MSL) | f32 (no native f64) | N/A | 32-bit OR |
| SPIR-V (Vulkan) | f64 via extension | N/A | 64-bit OR |

---

## 4. GPU-Specific Optimizations

### 4.1 Warp-Vote Epistemic Fast-Path

**Observation.** In many scientific workloads, large regions of data are well-characterized (high confidence, low uncertainty). Only boundary or anomalous regions require full uncertainty tracking. If all 32 lanes in a warp have valid data with uncertainty below a threshold, the shadow computation can be performed with reduced precision within a formal uncertainty budget.

**Mechanism.** The compiler generates dual-path kernels:

```ptx
// Phase 1: ballot — are ALL lanes valid?
vote.sync.ballot.b32  %r_ballot, %p_valid, 0xFFFFFFFF;
setp.eq.u32           %p_all_valid, %r_ballot, 0xFFFFFFFF;

// Phase 2: ballot — are ALL uncertainties below threshold?
setp.lt.f64           %p_eps_ok, %r_eps, THRESHOLD;
vote.sync.ballot.b32  %r_eps_ballot, %p_eps_ok, 0xFFFFFFFF;
setp.eq.u32           %p_all_ok, %r_eps_ballot, 0xFFFFFFFF;

@%p_all_ok bra FAST_PATH;

FULL_PATH:
    // 8 instructions per multiply (value + 5 variance + validity + provenance)
    ...
    bra MERGE;

FAST_PATH:
    // Budget-bounded uncertainty propagation (operator-specific bounds)
    // add/sub: u²_out ≤ 2·max(u²_a, u²_b)
    // mul:     u²_out ≤ (M_a² + M_b²)·max(u²_a, u²_b)  [requires operand envelopes]
    // div:     u²_out ≤ u²_a/m_b² + M_a²·u²_b/m_b⁴      [requires denom lower bound]
    // Validity: AND of inputs (preserved)
    // Provenance: OR of inputs (preserved)
    ...

MERGE:
    // Reconvergence
```

**Semantic invariant.** The fast path does *not* skip shadow computation — small uncertainty is not zero uncertainty. Instead, it uses operator-specific conservative bounds derived from operand envelopes. For addition/subtraction, $u^2_{\text{out}} \leq 2 \max(u^2_a, u^2_b)$. For multiplication, conservative bounds additionally require runtime or compile-time envelopes on operand magnitudes ($|a| \leq M_a$, $|b| \leq M_b$), giving $u^2_{\text{out}} \leq (M_a^2 + M_b^2) \max(u^2_a, u^2_b)$. For division, the bound requires a lower bound on the denominator ($|b| \geq m_b > 0$). These bounds are *sound with respect to a user-specified uncertainty budget* (Definition 1 below), not "GUM-compliant" in the metrological sense — the GUM provides the propagation law; the fast path is an optimization atop it.

This costs approximately 30% of full shadow overhead (the `budget_overhead` parameter in the divergence cost model), yielding a predicted speedup range of 1.3–1.7x depending on divergence percentage. The ballot overhead (2 instructions per check point) is amortized over the loop body.

**Definition 1 (Uncertainty Budget).** For a kernel output $y$, define the budget gap as:

$$\Delta(y) = u^2_{\text{fast}}(y) - u^2_{\text{full}}(y)$$

The fast path is *budget-sound* if $\Delta(y) \leq B$ for a user-specified absolute bound $B$, or equivalently $\Delta(y) / u^2_{\text{full}}(y) \leq \rho$ for a relative bound $\rho$. The budget is *local* (per-instruction): each operator's conservative bound overapproximates the full GUM result by at most a bounded factor. Composition across a kernel of $L$ instructions accumulates at most $L \cdot B_{\text{local}}$ total budget gap; tighter analysis via interval arithmetic is future work.

**Novelty claim.** `vote.sync.ballot` is a standard CUDA primitive used for reductions, stream compaction, and divergence management. Using it to evaluate *epistemic validity predicates* — "are all lanes' uncertainty values below a threshold?" — to gate *budget-bounded dual-path kernel execution* is, to our knowledge, a novel application.

### 4.2 Entropy-Gated Kernel Dispatch

**Observation.** The distribution of uncertainty values in the input buffer carries information about which kernel variant to dispatch. Concentrated uncertainty (all values similar) suggests the fast path will dominate; spread uncertainty suggests full GUM propagation is needed.

**Mechanism.** Before kernel launch, the host:

1. Samples $k$ uncertainty values from the input buffer (default $k = 256$)
2. Histograms them into $B$ bins (default $B = 16$)
3. Computes Shannon entropy: $H(\epsilon) = -\sum_{i=1}^{B} p_i \log_2 p_i$

| $H(\epsilon)$ | Dispatch | Rationale |
|-------------|----------|-----------|
| $< 1.0$ bit | Fast kernel | Concentrated; warp-vote fast path will dominate |
| $1.0 - 3.0$ bits | Adaptive | Mixed; both paths will activate |
| $\geq 3.0$ bits | Full GUM | Spread; full propagation needed everywhere |

**Novelty claim.** Existing adaptive kernel dispatch systems (Stream-K++ [8], KernelFoundry [9], Triton autotuning [10]) select variants based on *performance metrics* (throughput, latency) or *problem shape* (matrix dimensions). Using Shannon entropy of the *data uncertainty itself* for dispatch is, to our knowledge, novel.

### 4.3 Provenance-Aware DAG Scheduling

**Observation.** After conventional dependence analysis has ruled out read/write and explicit dataflow dependencies, kernels operating on data from *disjoint provenance sources* are treated as candidates for concurrent scheduling on separate CUDA streams. Provenance disjointness is a *secondary scheduling heuristic* for identifying concurrency opportunities among otherwise independent kernels — it does not substitute for alias analysis or read/write-set analysis.

**Mechanism.** The compiler:

1. Computes a provenance tag per kernel from its workload classification (bit-packed)
2. Builds a dependency graph: edge from $K_i$ to $K_j$ if $\text{prov}(K_i) \mathbin{\&} \text{prov}(K_j) \neq 0$ and they share parameters
3. Topologically sorts via Kahn's algorithm
4. Assigns streams via greedy coloring (lowest stream number without unsatisfied dependency)
5. Emits structured metadata as PTX comments:

```
// SOUNIO_DAG stream_count=3
// SOUNIO_STREAM kernel=0 stream=0
// SOUNIO_STREAM kernel=1 stream=1
// SOUNIO_DEP from=0 to=2
```

A host glue emitter reads this metadata at compile time and generates multi-stream launch code with `cudaStreamCreate`, per-kernel stream assignment, and dependency-based `cudaStreamSynchronize` insertion.

### 4.4 Speculative Epistemic Execution

When a kernel pipeline contains both coarse and refinement stages, the compiler classifies kernels by their computational cost (presence of transcendental operations, iteration depth) and inserts guards:

```
if aggregate_epsilon > threshold:
    launch_refinement_kernel(...)  // expensive
// else: coarse result is sufficient
```

The guard metadata (`SOUNIO_SPEC_GUARD`) is embedded in the PTX and interpreted by the host glue, enabling runtime decisions without recompilation.

### 4.5 Epistemic Kernel Fusion

Standard kernel fusion scores candidate pairs by parameter overlap and register pressure. Our fusion scoring adds a *provenance diversity bonus*:

$$\text{score} = \text{base\_score} + 60 \times \text{popcount}(\text{prov}_A \oplus \text{prov}_B)$$

Kernels with higher provenance diversity (many differing bits in the XOR) are favored as fusion candidates, as a heuristic for reduced source overlap. We note that provenance diversity does not prove statistical independence — different origin bits suggest low overlap of measurement classes, but do not guarantee uncorrelated uncertainties. A more interpretable metric (e.g., Jaccard distance between bitsets) could normalize by union size; we use popcount for simplicity.

---

## 5. Evaluation

### 5.1 Programmer Effort

We compare the code required to compute drug concentration with GUM uncertainty from four sensors (blood draw, CT scan, MRI, in-vitro assay):

| Component | CUDA C++ | Sounio |
|-----------|---------|--------|
| Dual struct (`value`, `variance`) | 10 lines | 0 (native type) |
| Propagation helpers (add, mul, div, sqrt) | 60 lines | 0 (compiler-generated) |
| Shadow array allocation | 25 lines | 0 (automatic) |
| Modified kernel body | 40 lines | 0 (same kernel) |
| Host-side GUM budget | 30 lines | 0 (built-in) |
| Provenance tracking | 25 lines | 0 (automatic) |
| **Total additional effort** | **~190 lines** | **0 lines** |

### 5.2 Overhead Comparison

| Method | Overhead factor | GPU? | Analytical? | Source |
|--------|----------------|------|-------------|--------|
| Sounio shadow lanes | 4–10x instr. count (model) | Yes | Yes (GUM) | Section 3.3 |
| Manual CUDA C++ | Same (if correct) | Yes | Yes | Manual |
| Python `uncertainties` [4] | ~1,400x (reported) | No | Yes | Issue tracker [6] |
| Julia `Measurements.jl` [5] | ~50–1,500x (reported) | No | Yes | Issue tracker [7] |
| Monte Carlo ($10^4$ samples) | $\sim$10,000x | Yes | No | By construction |
| Monte Carlo ($10^6$ samples) | $\sim$1,000,000x | Yes | No | By construction |

**Important caveats.** The shadow lane overhead (4–10x instruction count depending on operation mix) is a *model prediction* based on instruction counting (Section 3.3). Wall-clock overhead depends on instruction-level parallelism, register pressure, occupancy, and memory latency hiding — factors not captured by the instruction model. Hardware benchmarks are in progress (Section 7.1). The overheads reported for `uncertainties` and `Measurements.jl` come from user-filed issue reports, not controlled benchmarks; we cite them as motivation, not as rigorous comparisons. Unlike Monte Carlo, analytical propagation provides exact first-order results in a single pass.

### 5.3 Correctness

The implementation passes 22 test cases verifying:

- GUM algebraic properties: commutativity, scaling identity, quadrature, non-negativity
- Provenance properties: OR idempotency, commutativity, associativity, bit-tagging
- Sensor fusion: weighted mean uncertainty reduction below best individual source
- Entropy dispatch: correct variant selection for concentrated vs. spread distributions
- Per-module self-tests: 10/10 for each novelty module (fusion, speculative, DAG, warp-vote, entropy)

### 5.4 Implementation Scale

The epistemic GPU stack comprises 9,122 lines of Sounio across 10 self-hosted modules, including the shadow lane emitter, five optimization passes, and three host glue generators (PTX, Metal, SPIR-V). The compiler is fully self-hosted.

---

## 6. Related Work

**Automatic differentiation on GPU.** Enzyme [2] is the closest structural precedent: it inserts shadow registers in LLVM IR for reverse-mode AD through GPU kernels. Our shadow lanes differ in *what* they propagate (GUM variance, not Jacobians) and *how* they compose (quadrature sum, not chain rule). The two techniques are complementary — Enzyme could compute the sensitivity coefficients that GUM requires for correlated inputs.

**Uncertain types.** Uncertain\<T\> [3] demonstrated first-order uncertain types in managed languages, but used Monte Carlo sampling and had no GPU backend. Probabilistic programming languages (Pyro [11], Stan [12]) model uncertainty through Bayesian inference on GPU, but are inference frameworks rather than general-purpose languages.

**Tensor core error analysis.** Blanchard et al. [13] and Fasi et al. [14] characterized the numerical error of NVIDIA tensor cores through mathematical analysis. Our work differs fundamentally: we propagate *measurement uncertainty* (a runtime quantity attached to input data) through WMMA instructions, not *rounding error* (a hardware property).

**CPU uncertainty tools.** The GUM Tree Calculator [15], Python `uncertainties` [4], Julia `Measurements.jl` [5], and Puffin [16] all implement uncertainty propagation on CPU. None target GPU, and all suffer significant overhead (50-1,500x) compared to analytical propagation at the compiler level.

**GPU kernel dispatch.** Stream-K++ [8] and Triton's autotuner [10] select kernel variants based on performance characteristics. LithOS [17] atomizes kernels for fine-grained scheduling. None use information-theoretic measures of data properties for dispatch decisions.

---

## 7. Resolved Limitations

Three limitations identified in earlier versions of this work have been addressed:

**Second-order GUM propagation (resolved).** The shadow lane emitter now supports optional Hessian correction terms. For $y = f(x)$, the second-order variance is:

$$u^2(y) = (f'(x))^2 \cdot u^2(x) + \tfrac{1}{2} (f''(x))^2 \cdot u^4(x)$$

A curvature threshold gates activation: the correction is applied only when $|f''(x)| \cdot u^2(x)$ exceeds a configurable tolerance, avoiding unnecessary computation for near-linear operations. For the benchmark $y = x^2$ at $x = 3.0, u(x) = 0.1$: the first-order variance is 0.36, the Hessian correction is 0.0002 (0.056% improvement), and the combined variance is 0.3602 — matching the analytical second-order Taylor expansion.

**Assumption.** The second-order correction follows GUM Section E.3.2 and is valid under the assumption that input uncertainties are approximately normally distributed. For non-Gaussian inputs (highly skewed, multimodal, or heavy-tailed distributions), the Hessian term provides a curvature-aware *approximation* but not a rigorous statistical bound. Users with non-Gaussian measurement models should verify distributional assumptions or fall back to Monte Carlo propagation per GUM Supplement 1 (JCGM 101:2008) [18].

**Covariance matrix GPU propagation (resolved).** The compiler now supports correlated inputs via an upper-triangular covariance shadow structure in shared memory. For $N$ correlated variables, the full GUM Eq. 13 propagation is:

$$u^2(y) = \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 u^2(x_i) + 2 \sum_{i < j} \frac{\partial f}{\partial x_i} \frac{\partial f}{\partial x_j} u(x_i, x_j)$$

Storage is upper-triangular: $N(N+1)/2$ entries, capped at $N = 8$ correlated input sources (36 entries, 288 bytes in f64) to fit within GPU shared memory alongside kernel data. Here $N$ is the number of correlated *measurement sources* contributing to a single datum (e.g., $N = 4$ for four sensors measuring the same analyte), not the number of GPU registers or array elements. The cross-term contribution $2 j_0 j_1 \sigma_{01}$ is computed per-element and added to the diagonal-only result.

**Warp divergence cost model (resolved).** An analytical cost model quantifies the penalty of the dual-path warp-vote mechanism. The model comprises three components:

- *Serialization penalty*: $P_s = 1.0 + d/100$ where $d$ is the divergence percentage (0% → 1.0x, 100% → 2.0x)
- *Ballot overhead*: $P_b = n_b \cdot c_b / C_k$ where $n_b$ = number of ballots, $c_b$ = ballot cost in cycles (~2 on SM 8.x), $C_k$ = typical kernel cycles
- *Reconvergence cost*: $P_r = L \cdot d/100 \cdot 0.01$ where $L$ = instruction distance between diverge and reconverge points

The combined penalty $P = P_s + P_b + P_r$ feeds into a speedup estimator. With budget-bounded fast-path overhead of 0.3x (30% of full shadow cost) and 10% divergence, the predicted speedup is approximately 1.5x. The model populates the `estimated_speedup` field in `WarpVotePlan`, enabling cost-aware decisions about whether to apply the dual-path optimization. We emphasize that these are *model predictions* based on instruction counting; actual speedup depends on register pressure, occupancy, and memory access patterns that require hardware measurement.

### 7.1 Open Limitations

**Validity predicate incompleteness.** The current validity lane propagates inherited validity via `and.pred` but does not check all operational domain constraints. Division already guards against near-zero denominators, but sqrt does not yet verify non-negativity of the radicand. A complete validity semantics would require per-operation domain predicates (e.g., $a \geq 0$ for $\sqrt{a}$, $a > 0$ for $\ln a$). The current validity predicate means "all ancestors were marked valid," not "the GUM computation is well-defined for this specific operation." This is a known gap.

**Entropy dispatch scale-dependence.** The Shannon entropy thresholds for kernel variant selection are calibrated for raw epsilon values in $[0, 1]$. For inputs with different scales or physical units, the entropy values shift, potentially causing incorrect dispatch decisions. The correct fix is normalization to relative uncertainty ($\epsilon / |x|$) before histogramming, which we have not yet implemented.

### 7.2 Remaining Future Work

**Hardware benchmarks.** This preprint reports structural correctness and programmer effort reduction. Wall-clock benchmarks on NVIDIA Ampere (A5000, sm_86) and Ada Lovelace (L4, RTX 4000 Ada, sm_89) hardware are in progress and will be reported in the full paper.

**Higher-order covariance (implemented, pending empirical characterization).** Tiled covariance propagation extends support to $N = 128$ correlated variables via CUTLASS-style upper-triangular tile blocking. The $N \times N$ covariance matrix is partitioned into $T \times T$ tiles (where $T$ auto-tunes per SM architecture: 64 on Hopper sm\_90, 48 on Ada Lovelace sm\_89, 32 on Ampere sm\_86). Each tile fits in shared memory independently; results accumulate via tree reduction. For $N = 64$: 3 upper-triangular tiles at 4,224 bytes each. For $N = 128$: 10 tiles. The per-tile J$\cdot\Sigma_{\text{tile}}\cdot$J$^\top$ computation reuses the same propagation kernel as the monolithic $N \leq 8$ case. **Clarification:** $N$ here is the number of *correlated input uncertainty sources* per datum (e.g., $N = 4$ for blood draw + CT + MRI + assay), not the number of GPU registers or kernel dimensions. The covariance matrix is carried per logical measurement group, not per thread.

---

## 8. Conclusion

Epistemic shadow lanes demonstrate that first-order GUM uncertainty propagation through GPU kernels is a compiler problem, not a library problem. The transformation removes manual per-operation uncertainty propagation and metadata plumbing from GPU kernels; the instruction-count overhead is bounded by a moderate constant factor in the current model, though hardware characterization remains future work. The five optimizations we introduce — warp-vote fast-paths, entropy-gated dispatch, provenance-aware scheduling, speculative execution, and epistemic fusion scoring — show that uncertainty metadata is not merely overhead to be tolerated, but information that can guide execution-time decisions in ways unavailable to conventional GPU compilers.

---

## References

[1] JCGM 100:2008, "Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)," Joint Committee for Guides in Metrology, 2008.

[2] W. S. Moses, S. Churavy, et al., "Reverse-mode automatic differentiation and optimization of GPU kernels via Enzyme," in Proc. SC, 2021.

[3] J. Bornholt, T. Mytkowicz, and K. S. McKinley, "Uncertain\<T\>: A first-order type for uncertain data," in Proc. ASPLOS, 2014.

[4] E. O. Lebigot, "Uncertainties: a Python package for calculations with uncertainties," https://uncertainties.readthedocs.io/.

[5] M. Giordano, "Measurements.jl: Uncertainty propagation with linear error theory and real and complex numbers," J. Open Source Software, 2016.

[6] https://github.com/lmfit/uncertainties/issues/57 (1,400x overhead on 100K vectors).

[7] https://github.com/JuliaPhysics/Measurements.jl/issues/25 (50x overhead on 1000x1000 arrays).

[8] A. Dawar et al., "Stream-K++: Adaptive GPU GEMM Kernel Scheduling," arXiv:2408.11417, 2024.

[9] "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization," arXiv:2603.12440, 2025.

[10] P. Tillet, H. T. Kung, and D. Cox, "Triton: An intermediate language and compiler for tiled neural network computations," in Proc. MAPL@PLDI, 2019.

[11] E. Bingham et al., "Pyro: Deep universal probabilistic programming," JMLR, 2019.

[12] B. Carpenter et al., "Stan: A probabilistic programming language," J. Statistical Software, 2017.

[13] P. Blanchard, N. J. Higham, F. Lopez, T. Mary, and S. Pranesh, "Mixed precision block fused multiply-add: Error analysis and application to GPU tensor cores," SIAM J. Sci. Comput., 2020.

[14] M. Fasi, N. J. Higham, M. Mikaitis, and S. Pranesh, "Numerical behavior of NVIDIA tensor cores," PeerJ Computer Science, 2021.

[15] B. D. Hall, "GTC: The GUM Tree Calculator," Measurement Standards Laboratory of New Zealand.

[16] A. Gray, M. De Angelis, and S. Ferson, "The creation of Puffin, the automatic uncertainty compiler," Int. J. Approximate Reasoning, 2023.

[17] E. Goldstein et al., "LithOS: An operating system for efficient machine learning on GPUs," in Proc. SOSP, 2025.

[18] JCGM 101:2008, "Supplement 1 to the GUM — Propagation of distributions using a Monte Carlo method," 2008.
