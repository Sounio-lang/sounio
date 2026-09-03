# Octonions, non-associativity, and where "something more" actually lives

**A consolidated, adversarially-checked findings document.**
Scope: a multi-stage investigation testing the hypothesis that octonionic non-associativity
is a meaningful structure — in brain connectomics, as a machine-learning inductive bias,
and in fundamental physics. Every quantitative claim below was produced by a runnable
script (paths in §7) and, where central, cross-checked against an independent baseline,
a pre-committed decision rule, or a null model.

Date: 2026-06-30. Engine for Sounio artifacts: `souc-lean-single-x86_64`.

> **ERRATUM (2026-06-30, self-reported).** The explicit 8-component octonion product
> used in the numpy synthetic/brain scripts (§2–§3) is **not a genuine octonion
> algebra** — it fails composition, alternativity, and Re(associator)=0 (verified
> symbolically). It matched correct octonions on the basis triples originally
> spot-checked but differs on 5/35 basis triples and on general elements. **Impact:**
> (1) the **mass result (§4) is unaffected** — it is pure scalar arithmetic with no
> octonion multiplication; (2) the **Sounio artifact's associator is unaffected** — it
> uses the Fano-verified stdlib `algebra::octonion` (2.0/0.0); (3) the **§3
> inductive-bias result re-verifies with correct Cayley–Dickson octonions** — the fair
> ablation gives octonion−associative = **+40.0** on the octonionic task (vs the
> reported +35.4) and −3.2 on the associative task, so the double-dissociation
> conclusion *holds for genuine octonions*; (4) the **brain null (§2) survives** (a
> non-associative detector found nothing; recalibrated to +40). Verified octonions and
> the re-check: `scripts/octonion_cd_correct.py`. The §2–§3 numeric tables were
> computed with the flawed table; their *conclusions* are confirmed, their exact
> figures should be read as "computed with a non-alternative near-octonion, conclusion
> re-verified with true octonions."

---

## 0. One-paragraph verdict

Octonionic non-associativity is **real and decisive as a representational tool when the data
carries matching structure** (a fair, pre-committed, matched-capacity ablation gives a +35-point
balanced-accuracy gap over an associative control). It is **absent from brain functional
connectivity** (ABIDE rs-fMRI): three converging, fairly-powered tests — including an
end-to-end learned embedding that had every chance to find it — return null, while the *same*
machinery scores +35 on genuinely octonionic data. Its one suggestive anchor is in
**fermion masses**: **given the electric-charge center assignment** of the exceptional Jordan
algebra J₃(O_C), real PDG mass ratios are **consistent with** δ²=3/8 (the up sector's
spread/center ratio = √(3/8)/(2/3) to 0.1%; given charge-centers the 5-ratio cluster is
p < 10⁻⁵ vs random; the centers≈charges match is selective — 0/20000 scrambled spectra reproduce
it). **Important conditioning (identifiability audit, §4.10):** with the sector centers left
free, δ is *not* identified by the masses alone — χ²(δ) is flat; only fixing centers = electric
charges pins δ to √(3/8). So the honest claim is conditional: *charges ⇒ δ²=3/8*, not *masses
select δ²=3/8*. This is a genuine, quantified, conditional regularity — **not** a confirmed
predictive theory (δ is unidentified without the charge input; some relations miss by ~10%; the
source is a single-author program). Net: the octonion has "something more," but in physics,
not the brain — and as suggestive, properly-conditioned structure, not established law.

---

## 1. The question

Does octonionic non-associativity — the associator `[a,b,c] = (ab)c − a(bc)`, nonzero off the
Fano lines — encode something real? Three fronts were tested in sequence, each forcing the
next. The investigation began as a "correction run" for a claimed octonionic O-SSM autism
biomarker and ended in the fermion mass spectrum.

---

## 2. Front A — Brain connectomics: a robust null

### 2.1 The ROI biomarker screen is a clean, well-powered NULL
Model-free octonionic-associator ROI screen on real ABIDE CC200 (N=505 ASD / 518 control):
**0/200 ROIs survive Holm–Bonferroni**, both mean and max pooling; max |Cliff's δ| = 0.072;
feature non-degenerate (between-subject CV 0.287). No biomarker. The much-publicized
"O-SSM 58.7%" claim is unsupported: every repository result shows ~49.5%, octonion training
was disabled, and no trained checkpoint was ever written (the pipeline never serializes a model).

### 2.2 The chance-level results were feature engineering, not the model
Every architecture sits at chance on the project's 8×8 manifest (200 ROIs pooled → 8 groups,
time → 8 steps = 64 numbers): gru 49.4, lstm 50.3, tcn 50.3, **transformer 50.8**, O-SSM raw
49.95, H-SSM 49.91. A transformer at chance on the same features ⇒ the signal is destroyed
upstream. Restoring it: full CC200 functional connectivity (Pearson upper-triangle,
19 900 features) + a plain linear classifier → **65.93% balanced accuracy** (LOSO, 20 sites,
N=988). The 200→8 averaging discards the pairwise-connectivity structure where ABIDE signal lives.

### 2.3 The octonion recurrence destroys linearly-separable signal
At matched 64-dim input (per-fold PCA-64 of FC): linear **65.00%**, O-SSM (octonion) **49.72%**,
H-SSM (quaternion) **51.21%**. The 64-dim bottleneck is not the problem (PCA-64 linear ≈ full-FC
linear); the frozen-random-A octonion recurrence + tiny readout is an untuned Echo State Network
that scrambles separability.

### 2.4 Three converging brain tests; the tool is calibrated
A *fair* octonion-vs-associative ablation (identical recipe — both bracketings L=(o_i o_j)o_k,
R=o_i(o_j o_k) + quadratic readout; toggle only the product) on ABIDE FC:

| Test | octonion − associative |
|---|---|
| Frozen ESN (weak) | ~0 (chance) |
| Fixed PCA-64, fair ablation | **−1.78** |
| Learned end-to-end embedding (max power) | **+0.28** |
| *(reference: genuinely octonionic data)* | *+35.4* |

The same machinery reads +35 where structure exists and ≤0 on the brain across fixed *and*
learned embeddings. **Conclusion: brain FC carries no octonionic non-associative structure**
(bounded honestly to FC representations; a directed/effective-connectivity modality remains
formally untested). "Connectomics is non-associative by definition" conflates nonlinearity /
higher-order interactions (hypergraph, TDA — formally orthogonal, over associative groups)
with Cayley–Dickson non-associativity.

---

## 3. Front B — Octonion as a machine-learning inductive bias: vindicated, fairly

### 3.1 Double dissociation on synthetic data
Label = octonion associator-norm `‖[a,b,c]‖ > median` (genuinely octonionic) vs an
associative/linear label. Same 24-real inputs.

| Model | Octonionic target | Associative target |
|---|---|---|
| linear | 49.8 | 98.2 |
| real reservoir (generic nonlinearity, matched cap.) | 55.7 | 95.1 |
| octonion (associator feature) | **99.5** | 51.2 |

### 3.2 Not a degree artifact, not a capacity artifact
Real random-polynomial features matched to the target degree: deg-3 56.6, deg-4 59.5, deg-6 58.1.
A trained MLP up to **537 k parameters**: ~60%. None approach the octonion's 99.5% — the
associator is a specific, hard-to-learn target generic models miss.

### 3.3 The clean test: matched-capacity, both trained, toggle only associativity
*Pre-committed rule:* non-associativity helps ⟺ on the non-associative task octonion − associative ≥ 10,
AND on associative/neutral tasks |difference| ≤ 3.
- Weak readout (linear, final state): NA-task **+4.1** → fails the bar (architecture too weak to
  express the degree-6 norm).
- **Fair readout (both bracketings + quadratic, neither handed the associator):**
  NA **90.3 vs 54.9 = +35.4 (≫10 ✓)**, neutral −0.2 (✓), associative −4.1 (octonion slightly
  *worse* — reinforces the dissociation). **Rule passed.**

**Conclusion: octonionic non-associativity is a real, decisive, learnable inductive bias when
the data carries matching structure.** Bounded claim: this is representational capability given
matching structure, not evidence any real domain has it. (Design note: providing both
bracketings is a symmetric architecture choice that exposes non-associativity; fair because the
advantage is intrinsic to L≠R, but a choice worth stating.)

---

## 4. Front C — Physics: the real anchor

### 4.1 Literature state (deep, adversarially-verified survey)
No rigorously demonstrated, matched-capacity, cross-validated advantage for octonion / non-associative
algebras in ML. Quaternions (associative): no significant accuracy gain over real baselines at
matched parameters; benefit is parameter efficiency only. Octonion-positive results
(Deep Octonion Networks) are single-paper, unreplicated, never isolate non-associativity; a 2025
review calls octonion nets empirically unvalidated. The most-cited modern work (PHM/PHNN, ICLR'21)
*learns* the multiplication rule from data, treating the fixed algebra as a restriction. Octonions
appear rigorously by construction in the Standard-Model C⊗O programs (Furey, Todorov), the
exceptional Jordan algebra J₃(O_C)→E₆ (Singh), and G₂=Aut(O) lattice gauge theory (which shows
*no* exceptional thermodynamic signature vs SU(N)).

### 4.2 The Koide diamond
Koide Q = (Σm)/(Σ√m)² for charged leptons = **0.666661** vs 2/3 = 0.666667 (φ = 44.9997° vs 45°),
agreeing to ~5 significant figures. Derived bridge: the J₃(O_C) spectrum (q−δ, q, q+δ) with
**δ²=3/8 is exactly the Koide 2/3 point** (φ=45°). Quarks miss: up Q=0.849, down Q=0.731.
Caveat: Koide (1981) predates the octonion derivation — a retrofit, not a forward prediction.

### 4.3 Singh Table I (parameter-free √mass-ratio closed forms, δ²=3/8)
√(mτ/mμ) +1.0%; √(m_d/m_e) gen-1 −0.7%; √(mμ/me) −3%; √(mc/mu) +5%; √(ms/md) −7%;
√(mb/ms) −9%; **√(mt/mc) −26% (miss)**; gen-1 √(mu/me) +26% @MZ. "Few-to-tens of percent,"
as the paper itself states; precision agreement not claimed.

### 4.4 δ-consistency — the strongest pro-octonion result
Inverting each ratio (three functional forms, three sectors) for the δ it implies:
τμ→0.608, sd→0.635, bs→0.612, cu→0.614, tc→0.610. Cluster **0.6155 ± 0.0099** (std/mean 1.6%),
on top of √(3/8)=0.6124 (0.5% away). **Null model** (200 k random mass ratios, same fixed formulas):
P(cluster this tight)=10⁻⁴; P(mean this close to √(3/8))=0.036; **P(both) < 1/200 000 (<5×10⁻⁶)**.
Real fermion masses are *not* random with respect to these octonion formulas.

### 4.5 Cross-sector held-out prediction
Fit the single δ on the **quark sector only** → δ = **0.6124 ± 0.003 = √(3/8)**. Predict the
**lepton sector held-out** (zero lepton data): √(mτ/mμ) +1.4%, √(mμ/me) −2.0%. Reverse
(leptons→quarks): √(mt/mc) −0.4%, √(mc/mu) −10%. Pure group-forced equalities (zero δ):
Dynkin swap √(mτ/mμ)=√(ms/md) → **8.7% off** (a real miss); trace split √m_e:√m_u:√m_d = 1:2:3
→ observed 1:2.04:3.02 (good).

### 4.6 Neutrinos — the one surviving forward prediction
Singh predicts leptonic Jarlskog J_ℓ=0 ⇒ δ_CP ∈ {0, π}. NuFit-6.0 (2024): for **normal
ordering**, the global fit is consistent with CP conservation within 1σ → **prediction survives**;
for inverted ordering, δ_CP≈270° is favored at >3.6σ → would refute. Majorana nature untested
(0νββ only limits). Lightest-ν ≈ massless: consistent with tight cosmological Σm_ν bounds.
Decisive future arbiter: DUNE / Hyper-Kamiokande.

### 4.7 The residual confound
The δ-consistency null randomizes the *data*, not Singh's *formula-assignment* (which functional
form maps to which ratio). That assignment is derived from group theory in the paper but was not
independently re-derived mass-blind here. The cross-functional-form consistency (one δ across
three forms) is harder to reverse-engineer than per-ratio fitting, but a full kill requires a
mass-blind derivation of the assignment + a truly held-out sector.

**Enumeration bound on the assignment freedom (`assignment_enumeration.py`).** Triality forces
each single-edge adjacent-generation ratio to be one of three ascending edge types
(`b/a`, `c/a`, `c/b`) on the Sym³(3) triangle, with eigenvalues (c_s−δ, c_s, c_s+δ). Enumerating
*all* 3⁴ edge assignments for the four single-edge ratios (τ/μ, s/d, c/u, t/c) and demanding a
single cross-sector δ: only **16 of 81** are even valid (`c/b` always gives δ > c_s); the
lepton and down ratios are **forced to `c/a`** (the `b/a` alternative implies δ ≈ 0.76, grossly
inconsistent); and only **two** assignments have δ-spread ≤ 0.02, both landing at δ ≈ √(3/8)
(0.616, 0.623). The genuine residual freedom is a mild **two-fold (`b/a` vs `c/a`) ambiguity in
the up sector**, moving δ by ~±0.02 around √(3/8). So the assignment freedom on the single-edge
ratios is **small and funnels to √(3/8)**, not free to fit arbitrary δ.

### 4.8 The sector centers are the electric charges (recovered mass-blind)
The remaining structural inputs flagged in §4.7 — the per-sector eigenvalue centers c_s — are
**not free parameters fit to masses**. Fixing only the single algebraic constant δ=√(3/8) (zero
mass input for the center) and inverting each single-edge form for the c_s it implies
(`centers_from_charges.py`):

| ratio | sector | implied c_s | electric charge | dev |
|---|---|---|---|---|
| τ/μ | lepton | 1.0073 | 1 | +0.7% |
| s/d | down | 0.9651 | 1 (electron, via Dynkin swap) | −3.5% |
| c/u | up | 0.6651 | 2/3 | −0.2% |
| t/c | up | 0.6698 | 2/3 | +0.5% |

Given δ=√(3/8), the implied centers match electric charges for **two of three** sectors — lepton
1.007 vs 1 (+0.7%) and up 0.665/0.670 vs 2/3 (~0.1–0.5%). The down sector comes out ≈1, the
**electron** charge (via the Dynkin swap), **not** its own 1/3 — and partly because center 1/3 < δ
makes the form invalid, so this is forced by validity, not an independent charge recovery.
*(And see §4.10: this recovery is conditional on δ being fixed; with δ free the centers and δ are
co-determined.)* The δ inferred here is δ²≈0.379, **consistent with** 3/8=0.375 at the ~1% level,
not exactly selected. **Compound (two-edge) ratios** match
*forward* (δ and centers fixed, zero free parameter): b/s = (c/a)(c/b)|_{c=1} = 6.707 vs 6.690
(+0.3%); μ/e = (c/a)·(δ+⅓)/(δ−⅓) = 14.10 vs 14.38 (−2.0%) — the (δ±⅓) factor is the Dynkin-swap
1↔⅓ image (consistent; not independently re-derived here).

**Net after §4.7–4.8:** δ is the algebraic √(3/8); the centers are the electric charges
(recovered mass-blind to ~1%); the edge assignment is bounded and funnels to √(3/8); compound
ratios match forward to a few %. The selection confound is now **substantially closed** — the
free knobs that would have made this "numerology" are pinned to {√(3/8), electric charges}. What
remains for a full first-principles claim: an independent derivation of the Dynkin-swap (δ±⅓)
action and a genuinely held-out sector (neutrinos), neither computable today.

### 4.9 Formal obligation and the two-center reconciliation
`formal/DynkinSwapMassLadder.lean` separates what is **proved** (pure-ℝ algebra) from the one
**representation-theory obligation** left open. Every algebraic identity is verified symbolically
in `scripts/dynkin_swap_symbolic.py` (7/7 PASS):
- Koide closed form `Q(c,δ) = 2δ²/(9c²) + 1/3`, and `Q = 2/3 ⟺ δ² = (3/2)c²`, so **δ² = 3/8 at
  the diagonal center c = 1/2**.
- The μ/e swap factor `(δ+⅓)/(δ−⅓)` is exactly the `c/a` edge with **center and spread exchanged**
  (`caEdge δ ⅓`) — the algebraic signature of the Dynkin Z₂.
- μ/e and b/s expand to the Table-I closed forms.

**The two-center reconciliation (not a contradiction).** δ²=3/8 appears in two places with
*different* centers — the diagonal Koide spectrum (center **1/2**) and the edge-ratio ladders
(centers **1** and **2/3** = charges). These are consistent: the **same** δ = √6/4 simultaneously
makes the center-1 edge `(1+δ)/(1−δ) = 4.16` match √(mτ/mμ) **and** makes the center-1/2 diagonal
spectrum Koide-exact (Q = 2/3). One constant, two manifestations, linked through δ — verified
symbolically. The genuinely open step (stated, not encoded as an axiom): deriving that triality
*realises* the center↔spread exchange. This is the boundary between strong quantified regularity
and first-principles law.

### 4.10 Identifiability audit (adversarial — what survives, what doesn't)
Two break-it tests (`identifiability_audit.py`):

1. **δ is not identified by the masses alone.** §4.8 fixed δ=√(3/8) and then "recovered" the
   centers — near-circular. Profiling the fit χ²(δ) with the sector centers left **free**: χ²(δ)
   is **flat** across δ∈[0.40, 0.65] (max/min = 1.0). The lepton and down sectors (one ratio,
   one free center each) fit *any* δ exactly; the up sector pins only the scale-free ratio
   **δ/c_U = 0.918 = √(3/8)/(2/3) to 0.1%**, not δ itself. So δ=√(3/8) emerges **only** once the
   centers are fixed to the electric charges. The honest statement is conditional:
   *charge-centers ⇒ δ²=3/8*, **not** *masses select δ²=3/8*.
2. **The charge-center match is selective (not manufactured).** Real masses give centers within
   1.2% of {1, 1, 2/3}; over 20 000 scrambled mass spectra, **0** land that close. And the
   5-ratio δ-cluster (given charge-centers) sits at p < 10⁻⁵ vs random (§4.4). So conditional on
   the charge assignment, the agreement is real and unlikely by chance.

**Reconciled honest claim:** the load-bearing input is the **electric-charge center assignment**
(physically motivated, external). *Given* it, the fermion masses are consistent with the single
algebraic constant δ²=3/8, selectively (p<10⁻⁵) and cross-sector-predictively (~2%). Without it,
δ is unidentified. The remaining first-principles step (deriving the charge↔center and swap
structure from E₆ triality) is exactly where the content lives — not in the algebra we can already
prove.

---

## 5. Sounio artifact (reproducibility + formal anchoring)

`examples/physics/octonion_mass_delta.sio` — native Sounio (lean_single) port:
- §4.4–4.5 reproduced **bit-identically** to the Python reference (δ_quark=0.612100; lepton
  held-out predictions +1.3% / −2.0%; Dynkin swap 8.3%).
- Stage-2 uses the verified `algebra::octonion` product (the `NonAssoc` effect tracked by the
  compiler): |[e1,e2,e4]| = 2.000000 (off-Fano), |[e1,e2,e3]| = 0.000000 (Fano) — the forward
  direction of `formal/OctonionAssociator.lean`.

Running in Sounio **reproduces, does not improve** the numbers — confirming the throughline of
the whole arc: the binding constraint was never the tool. Sounio's value here is provenance,
a formally-verified non-associative algebra, and (next) ISO-GUM uncertainty propagation.

---

## 6. Honest verdict

| Domain | "Octonion has something more"? |
|---|---|
| Brain (ABIDE FC) | **No** — robust null (3 tests; tool calibrated at +35) |
| Octonion as inductive bias | **Yes** — decisive (+35) when data carries the structure |
| Fermion masses | **Conditional** — *given charge-centers*, masses are consistent with δ²=3/8 (selective, p<10⁻⁵; cross-sector ~2%). δ is **unidentified** without the charge input (§4.10). |
| Confirmed predictive theory | **Not yet** — assignment not re-derived mass-blind; ~10% misses; single-author |

The honest lesson, repeated at every front: a different tool, algebra, or language does not
rescue a limit that lives in the data/structure. The 8×8 manifest destroyed the brain signal
(not the model); the octonion is null on the brain across all embeddings (not a tool failure);
Sounio reproduces bit-identically (the language changes nothing). Where the octonion genuinely
has "something more" is where octonions provably live — the fermion mass spectrum — and even
there the status is *strong, quantified regularity*, not established law.

---

## 7. Methods / code (scratchpad) and key sources

Scripts (numpy/torch reference): `fc_baseline.py`, `fc_vs_octonion.py`, `octonion_positive_control.py`,
`degree_matched_control.py`, `mlp_baseline.py`, `assoc_ablation.py`, `assoc_ablation_fair.py`,
`assoc_ablation_fc.py`, `learned_embedding_fc.py`, `koide_jordan_test.py`, `singh_tableI_test.py`,
`delta_consistency.py`, `heldout_crosssector.py`. Sounio: `examples/physics/octonion_mass_delta.sio`.

Sources: Furey arXiv:1611.09182; Todorov arXiv:2206.06912 (Universe 2023); Singh arXiv:2508.10131;
G₂ lattice arXiv:1409.8305 (JHEP 03(2015)057); PHM arXiv:2102.08597 (ICLR 2021); quaternion null
arXiv:2409.00140; NuFit-6.0 arXiv:2410.05380 (JHEP 12(2024)216); Baez "The Octonions" (Bull. AMS 2002).

Pre-registered decision rules were fixed *before* running each ablation; results are reported
against those rules including where they failed (§3.3 weak readout; §4.3 mt/mc; §4.5 Dynkin swap).
