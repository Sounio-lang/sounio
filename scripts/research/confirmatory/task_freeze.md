# C2 — Frozen task definitions (Rfam OctTree confirmatory)

**Freeze date:** 2026-08-09 · **Status:** frozen before any confirmatory run; no outcome inspected
**Dataset:** C1 freeze (`scripts/research/confirmatory/freeze/`, manifest sha256 `50668b60…`, 105,888 eligible records, clan-held-out 70/15/15)

## Arms

### Task A — flip corruption (precedent arm)

Exact port of the exploratory corruption, frozen for comparability with the
87.1%/55.1% exploratory numbers. Negative examples: `n_flip = max(1, len//8)`
positions mutated as — with prob 1/3 flip an existing `(` to `)`; with prob 1/3
flip an existing `)` to `(`; otherwise turn a `.` into `(` or `)` (coin flip).
Known and predeclared weakness: this corruption unbalances bracket counts, so a
counting feature separates classes. Task A is kept ONLY as precedent; no claim
is promoted from Task A alone.

### Task B — balance-preserving subsegment swap (primary arm)

Negative examples preserve the exact multiset of `(`, `)`, `.` of the source
structure. Procedure (deterministic given the frozen RNG stream):

1. Parse matched pairs with a stack; enumerate maximal top-level substructures
   (contiguous segments from an opening `(` to its mate, inclusive).
2. If ≥ 2 top-level substructures exist, pick two distinct ones with length
   ratio in [0.5, 2] (closest qualifying pair in the RNG scan order) and swap
   their positions in the string. All symbol counts are unchanged; sibling
   order — a property counting cannot see — is destroyed.
3. Fallback (single-subsystem structures): take the longest internal matched
   substructure, reverse it, and complement every bracket (`(`↔`)`). Counts
   unchanged; nesting order inside the segment mirrored.

**Validity check (predeclared):** a frozen reference implementation
(`corruptions.py`) emits golden vectors (fixed seed, input → expected output);
the count-invariance and count-baseline-blindness properties are asserted on
every generated negative at run time (generation fails closed on violation).

### Task C — family classification (secondary arm)

50-way classification: the 50 families with the most eligible records in the
TRAIN split (list frozen in `families.json`, derived from the C1 freeze —
counts only, no outcomes). Input: dot-bracket tokens exactly as Tasks A/B.
Purpose: the arm where clan-held-out generalization actually bites.

### Negative-control arm (H4 analog)

All models are also trained/evaluated on fully random balanced strings
(Dyck-grammar samples with matched length distribution, same pipeline and
labels). Any model separability above chance there invalidates the run
(pipeline artifact), not the model.

## Frozen execution parameters

- L grid: {64, 128, 256, 512} (pad/truncate as exploratory; sampling rule:
  eligible records with L//4 ≤ len ≤ L per split, without replacement where
  the pool allows, with replacement otherwise — as exploratory)
- **Amendment 2026-08-09 (pre-results):** L=32 removed from the grid. Under
  the frozen sampling rule the L=32 pools are degenerate (train 108, val 0,
  test 226 records of length exactly 32); the exploratory L=32 numbers were
  drawn from ~100 structures with replacement and are not confirmable. No
  confirmatory result existed when this amendment was made.
- **Amendment 2, 2026-08-09 (pre-results; gate-6 failure caught by smoke):**
  the seed-0 / L=64 smoke cell (the only confirmatory cell inspected before
  this amendment) showed NEG-arm separability above chance (GRU test 0.604,
  OctTree 0.542; chance CI ±0.015 at n=4096) — exactly the pipeline-artifact
  signature gate 6 exists to catch. Population diagnostic (`diag_neg.py`,
  10k samples × 3 seeds × 4 L): the ( A ) B recursive Dyck sampler is not
  invariant under the Task B corruptions — 22% of L=64 random Dyck words
  fall into the mirror fallback (no qualifying swap pair), and the mirror
  reverses the sampler's first-component size bias, so corrupted and
  uncorrupted random Dyck words differ distributionally without any
  biological content. Fix, applied before any further confirmatory compute:
  (a) `gen_dyck_word` replaced by a UNIFORM Dyck sampler (cycle lemma;
  self-test `verify_uniform_dyck` runs fail-closed at runner start);
  (b) BOTH NEG classes are rejection-conditioned on admitting a swap, and
  NEG negatives therefore always take the swap path. Under uniformity the
  segment swap preserves the distribution (involution on words ×
  qualifying pairs; qualifying-pair count invariant), so the two NEG
  classes are exactly identical in distribution by construction — the null
  is exact, and any residual NEG separation can only be an implementation
  artifact. Arms A and B are untouched: data streams are derived per
  (arm, L, split, seed) via `derive_seed`, and `gen_dyck_word` is only
  called for NEG. All pre-amendment cells (the single smoke cell) are
  discarded as evidence-only (`results/_prefix_evidence/`) and the full
  grid is re-run from scratch under this amendment.
- Samples per arm per seed: train 16,384 / val 2,048 / test 4,096, drawn from
  the respective frozen split without replacement where pool size allows
- Seeds: 20 frozen seeds, `s_i = 2026080900 + i`, i = 0..19; the same 20 seeds
  drive every model (paired comparisons across seeds)
- Split discipline: val is used for early stopping / model selection; test is
  evaluated exactly once per model × seed, after all training decisions
- Training: epochs 50, lr 1e-2, batch 64 (exploratory hyperparameters, frozen;
  no search — this is a confirmatory, not an optimization)

## Models (C3 implements)

1. `CountBaseline` — logistic regression on (#`(`, #`)`, #`.`, length) — must
   fail on Task B (≤ 55%) or Task B is invalidated, not the models
2. `RealTree-8` — componentwise product (182 p)
3. `CliffTree-8` — fixed Clifford Cl(3) ≅ M₂(ℂ) product, dense associative,
   iso-parametric (182 p) — the decisive control
4. `LearnedBilinTree` — full-rank free bilinear tensor ℝ^{8×8×8} (512 p) +
   same gates/bias/readout (~670 p) — ceiling for "any learnable product"
5. `OctTree-8` — the hypothesis (182 p)
6. `GRU-8` — task ceiling

## Promotion criteria (predeclared; map to offload BLOCKERs 1–5)

The confirmatory claim "octonion multiplication provides an inductive bias
that *structure-blind associative products do not*" is promoted ONLY if, on
Task B test across the 20 paired seeds:

1. OctTree-8 − CliffTree-8 > 0 with paired 95% CI excluding 0, AND
2. OctTree-8 − RealTree-8 > 0 with paired 95% CI excluding 0, AND
3. OctTree-8 ≥ LearnedBilinTree − ε (ε = 2 pp), i.e. a free product does not
   match it, AND
4. direction replicates on Task A test (no CI requirement — precedent only),
   AND
5. CountBaseline ≤ 55% on Task B (task validity), AND
6. negative-control arm shows no separability (all models within chance CI).

Any failure demotes the claim to the corresponding honest statement:
(1) fails → "dense associative coupling suffices, non-associativity not
required"; (3) fails → "a learned product matches the octonion prior; the
prior is not special"; (5) fails → task redesign, results voided.
