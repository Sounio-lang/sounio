<!-- docs:meta
topic_id: repo.docs.research.zd-qec-prediction-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zd-qec-prediction-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# A testable physical prediction of the rupture programme: the sedenion zero-divisor crown-graph code

**Date:** 2026-07-26
**Status:** `PREDICTION` → `EXECUTABLE` (Q_GREEN reached 2026-07-26)
**Parents:** `docs/research/routon_zd_spec_2026-07-26.md` (exact 2-cycle criterion, nullity law), `docs/research/chingon_zd_spec_2026-07-25.md` (fiber/growth laws), `docs/research/g2_zd_fibers_spec_2026-07-25.md` (G₂/PSL(2,7) fiber action), `docs/research/rupture-programme-synthesis_2026-07-25.md`
**Harness:** `scripts/research/zd_qec_prediction_contract.py`
**Gate:** `scripts/ci/zd_qec_prediction_gate.sh`
**Prediction class:** option A (quantum error correction) of the rupture prediction programme.

---

## 1. The prediction in one paragraph

The 84 canonical sedenion zero divisors, organised by the rupture programme's fiber structure, form a graph that we prove to be the **crown graph** `H₇ = K_{7,7}` minus a perfect matching, plus one isolated vertex. That graph defines (i) a classical LDPC code `[42, 29, 4]` and (ii) a quantum CSS LDPC code `[[1960, 842, 4]]` (Tillich–Zémor hypergraph product). Both codes have **exactly enumerable low-weight logical spectra**, so their error-detection and error-correction failure rates are predicted to leading order *exactly*, with no free parameters:

- **Classical BSC, detection mode:** `p_undetected = 210 p⁴ + O(p⁵)` (exact coefficient: the 210 four-cycles of the crown).
- **Classical BSC, correction mode:** `p_logical = 840 p³ + O(p⁴)`.
- **Quantum depolarising, detection mode:** `p_undetected = 17640 (p/3)⁴ + O(p⁵)` (exact: 8820 X-type + 8820 Z-type weight-4 logicals, complete enumeration).
- **Quantum depolarising, correction mode:** `p_logical = 70560 (p/3)³ + O(p⁴)`.
- **Quantum bit-flip channel:** `p_undetected = 8820 p⁴ + O(p⁵)`, `p_logical = 35280 p³ + O(p⁴)`.

Furthermore the Cayley–Dickson tower forces the code family to **collapse**: at levels 5 and 6 the designs contain triangles (label identity `9 ⊕ 17 = 24`), so the quantum codes are `[[87336, 70226, 3]]` and `[[2308168, 2122850, 3]]` — the distance never grows past 4. **The family has no quantum error-correction threshold.** The distance-4 code is unique to the sedenion birth level, where the catastrophe is born. These statements are falsifiable by classical-channel experiments (immediately), quantum simulation (immediately), and quantum hardware of ~2000 qubits (near-term).

---

## 2. Derivation from the rupture programme

Every link is executed by the contract (clause in parentheses).

### 2.1 The zero-divisor design

At tower level `b` the canonical zero divisors are the 2-unit sums `e_i ± e_j` (`1 ≤ i < j < 2^b`) with singular `L_{e_i ± e_j}`, enumerated exactly by the 2-cycle criterion of the routon contract: with `S[i,j]` the Cayley–Dickson sign and `ℓ = i ⊕ j`, the pair is a zero divisor iff `p(k) = S[i,k]·S[j,k]·S[i,k⊕ℓ]·S[j,k⊕ℓ] = +1` for some `k`. The censuses `Z(b)/2 = 42, 294, 1518` at `b = 4, 5, 6` are the rupture programme's (Q1). The **ZD design** is the graph on the imaginary units `{1, …, 2^b − 1}` with an edge per index pair.

### 2.2 The crown theorem (level 4)

**Theorem (Q2).** The level-4 ZD design is `H₇ ⊔ K₁`: the crown graph `K_{7,7}` minus the perfect matching `{i, 8+i}_{i=1..7}`, on the bipartition `{1..7} | {9..15}`, plus the isolated vertex `8`.

**Proof.** Edge `{i, 8+j}` (`i, j ∈ {1..7}`) has xor-label `8 + (i ⊕ j)`. If `i = j` the label is `8`, a power of two, so the pair is not a zero divisor (fiber-label law: labels are `≥ 8` and not powers of two). If `i ≠ j` the label lies in `{9..15}`, so the pair is a zero divisor unless it is a defect pair; by the defect diagonal (chingon spec C7) the only missing pairs at level 4 are `{r, 8}`, `r = 1..7`, which never have the form `{i, 8+j}`. Hence exactly the `49 − 7 = 42` pairs `{i, 8+j}, i ≠ j` are edges. Within-side pairs have label `≤ 7` (not zero divisors); vertex 8 meets only the seven defect pairs. ∎

Consequences, all contract-verified:

- The graph is 6-regular on 14 vertices and bipartite; girth `≥ 4`, and 4-cycles exist, so **girth = 4** (Q3).
- The seven G₂ fibers (xor-labels 9–15) are precisely the seven **near-perfect matchings** `M_r = { {i, 8+(i⊕r)} : i ≠ r }` (Q2). The fibre action of `PSL(2,7)` (g2_zd_fibers contract) therefore acts inside `Aut(H₇) = S₇ × Z₂`, which contains `PSL(2,7)` (order 168 | 10080) — the rupture programme's G₂ structure is the symmetry of the crown.

### 2.3 The classical crown code

The cycle space of a graph is a binary linear code whose minimum distance is the girth. The crown gives a **`[42, 29, 4]`** code (rank of the incidence matrix `= 13`; `42 − 13 = 29`; distance `= 4`) (Q4). Its weight-4 codewords are *exactly* the 210 four-cycles — a weight-4 even subgraph is a 4-cycle, so the count is **complete**, not a lower bound. The dual (cut space) is `[42, 13, 6]`, enumerated exhaustively (`2^13`); its minimum-weight vectors are the 14 vertex stars.

### 2.4 The quantum crown code

Apply the Tillich–Zémor hypergraph product to the reduced crown incidence matrix `M` (`14 × 42`, rank 13): `H_X = [I₄₂ ⊗ M | M^T ⊗ I₁₄]`, `H_Z = [M ⊗ I₄₂ | I₁₄ ⊗ M^T]` (Q5). Verified:

- CSS commutation `H_X H_Z^T = 0`; ranks `559 + 559`; hence `k = 1960 − 1118 = 842`.
- Distance: the T–Z bound gives `d ≥ min(d₁, d₂, d₁′, d₂′) = min(4, 4, 14, 14) = 4`; explicit weight-4 logical operators (`c ⊗ e_b` X-type, `e_a ⊗ c` Z-type for each 4-cycle `c`) give `d ≤ 4`. Hence **`[[1960, 842, 4]]`**, quantum LDPC (check weight 8, qubit degree ≤ 6).
- The complete weight-4 logical census, obtained by exact pair-syndrome hashing over all `C(1960,2)` pairs: **8820 X-type + 8820 Z-type = 17640**, and no others (Q6). Minimum stabiliser weight is 6 (structural lemma + the exhaustive cut enumeration), so no weight-4 centraliser element is a stabiliser, and no Y-type weight-4 logical exists (its X- and Z-parts would have weight ≤ 2).
- All `1960` single-X and all `1960` single-Z error syndromes are distinct: every weight-1 error is uniquely correctable, and a weight-2 error never miscorrects (its corrected residue has weight ≤ 3 < min(4, 6)). Logical failure in correction mode therefore starts at weight 3.

### 2.5 Family collapse (no threshold)

At level `b ≥ 5` the label identity `9 ⊕ 17 = 24` (three valid fiber labels) forces triangles — the contract verifies the witness triangle `(2, 11, 26)` at levels 5 and 6 and counts `1092` (L5) and `19236` (L6) triangles (Q3). The tower embedding carries the witness to every `b ≥ 5`, while `d₁′ = 2^b − 2` grows, so the HGP distance is **3 at every level `b ≥ 5`** (T–Z bound `min(3, 3, 2^b−2, 2^b−2) = 3`, with triangle logicals attaining it). The family (Q7):

| level | quantum code | classical cycle code | triangles | 4-cycles |
|---|---|---|---|---|
| 4 (`𝕊`) | `[[1960, 842, 4]]` | `[42, 29, 4]` | 0 | 210 |
| 5 (`𝕋`) | `[[87336, 70226, 3]]` | `[294, 265, 3]` | 1092 | 17136 |
| 6 (`𝕀`) | `[[2308168, 2122850, 3]]` | `[1518, 1457, 3]` | 19236 | 703752 |

Quantum error-correction thresholds require code families with growing distance. This family has distance `4, 3, 3, 3, …`: **no threshold exists**. The distance-4 member is unique to the sedenions, the birth level of the catastrophe — a structural fact traceable to the fiber-label law, not a design choice.

---

## 3. Physical predictions (measurable quantities)

Noise models and decoder are stated explicitly; the coefficients are exact consequences of §2 and are the rupture programme's numerical fingerprints (84 = 4·21, 210, 8820 = 42·210, 17640 = 84·210).

### P1 — Classical channel (testable today)

Transmit the `[42, 29, 4]` crown code over a binary symmetric channel with crossover `p`, in two modes:

- **Detection** (discard frames with nonzero syndrome): undetected frame-error probability
  `p_u(p) = 210 p⁴ + O(p⁵)`.
  The leading coefficient is *complete* (all weight-4 codewords are 4-cycles; there are 210).
- **Correction** (bounded-distance decoder correcting all weight-1 errors): silent logical-error probability
  `p_L(p) = 840 p³ + O(p⁴)`,
  because each of the 210 weight-4 codewords has exactly 4 splittings into a weight-3 error plus its unique weight-1 syndrome-mate, and no smaller silent failure exists (min codeword 4, so weight-2 residues are never codewords).

### P2 — Quantum simulation / hardware (testable now in simulation; hardware ~2000 qubits)

Implement the `[[1960, 842, 4]]` code. Depolarising noise (each qubit: X, Y, Z each with probability `p/3`):

- **Detection mode** (postselect on trivial syndrome): `p_undet(p) = 17640 (p/3)⁴ + O(p⁵)`.
- **Correction mode** (unique-syndrome decoder): `p_L(p) = 70560 (p/3)³ + O(p⁴)` (each weight-4 logical contributes its 4 splittings; nothing smaller fails — min logical 4, min stabiliser 6, single-error syndromes distinct).
- **Pure bit-flip channel:** `8820 p⁴` (detection), `35280 p³` (correction). Same numbers for pure dephasing by the X/Z symmetry of the construction.

The formal leading-order crossing `p_L(p) = p` sits at `p ≈ 5.3×10⁻³` (bit-flip) and `p ≈ 2.0×10⁻²` (depolarising); these are *pseudothresholds*, not thresholds, and their precise values are to be measured, not trusted from the leading term (higher-order terms are non-negligible there). The robust predictions are the **exponents** (4 detection / 3 correction) and the **leading coefficients**.

### P3 — No-threshold theorem (family level)

Any experiment or simulation comparing logical rates across the family must find distance-driven behaviour consistent with `d = 4, 3, 3, …`; in particular, increasing the tower level does *not* suppress logical errors asymptotically. A demonstrated threshold in this family would falsify the rupture combinatorics at levels ≥ 5.

---

## 4. Experimental protocol

**Test 0 — computational verification (done, CI-gated).**
`bash scripts/ci/zd_qec_prediction_gate.sh` → `ZD_QEC_PREDICTION_GATE_OK` (~45 s). Rebuilds the ZD designs from the Cayley–Dickson sign table, proves the crown theorem, constructs both codes, enumerates the logical spectrum exactly, and assembles the coefficients.

**Test 1 — classical BSC testbed (weeks, no special hardware).**
1. Fix the generator matrix of the `[42, 29, 4]` cycle code (contract emits the design; any basis of `ker M`).
2. Software BSC: for `p ∈ {10⁻³, 3×10⁻³, 10⁻², 3×10⁻², 10⁻¹}`, transmit `≥ 10⁹/p⁴`-scaled frame counts; record undetected frames (detection mode) and silent miscorrections (correction mode).
3. Fit `log p_u` vs `log p`: **predicted slope 4.00, intercept `log 210`**; correction mode **slope 3.00, intercept `log 840`**. Binomial 95 % confidence intervals must contain the exact coefficients.
4. Optional physical channel: any SDR or wireline testbed with calibrated effective `p` (or a BSC-emulating optical/link bench) — same measurement.

**Test 2 — quantum error-correction simulation (weeks, classical compute).**
1. Build `H_X, H_Z` (contract construction); implement the unique-syndrome lookup decoder (5880 single-error syndromes, all distinct — verified).
2. Monte-Carlo sample depolarising noise at `p ∈ [10⁻⁴, 5×10⁻²]`, `≥ 10⁷` shots per point; measure `p_L(p)` in correction mode and the postselected undetected rate in detection mode.
3. **Predicted:** leading exponents 3 and 4; coefficients `70560 (p/3)³` and `17640 (p/3)⁴` in the small-`p` regime.

**Test 3 — quantum hardware (2027–2029 window).**
Platforms requiring ≳ 2000 qubits with mid-circuit measurement and reset (superconducting or neutral-atom roadmaps both project this scale). One round of syndrome extraction (1118 checks, weight 8, Tanner degree ≤ 6 — LDPC, so extraction circuits are shallow); logical preparation of `|0⟩^L` on the 842 logical qubits; natural or injected calibrated noise. Measure the logical failure rate vs. physical error rate in both modes. **Predicted exponents 3 (correction) and 4 (detection), no threshold crossing as the code is scaled within the family.**

**Falsifiers** (full clause-level list in `zd_qec_prediction_falsifiers_2026-07-26.md`):

- **F1 (exponent):** measured leading exponent `≠ 4` (detection) or `≠ 3` (correction), in any of Tests 1–3, with the stated noise model and decoder.
- **F2 (coefficient):** measured leading coefficient outside the exact values `210 / 840 / 8820 / 17640 / 35280 / 70560` (with statistical CIs), e.g. discovery of an eleventh weight-4 codeword class beyond the 210 four-cycles, or any mixed-support weight-4 logical (the contract proves none exist).
- **F3 (weight-3 event):** observation of an undetected logical error of weight ≤ 3 in the quantum code — this directly falsifies `d = 4`, hence the girth-4 crown theorem, hence the level-4 ZD census.
- **F4 (threshold):** demonstration of an error-correction threshold in the ZD-code family — falsifies the triangle-forcing identity `9 ⊕ 17 = 24` and the level ≥ 5 fiber structure.
- **F5 (crown structure):** any deviation from `H₇ ⊔ K₁` in the level-4 design (e.g. vertex 8 not isolated, a within-side edge, a 7-regular vertex) — falsifies the fiber-label law or defect diagonal of the parent contracts.

---

## 5. What this is NOT

- **Not a claim that nature "uses" sedenions.** The prediction is about codes whose *parameters and performance* are forced by the rupture combinatorics; the experiments test error-correction physics on engineered systems, not fundamental physics.
- **Not a new threshold theorem.** The family provably has none (§2.5); the prediction is the exact low-`p` scaling and the *absence* of a threshold.
- **Not an optimality claim.** `[42, 29, 4]` and `[[1960, 842, 4]]` are not claimed to be best-in-class; the claim is that they are *exactly* the rupture-derived codes with *exactly* the stated spectra.
- **Not a proof of the T–Z distance bound.** We cite Tillich–Zémor (2009/2014) for `d ≥ min(d₁, d₂, d₁′, d₂′)` and verify its four hypotheses exactly; the matching upper bound is exhibited, not assumed.
- **Not the full ZD variety.** Only canonical 2-unit zero divisors enter the design, as in the parent contracts.
- **Not a clinical claim.**

---

## 6. Novelty

- The identification of the sedenion ZD design with the crown graph `H₇ ⊔ K₁`, and of the G₂ fibers with its near-perfect matchings, is new (the parent rupture contracts tabulated fibers but never identified the graph).
- The hypergraph-product code on this design, its parameter set `[[1960, 842, 4]]`, and its exact weight-4 logical spectrum (8820 + 8820, complete) are new to the coding literature we are aware of (literature search 2026-07-26: no sedenion/ZD-based QEC codes; crown graphs appear in combinatorics, not as QEC Tanner graphs).
- The triangle/4-cycle censuses (`0/1092/19236`; `210/17136/703752`) are new invariants of the tower.
- The physical predictions (exponents + exact coefficients + no-threshold) are new.

---

## 7. Reproduce

```bash
python3 scripts/research/zd_qec_prediction_contract.py
# expect: Q1..Q8 PASS, ZD_QEC_PREDICTION_VERDICT Q_GREEN   (~45 s)

bash scripts/ci/zd_qec_prediction_gate.sh
# expect: ZD_QEC_PREDICTION_GATE_OK
```

Pure Python + NumPy, self-contained. The ZD census uses the exact integer 2-cycle criterion (no floating point); all code-theoretic linear algebra is over F2.

---

## 8. Assumptions register

1. **Construction choice:** the hypergraph-product map from design to quantum code (Tillich–Zémor). Other maps (bicycle, lifted product) give other codes; the *classical* predictions (P1) are construction-free once the crown theorem holds.
2. **Noise models:** BSC (classical), independent depolarising / bit-flip (quantum), phenomenological (no gate errors in the stated coefficients; circuit-level noise changes coefficients, not the distance-driven exponents below the pseudothreshold).
3. **Decoder:** unique-syndrome bounded-distance decoder; the correction-mode coefficient `70560` counts exact miscorrections for this decoder (any decoder correcting all weight-1 errors agrees, since single-error syndromes are distinct).
4. **Distance lower bound:** Tillich–Zémor theorem cited; its hypotheses (`d₁ = 4`, `d₁′ = 14`, etc.) are verified exactly.

---

## 9. AI disclosure

Prediction, derivation, harness, and this spec drafted under human direction (2026-07-26). Math-facing claims are bounded by the named gate and by the cited Tillich–Zémor theorem. No clinical content. GAIDeT-ICMJE 2025.
