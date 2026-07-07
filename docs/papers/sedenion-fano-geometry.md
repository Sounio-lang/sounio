<!-- docs:meta
topic_id: repo.docs.papers.sedenion-fano-geometry
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.sedenion-fano-geometry
-->

# Paper 2: The Fano Geometry of the Sedenion Zero-Divisor Set, Executed and Triple-Certified

## Abstract

The sedenion algebra 𝕊 (dimension 16, the Cayley–Dickson double of the octonions) has zero divisors,
and their count `168 = |PSL(2,7)|` is the same number that counts the octonion non-associative
("non-Fano") triples — the "168-theorem" (Paper 1). This paper executes the **geometry** behind that
number, exactly, in the running Sounio language, and certifies every result on **three independent
legs** — souc execution (decidable ℤ-equality), an independent Python oracle, and Lean `native_decide`.
The picture that emerges is a single object seen from several sides, all organized by one distinguished
element, the octonion→sedenion doubling unit **e₈**:

- The 84 participating primitive zero divisors split into **7 fibers** indexed by `L = lo⊕hi ∈ {9..15}`,
  each a **`K_{6,6} − 3·K_{2,2}`** graph (12 vertices, degree 4, bipartite, `168 = 7 × 24`).
- The same 168 pairs group into **42 support-quartets** of shape 2-lower + 2-upper, four pairs each
  (`168 = 42 × 4`), and the quartets, as edges on the 7 fibers, form the doubled complete graph **`2·K₇`**.
- The **boundary** of participation is exactly the e₈ family: the 28 excluded mixed-half primitives are
  precisely `{hi = 8} ∪ {lo⊕hi = 8}`.
- On the **associator side**, the ordered non-associative sedenion basis triples number
  `1848 = 11 × 168`, and by output grade the doubling grade 8 carries one octonion `168` while the
  other 14 grades carry `10 × 168` — so `11 = 10 + 1`.
- **The unifying theorem**: the `168` is a genuine group — the sedenion signed-automorphism group
  `Aut(𝕊) ≅ PSL(2,7)`, order 168, sitting in `GL(4,2) ≅ A₈` at index 120 — and it **fixes e₈**. The 7
  fibers **are the 7 points of the Fano plane `PG(2,2)`**, and `Aut(𝕊)` acts on them as the full Fano
  collineation group `PGL(3,2)`. So the zero-divisor geometry *is* the Fano plane, decorated by the
  doubling, with e₈ its unique fixed point.

## 1. Setup

The Cayley–Dickson sign `σ(i,j) ∈ {±1}` gives `e_i·e_j = σ(i,j)·e_{i⊕j}` on basis units; it is reused
verbatim from the runtime (`stdlib/algebra/cayley_dickson.sio::cd_sigma`) and is field-independent.
A **primitive** two-support imaginary is `v = e_lo (±) e_hi`; it is **mixed-half** when
`lo ∈ {1..7}`, `hi ∈ {8..15}`. There are 112 such (signed). Two primitives form a **zero-divisor pair**
when their exact product vanishes on all 16 components — decidable integer equality, no tolerance.

## 2. The boundary: e₈ (executed)

Of the 112 mixed-half primitives, exactly **84 participate** in a zero-divisor pair and **28 do not**,
and the 28 are exactly `{hi = 8} ∪ {lo⊕hi = 8}` — the vectors containing the doubling unit `e₈ = ℓ`
(`𝕊 = 𝕆 ⊕ 𝕆·ℓ`) or lying on its xor-grade-8 diagonal `e_lo·(1 ± ℓ)`. The zero-divisor geometry lives
strictly away from the doubling seam.
*(`tests/run-pass/sedenion_e8_boundary.sio`; `docs/research/sedenion_e8_boundary.md`.)*

## 3. Two factorizations of 168 (executed)

**Fibers.** The 84 vertices split into 7 fibers by `L = lo⊕hi ∈ {9..15}`; annihilation never crosses
fibers (`a·b = 0 ⟹ L(a) = L(b)`); each fiber has 12 vertices, 24 edges, degree 4, and is connected and
bipartite (6,6). `168 = 7 × 24`.
*(`sedenion_zd_fibers.sio`; `docs/research/sedenion_zd_fibers.md`.)*

**Fiber shape.** Each fiber is isomorphic to `K_{6,6} − 3·K_{2,2}` — certified by its common-neighbor
profile `(4:6, 2:24, 0:36)` and, in the oracle/Lean, by the complement being three 4-cycles.
*(`sedenion_zd_fiber_identity.sio`; `docs/research/sedenion_zd_fiber_identity.md`.)*

**Quartets.** The 168 pairs group by support-union into **42 quartets**, each a 4-set of 2 lower + 2
upper indices, four pairs each. `168 = 42 × 4`.
*(`sedenion_zd_quartets.sio`; `docs/research/sedenion_zd_quartets.md`.)*

**Interlock.** Each quartet's four pairs split 2 + 2 across exactly two fibers; the 42 quartets, as
edges on the 7 fibers, form the doubled complete graph **`2·K₇`** (all 21 = C(7,2) fiber-pairs, two
quartets each, every fiber of incidence-degree 12).
*(`sedenion_quartet_fiber_incidence.sio`; `docs/research/sedenion_quartet_fiber_incidence.md`.)*

## 4. The associator side: 1848 = 11 × 168 (executed)

For basis units the associator `[e_i, e_j, e_k]` is the single component `e_{i⊕j⊕k}` with coefficient
`A(i,j,k) = σ(i,j)σ(i⊕j,k) − σ(j,k)σ(i,j⊕k) ∈ {−2,0,+2}`. Of the `2730` ordered distinct triples of
imaginary units, exactly **`1848 = 11 × 168`** are non-associative. By output grade `i⊕j⊕k`, each of
the 14 grades `≠ 8` carries 120 and the doubling grade 8 carries 168 (the octonion associator count),
so `11 = 10 + 1`. The 455 unordered triples split 35 associative + 168 semi (exactly 2 orderings) + 252
fully non-associative; grade 8 is fully non-associative on all its support-triples. This confirms the
open conjecture of the zero-divisor-geometry report — the factor 11 lives on the **associator** side,
not the zero-divisor side — and shows it is combinatorial, not a group order (see §5).
*(`sedenion_associator_1848.sio`; `docs/research/sedenion_associator_1848.md`.)*

## 5. The unifying theorem: 168 is the Fano collineation group, e₈ is its fixed point

A linear map `M ∈ GL(d,2)` (acting on indices as `F₂^d` vectors, preserving `⊕`) is a **signed
automorphism** iff `σ(Mi,Mj)·σ(i,j)` is a coboundary `ε(i)ε(j)ε(i⊕j)`. Counting them (rigorous F₂
elimination):

- octonions `{1..7}`: **168** = `|GL(3,2)|` (all of it);
- sedenions `{1..15}`: **168** — of `|GL(4,2)| = 20160 = |A₈|`, at index 120 — the **same** 168;
- every sedenion automorphism **fixes e₈** (orbit `{8}`); the orbit partition is `{1..7} ∪ {8} ∪ {9..15}`.

Hence the pervasive `168 = |PSL(2,7)|` is the group `Aut(𝕊)`, and because it fixes `e₈` it acts on the
fiber labels `L = 8 ⊕ t` by `t ↦ M(t)` on `t = L∧7 ∈ F₂³∖{0}`. The seven nonzero vectors of `F₂³` are
the seven points of the Fano plane `PG(2,2)`; the action is faithful, transitive, and permutes the
seven Fano lines `{a,b,a⊕b}`. **So the 7 fibers are the 7 Fano points, and `Aut(𝕊)` is the full Fano
collineation group `PGL(3,2)`.** The number `1848 = 11 × 168` is *not* a group order: `168` is the
group, `11` is the combinatorial grade factor of §4.
*(`sedenion_automorphism_168.sio`, `docs/research/sedenion_automorphism_168.md`;
`SounioSedenionFano.lean`, `docs/research/sedenion_fano_fibers.md`.)*

**The e₈ throughline.** e₈ *bounds* the zero-divisor set (§2), *carries* the extra octonion 168 on the
associator side (§4), and is the *unique fixed point* of the symmetry group (§5). These are three faces
of one fact: e₈ is the doubling direction the Fano plane does not see.

## 6. Method: three independent legs, and an honesty perimeter

Every result is certified on three legs:

1. **souc execution** — a self-contained `tests/run-pass/*.sio` computing the claim by decidable
   ℤ-arithmetic (no float; self-contained to avoid the cross-module aggregate hazard #637).
2. **Python oracle** — an independent transcription of the sign law (`scripts/research/*_oracle.py`),
   diffed element-wise against the souc output by a CI gate (`scripts/ci/*_gate.sh`).
3. **Lean `native_decide`** — a Mathlib-free proof (`formal/lean4/Sounio*.lean`) checked by Lean's
   kernel evaluator, an implementation independent of both.

The first two transcribe the same sign law, so their agreement certifies *implementation-agreement*
against a miscompile; Lean is the independent-spec leg (as in Paper 1's census). Two caveats are
recorded honestly. First, the counts of §§2–4 (7 fibers, 42 quartets, `1848`) were established
empirically in an earlier Python geometry report; this work *executes* them exactly, pins their
closed forms (`L = lo⊕hi`, `K_{6,6}−3K_{2,2}`, the grade decomposition), and adds the group theorem of
§5. Second, the triple-certification is not decorative: it **caught a real compiler defect** — the
committed `bin/souc` miscompiles the `GL(4,2)` automorphism sweep (returning `17882`/`432`), while a
fresh stage2 souc, the Python oracle, and Lean all agree on `168`. The certified computation uses a
rigorous, order-independent F₂ elimination.

## 7. Reproduce

```bash
# one example per section; see each docs/research/*.md for the full trio
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_e8_boundary.sio
python3 scripts/research/sedenion_associator_1848_oracle.py
(cd formal/lean4 && lake build SounioSedenionFano)   # non-default: the Fano collineation theorem
```

## References

- Paper 1: `docs/papers/exact-168-executable.md` (the counts and the measure layer).
- Per-result notes: `docs/research/sedenion_{e8_boundary, zd_fibers, zd_fiber_identity, zd_quartets,
  quartet_fiber_incidence, associator_1848, automorphism_168, fano_fibers}.md`.
- Formal layer: `formal/lean4/SounioZeroDivisorBridge.lean`, `SounioCayleyDickson.lean`, and the
  `SounioSedenion{E8Fibers, FiberIdentity, Quartets, Incidence, Associator1848, Automorphism, Fano}.lean`.
