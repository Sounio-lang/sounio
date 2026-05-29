# Erdős unit-distance / chromatic-number examples

Sounio programs around two Erdős-flavoured problems on unit-distance graphs in the
plane, with an emphasis on **exact arithmetic** (no floating point in the geometry)
and **native SAT/UNSAT certification** (the in-repo CDCL solver in
`stdlib/theorem/smt.sio`, no external solver, no third-party DRAT).

Build & run any example:

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc examples/erdos/<file>.sio /tmp/out.elf && /tmp/out.elf
```

---

## Thread 1 — Hadwiger–Nelson / de Grey (Erdős #508): χ of the plane

de Grey (2018, arXiv:1804.02385) gave the first 5-chromatic unit-distance graph,
settling χ(ℝ²) ≥ 5. His graph is built from **Moser spindles** assembled under a
fixed group of rotations. The key fact we exploit: the whole construction lives in
an **exact algebraic number field**, so it can be represented with integer tuples —
no rounding, no epsilon tolerances.

### The field tower (resolved from the literature)

de Grey's graph derives from the ring `N = Z[ω₁, ω₃, ω₄, ω₁₆]`, where
`ω_t = exp(i·arccos(1 − 1/2t))`, so `cos θ_t = 1 − 1/2t` and `sin θ_t = √(4t−1)/(2t)`:

| rotation | cos | sin | surd introduced |
|---|---|---|---|
| ω₁  | 1/2   | √3/2     | √3 |
| ω₃  | 5/6   | √11/6    | √11 |
| ω₄  | 7/8   | √15/8    | **√5** (√15 = √3·√5) |
| ω₁₆ | 31/32 | √63/32 = 3√7/32 | **√7** (√63 = 9·7) |

So:

- the **Moser spindle** and its ω₁,ω₃ family (Golomb, V, G₂₁, G₄₃, G₄₉) live in
  **ℚ(√3, √11)** (degree 4);
- de Grey's **full 1581-vertex 5-chromatic graph** needs **ℚ(√3, √5, √7, √11)**
  (degree 16) — the auxiliary surds √5, √7 come from ω₄, ω₁₆.

### Files

| File | What it does | Field | Result |
|---|---|---|---|
| `degrey_q3q11_spindle.sio` | Moser spindle, exact coords (scale ×12) | ℚ(√3,√11) | χ = 4 (brute force **and** native 3-col UNSAT / 4-col SAT) |
| `degrey_fragment_q3q11.sio` | glues a 2nd spindle by a 60° rotation; checks field closure | ℚ(√3,√11) | 11-vtx graph, all unit edges dist²=576 exact, no auxiliary surd; native 3-col UNSAT / 4-col SAT |
| `degrey_fieldtower.sio` | extends the kernel to the degree-16 field; XOR multiplication law; realizes ω₄ (√5) and ω₁₆ (√7) unit edges exactly | ℚ(√3,√5,√7,√11) | 5/5 arithmetic + rotation checks pass |
| `native_sat_scale_demo.sio` | native CDCL on graphs of known χ past the old 256-var cap | — | 6/6 correct (cycles to 1022 vars) |

### The degree-16 kernel (`degrey_fieldtower.sio`)

A ℚ-basis of ℚ(√3,√5,√7,√11) is the 2⁴ = 16 monomials `√(∏ S)` for `S ⊆ {3,5,7,11}`,
indexed by a 4-bit mask (bit0=√3, bit1=√5, bit2=√7, bit3=√11). Multiplication is
**pure XOR** — the same algebra as Cayley–Dickson, with positive square-root coeffs:

```
√(∏S) · √(∏T) = ( ∏_{p ∈ S∩T} p ) · √(∏ (S △ T))
```

i.e. basis `i · j` lands on basis `i XOR j` with rational coefficient = product of the
primes selected by `i AND j`. An element is an integer 16-tuple; one `O(16²)` loop
implements exact multiplication. Pairwise coprimality of {3,5,7,11} guarantees the 16
monomials are linearly independent (degree exactly 16, no collapse). Math-reviewed
(xai / Grok 4.1, 2026-05-28): the multiplication is the standard multiquadratic
relation and all identities/realizations hold. See `.claude/llm_offload_log.md`.

### Native χ certification

A `k`-colouring is encoded as boolean CNF (one var per vertex×colour; at-least-one +
at-most-one per vertex; per-edge same-colour exclusion) and handed to the in-repo
CDCL solver. χ ≥ k+1 is certified by a `k`-colouring **UNSAT**; a colouring exists
iff **SAT**. The spindle's χ = 4 is `3-col UNSAT ∧ 4-col SAT`, both native.

### Solver scaling — status and BLOCKER

`native_sat_scale_demo.sio` validates the solver past the old 256-variable cap on
graphs with rigorously known χ (even cycles SAT, odd cycles UNSAT, complete graphs
UNSAT). Current state of `stdlib/theorem/smt.sio`:

- **boolean variable cap: 1024** (raised from 256 in this work) — stable; validated
  across the full demo sequence (a wrong bound flips a known-χ case) and the
  `test_smt_solver_basic` regression (incl. LIA) stays green.
- **clause budget: 4096 clauses / 16384 literals (unchanged).**

> **OPEN ISSUE (gates the de Grey-scale, χ ≥ 5 path).** de Grey's minimised graphs
> (~510 vertices) are *dense*: a 4-colouring is ≈ 2040 vars **and ≈ 10 000 clauses**.
> That needs both a bigger clause budget and more literals. When `clause_data` was
> temporarily enlarged to 65 536 (`SmtContext` ≈ 800 KB), a known-SAT even cycle
> (C₄₀₀, 800 vars) returned spurious **UNSAT** in one specific build of
> `native_sat_scale_demo.sio`. **This was NOT reliably reproducible**: minimal
> struct-return probes (512 KB array, trailing scalars after the array, deep
> recursion over a live large struct) all initialise correctly, and the exact demo
> sequence at the 512 KB config later returned the correct SAT. The symptom — a
> layout-sensitive wrong value from a by-value struct return, with no crash — is
> consistent with the known compiler SRET struct-return corruption family
> documented in `docs/audit/r2_3_compiler_tuple_return_bug/` and the large-struct
> follow-up in `docs/audit/sret_large_struct_smtcontext/`. Pending a *reproducible*
> diagnosis, the clause budget is kept at the proven-stable size (4096/16384) and
> only *sparse* graphs reach >256 vars. **No χ ≥ 5 certificate is claimed here.**

### What is NOT done (honesty boundary)

- We do **not** construct de Grey's actual 510/1581-vertex graph (its vertex data is
  not in this repo and is **not** fabricated here).
- We do **not** certify χ(ℝ²) ≥ 5. The pieces in place are: the exact field
  arithmetic the construction requires, field-closure of spindle gluing, and a
  native χ pipeline proven on small graphs and (sparsely) past 1000 vars. The gap to
  a real χ ≥ 5 certificate is: (1) de Grey's graph data, (2) the clause-budget /
  large-struct fix above.

---

## Thread 2 — `168_*.sio`: chromatic / orbit separation

The `168_*` family explores chromatic obstructions and orbit structure on
unit-distance / Cayley-style graphs (Erdős #168 neighbourhood). See
`docs/research/erdos-168-chromatic-separation.md` for the scope: these demonstrate a
"Level-2 algebraic separation" and explicitly make **no** claim to resolve #508/#704
or to improve any published bound.

---

## Provenance

- Field tower & arithmetic: math-reviewed via `bin/llm-offload -t math-review -p xai`
  (logged in `.claude/llm_offload_log.md`).
- de Grey, *The chromatic number of the plane is at least 5*, arXiv:1804.02385 (2018).
- Polymath16 project wiki (rotation tower, ω_t generators).
