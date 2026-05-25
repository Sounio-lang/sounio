# HeuleGraph510 — χ = 5 certificate bundle (Phase C)

Certifies **χ(HeuleGraph510) = 5**. The two bounds have **different trust bases** —
stated plainly because they are not equal:

| bound | claim | how certified | trust base |
|-------|-------|---------------|------------|
| χ ≤ 5 | proper 5-colouring exists | `formal/lean4/SounioHeule510Chromatic.lean` `native_decide` | **Lean kernel** |
| χ ≥ 5 | not 4-colourable (UNSAT) | clausal proof, below | SAT solver + `drat-trim` (community standard, **not** the kernel) |

## Pipeline (auditable, chained to Phase B)
1. `mkcnf.py` parses the edge list `E` **directly from `formal/lean4/SounioHeule510.lean`**
   (the Phase-B-certified 2504-edge graph) and emits the 4-colouring CNF
   `heule510_4col.cnf` (2040 vars, 13586 clauses). So the UNSAT result is about exactly
   the edge set Phase B certified.
2. `solve_verify.py` (PySAT + Glucose 3) solves it → **UNSAT**, emitting a 2,218,256-line
   DRAT proof `heule510_4col.drat`.
3. `drat-trim heule510_4col.cnf heule510_4col.drat -L heule510_4col.lrat` → **`s VERIFIED`**
   (see `VERIFIED.txt`), also emitting a 1,034,911-line LRAT.

The DRAT (324 MB) and LRAT (402 MB) are **not committed** (size; regenerable). Their
sha256 + the `s VERIFIED` line are pinned in `VERIFIED.txt`. The 5-colouring SAT model
is `coloring5.json` (consumed by the Lean upper-bound proof).

## Reproduce
```bash
pip install --break-system-packages python-sat
git clone https://github.com/marijnheule/drat-trim && (cd drat-trim && gcc -O2 -o drat-trim drat-trim.c)
python3 mkcnf.py && python3 solve_verify.py
./drat-trim/drat-trim heule510_4col.cnf heule510_4col.drat   # expect: s VERIFIED
```

## Upgrade path (Phase C.1, not done)
Replace `drat-trim` (trusted C) with a **formally-verified** LRAT checker (cake_lpr, or
Lean's `bv_decide` LRAT checker) consuming `heule510_4col.lrat` — that would lift the
χ ≥ 5 half to a kernel-grade trust base. Plumbing an arbitrary CNF+LRAT through a verified
checker is a separate task.
