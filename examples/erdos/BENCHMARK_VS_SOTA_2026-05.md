<!-- docs:meta
topic_id: repo.examples.erdos.benchmark-vs-sota-2026-05
authority: research
audience: researchers
last_validated: 2026-05-29
validated_by: agent
-->

# souc_sat vs kissat / CaDiCaL — measured benchmark (2026-05-29)

**No-BS calibration.** This is the honest, reproducible head-to-head requested before further
solver work. Numbers are real (sandbox machine, single run each); **nothing is fabricated**. The
verdict is plainly stated: on raw solving speed `souc_sat` is **one to two orders of magnitude
slower** than kissat/CaDiCaL. The defensible Sounio contribution is *not* raw speed.

## Setup

- **souc_sat**: current `examples/erdos/souc_sat.sio` @ `main` (post-E4c LRB-pick-cache). Always
  streams a DRAT proof. Note: `souc_sat` only solves instances it generates internally (graph
  k-colouring / pigeonhole); it does **not** read external DIMACS, so it is timed on its native
  runs while kissat/CaDiCaL are timed on the **identical CNF** that souc_sat emitted.
- **kissat 4.0.4** (`git describe` = rel-4.0.4) — latest release; SAT-Comp-2025 UNSAT runner-up family.
- **CaDiCaL 3.0.0** (master). *Caveat:* this is master, **not** the exact `CaDiCaL-SC2025`
  competition build (that lives on a dev branch with ported Kissat inprocessing). Treat as a strong
  current-CaDiCaL reference, not the literal 2025 UNSAT winner.
- All proofs ASCII DRAT (`--no-binary` / `--binary=false`), each **verified by `drat-trim`
  (`s VERIFIED`)**. Build: `git clone … && ./configure && make`.

## Results (identical CNFs; wall-time includes proof emission for all three)

| Instance (CNF) | souc_sat | kissat 4.0.4 | CaDiCaL 3.0.0 | souc/kissat | all `s VERIFIED`? |
|---|---:|---:|---:|---:|:--:|
| K₇/₆ pigeonhole `p cnf 42 133` | 107 ms (1 794 conf) | **6 ms** | 6 ms | ~18× | yes |
| K₈/₇ pigeonhole `p cnf 56 204` | 842 ms (52 571 conf) | **15 ms** | 9 ms | ~56× | yes |
| **G₅₂₉ + SB** (flagship) `p cnf 2116 11212` | 30 466 ms (300 218 conf) | **2 173 ms** | 2 547 ms | **~14×** | yes |
| G₅₂₉ raw (no SB) `p cnf 2116 11209` | **DNF (>300 s)** | **74 253 ms** | 81 978 ms | — | yes |

Proof size on the flagship: souc_sat **66 MB** vs kissat **10 MB** vs CaDiCaL **12 MB** (souc_sat's
proof is ~6× larger — weaker clause minimisation, no inprocessing).

## Honest verdict

1. **souc_sat is ~14–56× slower than kissat/CaDiCaL** on these instances, including on its own
   flagship G₅₂₉+SB (30.5 s vs 2.2 s). It is **not** speed-competitive with SOTA. Saying otherwise
   would be BS.
2. **souc_sat cannot solve raw G₅₂₉ (no symmetry breaking)** within 300 s; kissat does it in 74 s.
   souc_sat *depends* on the precolour symmetry break to be tractable here.
3. **The symmetry break is a real, transferable contribution:** the 3 triangle-precolour unit
   clauses speed up *kissat itself* by ~34× (74 s → 2.2 s). The domain encoding/SB work has value
   independent of the solver engine.
4. souc_sat's proofs are ~6× larger (no recursive/inprocessing-driven minimisation parity).

## What this means for the plan (`SOTA_LITERATURE_AND_PLAN_2026-05.md`)

This **confirms the honest positioning**, it does not refute it:

- The gap is exactly the missing **inprocessing (BVE + vivification)** + **cache/arena layout**
  (plan items S2, S3). Even with those, *matching* kissat on general instances is a multi-year
  effort; we do **not** claim it.
- The **defensible, genuinely-novel target is the verification-integration niche (plan track B):**
  a fully self-hosted solver whose refutation is checked **end-to-end inside a theorem prover with
  no Mathlib** — the first Lean-internal χ(ℝ²)≥5. kissat/CaDiCaL, for all their speed, produce **no
  such artifact**. That is where Sounio overreaches the field, not on PAR-2.
- Immediate solver work worth doing is the **cheap, proof-safe** kind (S3 arena/inline; S1 SBVA as
  an external DRAT-composable preprocessor), plus this benchmark as the baseline to beat — not a
  promise to dethrone kissat.

## Reproduce

```bash
# solvers
git clone --depth 1 https://github.com/arminbiere/kissat  && (cd kissat  && ./configure && make)
git clone --depth 1 https://github.com/arminbiere/cadical && (cd cadical && ./configure && make)
# CNFs from souc_sat (writes souc_sat_worker.cnf at end of solve)
SOUC_BIN=artifacts/self-hosted/souc-self-hosted-x86_64
SOUNIO_SOUC_BIN=$PWD/$SOUC_BIN $SOUC_BIN examples/erdos/souc_sat.sio /tmp/s.elf
/tmp/s.elf 0 4 1 1 examples/erdos/data/degrey_529.edge   # G529+SB -> souc_sat_worker.{cnf,drat}
# head-to-head on the identical CNF
kissat/build/kissat  --no-binary souc_sat_worker.cnf k.drat   && drat-trim souc_sat_worker.cnf k.drat
cadical/build/cadical --no-binary souc_sat_worker.cnf c.drat  && drat-trim souc_sat_worker.cnf c.drat
```
