# Hypercomplex Algebra & State-Space Modeling — Lane

Dedicated lane for hypercomplex (octonion/sedenion/Cayley-Dickson) algebra and
non-associative state-space models (O-SSM / S-SSM / H-SSM). Kept separate from
other verticals so the write set is disjoint and the work is auditable on its own.

All probes use a **valid** octonion table (e2·e5 = +e7, verified at runtime by
`oct_truth`/`oct_algebra`) — none carry the retracted e2·e5 sign error (PR #1024).

## Parallel Lane Contract

```text
Lane: hypercomplex-algebra-ssm
Owner: Demetrios
Base: origin/main
Branch: feat/hypercomplex-algebra-ssm
Write-Set:
  scripts/research/oct_truth.sio
  scripts/research/oct_algebra.sio
  scripts/research/ossm_recover.sio
  scripts/research/ossm_separation.sio
  scripts/research/HYPERCOMPLEX_SSM_LANE.md
  scripts/ci/octonion_probes_gate.sh
  examples/epistemic/rk4_correlated_uncertainty.sio   (bundled epistemic/GUM example)
  (future: examples/*ossm*.sio, examples/*octonion*.sio, stdlib/ssm/*, stdlib/{algebra,math}/octonion.sio, docs/briefings/OSSM_*.md)
Read-Set:
  stdlib/ssm/lib.sio, stdlib/algebra/octonion.sio, formal/OctonionAlgebra.lean,
  formal/OctonionGraph.lean, experiments/non_assoc_connectomics/ (prior empirical nulls)
Not-Touched:
  examples/associativity_probe_benchmark.sio — main already carries the gated
  run-pass version (//@ expect-stdout: ALL PASS); this lane defers to it and does
  NOT revert it. The provable separation lives in ossm_separation.sio.
Required-Gates: scripts/ci/octonion_probes_gate.sh green (compile+run every probe
  under lean_single, assert invariants in stdout) -> OCTONION_PROBES_GATE_OK.
Merge-Target: main
Known-Blockers:
  BLK-20260714-madaros-print_int-f64 (issue #891, owner codex-2) + sibling Madaros
  v0.80.0 codegen bugs. Probes are verified on the lean_single engine.
```

## Assets (all compile + run green on `lean_single`, current main)

| file | proves |
|---|---|
| `oct_truth.sio` | 168/512 non-assoc basis triples (=\|PSL(2,7)\|); **norm-mult error 0, alternative error 0** |
| `oct_algebra.sio` | 7 Fano lines, 168/42 decomposition; **associator antisymmetry 0, alternative 0, flexible 0, Moufang 0** |
| `ossm_recover.sio` | associator-recovery positive control: octonion **987** permil vs commutator held at chance (521) |
| `ossm_separation.sio` | **O-SSM(oct) 1000 vs H-SSM(assoc) ceiling 500 permil** on non-assoc triples — a *representational-capacity* statement (an associative algebra is bracket-blind), the honest #907 reframe, NOT an ML-win claim |
| `rk4_correlated_uncertainty.sio` | bundled epistemic example: CorrelatedValue sd 3.28 = Monte-Carlo truth vs independent Epistemic overestimate (5.0) |

`168 = |PSL(2,7)|` is recovered independently in three of the probes.

Note: `ossm_separation.sio` carries `with Mut` on `oct_class`/`left_class`/`right_class`/
`main` to satisfy the current (stricter) effect checker — product and output unchanged.

## Run everything

```bash
bash scripts/ci/octonion_probes_gate.sh          # asserts every invariant above
# or individually:
for f in oct_truth oct_algebra ossm_recover ossm_separation; do
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile scripts/research/$f.sio -o /tmp/x.elf && /tmp/x.elf
done
```

## Coding rules in this lane (Madaros v0.80.0 codegen workarounds)

1. Octonion/sedenion ops as **struct-returning, call-free-inner** functions
   (load-compute-store); never hold many f64 locals across many `wg/hg/hs` calls.
2. Cross-function scalar state → `[i64; 2]` array globals indexed `[0]`
   (unit-fn scalar writes don't persist; `[i64; 1]` SIGSEGVs).
3. Never `print(<f64>)`; compute to i64 (permil) and print via `print_int` only.

See issue #891 for the underlying compiler bugs (owner: codex-2).
