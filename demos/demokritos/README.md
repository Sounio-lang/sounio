# NCSR Demokritos — run-of-show

One command runs everything shown in the talk, on a fresh clone:

```bash
bash scripts/talk.sh            # every live demo, in order
bash scripts/talk.sh --fast     # demos under 20 s live; the rest checked against golden/
bash scripts/talk.sh --list     # the order below, with engines and expected times
```

Each step pins its engine and passes only if its own output contains a fixed
sentinel. One PASS/FAIL line per step, wall time included; the script exits
non-zero on any failure. A missing file is a failure, not a skip.
`--fast` does not re-run the long steps: it checks that the committed golden
output under [`golden/`](golden/) exists and carries the sentinel, and prints
when and at which commit it was recorded (`bash scripts/talk.sh --record`).

Linux x86-64 only. `bin/souc` is Madaros (the default engine);
`SOUNIO_SOUC_ENGINE=lean_single` selects the bootstrap engine. Where a step runs
on only one engine, it says so.

| # | Step | Engine | What it shows | What it does not claim |
|---|---|---|---|---|
| 1 | `demo_incerteza.sio` | Madaros | First-order GUM propagation through `+` and `*`: perimeter σ = 0.223607, area σ = 0.707107 (checkable by hand). Output is in Portuguese. | Not second-order; correlated inputs are not shown here. |
| 2a–c | `demos/quantum/no_cloning_*.sio` | Madaros | A qubit as a `linear struct`: one use compiles and runs (positive control); two uses → `E039`; never consumed → `E040`. Refused at compile time, before any run. | A type-system analogue of no-cloning/no-deleting, not a quantum simulator. Destructuring a linear struct currently gives false E039/E040 on Madaros (#2766), so the demo reads a single array field. |
| 3 | `demos/hydrogen/mh7_coupled_ceiling.sio` | Madaros | Thermodynamic ceiling of the seven-stage metal-hydride compressor from measured van 't Hoff data (Gkanas et al. 2020); the bottleneck is S6→S7 in all six cases; 15/15 concordant pairs. | A ceiling orders the cases; it does not predict the kinetic gap to the COMSOL delivered pressure. |
| 3x | `demos/hydrogen/tools/mh7_ceiling_crosscheck.cpp` | g++ -std=c++23 | Independent C++ implementation written from the protocol; all six ceilings agree with Sounio's to its printed precision (≤ 0.005 bar). | Sounio prints two decimals, so agreement is shown to two decimals, not four. |
| 4a–b | `examples/hydrogen/mhhc_batch_margins.sio` | both | Batch tolerance per stage, identical on both engines. | |
| 5a–b | `demos/hydrogen/trieres_chain.sio`, `valley_chain_epistemic.sio` | Madaros | Probability boxes carried through a process chain. | |
| 6 | `demos/hydrogen/uhs_brine_calcite.sio` | lean_single | Underground hydrogen storage: brine–calcite equilibrium. | Madaros stops with `arena full` (P0.4); this step runs only on lean_single today. |
| 7 | `examples/chemistry/h2_ignition_uq_demo.sio` | Madaros | H₂ ignition delay (GRI-Mech 3.0 H/O, 10 species, 29 reactions) with native first-order uncertainty. Prints identical output on Madaros and lean_single (measured 2026-10-06). | No in-program assertion: its Cantera parity lives in `benchmarks/chemistry`; here the run must reproduce the golden byte for byte. ~4.5 min on Madaros (~29 min on lean_single), so shown from golden in both modes. |
| 8a–d | SNOMED-style, traceability, full GO, OAEI anatomy | Madaros | EL+ classification; full Gene Ontology closure (38,245 classes, 92 roles, 2,135,207 role edges) in seconds; OAEI 2016 mouse↔human alignment repair. | Completeness of the closure is with respect to the derivation calculus `Der` (see `formal/README.md`); `stdlib/ontology/elplus.sio` is a hand-written mirror, not extracted from Lean. |
| 8e–g | Cell Ontology, UBERON, GO roots | Madaros | The same closure on three more ontologies. | 1–5 min each: shown from golden output in both modes. |
| 9 | `scripts/ci/dissertation_pbpk28_parity_gate.sh` | gate | 28-compartment PBPK in Sounio against an independent Node implementation: five drugs, 22 `_PASS` verdicts, RMSE < 1 %. | Golden in `--fast`. The gate rewrites two evidence CSVs (P1.9); `talk.sh` restores them. |
| 10 | `demos/quantum/h2_vqe_2q.sio` | Madaros | Two-qubit H₂ VQE against exact diagonalisation (O'Malley et al., PRX 6, 031007, 2016): E_exact = −1.851199 Ha (electronic, no nuclear repulsion); the VQE reaches it to 4e-16; the second-order GUM band agrees with a 20,000-sample Monte Carlo to within 2 % at three gate uncertainties; a C++23 cross-check agrees. | ≤ 4 qubits, statevector, no noise model, no hardware backend. Lands with #2770. |
| 11 | `make build` | make | The self-hosted compiler compiles itself to a bit-identical fixed point. | Starts from the prebuilt `bin/souc-linux-x86_64`, not from `bootstrap/stage0.c` (that chain is `scripts/ci/bootstrap_chain_gate.sh`). |

Every demo is seeded: a full run fails any step whose output differs from its committed golden, so a printed number cannot move silently. Re-record one step with `bash scripts/talk.sh --record --only <id>`.

Measured 2026-10-06 on the shared 8-CPU workspace: `--fast` 77–117 s (load-dependent), a full run ~7 min there (about 5.5 min with the 2-core timings of 2026-10-05: `make build` 32 s, PBPK gate 82 s). Anything that cannot run live is listed in `TOUR.md` §6 with its issue.
