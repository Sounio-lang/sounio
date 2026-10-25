# SIO-173: checked FEM and geometric Poisson multigrid

This increment supplies actual-size, caller-owned numerical solvers and repairs the existing bounded FEM/multigrid implementations. It does not claim a general unstructured FEM package, arbitrary boundary conditions, certified error bounds or cross-backend qualification.

## Public contracts

`math::fem_checked::fem1d_solve_into` solves `-(EA u')'=q` using piecewise-linear elements on finite, strictly increasing 1D nodes, positive piecewise-constant EA and piecewise-constant load. Both endpoint values are imposed by strong Dirichlet elimination. Thomas elimination requires positive finite pivots. The routine recomputes the physical interior residual before committing the solution.

`fem1d_workspace(n)` reports five length-n binary64 vectors, including the output: solution plus four scratch vectors diag/upper/rhs/trial, 40n payload bytes. Nodes and element input arrays are additional. All five buffers must be distinct; the mutable borrow contract prevents overlapping mutable aliases. Inputs must have exactly the reported lengths. Scratch may change on refusal; the solution changes only on success. `max_elements` bounds assembly before scratch writes. Status 0 means available, 1 invalid layout/options, 2 invalid domain/nonpositive pivot, 3 element budget, 4 unavailable arithmetic. `residual_max` is a measured binary64 residual, not an enclosure or proof of accuracy. The scaled load helper preserves representable products for subnormal q and huge h rather than prematurely halving q to zero.

`math::poisson_multigrid::poisson_solve_into` solves the unit-square five-point negative Laplacian with zero Dirichlet boundaries. The interval count must be a power of two in [2, 1048576]; this is a shape limit, not a promise that the largest allocation fits. `poisson_workspace` reports the finest vector length, the sum of all level lengths, and the output plus three packed scratch vectors. Callers check this query against their memory budget before allocation. The input RHS is additional. The hierarchy uses red-black Gauss-Seidel smoothing, full-weighting restriction, bilinear prolongation and an exact single-interior-point coarse solve.

The initial guess is zero. The physical infinity residual must be at most `atol + rtol * max(abs(rhs))`; options require a positive finite resulting tolerance. Both boundary RHS and solution represent homogeneous Dirichlet values. Status 0 means converged, 1 invalid layout/options, 2 invalid/nonfinite RHS or boundary data, 3 cycle/work budget, 4 unavailable arithmetic. Only a converged staged solution is copied to the output. Setup initializes packed scratch separately; `point_work` counts updates, residual evaluations, restrictions and prolongations, including the coarse solve. `setup_points` reports setup independently. No cycle allocates a buffer. The work limit may refuse partway through scratch operations while preserving output.

The legacy `fem.sio` retains its bounded FEResult API. Direct elimination uses row equilibration and relative pivots; singular systems refuse instead of returning zero unknowns. CG validates symmetry and positive Cholesky pivots, scales K and F separately, performs actual iterations and reports unavailable nonrepresentable results. It remains a bounded legacy interface. Existing 12 FEM and 12 multigrid scientific cases now import production implementations instead of carrying copied solvers. Canonical square root and stable residual norms replace the local toy routines.

## Reproduce

Run on an allocated Linux CPU worker with GCC supporting C++23, libquadmath and the identified raw compiler:

```sh
MADAROS_RAW_BIN=/absolute/path/to/madaros.elf \
SEED_RAW_BIN=/absolute/path/to/souc-lean-single-x86_64 \
bash scripts/stdlib/fem_poisson_gate.sh /absolute/fresh/result
```

The optional second argument is an immutable before-stdlib directory containing the original FEM and first checked FEM candidate, used only to retain failing before witnesses. The portable gate compiles and executes Madaros ELFs, compiles the independent C++23 binary128 oracle, checks committed-seed diagnostics when provided, verifies exact negative-control statuses, and checks source/compiler hashes. Seed checking alone is not native seed qualification; the extra run10 launcher in the receipt additionally compiles and executes the actual seed ELFs against the same independent reference.

## Qualification and source custody

All numerical jobs ran on Slurm cpu-ops / cpuops-t560-proxmox, UID 1000, one CPU, 2 GiB reservation, 512 MiB stack. Immutable source packets and failed attempts remain under `/scratch/sounio-stdlib-inference-scale-20261024/SIO173`; quota and unrelated jobs were unchanged.

The source base is main `c21b0e139830ca4949f80cba62887a8922a972d7` plus this feature. The raw compiler is the CI source-built artifact from run 40144792765, Witness job 121024346990, artifact 12580970584; ELF SHA256 `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`. Its 5295 input hashes/path catalog were verified against exact Git source. The separate committed seed SHA256 is `8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`; it is not described as a fresh rebuilt artifact.

Slurm 17284 qualified the full gate on immutable source09. Slurm **17302** repeated the final portable gate on source10 with C++ toolchain/reference hashes, then separately compiled and executed all seven positive fixtures using the committed seed. Both result10/STATUS and seed-native10/STATUS are zero, with independent C++ concordance and source/executable integrity checks. The reference uses binary128 analytic solutions and an independently evaluated physical stencil; it does not call the production solvers.

| Qualified fixture | Evidence |
|---|---|
| 1D FEM, 277 nodes | maximum absolute error 1.3100631690576847e-14 |
| Polynomial Poisson, 67517 nodes through 256 intervals | maximum absolute error 1.5778212070216568e-12; measured physical residual 5.9166893606743542e-11 |
| Sinusoidal refinement, 5684 nodes at 8/16/32/64 intervals | continuum error ratios 4.0232649236 / 4.0057912559 / 4.0014462671; discrete-reference error <=7.204e-13 |
| Subnormal load times huge geometry | expected and actual interior displacement 6.17582057301558187e-17 |
| Legacy imports | all original 12 FEM and 12 multigrid cases pass through production modules |
| Legacy scaled CG | nine rows verify flags, actual iteration count, returned/result displacement bits and atomic unavailable output |

The native polynomial fixture took 2.72 seconds / 3072 KiB maximum RSS; the refinement fixture 0.27 seconds / 1100 KiB in run09. These observations include printing every row and are not isolated solver benchmarks or a speedup claim.

Four corrupted inputs are stored separately from verifier logs: changed FEM node, removed completion marker, changed Poisson node, changed work count. They are rejected with exact exits 8 / 9 / 8 / 13. The numerical checks cannot pass merely because a mutation file was truncated by log redirection. Before witnesses preserve the original false-success singular solve, scaled-CG failure and prematurely underflowed element load; native/referee outcomes are recorded independently.

Earlier attempts are retained: 17216 exposed a nested mutable/immutable borrow in the fixture, 17221 failed reference-only launcher input selection, 17230 exposed the penalty-stiffness regression in the first global-scaling candidate. These are not counted as passes; row equilibration fixes the original bar case without relaxing expected values. Repeated controls after this correction are in result08 onward.

`native-cpp-text.tar.gz` contains text outputs, source/compiler/executable hashes, phase statuses, resources, run10 launcher and retained failure logs, excluding ELFs and reference executables. SHA256: `0a80794d00ba129d90e9e6043aa19026a3aacfdccd10be374cd434edd4920074`.

This is focused scientific qualification of this feature. It does not establish integration of all open sprint PRs, universal conditioning guarantees, other targets or the entire compiler corpus.

## Final review corrections and registered replay

Independent source review found two replay/safety issues: a legacy connectivity assertion mixed index validation and node reads, and an inherited `SOUNIO_SOUC_BIN` could take priority over the compiler whose hash the gate recorded. Index validation is now a separate assertion before node reads. The gate clears the inherited raw override, exports the identified Madaros raw path for resolver clients, and explicitly binds both seed aliases for seed checking.

Slurm **17306** replayed all native/independent-reference fixtures after the index correction and executed the exact five-line central registration block. Slurm **17307**, 57 seconds, then replayed the final pinned gate under the deliberately hostile inherited `SOUNIO_SOUC_BIN=/bin/false`: the full Madaros gate, all seven actual committed-seed native fixtures, independent concordance, and the exact extracted central registration all passed. The central invocation runs the child gate again with a fresh output directory; it is an execution of the actual registered code, not a textual registration check. Native scientific inputs and expected values were unchanged.

`final-17307-text.tar.gz` retains result11/12, seed-native11/12, central11/12, launchers and logs, excluding executables and generated source concatenations; SHA256 `0681e5ada0de62957df0d1f63df529b295bb12e6bcc6dc56af5f8b161642e723`. Launcher metadata detail: after the seed phase it changes the shell variable `out` to the central output directory; its EXIT trap therefore writes central11/12/STATUS, not seed-native11/12/STATUS. Seed execution evidence is the complete per-phase RESULTS.tsv, concordance outputs and `FEM_POISSON_SEED_NATIVE_PASS` in the Slurm log; no missing STATUS file is represented as a seed status receipt. The earlier run10 has its own correctly retained seed STATUS.

The feature branch is subsequently synchronized with main `7ce5eddeef78c76c7e744150ed9c169c53a3d3b0`. The exact c21-to-7ce Git diff contains only `.github/workflows/stdlib-reliability.yml` and `changelog.d/stdlib-reliability-workflow.internal.md`; all numerical and compiler input trees are identical. The a292 compiler remains identified as a c21-built artifact, not a 7ce rebuild. The focused central registration must be retained additively when integrating the other sprint PRs.
