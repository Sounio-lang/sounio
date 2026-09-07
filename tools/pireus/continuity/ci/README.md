# Pireus CI execution and historical custody

The ordinary run-pass harness executes the current stdlib. Eleven historical
Pireus fixtures require input files; their sibling .sio.args files now provide
literal argv through both the harness and the native compiler wrapper.

The imported Pireus modules also require sha256_digest_byte, recovered verbatim
from efc2ed41c2f7c6e8ef1e3940827aac742d23e2a0. Three digest builders now allocate
the [u8; 256] argument required by sha256_update and still hash exactly eight
bytes. The accessor is checked against the complete SHA-256 abc digest and
out-of-range indices, alongside the existing crypto vectors.

Historical parent inputs are data. The three .sio.snapshot files are never
imported or compiled by these tests; their code executes from current stdlib.
Two older execution-engine documents are also required by the frozen parent
hashes. historical-parents/manifest.json identifies every original Git commit,
path and byte hash. prepare_historical_inputs.py verifies these copies and the
two external Intel inputs against pinned SHA-256 values. Existing corrupt files
are refused without overwrite. Download failures remain failures.

Eight larger semantic witnesses retain their original assertions and source
bytes. Their .sio.timeout files declare a 1200-second execution ceiling. This is
an execution budget, not a performance result. route_slow_tests.py partitions
the actual full-suite inventory into ordinary tests and exactly these eight
witnesses. The eight matrix jobs use the same native compiler artifact as the
ordinary suite and are required by evaluate_ci_decision.py. A failed, missing,
cancelled or skipped selected slow job keeps CI Decision red.

Gate custody and execution are different checks:

- Pireus Gate Custody names 80 imported scripts and verifies their exact Git
  versions plus 456 literal tracked input paths at baseline 04dae64c915b.
  It reports current-tree drift and does not execute those historical scripts.
  The input inventory is partial: dynamic paths, external files, host identity,
  authority runtime, and historical toolchains still require actual replay.
- Gate Wave 0 executes six current Pireus compiler gates with Madaros built from
  the PR source: Walsh spectrum, twist factorization, cross-architecture
  synthesis, typed GPU lowering/refusals, XOR HLIR and sedenion contracts.
- The frozen-authority wrappers retain their original checks. Custody does not
  assert that their old host, kernel, compiler, Lean toolchain or material
  hardware environment can currently be replayed.

The existing unnamed-gate ratchet still measures workflow references, not
behavioral coverage. Its observed change from 551 to 470 includes references
to custody verification. It must not be described as 81 new runtime gates.

No result here proves a fresh Spark run, speedup, HTTP serving, general 16K
inference, or closure of V13/V14. The full 3-condition × 3-round × 32-proposal
pilot remains dependent on successful current integration validation.
