# Stdlib service-wave evidence

Base: `a95536ac032cb0ff1b3418c3232fa8c07fba09ac`.
Branch: `devpass/stdlib-wave3-services-20261027`.

## Arrow oracle migration

The canonical oracle is `tests/reference/arrow_cpp_interop.cpp` compiled as
C++23 against official Arrow 26.0.0. The required preservation map is
`tests/reference/arrow_legacy_case_inventory.tsv`. Legacy Python files are
read-only historical inputs; no Python judge is part of acceptance.

Slurm 18105 preserved the initial C++ compilation failure. Slurm 18109 built
the corrected judge with `-Wall -Wextra -Werror`, confirmed
`ARROW_CPP_VERSION 26.0.0`, and recorded its executable and bound libraries.
Receipt root on DL380p:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-cpp-02/results`.

This is build and SDK binding evidence. Full producer/oracle interoperability
requires a separate completed gate receipt; none is asserted by the build.
The SDK was independently provisioned and qualified by the root lane
(Slurm 18103). No system packages were installed by this lane.

## Full Arrow interoperability receipt

Slurm **18126** completed with status 0 on DL380p. It required all five
producer markers, all 30 inventory case groups, the SDK-absence refusal,
`ARROW_CPP_INTEROP_OK`, `ARROW_CPP_ID_MUTANT_REJECTED` and
`ARROW_CPP_GATE_OK`.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-full-02/results/gate`.
`INPUTS.sha256`, `COMPILER.sha256`, `EXECUTABLES.sha256`,
`BOUND_LIBRARIES.sha256` and `ARTIFACTS.sha256` have successful final
verification files. `FIXTURE_RELOCATION.json` records each original/derived
source hash, literal path mapping and byte-exact reverse check.
`EXPECTED_CASES.txt` and `OBSERVED_CASES.txt` agree exactly.

The raw compiler was built from base a95536ac and pinned to SHA-256
`7df87ae9d9b4b3c0cefcd18b892512b6bf7e5c6fde6690ca5223669a9a1a2639`.
The initial full job 18123 passed the semantic checks but failed inventory
closure because the inventory had an extra blank row. Job 18126 used the
same assertions, inputs and oracle after removing that blank row and passed
the complete gate. This is a focused Arrow receipt, separate from the
whole-stdlib acceptance run.
