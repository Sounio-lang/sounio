<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-172.compiler-integration-20261025.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-172.compiler-integration-20261025.readme
-->

# SIO-172 integration of the reviewed private-borrow compiler fix

This integrates independent compiler [PR3119](https://github.com/Sounio-lang/sounio/pull/3119),
exact9feb2e1a8a3e7fdc3e9be790f735c72043e0b4b3, into the OWN phonon feature branch.
It does not merge anything into main or change the scientific implementation.

The starting phonon HEAD is55b80344d4995b7ec53a9f19d0b7d760e5baad22.
All40 incoming paths were claimed after a clean-worktree check and the branch
intent was announced on the coordination bus. The merge6966d33ceb0961128cd8414401777f623d89f5d0
needed no conflict resolution; the Linux CI registration is additive.

## Existing science and tests are byte-identical

EXISTING_SCIENCE_TESTS.sha256 records all8583 tracked stdlib/test files before
the integration. Every entry verifies after the merge. Existing production
sources, physical parameters, expected numerical values, refusal cases,
tolerances, test annotations and known-failure baseline are unchanged.
The only incoming test files are the new private-borrow regressions.

The complete raw verification output remains in the remote workspace;
VERIFIED_EXISTING.txt records its SHA256, the manifest identity and the all-OK
coverage check, without duplicating thousands of identical OK lines in Git.

## Qualified compiler reuse, explicitly source matched

The standalone lean_single source and canonical host/build/comparison helpers
match the already qualified private-fix build inputs byte for byte.
QUALIFIED_COMPILER_INPUTS.verify.txt records each comparison.

The previously source-built host,Stage2,Stage3 are retained at:
 /scratch/sounio-stdlib-inference-scale-20261024/SIO172private/after04/host_gate/artifacts.

All three have SHA256:
 13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b.

The integration gate rechecks actual Stage2/Stage3 byte equality and artifact
checksums. It reuses this qualified artifact instead of repeating the bootstrap
or copying binaries. This is not a claim of a fresh compiler build on this
feature merge; the original source-built fixed-point custody remains in the
independent compiler evidence. The source match is the explicit basis for reuse.

## Actual merged-source execution

Slurm17391 completes in38s on UID1000 with one CPU,4GiB job memory and
524288KiB stack. It runs current merged inputs with the matched new Stage2:

- The complete private-field gate passes44 target E259 cases plus public controls.
- The byte-exact archived original phonon negative refuses with E259/no ELF.
- The current unchanged phonon_factor_private fixture also exits1 with E259.
- Imported phonon core171 values, SI parity30 values, and13 guard values/31
  statuses pass native execution and the unchanged C++23 decimal100 oracle.
- The entire existing modular phonon gate passes separately, including the
  original wrong-variance witness and missing/corrupted-oracle controls.

Source/compiler checksums verify after execution. The staged packet contains
sources and launchers; no compiler archives or duplicate compiler ELFs are
materialized. The original compiled negative is never executed.
The ARM64 public-global borrow limitation remains separately documented in the
compiler PR; this integration does not claim universal compiler enforcement.

Worker receipt root:
 /scratch/sounio-stdlib-inference-scale-20261024/SIO172integration/merged01.

The PR is now explicitly stacked on the reviewed compiler fix. Hosted CI on
the updated feature HEAD remains a separate result. Draft/human-review status,
main merge and publication must be observed independently; a local gate pass
does not imply them.
