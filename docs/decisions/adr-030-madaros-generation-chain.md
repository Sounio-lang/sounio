<!-- docs:meta
topic_id: repo.docs.decisions.adr-030-madaros-generation-chain
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-030-madaros-generation-chain
-->

# ADR-030: The Madaros generation chain (gen2 to gen7) and what each layer shows

**Status**: Proposed
**Date**: 2026-11-10

## Context

`scripts/ci/madaros_fixed_point_gate.sh` takes gen1 to be the Madaros ELF named in `MADAROS_BIN`, which the lean_single seed lineage built. gen2 is gen1 compiling `self-hosted/compiler/main.sio`, and gen3 is gen2 compiling the same source. The gate stops at gen3. gen2 and gen3 are compared with `cmp -s`, byte for byte on Linux (on macOS code signatures are stripped first). The gate passes when the rung it reaches equals the recorded expectation, even when that expectation is below `fixpoint`.

Measured in CI, from the last passing run on main (run 42641866996, commit 37fdea3e, 2026-11-09):

- gen1 compiles the compiler: the rung step took about 71 minutes and reported `into_acc_done=130`.
- gen2 exists and runs. The gate reported rung `run` as reached and the next wall as `gen3`, because gen2 could not compile the source it was built from (rc=139).
- gen3 has never been produced in CI.
- gen2 == gen3 has never been measured in CI.

Since 2026-11-09 19:24 UTC, 13 consecutive runs on main have failed in the job that contains the rung step. The step has no conclusion and the job has no log. The runner reports that it lost communication with the server. Two merges lie between the last pass and the first failure: #3046 (`fix/2966-gen3-arena-checker-init`) and #3494 (documentation text). Not known: whether the runner pod was killed for memory, whether #3046 made gen3 compile, and how much memory a generation needs now.

One memory figure exists. On 2026-11-10 a Slurm job that was building the compiler from a branch off current main requested 64G and ended OUT_OF_MEMORY after 36 minutes 53 seconds, with a recorded MaxRSS of 67,106,840K (about 64 GiB); the kernel log shows the killed process was `madaros`. In the last passing run on main, one day earlier, the same kind of compile finished in about 71 minutes inside a runner pod limited to 48Gi. So the memory a Madaros compile of the compiler needs rose from under 48 GiB to over 64 GiB between those two points, and the true peak is still unmeasured because the process was killed. The kernel log of the CI runner node itself has not been read.

ADR-006 names the gen2 == gen3 check as the trust anchor for compiler changes. ADR-029 names the committed seed `bin/souc-lean-single-x86_64` as the provenance root, and states that the requirement that the sha256 of Madaros gen2 through gen7 be identical applies to Madaros, whose chain starts from that seed. Every Madaros build today begins by compiling `lean_single.sio`, so Madaros is built by lean_single on every build.

The owner wants a chain that is bit-identical from gen2 to gen7, described as a fractal fixed point. He has not decided whether to retire `lean_single.sio` and wants measurements first.

## Decision

The chain is demonstrated in four layers, in this order. Each layer depends on the one before it.

### Layer 0: gen3 exists

A prerequisite. gen2 must compile `self-hosted/compiler/main.sio` and produce gen3. A pass shows only that the generation exists. It says nothing about equality with gen2.

### Layer 1: determinism in depth

- **Built**: gen2 through gen7, each compiled from the same source by the previous generation, starting from a gen1 built by the lean_single seed lineage.
- **Compared**: each generation with gen2, byte for byte, with the sha256 recorded in its receipt.
- **A pass shows**: for a deterministic compiler, gen N == gen N+1 once already implies that every later generation is equal. Five further equal generations are therefore evidence about determinism: no hidden state (uninitialised memory, hash order, address-dependent output) reaches the output.
- **It does not show**: a fixed point stronger than gen2 == gen3. A fixed point reached from one seed inherits whatever that seed contains.

### Layer 2: attractor

- **Built**: the same fixed point reached by a second route that does not share the first route's starting binary.
- **Compared**: the settled binaries of the two routes, byte for byte, with sha256 recorded.
- **A pass shows**: the fixed point does not depend on which binary the chain started from. This is the property that diverse double-compiling (D. A. Wheeler) relies on.
- **It does not show**: independence of origin. The repository has two compilers for the same language with almost no shared code (4 of 973 function names in common), and that independence of implementation is the asset here. But both were written in this project, and a committed Madaros prebuilt is itself a descendant of the lean_single seed, so starting a second chain from it varies the starting state, not the lineage. The route with real discriminating power is the cross route: each compiler compiling the other's source and then that source compiling itself, arriving at the same binary as the compiler's own chain. Whether Madaros can compile `lean_single.sio` is not known.
- A divergence here can be attributed to the starting point only after layer 1 has passed. Without determinism it cannot be told apart from nondeterminism.

### Layer 3: self-similarity by scale

- **Built**: the comparisons of layers 1 and 2 repeated per module and per function.
- **Compared**: module and function outputs between generations and between routes.
- **A pass shows**: the fixed point holds at every scale, not only for the whole binary.
- **Its use**: it localises a divergence. When two generations differ, it says where.

### Receipts

Each generation has one receipt, and a receipt is written even when the generation fails. Fields: generation number; sha256 of the produced binary; sha256 of the compiler that produced it; source commit; sha256 of the source closure; host; wall time; peak memory; exit status.

Peak memory is the field with one data point today (a compile killed at 64 GiB after 37 minutes) and no completed measurement.

### Where it runs

Nightly or on manual dispatch, not as a pull request check. On current evidence one generation costs on the order of 70 minutes and tens of GiB, so gen2 through gen7 is several hours. The existing rung step already runs only on push to main, on schedule and on `workflow_dispatch`.

## What this ADR does not decide

- It does not retire `lean_single`. The owner has not decided and wants measurements first. A retirement needs its own ADR, and the question of what replaces the seed as the build's starting point with it.
- It does not change the provenance root of ADR-029.
- It does not amend ADR-006 or ADR-029. ADR-006 keeps gen2 == gen3 as the check and ADR-029 keeps the seed as the root. This ADR adds a measured chain beside them.

## Consequences

- Layer 0 blocks every later layer. Until gen3 compiles in CI, nothing past it can be measured.
- Each generation leaves a receipt with recorded hashes, so each claim can be checked against files.
- A failure is a result. A layer 1 failure indicates nondeterminism. A layer 2 divergence points at the starting binary only after layer 1 has passed. Layer 3 locates it.
- The runner pool gives each pod 48Gi (request 24Gi), and the swap step used on GitHub-hosted runners is skipped on that pool. Peak memory per generation decides whether a generation fits.

## Open questions

1. The cause of the 13 consecutive failed runs on main is not established. The memory figure above makes exhaustion of the 48Gi runner pod the leading explanation, and #3046 the leading candidate for the rise, but neither has been measured directly. Measuring peak memory of the compile at the commit before #3046 and at the commit after it would settle both.
2. Memory per generation is not measured. One process was killed at about 64 GiB, so the true peak is higher than that and above the 48Gi runner limit. It is the first measurement required.
3. The second route for layer 2 is not chosen. Candidates: an older committed Madaros prebuilt (varies the starting state only), and the cross route through `lean_single.sio`.
4. The granularity for layer 3 is not defined: which units, and how a function is matched across builds.
