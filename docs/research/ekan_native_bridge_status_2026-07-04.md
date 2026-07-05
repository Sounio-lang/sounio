<!-- docs:meta
topic_id: repo.docs.research.ekan-native-bridge-status-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ekan-native-bridge-status-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# E-KAN native bridge status - 2026-07-04

Current branch: `gpu/epistemic-tensor-core-next`.

## Current bridge audit

The default native-v2 path still compiles E-KAN witnesses, but current evidence
does not support a native-v2 E-KAN run-pass claim. Even the compact epistemic KAN
smoke currently compiles and then segfaults when the generated ELF is executed:

```bash
./bin/souc run tests/run-pass/ekan_knowledge_basic.sio
```

Observed current classification:

```text
EKAN_NATIVE_FRONTIER_PROBE_KNOWLEDGE_BASIC=KNOWN_BLOCKER
```

The fixed-point reductions that were previously explored as native-v2 run-pass
candidates have been demoted to `tests/known_failures/` because the current
default path compiles them but the generated ELF segfaults. The current
frontier snapshot is:

```bash
./scripts/ci/ekan_native_frontier_matrix.sh
```

Observed current result:

```text
EKAN_NATIVE_FRONTIER_PROBE_KNOWLEDGE_BASIC=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_EDGE=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_HIDDEN_COMPACT=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_HIDDEN_VERBOSE=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_LINEAR_READOUT_COMPACT=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_LINEAR_READOUT_VERBOSE=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_HAT3_READOUT=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_HAT5_READOUT=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_TWO_HIDDEN=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_TWO_HIDDEN_READOUT=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_THREE_INPUT_TWO_HIDDEN=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_FOUR_INPUT_ONE_HIDDEN=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_PROBE_FOUR_INPUT_TWO_HIDDEN=KNOWN_BLOCKER
EKAN_NATIVE_FRONTIER_MATRIX_PASS
```

The CI guard is therefore a bridge audit, not a native-v2 success claim:

```bash
./scripts/ci/ekan_native_core_gate.sh
```

Observed result:

```text
EKAN_NATIVE_CORE_GATE_PASS
```

It proves the surrounding compiler can run a simple native executable
(`examples/hello.sio`), proves the E-KAN examples on the preserved
`lean_single` path, and classifies the native-v2 E-KAN reductions as bridge
blockers instead of promoting flaky or failing witnesses.

## Full showcase surface

`examples/epistemic_kan.sio` is the fuller 4 -> 6 -> 2 E-KAN showcase. It now
typechecks after returning `failed as i32` from `main`, and its output avoids
`print(f64)` because the current runtime can segfault while printing float
values.

The full showcase passes on the preserved `lean_single` path:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/epistemic_kan.sio
```

Observed result:

```text
ALL PASS
```

`examples/epistemic_kan_fixed_point.sio` is a fixed-point E-KAN witness using
`i64` scaled arithmetic (`1000 == 1.0`). It composes two hat-basis input edges
into a hidden node, then one output edge, and propagates uncertainty by
quadrature across the composed path. It also passes on `lean_single`:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/epistemic_kan_fixed_point.sio
```

Observed result:

```text
ALL PASS
```

## Native-v2 blocker

The full 4 -> 6 -> 2 showcase still fails on the current native-v2 bridge:

```bash
./bin/souc run examples/epistemic_kan.sio
```

Observed failure:

```text
Segmentation fault
```

Observed failures now consistently appear after native-v2 compilation, when the
generated ELF runs. This is not a scientific E-KAN failure; it is a compiler
bridge/runtime boundary around E-KAN-shaped floating-point or fixed-point edge
evaluation, array references, struct returns, and uncertainty composition.

The reductions currently preserved under `tests/known_failures/` cover:

- fixed-point edge-only E-KAN
- compact and verbose hidden-node composition
- compact and verbose linear readout
- 3-knot and 5-knot hat-basis readout
- two-hidden width expansion
- two-hidden downstream readout
- three-input/two-hidden input-width expansion
- four-input/one-hidden and four-input/two-hidden expansion

The next implementation step should repair the native-v2 generated-ELF runtime
failure starting from the smallest reduction:

```text
tests/known_failures/ekan_fixed_point_edge_native_v2_probe.sio
```

Only after that probe repeats green should any fixed-point E-KAN witness be
promoted back to `tests/run-pass/`.
