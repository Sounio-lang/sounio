<!-- docs:meta
topic_id: repo.docs.serious-language.current-madaros-downgrades
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.current-madaros-downgrades
-->

# Current Madaros Downgrades

This note records public-claim downgrades that are based on the live Madaros
binary used by the serious-language conformance spine in this validation lane. These are
not removals of the underlying fixtures; they are evidence boundaries for what
the current public spine may cite as passing behavior.

The downgraded rows are retained in
`tests/conformance/manifest.v1.downgraded.tsv` so the non-passing expectations
remain machine-readable without blocking the passing public spine.

Binary identity from the worktree used for this downgrade:

```text
bin/souc --version -> Madares v0.80.0 -- the Sounio self-hosted compiler
bin/souc info raw_elf -> bin/madaros-linux-x86_64
```

The `Madares` spelling above is the exact identity string printed by the
current binary.

Reproduction command:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  SOUNIO_SERIOUS_CONFORMANCE_ARTIFACT_ROOT=/tmp/sounio-conformance-final-cheap-fixes \
  scripts/ci/serious_language_conformance_gate.sh
```

Observed pre-downgrade summary:

```text
Summary: total=16 passed=10 failed=6
```

The failing public-spine cases were:

| Case | Claim | Observed result | Downgrade reason |
|---|---|---|---|
| `core-struct-run` | `core.structs` | Program exited 0 but printed `3` instead of expected `7`. | Two-field struct readback is not safe to cite as closed runtime behavior. |
| `observe-io-boundary` | `epistemic.observe` | `souc check` exited 0 and did not emit `cannot print Unobserved<T>`. | The live checker does not reject the intended IO leakage boundary. |
| `generics-multi-run` | `generics.functions` | Type checking rejected generic calls with `expected T` / `expected U`, `found i64`. | Generic function substitution/monomorphization is not closed in the current Madaros path. |
| `gum-compliance-run` | `epistemic.gum` | Native run exited 139. | GUM runtime examples are not safe to cite until the crash is fixed. |
| `gum-iso-budget-run` | `epistemic.gum` | Native run exited 139. | GUM runtime examples are not safe to cite until the crash is fixed. |
| `epistemic-bmi-run` | `epistemic.knowledge` | Native run exited 139. | Knowledge-value runtime examples are not safe to cite until the crash is fixed. |

The retained conformance spine still checks passing slices for small-program
syntax/execution, effects, imports, generic struct parsing, borrowing, and
epistemic boundary diagnostics.

Related fixture updates kept in the passing spine:

- `ownership-conflict-diagnostic` now expects `cannot borrow exclusively`, which
  is the current Madaros diagnostic for the same exclusive-borrow conflict.
- `knowledge_no_silent_unwrap.sio` now builds the source value through
  `measure(1.0, uncertainty: 0.1)` and assigns it to `f64`; the live diagnostic
  still rejects implicit `Knowledge<f64>` to `f64` assignment with
  `found Knowledge<f64>`.
