<!-- docs:meta
topic_id: repo.docs.audit.type-archaeology-family-b-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.type-archaeology-family-b-2026-08-19
-->

# Type archaeology family B — fixture index (protocol v3)

**This is a census converted to fixtures.** It does not reclassify `docs/internal/concepts/registry.tsv`. It does not store a ladder position.

**Index:** [`TYPE_ARCHAEOLOGY_FAMILY_B_2026-08-19.tsv`](TYPE_ARCHAEOLOGY_FAMILY_B_2026-08-19.tsv)  
**Derive / CI gate (writes nothing; prints the table that used to be handwritten):**

```bash
bash scripts/ci/typekind_archaeology_b.sh
# report-only (never fails the ladder criteria):
TYPEKIND_B_REPORT_ONLY=1 bash scripts/ci/typekind_archaeology_b.sh
```

**Fixtures:** `tests/typekind-archaeology/family-b/`  
That tree is **not** in the `run_sio_test_suite.sh` globs. The CI gate is the runner:

```bash
bash scripts/ci/typekind_archaeology_b.sh
```

Wired in `.github/workflows/ci.yml` (Contracts → TypeKind archaeology family B). The older report-only derive remains at `scripts/dev/typekind_family_b_derive.sh`.

**Engine:** this worktree `bin/souc` (Madaros). Inherited `SOUC_BIN` / `SOUNIO_SOUC_ENGINE` unset. `SOUNIO_STDLIB_PATH` pinned here. `MADAROS_STACK_KB=524288`, `ulimit -s 1048576`.

## What the index stores

`kind`, `pass_path`, `refuse_path`, `expected_diagnostic`, `deepest_layer`

It does **not** store Garden / Hypothesis / Reserved / Executable / Claim-ready. Those five are the output of running the two programs.

Derivation (protocol v3):

| pass fixture | refuse fixture | derived |
|---|---|---|
| absent | absent | Garden |
| only one present | | Hypothesis |
| `souc check` OK | `souc check` fails and log contains `expected_diagnostic` | Claim-ready |
| `souc check` fails | fails, named diagnostic | Reserved |
| `souc check` OK | `souc check` OK | Executable |
| anything else with both files | | Hypothesis |

A refuse that starts to pass is printed as `refuse_named=xpass` (the known-failure pattern that landed as `scripts/ci/known_failure_madaros_recheck.sh`). A pass that starts to fail moves a previous Claim-ready row to Reserved or Hypothesis on the next derive. Nobody edits a position column.

`deepest_layer` is a **source-name** inventory: parser `TypeExprKind`, checker `TypeKind`, HLIR `HlirTypeKind`, IR `IrOpcode` / `Ir*Info`, codegen `native/lower_ir.sio`. Family B: all sixteen are named in parser and checker; only Contest is `HlirTypeContest`; several reappear as IR opcodes and codegen MOV wrappers. Both the HLIR hole and the IR/codegen resurrection are debt. This column is not a claim that a fixture compiled through that layer.

## Conversion from the v2 prose (not live positions)

The v2 document wrote positions by hand. v3 keeps the programs that justified those sentences and throws the sentences away.

| v2 handwriting | What v3 leaves |
|---|---|
| Contest Claim-ready | `Contest.pass.sio` (constructor that checked OK) + `Contest.refuse.sio` (E059) |
| AcquisitionPlan / RecoursePlan / AlternativeSet / AlternativeOption / TransitionPlan / Deferred Reserva | intended constructor as `*.pass.sio` (does not check today) + named `*.refuse.sio` |
| DeferralPolicy Hypothesis | `DeferralPolicy.pass.sio` (item DSL that did not parse) + `DeferralPolicy.refuse.sio` (E097) |
| Eight Garden kinds (six Policies + ObservedTransition + RollbackCertificate) | no fixtures; index paths are `-` |

Garden is not a failure of the census. A new TypeKind with no fixtures will derive Garden without a meeting.

## What this does not do

- Does not treat `fn takes(x: Kind<i64>)` or `let x: Kind<i64> = 1` as either fixture.
- Does not point at `tests/frontend/*_basic.sio` (policy-item DSL parse-fail is not a named TypeKind refuse).
- Does not use lean_single as a semantic authority.
- Does not claim that `Contest.pass.sio` compiles or runs. The v2 turn saw `souc compile` die at `native-v2 bridge compilation failed`. The CI gate records `pass_run` when check is OK but does **not** fail on a non-zero run. Claim-ready remains a check fact until `pass_run` is 0.
- The CI gate fails on refuse xpass, refuse missing named diagnostic, or Contest pass-check regression. Reserved pass fixtures are allowed to fail check.

## Pairing

The Policy/Plan rhyme is still not a live Madaros relation. The constructors that would witness “a policy declares, a plan executes” are the `*.pass.sio` files for the Plan/Deferred side. They fail before naming the Policy TypeKind. The six undocumented Policy TypeKinds remain Garden (no fixtures).
