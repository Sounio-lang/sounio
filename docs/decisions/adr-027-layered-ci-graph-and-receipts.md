<!-- docs:meta
topic_id: repo.docs.decisions.adr-027-layered-ci-graph-and-receipts
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-027-layered-ci-graph-and-receipts
-->

# ADR-027: Layered CI, graph-directed test selection and receipts as the unit of evidence

**Status**: phase 1 and current-source compiler fan-out implemented in this PR; remaining phases proposed
**Date**: 2026-11-01
**Builds on**: [ADR-018](adr-018-release-cycle-and-versioning.md) (release trains, changelog fragments), [ADR-009](adr-009-verified-foreign-reference.md) (verified foreign reference oracles, the MPFR witness pattern §4.2 generalizes)
**Grounded in**: the measurement in §1, reproducible with the commands given there

## 1. Context: what the CI measures today

The following tables preserve the historical 2026-11-01 report of 300 runs
of `.github/workflows/ci.yml` (2026-10-29T12:43Z to 2026-11-01T07:37Z).
The original 300-run manifest and the 12 PR / 8 merge-group job sample were
not retained with that report. These historical distributions have not been
independently reproduced in this revision; the old one-page command below
retrieves at most 100 runs and does not reproduce them. Do not use those
tables as candidate performance acceptance. The dated 2026-11-02 receipt
below provides the retained two-run baseline and C++23 analyzer.

Historical discovery command (not a reproduction of the tables):

```bash
gh api 'repos/Sounio-lang/sounio/actions/workflows/ci.yml/runs?per_page=100&page=1' \
  -q '.workflow_runs[] | [.event,.conclusion,.run_started_at,.updated_at] | @tsv'
```

| Event | Runs | Outcome | Wall time p50 / p90 / max (min) |
|---|---|---|---|
| `pull_request` | 166 | 93 success, 33 failure, 40 cancelled | 39 / 81 / 171 |
| `merge_group` | 79 | 75 success, 1 failure, 3 cancelled | 40 / 64 / 154 |
| `push` to `main` | 53 | 15 success, **38 cancelled** | 43 / 46 / 86 |

Per-job medians over the 12 green PR runs and 8 green merge-group runs in
that window:

| Job | Median (min) | Max (min) |
|---|---|---|
| Native Self-Host (macOS arm64) | 28.4 | 31.8 |
| Madaros Current-Source f64 Lowering | 26.1 | 41.9 |
| Stdlib Deep Science | 17.3 | 19.6 |
| Contracts Core | 15.5 | 18.2 |
| Full Test Suite | 13.0 | 13.4 |
| Lean Proofs | 11.0 | 14.2 |
| Madaros Witness Gate | 10.9 | 11.5 |
| Gate Wave 0 | 9.2 | 9.3 |
| Source-Bootstrap Self-Host (Linux x86_64) | 5.6 | 6.3 |
| Sounio Lint | 1.9 | 2.0 |

Three facts follow.

1. **`main` had no complete verdict for 72% of its commits.** The
   `concurrency` block cancelled the previous run on every push to `main`,
   including pushes made by the merge queue itself. A regression that
   reached `main` was detected only when a later run happened to survive.
2. **The critical path is compiling the compiler.** The two slowest jobs
   rebuild Madaros from source on every selected run. The macOS lane also
   waits for the Linux source-bootstrap lane, so its chain is 34 min on a
   green run.
3. **Cheap verdicts arrive late.** Five gates that each take under a
   minute (epistemic shape ratchet, language gap ratchet, effect-name
   closed list, EISA origin, knowledge-annotation silent drop) ran as five
   separate workflows, each with its own checkout and prebuilt
   materialization, and outside the one required check (`CI Decision`).
   Meanwhile, on the 20 red PR runs sampled, the earliest hard failure
   was most often Contracts Core, 15 min in.

The repository also carries 29 workflows, 792 scripts under `scripts/ci/`,
3,433 test programs (2,542 run-pass, 862 compile-fail, 29 known failures),
581 modules in `self-hosted/` and 1,705 in `stdlib/`. Nothing in the CI
uses the module graph that the compiler itself computes.

## 2. Decision, phase 1 (landed with this ADR)

### 2.1 Cancellation is a pull-request property

A pull request keeps a stable group (`ci-<PR number>`) with
`cancel-in-progress: true`, so a newer push still supersedes the older
run. Every other event (push to `main`, merge group, schedule, dispatch)
gets a group of its own keyed by `github.run_id`, so it is neither
cancelled nor queued behind a sibling. Turning `cancel-in-progress` off
alone would not have been enough: a concurrency group holds at most one
pending run, so with run A in progress, B pending and C arriving, B is
dropped without a verdict (caught in review of the first draft). The
merge group already keyed on its own ref and was never affected.

### 2.2 One `Fast Checks` job, first, inside the required check

The five sub-minute gates are steps of a single `fast-checks` job in
`ci.yml` with no `needs:`, so it starts at t=0 and gives a verdict in
about three minutes. It is a key of the `required` map in
`scripts/ci/evaluate_ci_decision.py` and a `needs:` entry of
`ci-decision`, so a red ratchet is a red `CI Decision`. The two
accusation gates keep `continue-on-error: true` and upload their
receipts, exactly as their audits specify. The five workflow files are
removed.

### 2.3 macOS arm64 is event-selected

`native-selfhost-macos-arm64` runs in the merge group, on `main` and on
dispatch, not on `pull_request`. The evaluator treats `skipped` as correct
on a PR and as a failure on the other events; `impact_ci_selftest.sh`
pins both directions. An arm64 regression still blocks the merge, one
stage later, which is the trade-off Rust makes with its full platform
matrix in bors and not in PR CI.

### 2.4 Measured expectation

With 2.1–2.3 the PR critical path drops from the 34 min macOS chain to the
26 min current-source lane; the main-cancellation rate drops to zero by
construction. The remaining 26 min is the subject of phase 2.

## 3. Decision, phase 2: the compiler is the build graph

A self-hosted compiler already knows which module imports which. CI should
ask it instead of re-deriving impact from path globs.

- **`souc graph --json`** emits the module import graph (nodes: modules;
  edges: imports). Deliverable: a flag on both engines and a fixture test
  that the graph of `tests/multifile/` matches the expected edges.
- **Test selection by reverse reachability.** For a PR, the set of tests
  to run is `rdeps(changed modules) ∩ tests`. A change to
  `self-hosted/ir/lower.sio` runs the tests whose compilation reaches
  `lower`; a change to `stdlib/darwin_pbpk/` runs the tests that import it.
  The `Impact` classifier keeps its coarse buckets as a fallback and as a
  cross-check: a test selected by the graph but not by the bucket is a
  classifier bug and fails the self-test.
- **Content-addressed artefact cache.** Every compiled artefact (ELF,
  HLIR, test verdict) is keyed by the hash of its transitive source closure
  plus the hash of the compiler that produced it. Two PRs that do not touch
  the parser share the compiled parser. Storage is `actions/cache` keyed by
  those hashes; a self-hosted store can replace it without changing the
  contract.
- **Madaros from artefact, never from source, on a PR.** The Madaros of the
  last green `main` commit is published with its receipt (source SHA, ELF
  SHA, md5) and every PR job downloads it. Only a change under
  `self-hosted/` triggers a rebuild, and that rebuild is one job whose
  output the other jobs consume. #3224 is the first implementation of this
  rule; the six Copilot drafts #3181–#3188 overlap it and one line is
  chosen.
- **Topological sharding.** The full suite is partitioned by graph depth,
  not by count, so the minimum wall time is the depth of the graph.

Target: PR p50 under 10 min.

## 4. Decision, phase 3: bootstrap as a proof, semantics as a property

### 4.1 Diverse double-compiling

The committed ELF has already diverged from the source once
(`CLAUDE.md` §7: `assert!` accepted by the shipped binary, refused by the
source). The merge group compiles stage1 and stage2 with both engines
(Madaros and lean_single) and requires the two stage2 binaries to be
bit-identical. The result is `bootstrap-receipt.json` (seed SHA, stage1
SHA, stage2 SHA, per engine). A release ships only a binary whose receipt
closes. This is the standing answer to "is the committed binary older than
the fix?" (board cards 353, 354).

### 4.2 Four test families the suite lacks

- **Differential, by generated program.** A generator of `Knowledge<f64>`
  programs (arithmetic, calls, branches) runs nightly on both engines and
  reports any program where variance, sensitivity or value differ. Reduce
  with the existing `scripts/dev/run_kaxi_fuzz.sh` harness. Card 364
  (`sensitivity_of` returns 0 over a sum) is the class of defect this
  finds in one night.
- **Metamorphic, for uncertainty propagation.** Properties that must hold
  regardless of implementation: `var(a+b) = var(a)+var(b)` for independent
  inputs, `var(k·a) = k²·var(a)`, chain rule for composed GUM. Generated,
  not hand-written.
- **Oracle against MPFR for every numeric builtin**, generalizing the
  interval witness of card 224. ULP regressions block.
- **Compile-fail with anchored diagnostics.** The 862 compile-fail tests
  assert "refused"; they should assert code, line, column and message, as
  rustc's `ui` tests do. Cards 347, 348 and 429 are message-quality
  defects this catches.

## 5. Decision, phase 4: receipts are the unit of evidence

Every job emits `receipt.json` with a fixed schema (what ran, with which
binary, on which machine, result, signed with Sigstore). `ci-decision`
reads receipts, not job statuses. Consequences:

- A GPU gate run under Slurm on the DGX Spark produces the same receipt as
  a GitHub job; card 380 closes when its receipt exists, not when someone
  pastes evidence.
- A release carries the receipts of everything that validated it, with
  SLSA level 3 provenance on the published ELF
  (`actions/attest-build-provenance`) and an SBOM.
- The board reconciliation task reads receipts: `Done` means "the receipt
  for the acceptance criterion exists", which is what the `Acceptance`
  field already asks for.
- Performance is a gate with honest statistics: N runs of base and PR,
  Mann-Whitney, alert only on a significant shift above threshold, with a
  public time series (self-compile wall time and RSS, PBPK-28 check time,
  L4 decode tokens/s).

## 6. Flakiness as data

Every test failure is appended to the ledger with `{test, commit, engine,
runner, duration, stderr}`. A test that fails and passes on the same commit
is quarantined automatically (runs, does not block) and gets a card with
the history; after 14 days unrepaired it becomes a `known_failure` with the
evidence attached. A red `main` is bisected by a bot using the artefact
cache of §3, so no commit is recompiled.

## 7. Metrics and targets

Four numbers, published weekly by the same scheduled task that reconciles
the board:

| Metric | 2026-11-01 | 60-day target |
|---|---|---|
| PR wall time p50 | 39 min | 10 min |
| PR red rate | 25% | 10% |
| `main` cancelled rate | 72% | 0% |
| Merge-group wall time p50 | 40 min | 25 min |

## 8. Consequences

- Phase 1 changes no gate's script, flags or blocking semantics; it changes
  where and when they run. The removed workflows' audits carry a note
  pointing here.
- A PR touching only arm64 codegen gets no macOS verdict until the merge
  group. The `Impact` classifier may later add an `arm64` bucket that
  re-enables the lane on PR for those paths.
- Phases 2–4 each change the evidence contract and get their own ADR or an
  amendment here before landing.

## 9. Alternatives considered

- **Keep five workflows and add them to the ruleset.** Rejected: five
  checkouts and five prebuilt materializations per PR for three minutes of
  work, and the ruleset would need editing on every rename.
- **Keep macOS on PR but behind a path filter.** Deferred to the `arm64`
  bucket in §8; the current classifier has no such bucket and a wrong
  filter would silently skip the lane.
- **Never cancel on main, never cancel on PR either.** Rejected: 166 PR
  runs in three days at 40 min each; superseded PR runs are wasted
  minutes with no verdict anyone reads.

## 2026-11-02 implementation receipt

The workflow now builds one receipt-bound current-source Linux compiler in
`current-source-madaros`. Witness, Wave0, f64 and Deep Science depend directly
on that producer and verify the artifact with a constrained composite action.
Deep Science no longer waits for the entire Witness test job. SCO runs in a
separate required job with its existing independent rebuild and mutated-source
refusal controls preserved. Source-bootstrap and native self-host acceptance
remain independent. No scientific gate was removed.

CI Decision now requires both new jobs when selected, and the f64 evaluator
includes stdlib impact to match its workflow guard. Positive compiler/stdlib
fixtures and isolated failed/cancelled/skipped/missing controls cover selection.
The action text is explicitly pinned by a small repository wiring contract; it
is not presented as a general shell parser.

Baseline metadata from PR #3413: head run 41433054775 took 2401 seconds;
merge-group run 41438509619 took 2081 seconds. In the head run f64 took
2193 seconds, macOS 1866, Deep Science 1023; independent Wave0 and Witness
build steps took 428 and 369 seconds. The C++23 analyzer and projections are
in docs/evidence/ci-latency-20261102. Skipped jobs with inverted timestamps
are excluded explicitly. The analyzer ran on the Mac using Xcode C++23 after
a Slurm metadata-analysis attempt failed; no Sounio compiler was built locally.
These are baseline observations, not measured candidate speedups. Hosted CI
and a completed merge-group run remain the acceptance gate for the new graph.

Validation: receipt same-SHA reuse and tamper controls, impact/decision
self-tests, eight action/graph mutation controls, actionlint 1.7.12 (shellcheck
and pyflakes disabled), and diff whitespace checks. The optional coder-next
review through darwin-llm failed at the MCP transport; no independent model
approval is claimed. Primary review retained all SCO gate calls and the
existing knowledge-runtime paths.

### Follow-up: hidden builds and deep-science partitions

The full-gate dereference subgate now receives its parent compiler explicitly.
The SCO runtime may consume a receipt-verified compiler artifact; invalid
artifacts fail without falling back to a build. Standalone builds remain
available, and the exact-integral private modified-source rebuild remains.
Deep Science runs mandatory core/integration matrix jobs with fail-fast false
and unique artifacts. Native fixtures remain paired with their scientific
oracles in core; Arrow and logarithm integration are independent. The original
scientific bodies are byte-preserved under function wrappers; standalone all
preserves their original order. Hosted validation and comparison of the union
of result rows against run 41462324982 remain acceptance gates.
