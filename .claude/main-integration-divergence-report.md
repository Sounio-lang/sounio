# `main` vs `integration/sounio-dev-ready-base` — Divergence Inventory

Read-only archaeology. Merge-base: `2a530d031d52e2afae5472050857944de92484a6` (2026-08-22).
`main` tip at the time of this analysis: `e6cc1e4bf1100fb5e57193f2c59e172e57fa9ae9` (2026-09-22, merge of PR #2600 -- `origin/main` is a moving ref, this SHA is what the commit/file counts below were actually computed against). `integration` tip: `d502e3d5a1002514e11453707af97831e961ef19`.
No merge/resolution decision is made or proposed here.

---

## Executive summary

- **Scope**: `main` has 989 unique commits touching 3,563 files since the merge-base; `integration` has 223 unique commits touching 277 files. Both sides merged real PRs continuously through today — this is a full month of parallel, non-trivial development on both branches, not a stale side-branch.
- **File overlap is small in absolute terms**: only **40 files** were touched by both sides (~1.1% of `main`'s changed files, ~14% of `integration`'s). ~3,760 files are "safe" (changed on only one side).
- **Highest risk, confirmed**: both compiled ELF binaries (`bin/madaros-linux-x86_64`, `bin/souc-lean-single-x86_64`) were rebuilt independently on both branches and now hold **different final blobs** — these cannot be textually merged and must be rebuilt/chosen deliberately. `self-hosted/compiler/lean_single.sio` (main: 86 commits, +6353/‑585; integration: 8 commits, +768/‑76), `self-hosted/check/check.sio` (main: 79 commits, +4615/‑505; integration: 3 commits, +177), `self-hosted/ir/lower.sio` (main: 77 commits, +7376/‑796; integration: 6 commits, +450/‑5), plus `self-hosted/check/types.sio`, `self-hosted/parser/types.sio`, `self-hosted/native/codegen_x86_linux.sio` are all independently, heavily edited compiler-internals files on both sides. This is the reconciliation core.
- **One genuine cross-pollination event found, not blind duplication**: the "IndepKnowledge / d-separation / conditional-independence" epistemic-type feature (PR #1758, `feat/independencia-na-composicao`) was developed on `main` and then **manually hand-ported into `integration`** by the same author (commit `e5e742428 "fix(indep): resolve #1758's real merge conflicts with integration/sounio-dev-ready-base"`), with matching commit messages replayed on both sides (e.g. `[epistemic] Quadrature requires a discharged proof, not an assertion` appears verbatim on both). Several `tests/run-pass/*` and `tests/gpu/*` files are **byte-identical** between the two tips as a result. This is intentional, already-reconciled parity work, not a duplicated-effort risk — but it explains ~19 of the 40 overlapping files.
- **A real silent-conflict found inside that "reconciled" feature**: 4 of the `tests/compile-fail/*.sio` files that are otherwise identical assert **different error codes** on each branch for the same test (e.g. `dsep_fork_unconditioned.sio`: `E255` on `main` vs `E176` on `integration`; `cond_indep_violation.sio`: `E072` vs `E209`). The diagnostic-code numbering in `check.sio` has drifted independently between the branches even where the underlying feature logic is shared — a naive merge would need every affected `error-pattern:` re-verified, not just the `.sio` source.
- **CI is materially different, not just reformatted**: `main`'s `.github/workflows/ci.yml` gained ~25 new gates integration doesn't have (backend-claim parity, PIREUS admission engine, measured-claims gate, GUM-covariance gate, duplicate-definition gate, etc.) — 93 commits vs 4 on `integration`. Merging without reconciling CI would silently drop coverage that has been load-bearing on `main` for a month.
- **Docs/governance/dataset files touched on both sides** (`KNOWN_LIMITATIONS.md`, `DOCS_AUTHORITY_MATRIX.md`, `topic-registry.v1.json`, the two `datasets/sounio-code-examples/*.jsonl`) are large diffs on `main`, small on `integration` — these are low-*mechanical*-risk (text/JSON merges are tractable) but will need editorial reconciliation since both sides independently documented compiler status/known-limitations.
- **Thematically the branches are disjoint by design**: `main`'s unique work is dominated by particle-physics/GUM epistemic splits (WP20/WP25 HVP), octonion/sedenion associator and anti-garbling research, CI/gate hardening, and heavy `self-hosted/{compiler,check,ir,native}` churn. `integration`'s unique work is a TCP/TLS/X.509/crypto networking stack on Madaros (`stdlib/{x509,crypto,tls,net,hash,bignum,asn1}`, ~90 commits) plus the IndepKnowledge/d-separation type system and light CUDA/GPU backend fixes.
- **Risk bucketing of the 40 overlapping files** (8+6+12+3+11 = 40, reconciled against a direct recount): 8 **high-risk** (2 binaries + 6 compiler-internals `.sio` files, §3a), 6 **compile-fail tests with diagnostic-code drift** (§3c: 5 with a mismatched `error-pattern:` code + 1 non-conflicting enrichment), 12 **medium-risk** (docs/governance/dataset/CI/script files, §3d — mechanically mergeable but need editorial reconciliation), 3 **medium-risk** `stdlib/epistemic` files with divergent line-counts on an already-shared feature (§3e), 11 **low-risk** test fixtures already confirmed byte-identical (§3f).

---

## 1. `main`'s unique commits (989), by subsystem

Bucketed by top-level/second-level directory touched (a commit can appear in multiple buckets; "commits" = distinct commits touching that bucket, not file count).

| Bucket | ~Commits | Primary directories | Example commit subjects |
|---|---:|---|---|
| Docs | 245 | `docs/` | `docs(compiler): reconcile known limitations by engine (rebased; a claim was falsified, not just stale) (#2233)`; `docs(audit): ...` (29 commits) |
| CI / tooling | 224 (scripts/ci) + 92 (`.github`) | `scripts/ci/`, `.github/workflows/` | `ci: raise per-test timeout 90 -> 180s for the changed-tests gate`; `ci(epistemic): run worldline merge ABI gate` |
| Build artifacts / receipts | 193 | `artifacts/` | seed-refresh receipts, madaros gate-receipts (mechanical, regenerated by scripts) |
| Test suite (run-pass) | 189 | `tests/run-pass/` | new `.sio` fixtures pinning fixed behavior |
| Compiler frontend/middle: `self-hosted/compiler` | 103 | `self-hosted/compiler/` | `feat(madaros): ...` (30), lean_single.sio churn |
| Compiler: `self-hosted/check` | 84 | `self-hosted/check/` | `fix(check): ...` (16), diagnostic-code remaps (`fix(diagnostics): remap colliding Madaros E2xx codes #2191`) |
| Compiler: `self-hosted/ir` | 78 | `self-hosted/ir/` | `fix(lower): ...` (15), e-graph rewrite work |
| Test suite (compile-fail) | 64 | `tests/compile-fail/` | new negative-test fixtures |
| Compiler binaries | 59 | `bin/` | seed refreshes, ELF rebuilds |
| Dev scripts | 44 | `scripts/dev/` | test-suite harness v2, workspace doctor |
| `.claude/` agent contracts | 39 | `.claude/` | agent-bus, offload policy updates |
| Compiler: `self-hosted/native` | 35 | `self-hosted/native/` | `fix(ffi): ...` (9), x86 codegen fixes |
| Benchmarks | 33 | `benchmarks/` | perf tracking |
| Compiler: `self-hosted/parser` | 31 | `self-hosted/parser/` | grammar/type-annotation work |
| Datasets | 30 | `datasets/` | training-example JSONL regeneration |
| Formal (Lean proofs) | 26 | `formal/` | `formal(lean): ...` (8) — includes the octonion Mathlib-linked port seen on the current working branch |
| Particle physics / GUM epistemic split | ~17 | `stdlib/particle_physics`, `examples/particle_physics` | WP20/WP25 HVP window splits, `a_mu` GUM decomposition (`feat(particle): LD(ud) splits; W pull is larger than LD, LD larger than SD`) |
| Octonion/sedenion algebra research | ~9 (feat) + 9 (research) | `stdlib/algebra`, top-level research commits | `research: Anti-Garbling Completeness — two orthogonal axes unify noise-sets and the associator (Lean-verified) (#2098)`; `feat(algebra): sedenion Artin probe fails; octonion embedding holds` |
| KMC / chemistry / hydrogen examples | ~28 | `examples/kmc`, `examples/chemistry`, `stdlib/chemistry` | `feat(kmc): ...` (11) |
| Epistemic stdlib | 22 | `stdlib/epistemic/` | includes the shared IndepKnowledge feature (see §3) |
| Math stdlib | 18 | `stdlib/math/` | — |
| GPU backend | 4 | `self-hosted/gpu/` | small relative to the CUDA-ABI fix shared with integration |
| Website / demos | ~16 | `website/`, `demos/` | — |

## 2. `integration/sounio-dev-ready-base`'s unique commits (223), by subsystem

| Bucket | ~Commits | Primary directories | Example commit subjects |
|---|---:|---|---|
| Test suite (run-pass) | 113 | `tests/run-pass/` | new fixtures, largely TLS/crypto and IndepKnowledge |
| Docs | 79 | `docs/` | `docs(audit): ...` (5), architecture/status notes for the networking stack |
| X.509 / PKI | 32 | `stdlib/x509/` | `[x509/crypto] Fix 6 minor findings from P-384 ECDSA final review`; `fix(x509): flip stale P-384-rejection test to assert correct verification` |
| Crypto primitives | 30 | `stdlib/crypto/` | SHA-384/HKDF, cipher suites for TLS 1.3 |
| TLS | 9 | `stdlib/tls/` | `Merge pull request #2133 from Sounio-lang/tls-on-madaros` |
| Hash | 8 | `stdlib/hash/` | — |
| Bignum | 7 | `stdlib/bignum/` | big-integer support underpinning RSA/ECDSA |
| ASN.1 | 6 | `stdlib/asn1/` | DER encode/decode for certs |
| Networking | 6 | `stdlib/net/` | TCP socket layer under Madaros |
| Interop tests | 10 | `tests/interop/` | cross-engine/cross-protocol checks |
| IndepKnowledge / d-separation (epistemic type system) | ~12 | `stdlib/epistemic/`, `self-hosted/check`, `self-hosted/parser`, `self-hosted/compiler` | `Merge pull request #1758 from Sounio-lang/feat/independencia-na-composicao`; `[parser] Grammar for IndepKnowledge<T, A, B \| Z>`; `[check] Declare the causal graph in source; discharge independence by d-separation` — **this cluster is the shared/ported feature, see §3** |
| GPU/CUDA backend | 2 | `self-hosted/compiler` (lean_single.sio) | `[backend] Generalize simple GPU PTX parameter matching`; `[backend] Fix CUDA runtime ABI launch packing` — matched by a *combined* commit on `main` (`[backend] CUDA runtime ABI launch packing + GPU PTX param matching`), i.e. the same fix, same author, landed as one commit on `main` and two commits on `integration` |
| CI | 4 | `.github/workflows/ci.yml` | worldline-merge ABI gate, timeout increases only — far lighter-touch than `main`'s CI work |

## 3. File-level overlap (40 files touched by both sides)

Full list and per-file commit/line counts were computed via:
```
git diff --name-only 2a530d031d e6cc1e4bf1100fb5e57193f2c59e172e57fa9ae9
git diff --name-only 2a530d031d d502e3d5a1002514e11453707af97831e961ef19
comm -12 <(sort ...) <(sort ...)
```
(pinned SHAs, not the moving `origin/main`/`origin/integration/...` refs -- using the moving refs re-run later picks up unrelated commits merged after this analysis, including this very PR's own eventual merge)

### 3a. Highest risk — compiler internals & binaries (8 files)

| File | `main` commits / diff | `integration` commits / diff | Note |
|---|---|---|---|
| `bin/madaros-linux-x86_64` | 9 commits | 2 commits | **Different final blob on each tip** (confirmed via `git rev-parse <tip>:<path>`). Binary, cannot be textually merged. |
| `bin/souc-lean-single-x86_64` | 47 commits | 2 commits | **Different final blob on each tip.** Same as above; `main` has iterated on this seed far more (47 vs 2 commits). |
| `self-hosted/compiler/lean_single.sio` | 86 commits, +6353/‑585 | 8 commits, +768/‑76 | `integration`'s 8 commits are almost entirely the IndepKnowledge port + the shared CUDA/GPU-PTX fix (see §3b) plus merge-conflict-resolution commits. `main`'s 86 commits are broad, unrelated compiler work. Reconciling this file means replaying `main`'s much larger changeset on top of a base that already contains the ported feature. |
| `self-hosted/check/check.sio` | 79 commits, +4615/‑505 | 3 commits, +177 | Same pattern — integration's 3 touches are the IndepKnowledge grammar/check additions (pure insertions, no deletions), main's are broad and much larger. |
| `self-hosted/ir/lower.sio` | 77 commits, +7376/‑796 | 6 commits, +450/‑5 | Two of integration's 6 commits are explicitly reconciliation commits (`fix(lower): reconcile Knowledge-shadow fix with main's independent patch` — the author was *already aware* of and manually reconciling a main-side fix here). |
| `self-hosted/check/types.sio` | 9 commits, +468/‑112 | 1 commit, +1/‑1 | Integration's touch is trivial (1-line); low actual conflict surface despite appearing in the overlap list. |
| `self-hosted/parser/types.sio` | 13 commits, +354/‑39 | 2 commits, +135 (pure insertion) | Integration's insertions are the IndepKnowledge grammar; no deletions, so a line-level merge is plausible but should be verified against main's independent 354-line churn. |
| `self-hosted/native/codegen_x86_linux.sio` | 38 commits, +1903/‑156 | 1 commit, +5/‑5 | Integration's touch is trivial; main did the heavy lifting alone. |

**This is the highest-risk reconciliation area, as flagged in the task brief.** Both binaries independently rebuilt and diverged; the underlying `.sio` sources also diverged, though for most of these files `integration`'s side of the diff is small and already partly reconciled (see §3b) while `main`'s side represents a month of largely orthogonal, much larger compiler work.

### 3b. The IndepKnowledge / d-separation feature — confirmed intentional cross-port, not blind duplication

Evidence: identical commit *subject lines* appear independently on both branches for this feature, e.g.:
- `[epistemic] Quadrature requires a discharged proof, not an assertion` — on `main` as `4624d5321`, on `integration` as `c5b13229b`.
- `[epistemic] Quadrature is the independence law: make the assumption explicit` — on `main` as `0349a4303`, on `integration` as `66db7dd43`.
- `[parser] Grammar for IndepKnowledge<T, A, B | Z>` and `[check] Declare the causal graph in source; discharge independence by d-separation` appear on `integration` only, but `integration`'s history contains an explicit commit `e5e742428 "fix(indep): resolve #1758's real merge conflicts with integration/sounio-dev-ready-base"` — i.e. PR #1758 (`feat/independencia-na-composicao`, merged to `main` as `8d56129c5`, notably also the current working branch's own ancestor via `integration`) was **manually replayed onto `integration` by hand**, conflict-resolved commit by commit, by the same author.

Confirmed via direct content diff (`git show <tip>:<path>`) that these files are **byte-identical** between the pinned `main` SHA (`e6cc1e4bf1`) and the `integration` tip (`d502e3d5a1`):
- `tests/run-pass/causal_graph_dsep.sio`
- `tests/run-pass/gum_independence_required.sio`
- `tests/gpu/epistemic_runtime/manifest.tsv`
- `tests/madaros/source_to_elf/knowledge_field_shadow_exit0.sio`
- `tests/madaros/source_to_elf/manifest.tsv`
- `tests/run-pass/gpu_param_names_read_mut.sio`, `gpu_param_names_readonly_pair.sio`, `gpu_readonly_multi_slice.sio` (part of the shared CUDA/GPU-PTX fix, see below)
- `tests/run-pass/graded_compose_needs_witness.sio`, `indep_knowledge_grammar.sio`, `knowledge_layout_shadows_user_field_name.sio`

**Conclusion: no genuine duplicated-effort risk here** — this is already-reconciled parity, done deliberately by a human, not two independent implementations that now need arbitrating.

**GPU/CUDA fix**: `main`'s single commit `26c0e0581 "[backend] CUDA runtime ABI launch packing + GPU PTX param matching"` corresponds to `integration`'s two commits `8d203709e "[backend] Generalize simple GPU PTX parameter matching"` + `a9c5f7d85 "[backend] Fix CUDA runtime ABI launch packing"`. Same fix, evidently split differently when ported. Also already-reconciled, not a duplication risk.

### 3c. Real (small) divergence inside the "reconciled" feature — error-code drift

Five `tests/compile-fail/*.sio` files are identical except for the asserted diagnostic code, meaning `check.sio`'s error-code numbering has drifted independently between branches even on shared logic:

| File | `main` expects | `integration` expects |
|---|---|---|
| `tests/compile-fail/cond_indep_violation.sio` | `E072` | `E209` |
| `tests/compile-fail/dsep_fork_unconditioned.sio` | `E255` | `E176` |
| `tests/compile-fail/dsep_collider_conditioned.sio` | `E255` | `E176` |
| `tests/compile-fail/indep_var_not_in_graph.sio` | `E254` | `E175` |
| `tests/compile-fail/quadrature_needs_proof.sio` | `E255` | `E176` |

`main` independently ran a diagnostic-code remap (`fix(diagnostics): remap colliding Madaros E2xx codes #2191`) after the feature was ported, which is the most likely explanation for the drift. **This means a merge cannot simply take either side's `check.sio` wholesale for the shared feature — every affected `error-pattern:` assertion needs to be re-verified against whichever `check.sio` wins.**

This makes 6 overlapping `tests/compile-fail/*.sio` files in total once `refinement_subtype_strict.sio` is included (5 with error-code drift + 1 enrichment, see next paragraph) -- reconciling `check.sio`'s numbering.

`tests/compile-fail/refinement_subtype_strict.sio` also overlaps, but non-conflicting: `main` added a `requires: lean_single` engine-scoping annotation plus ~27 lines of forensic commentary explaining a genuine Madaros/lean_single behavioral difference; `integration`'s version is the earlier, shorter version of the same test. This is an enrichment, not a conflicting edit — main's version should likely win, but that is not this report's call.

### 3d. Docs / governance / datasets / CI / scripts (12 files) — mechanically tractable, editorially non-trivial

| File | `main` | `integration` | Note |
|---|---|---|---|
| `.github/workflows/ci.yml` | 93 commits, +729/‑33 | 4 commits, +29/‑1 | See §4 — main added ~25 gates integration lacks. |
| `docs/compiler/KNOWN_LIMITATIONS.md` | 63 commits, +234/‑589 (net shrink — stale claims retracted) | 3 commits, +11 (pure addition) | Both sides independently documented compiler status; will need editorial merge, not just textual. |
| `docs/governance/DOCS_AUTHORITY_MATRIX.md` | 100 commits, +205 | 2 commits, +34 | |
| `docs/governance/topic-registry.v1.json` | 105 commits, +8074/‑3430 | 2 commits, +786/‑4 | Structured JSON, large main-side churn; likely regenerated/derived rather than hand-edited — check for a generator script before manual reconciliation. |
| `datasets/sounio-code-examples/train.jsonl` | 27 commits, +1879/‑197 | 2 commits, +3/‑3 (trivial) | Low actual conflict; integration's touch is incidental. |
| `datasets/sounio-code-examples/validation.jsonl` | 17 commits, +260/‑73 | 2 commits, +2/‑2 (trivial) | Same. |
| `artifacts/seed-refresh/SeedReceipt.latest.json` + `.txt` (2 physical files, one row) | 33 commits each, pure additions | 1 commit each, pure additions | Mechanically regenerated receipts (see §4, bootstrap-seed status) — should be regenerated post-merge, not hand-merged. |
| `artifacts/self-hosted/madaros.gate-receipt` | 9 commits, +4/‑4 | 1 commit, +3/‑4 | Same — regenerate, don't hand-merge. |
| `scripts/ci/souc-native-wrapper.sh` | 2 commits, +20/‑1 | 1 commit, +18/‑1 | Small, plausibly compatible additive changes on both sides — verify no logic collision. |
| `scripts/dev/run_sio_test_suite_v2.sh` | 8 commits, +173/‑9 | 1 commit, +13/‑1 | Integration's touch is small relative to main's; low risk. |
| `scripts/gpu/run_native_cuda_smoke.sh` | 1 commit, +68/‑1 | 1 commit, +68/‑1 (**same line count**) | Worth a direct byte-diff check before assuming duplication — same size diff on both sides is suspicious and wasn't independently content-verified in this pass. |

### 3e. Stdlib epistemic files with divergent sizes on a shared feature (3 files)

| File | `main` | `integration` |
|---|---|---|
| `stdlib/epistemic/combine.sio` | 1 commit, +7/‑1 | 1 commit, +7/‑1 (same size, not verified byte-identical) |
| `stdlib/epistemic/graded_effects.sio` | 5 commits, +162/‑17 | 5 commits, +176/‑20 | Both sides did the same "witness of independence" feature work independently in parallel (commit messages overlap in wording but not hash), then integration additionally reconciled — see §3b for the matched commit-message evidence. The extra 14 lines/3 deletions on integration are most likely the manual-port adjustments, not a second independent design. |
| `stdlib/epistemic/invariants.sio` | 1 commit, +37/‑2 | 3 commits, +50/‑2 | Same pattern as above — integration has one extra reconciliation commit layered on the same underlying change. |

### 3f. Low-risk, already-reconciled test fixtures (11 files)

The remaining overlap files (`tests/run-pass/causal_graph_dsep.sio`, `graded_compose_needs_witness.sio`, `gum_independence_required.sio`, `indep_knowledge_grammar.sio`, `knowledge_layout_shadows_user_field_name.sio`, `gpu_param_names_read_mut.sio`, `gpu_param_names_readonly_pair.sio`, `gpu_readonly_multi_slice.sio`, `tests/gpu/epistemic_runtime/manifest.tsv`, `tests/madaros/source_to_elf/{knowledge_field_shadow_exit0.sio,manifest.tsv}`) were spot-checked and found **byte-identical** between the two tips (see §3b). These are genuinely safe.

---

## 4. Other structurally important findings

- **Compiler naming/version**: `main`'s `docs/compiler/KNOWN_LIMITATIONS.md` references resolved-status entries stamped `v0.99.0`; `integration`'s copy of the same file references `v0.66.0`. The two branches' compilers have advanced to different version numbers independently — any merge needs an explicit decision on which version string (and the underlying `main.sio` version constant) wins, consistent with the project's "Madaros" naming convention fixed on 2026-07-11 (per this repo's `CLAUDE.md` §4).
- **Binaries diverged independently** (§3a) — both `bin/madaros-linux-x86_64` and `bin/souc-lean-single-x86_64` have different final blobs on each tip and cannot be textually merged; a merge will need to rebuild from source and re-verify the fixed point (`make build`), not attempt to reconcile the ELFs themselves.
- **CI gates are not at parity**: `main` added roughly 25 additional named gates to `ci.yml` (backend-claim parity, PIREUS admission engine, measured-claims gate, ep_sqrt range gate, GUM-covariance gate, duplicate-definition gate, and their self-tests, among others) that do not exist on `integration`. `integration`'s CI changes in the same window are limited to a worldline-merge ABI gate and timeout bumps. **A merge that just takes one side's `ci.yml` will silently drop real coverage that has been catching regressions on `main` for a month** — this needs an explicit reconciliation pass, not a pick-one.
- **The `tests/compile-fail/*.sio` add/add cases are the SAME tests, not a naming collision** (confirmed in §3c) — they originate from the shared IndepKnowledge feature intentionally ported across both branches by the same author. The divergence between them is exclusively in the expected error code, tracking each branch's independent diagnostic-code numbering in `check.sio`. This is not "two different tests happen to share a filename" — it is one test whose expected-output assertion has drifted.
- **`scripts/gpu/run_native_cuda_smoke.sh` has identical diff line-counts on both sides** (+68/‑1) — this was *not* independently byte-verified in this pass (unlike the IndepKnowledge test files) and should be checked directly before assuming it is the same content; same-size diffs are suggestive but not conclusive of identical text.
- **`artifacts/seed-refresh/*` and `artifacts/self-hosted/madaros.gate-receipt` are generated receipts**, not hand-authored — per this repo's own tooling (`scripts/dev/souc-build-lock.sh`, seed-refresh scripts) these should be regenerated post-merge rather than manually merged; treating them as ordinary text-merge targets would produce a receipt that doesn't match either branch's actual rebuilt artifact.
- **`docs/governance/topic-registry.v1.json`** shows an unusually large main-side diff (+8074/‑3430 across 105 commits) — worth checking whether this file is machine-generated from source comments/registries before attempting any manual reconciliation; hand-editing a generated registry would be wasted and error-prone effort.
