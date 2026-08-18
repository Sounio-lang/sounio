<!-- docs:meta
topic_id: repo.docs.audit.gum-fo-propagation-hidden-by-main-vacuous-harness-dispatch-2026-08-10
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gum-fo-propagation-hidden-by-main-vacuous-harness-dispatch-2026-08-10
-->

# Dispatch — first-order GUM propagation bugs, pre-existing on `main`, masked by `main`'s own vacuous-match harness bug

**Filed:** 2026-08-10 · **Status:** OPEN (dispatch; not fixed) · **Protocol:** CLAUDE.md §8 (`self-hosted/` and epistemic-checker findings require a forensic dispatch before code) and §10 (math/GUM claims require `bin/llm-offload -t math-review` before a fix lands — not performed here, since no fix is proposed).

## Summary

Merging `main` into `claude/refine-local-plan-sgxlte` (PR #1531) surfaced
**14 `run-pass` tests that genuinely compute wrong first-order GUM
(`Knowledge<T>`) variance values, on both shipped engines** — this is not a
vacuous-annotation artefact, and not an engine-availability gap. All 14 were
"passing" on `main`'s own CI before this merge, for the same reason the
original 36 this PR resolved were: `main` still carries the vacuous
`expect-stdout`/`error-pattern` extraction bug (`git merge-base
--is-ancestor 4d8f6ce2 origin/main` fails), so the assertions never fired.

This dispatch went through three rounds of correction before landing on the
above, each one caught by measuring the next claim rather than trusting the
previous one. The full trail is kept below rather than compressed away,
because the mechanism that caused each error — testing against the wrong
compiler artefact — is itself the reusable lesson.

## Evidence

**14 of the (eventually) 16 files CI flagged trace to one commit.**
`git log --follow --diff-filter=A` on each:

```
15445025 research: FO second-order — PK driver gates, GUM trust, residual stack
```

introduced `madaros_gum_fo_deep_field`, `_eight_param`, `_field_call`,
`_if_helper`, `_import`, `_impure_ctor`, `_knowledge_ops`, `_let_bytecode`,
`_let_ctor`, `_mutual`, `_mutual_deep`, `_nested_field`, `_nonpure_ctor`,
`_struct_field` (all `tests/run-pass/`). The commit message ("research: ...
residual stack") and several files' own header comments ("residual was FO 0",
"freezes Call paths") read as an in-progress research batch, not a finished,
audited feature.

**Confirmed failing on both engines, against a genuinely fresh build —
not a stale one.** Full Test Suite under `souc-stage2` (CI's actual
lean_single engine — confirmed by reading the build step itself:
`scripts/ci/selfhost_host_gate.sh` produces `souc-stage2` by compiling
`self-hosted/compiler/lean_single.sio`, not Madaros) reported these 14
failing. A first pass in this dispatch tested the Madaros side against a
locally prebuilt `bin/souc` — which turned out to be stale, built before this
session's merge landed the many `fix(madaros)` commits now on `main`, an
instance of the exact pitfall CLAUDE.md's operating principle 15 already
warns about. Rebuilt Madaros from current source (`make build-madaros`,
99,224,615-byte ELF, `self-hosted/compiler/main.sio` at 10,773 functions) and
retested all 14 against it:

```
madaros_gum_fo_deep_field    v_chain=0 v_call=0 v_call_flat=0 v_id=0 ...   FAIL
madaros_gum_fo_eight_param   v5p=0 v8s=0 vw=0                             FAIL
madaros_gum_fo_import        v_imp_css=0 v_peel_css=0                     FAIL
madaros_gum_fo_mutual        v_me=0 v_mo=0 v_top=0                        FAIL
madaros_gum_fo_field_call, _if_helper, _impure_ctor, _let_bytecode,
_let_ctor, _mutual_deep, _nested_field, _nonpure_ctor, _struct_field        FAIL
madaros_gum_fo_knowledge_ops                              SEGFAULT (bin/madaros:634)
```

All 14 fail on both engines against verified-fresh builds. One
(`_knowledge_ops`) is worse under Madaros than under lean_single — a crash,
not a wrong value — which if anything strengthens rather than weakens the
"genuinely broken, not an engine gap" reading.

## Two files are a real engine divergence, not part of the 14

`gum_correlated.sio` (introduced independently by `9df5ab6b`, #1403 — predates
and is unrelated to the `15445025` research batch) documents in its own
header:

```
// GUM §5.2: correlated inputs — same variable on both sides of an operator.
// var(h+h) = 4·var(h) for Y=2h (FO ∂Y/∂h=2), NOT 2·var(h) (independent).
```

Under `lean_single`: `var(h+h)=0.000200` — the wrong, *independent*-inputs
value the comment names as wrong. Under a fresh from-source Madaros:
`var(h+h)=0.000400` — exactly correct. This is a genuinely implemented
Madaros feature absent from the frozen `lean_single` seed (which CLAUDE.md
documents as deliberately not touched — `docs/compiler/KNOWN_LIMITATIONS.md`),
the same category as the `E108`/`E120`/`E121` tests fixed earlier in this PR.
Tolerated in `tests/vacuous_expect_baseline.txt` (kept there rather than
converted to `//@ requires: madaros`, matching the disposition already landed
on this branch for it).

Four further files already sat in this branch's baseline before this
investigation started — `madaros_gum_fo_deep_poly.sio`, `_div_if.sio`,
`_interproc.sio`, `madaros_gum_multichannel_fo.sio`, each introduced by an
earlier `fix(madaros):` commit dated 2026-07-25/26 (channel-expansion,
quotient-rule, inter-procedural, and multi-channel FO work respectively).
Verified against the same fresh Madaros build: all four now compute the
correct, matching values and print their `_PASS` marker — the same
engine-divergence category as `gum_correlated.sio`. Left in the baseline
as-is; not re-litigated here.

## A misclassification caught and corrected during this same investigation

An earlier revision of the baseline file (commit `70d344db` on this branch,
landed by a concurrent session while this dispatch was being written) also
placed `tests/compile-fail/ontology_property_weakening.sio` alongside
`gum_correlated.sio` as a second instance of engine divergence. That was
wrong: under `lean_single`, the checker emits `error[E160] ... ontology
subsumption could not be verified: child property constraint weakens
parent`, which *contains* the file's pinned `//@ error-pattern: ontology
subsumption could not be verified`. Confirmed three independent ways — the
raw `souc check` output contains the substring; the harness itself
(`--filter-exact ontology_property_weakening.sio --format junit`, matching
CI's own invocation) reports `Pass: 1, Fail: 0`; and the harness's own
stale-entry detector (built earlier in this same PR specifically to catch
this class of drift) flagged it under "Vacuous-annotation baseline entries
that passed in THIS run." Removed from the baseline rather than left as a
harmless extra tolerance, since a passing test parked in a file whose whole
purpose is "annotation vs. reality" drift detection is exactly the failure
mode that file exists to prevent.

## What this dispatch does not do

**No fix is proposed or attempted here.** This is `Knowledge<T>`/GUM
uncertainty-propagation correctness — CLAUDE.md §10 makes
`bin/llm-offload -t math-review -p xai` mandatory before any such change
lands, and operating principle 6 requires tightening the implementation or
reporting `FAIL_HONEST` rather than adjusting a tolerance to admit the
failure. Given the 14 originate from a commit explicitly labeled `research:`
and evidently still in progress, this reads as active, unfinished work rather
than a regression this PR introduced or should absorb.

## Disposition for CI

The 14 genuinely-broken files are baselined in
`tests/vacuous_expect_baseline.txt` (not
`tests/known_failures/hardened_diagnostics_full_suite.txt` — an earlier draft
of this dispatch used that file instead, before the reconciliation above;
the vacuous baseline is the mechanism actually landed on this branch and is
adequate, since it is always active rather than gated to `--format junit`).
`gum_correlated.sio` and the four earlier `madaros_gum_fo_*`/
`madaros_gum_multichannel_fo` files are baselined too, as genuine engine
divergence rather than defects.

**Consequence for `main`.** `main`'s own CI will not see any of this until
`main` gets the vacuous-match harness fix independently (a separate PR, not
filed here). The 14 broken tests will keep silently "passing" there until it
does. Worth flagging to whoever owns landing the harness fix on `main`
directly, since that is the moment `main`'s CI turns red on this same
finding.

## Recommendation

1. Do not fix the GUM propagation gap as part of #1531 or this dispatch.
2. Route to the owner of the `research: FO second-order` work (commit
   `15445025`) and to `bin/llm-offload -t math-review` before any fix.
3. When landing the harness fix on `main` independently, expect these same
   14 failures (plus the 5 engine-divergence files, if `main`'s CI is ever
   pointed at Madaros for full-test-suite) to appear there and link them to
   this dispatch rather than re-diagnosing from scratch.
4. Consider converting the 5 engine-divergence entries
   (`gum_correlated.sio` + the four `fix(madaros):`-introduced files) to
   `//@ requires: madaros` instead of baseline entries, for consistency with
   the `E108`/`E120`/`E121` precedent elsewhere in this PR — not done here to
   avoid re-litigating a stylistic choice already landed by a concurrent
   session; a candidate for a small follow-up, not a defect.
5. Remove a `known_failures`/baseline entry only after confirming under the
   CI-calibrated engine specifically — three separate measurements in this
   dispatch's own history were wrong because they used Madaros, a stale
   prebuilt artifact, or (for `ontology_property_weakening.sio`) a
   transcription slip, instead of the harness's own CI-equivalent
   verification path.
