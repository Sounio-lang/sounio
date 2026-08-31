<!-- docs:meta
topic_id: repo.docs.audit.known-failure-needs-an-engine-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.known-failure-needs-an-engine-2026-08-19
-->

# A known-failure without an engine is a numeral without an engine

> **Status**: measurement receipt | **Last validated**: 2026-08-19 | **Source**: live `souc compile` / `souc run` of the 19 XFAIL-lean / XPAS-Madaros files

The harness can say `requires: madaros`. It cannot say "this known-failure is about the seed". That is the same class as a published number with no engine: the tag looks like a claim about *today's compiler* and is quietly about a different one.

#1930 is a separate instrument question. This file does not touch it.

## Population

The 19 suite-visible `//@ known-failure` files that were XFAIL on lean_single and XPAS on default Madaros in the 2026-08-18 census (N=47, `docs/audit/KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md` on #1910). Criterion written first at 2026-08-19T00:38:14Z, before this re-run.

Instrument: `bin/souc` default → `artifacts/self-hosted/madaros` (2026-08-17 17:01), not an E230 source-build. lean via `SOUNIO_SOUC_ENGINE=lean_single`. `souc compile` / `souc run` only. `hello` rc=0 on both before the 19.

## Question

Does Madaros XPAS because the test's contract holds, or because something else rejects first?

| Class | Meaning |
|---|---|
| CONTRACT | compile-fail: pinned pattern present. Not vacuous. |
| EARLY | compiler failed, pinned pattern absent |
| REAL_RUN | run-pass rc=0 and the test's own assertion held |
| SEED_CHECK_FAIL | lean never reached runtime |

## Verdict — 17 + 1 + 1

**None of the 17 compile-fails are EARLY on Madaros.**

| n | Files | Madaros | lean |
|--:|---|---|---|
| 16 | every `error-pattern: error[E218]` f128/f256 reservation in the 19 | `compile` rc=1, `error[E218]` *f128/f256 is reserved… V0-A* | 15× `compile` rc=0 (accepted, no E218). 1× (`f128_v0b_implicit_conversion_rejected`) rc=1 `tail type mismatch`, still no E218 |
| 1 | `turbofish_type_arg_arity` | rc=1, `wrong number of arguments` / E010 | rc=0 — accepts `first::<i64>(42, 99)` |
| 1 | `uninit_fixed_array_zero_init` | **REAL_RUN** rc=0: zero-init assertions held | SEED_CHECK_FAIL: `unknown identifier s` / E200 |
| 1 | `vec_new_nonexistent_type_eval_zero` | **REAL_RUN** rc=0: prints `0` and `DOCUMENTS_FABRICATION` | SEED_CHECK_FAIL: `unknown identifier Vec` |

The 16 E218 tags already say `lean_single-only gap`. The Madaros XPAS is the reservation doing its job, not a reject-before-the-interesting-part. "Madaros already rejects" is true **and** it rejects for the reason the test names. That is **noise** (tag about the seed, running on the oracle), not **debt** (a hidden live pass of a run-pass).

The two run-pass files really execute on Madaros. They are not the same as each other:

- `uninit_fixed_array_zero_init` — the #1548 contract holds. The tag text is about a stale prebuilt Madaros. **Debt.** Drop the tag on the oracle, or scope it to the seed. The in-file comment that "lean_single zero-initializes" is false on this seed: lean never typechecks the file.
- `vec_new_nonexistent_type_eval_zero` — Madaros still fabricates `0` (W44 never fires). The pass *is* the census. Not a fixed bug wearing a known-failure. Lean does not know `Vec`.

## `requires: lean_single` does not exist

`scripts/dev/run_sio_test_suite_v2.sh` accepts `requires: gpu|llvm|madaros` only. Any other token is a hard fail, not a skip. No `requires: lean_single`, `requires: lean`, or `requires: seed` anywhere in the tree.

It is also the **wrong** tool for the 17. Those tags already say "keep as a Madaros regression guard until fixed-point rebootstrap". `requires: lean_single` would skip them on Madaros and drop the E218/E010 guard. `requires:` answers *may this test run*. The missing annotation answers *on which engine is this failure claimed*.

## Proposal (not landed)

```
//@ known-failure-on: lean_single
```

| Engine | Meaning |
|---|---|
| named engine | today's XFAIL / XPAS rules |
| any other engine | evaluate the test as if the known-failure line were absent |

That is the dual of `requires: madaros`. A known-failure with no engine is the same honesty class as a numeral with no engine: the panel lesson applied to the harness.

Do not implement in this lane. Do not retag the 19 here.

## Not done

- No harness edit.
- No tag edit.
- No #1930 follow-on.
- CAP / token table / handle table / E230 not touched.
