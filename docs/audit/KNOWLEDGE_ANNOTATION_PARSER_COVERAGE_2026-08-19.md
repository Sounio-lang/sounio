<!-- docs:meta
topic_id: repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
-->

# `Knowledge<…>` annotation components — parser coverage against the AST enums

## Provenance of this document

Measured **blind** by `glm-cli2` on 2026-08-19: the lane was given the question
and explicitly not shown a parallel measurement by `claude-1`, so that agreement
would carry weight and disagreement would be a finding.

It was originally filed as `#2005` / `#2006`, which also carried +5,075,769 lines
of accidentally committed content — `uberon.owl`, `cl.owl`, ChEBI dumps, session
state, and edits to `CLAUDE.md`, `AGENTS.md`, `FOUNDER_INTENT.md`, `.gitignore`,
`.githooks/` and `ci.yml`. Those PRs were closed by founder decision. **This is
the measurement, without the freight.** No finding is altered.

## Findings

| family | declared | reachable from source | unreachable |
|---|---:|---:|---|
| `AstValidityKind` | 3 | 3 | 0 |
| `AstProvenanceKind` | 6 | 3 | 3 — `AstProvSource`, `AstProvLiterature`, `AstProvInput` |
| `KnowledgeConstraintKind` | 6 | 6 | 0 |
| `ValueKind` | 4 | 4 | 0 |

**The hole is specific to provenance.** Three other families in the same parser
are wired end to end, which rules out "the annotation parser is generally
unfinished".

- The three unreachable provenance cases have **no lexer words and no parser
  branches, and never had any** — `git log -S` over the parser files returns
  nothing. The gap is duplicated in `self-hosted/bootstrap/bootstrap_v0.sio`.
- Unrecognised components are **silently skipped**: identifiers are greedily
  eaten as epsilon bounds, and everything else reaches
  `} else { // Unknown component — skip }`.
- `check/epistemic.sio` `provenance_from_ast` has explicit branches for
  `Source`, `Literature` and `Input`, each with its own runtime constant. Their
  pipeline is therefore **dead at the front end only**: consumers exist,
  producers do not.

## Controls

- **Positive** — `ValidUntil` → `AstValidityKind::ValidUntilTime`, exercised by
  `tests/run-pass/covid_2020_kernel.sio`. Without it, a count of zero reachable
  cases would be indistinguishable from a broken instrument.
- **Negative** — declarations and consumer match arms excluded from the
  reachability count, so that a case which is merely *declared* and *matched*
  cannot be counted as reachable from source.

## The lane's verdict, and why it was superseded

The lane concluded **"the enum grew by anticipation, not parser lag"**, on the
grounds that all six entered in one commit with three branches and no versioned
`.sio` or doc ever used the missing three.

`claude-1` had concluded the opposite from commit history. Neither verdict
survived a third measurement that the disagreement provoked — a layer-by-layer
count (`docs/audit/PROVENANCE_LAYER_STAIRCASE_2026-08-19.md`), which found that
the three unreachable cases carry **runtime constants and consumer branches**. A
speculative enum does not acquire those. The reading is **lag**.

The lane's exclusion of consumer match arms was correct for the question it
asked, and is exactly what hid the evidence that settled it. Two correct
measurements of different things; the disagreement was the finding.

## Note on execution

Slurm jobs 10392/10394 were submitted per the standing directive, but the cluster
was degraded (jobs failing with signal 53). The measurement is static text
analysis over versioned files, performed as reads, so no pod compute was consumed
and the directive's purpose was met by other means. Recorded rather than omitted.

## Addendum 2026-08-23 — empirical Madaros discriminators; wrapper surface refined

Continuation of the parser-*surface* measurement. Still no parser invention: no new surface words, no diagnostic, `self-hosted/` untouched. This does **not** reopen the anticipation-vs-lag verdict of #2015 / `PROVENANCE_LAYER_STAIRCASE_2026-08-19.md`. Parser surface remaining 3-of-6 is compatible with lag at the checker/runtime layer (those three cases already have consumer arms and runtime constants).

Two clocks, labeled:

| Clock | What it is | What it is not |
|---|---|---|
| **STATIC** | Current `self-hosted/parser/{ast,types,parser}.sio` + `self-hosted/lexer/tables.sio` | A claim that the shipped ELF was built from this SHA |
| **DYNAMIC** | `env -u SOUC_BIN -u SOUNIO_SOUC_BIN ulimit -s 524288 ./bin/souc check` (Madaros v0.80.0 shipped ELF) | A from-source Madaros rebuild; not Slurm |

Re-run: `bash scripts/dev/knowledge_annotation_parser_coverage.sh`. Probes live in `docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19/` (not on the test-suite glob). Skip ELF checks with `SOUNIO_KCOV_SKIP_DYNAMIC=1`.

### STATIC — pin (unchanged 3-of-6, now including the live Madaros lexer table)

`AstProvenanceKind` still declares six cases; `types.sio` still constructs only `AstProvDerived` / `AstProvComputed` / `AstProvMeasured`. `self-hosted/lexer/tables.sio` (the Madaros keyword table) matches `parser.sio`: `Derived`, `Computed`, `Measured`, `Valid`, `ValidUntil`, `ValidWhile` are keywords; `Source`, `Literature`, `Input` are not. The original 2026-08-19 read of `parser.sio` alone was not the whole lexer.

`parse_knowledge_type` runs the comma-component loop **unconditionally** after the inner type — the comment says “only in bracket form”, but `Knowledge<f64, Derived>` is a live angle-form path. `parse_epistemic_wrapper_type` still gates that loop on brackets, and still has no `ValidUntil` / `ValidWhile` arms.

### DYNAMIC — discriminator matrix (shipped ELF, 2026-08-23)

Command for each probe: `ulimit -s 524288`; `env -u SOUC_BIN -u SOUNIO_SOUC_BIN SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check <probe>`.

| Probe | Type | ELF | Source interpretation |
|---|---|---|---|
| `derived.sio` | `Knowledge[f64, Derived]` | check OK | reachable provenance |
| `computed.sio` | `Knowledge[f64, Computed]` | check OK | reachable; first versioned use of this word |
| `measured.sio` | `Knowledge[f64, Measured]` | check OK | reachable; first versioned use of this word |
| `valid.sio` | `Knowledge[f64, Valid(1.0)]` | check OK | reachable validity |
| `validuntil.sio` | `Knowledge[f64, ValidUntil("2020-03-31")]` | check OK | already exercised by `covid_2020_kernel.sio` |
| `validwhile.sio` | `Knowledge[f64, ValidWhile(true)]` | check OK | reachable, still unused in product code |
| `knowledge_angle_derived.sio` | `Knowledge<f64, Derived>` | check OK | angle-form components are live for Knowledge |
| `source.sio` | `Knowledge[f64, Source]` | check OK | **silent** — not `AstProvSource` |
| `literature.sio` | `Knowledge[f64, Literature]` | check OK | **silent** |
| `input.sio` | `Knowledge[f64, Input]` | check OK | **silent** |
| `typo_ident.sio` | `Knowledge[f64, Sourc]` | check OK | misspelling is also silent |
| `source_eps.sio` | `Knowledge[f64, Source < 0.05]` | check OK | `Source` is Ident, so this is an epsilon bound |
| `derived_eps.sio` | `Knowledge[f64, Derived < 0.05]` | parse fail | `Derived` is a keyword; it cannot be an epsilon ident |
| `int_skip.sio` | `Knowledge[f64, 123]` | check OK | non-ident unknown component is skipped |

The pair `source_eps` (OK) vs `derived_eps` (parse fail) is the empirical discriminator for the Ident-epsilon sink. It does **not** prove which AST field `Source` occupies; it proves `Source` is not the `Derived`/`Computed`/`Measured` keyword class. Combined with the lexer having no `Source` word, the silent-OK probes are Ident capture or skip, not `AstProvSource`.

### Wrapper surface — do not treat the 3-of-6 copy as independently exercised

Under the same ELF:

| Probe | ELF |
|---|---|
| `Intervention<f64>` | check OK |
| `Intervention<f64, Derived>` | check OK |
| `Intervention[f64]` | parse fail (8 parser errors) |
| `Validated[f64, Derived]` | parse fail (13 parser errors) |
| `Validated<f64>` | check OK |

Current source sends `Intervention` / `Counterfactual` through `parse_epistemic_wrapper_type` (bracket component loop) and `Validated` through that loop only when the next token is `[`. Empirically the **bracket** form does not round-trip on this ELF, so the wrapper’s provenance arms are not a live Madaros surface in this measurement. `Intervention<f64, Derived>` checking OK is **not** evidence that `Derived` became `AstProvDerived` on an Intervention: the angle path in current source expects `>` immediately after the inner type, and the shipped ELF has no `--show-ast`. Possible readings (unresolved here): generic type-args (`knowledge_info: None`), ELF/source drift, or recovery. Do not collapse them.

Wrapper probes are **not** in the census pin for that reason. Closing that question needs a source-built Madaros or an AST dump.

### What this addendum does not do

- Does not add `Source` / `Literature` / `Input` keywords or parser arms.
- Does not add a diagnostic for the silent skip / Ident-epsilon sink (that remains the honesty gap).
- Does not promote the silent-OK probes into `tests/run-pass/` — they are receipts of current behaviour, not features. The day `source.sio` starts failing check, the census fails and the gap has moved.
- Does not claim the shipped ELF matches this checkout’s `self-hosted/` bit-for-bit.
- Does not reopen #2015's lag verdict.

## Addendum 2026-08-23 (E241) — unknown components refuse

The silent Ident-epsilon / skip sinks are closed in `self-hosted/parser/types.sio`. Still no `Source` / `Literature` / `Input` keywords.

| Input | Before (shipped ELF) | After (source, E241) |
|---|---|---|
| `Knowledge[f64, Source]` | check OK (Ident → default CmpLt 0.0) | `error[E241]` |
| `Knowledge[f64, Sourc]` | check OK | `error[E241]` |
| `Knowledge[f64, 123]` | check OK (skip) | `error[E241]` |
| `Knowledge[f64, Source < 0.05]` | check OK (epsilon bound) | check OK (unchanged) |
| `Knowledge[f64, Derived]` | check OK | check OK (unchanged) |

Clocks:

- **STATIC** (Contracts): `bash scripts/dev/knowledge_annotation_parser_coverage.sh` — 3-of-6 still holds; E241 helper present; skip comment gone.
- **DYNAMIC / source-built** (Madaros Witness): `bash scripts/ci/knowledge_unknown_component_live_refuse.sh` against `MADAROS_RAW_BIN`. The committed ELF is expected to still swallow these until it is rebuilt.

`tests/compile-fail/knowledge_unknown_component_{ident,int}.sio` are `//@ requires: madaros` so the default suite does not score them against the stale ELF.

## Addendum 2026-08-27 (Input) — one of the three unreachable cases is now reachable

`Input` left the unreachable set. Under the founder ruling of 2026-08-19
(`asserted → Input`) the lexer gained `TokenKind::Input` and both Knowledge
component loops in `self-hosted/parser/types.sio` gained an arm constructing
`AstProvenanceKind::AstProvInput` (PR #2062). `Source` and `Literature` are
unchanged: still declared, still not keywords, still with no construction site.

The census line therefore moves **3-of-6 → 4-of-6**, and
`scripts/dev/knowledge_annotation_parser_coverage.sh` now pins
`constructed=4 unreachable=Source,Literature`. That pin was re-pointed, not
removed: it still enumerates exactly which provenance words the parser may
construct, so a fourth one cannot appear without a human reading the diff.

The 2026-08-23 rows above stay as written — they were measured before the
keyword existed. What they no longer describe is `Input` specifically:

| Input | 2026-08-23 (E241, no keyword) | 2026-08-27 (E241 + Input keyword) |
|---|---|---|
| `Knowledge[f64, Input]` | `error[E241]` (bare Ident, no comparison) | parses, provenance = `AstProvInput` |
| `Knowledge<f64, Input>` | parse error (`Input` was `Ident`, `>` eaten as `CmpGt`) | parses |
| `Knowledge[f64, Source]` | `error[E241]` | `error[E241]` (unchanged) |

## Addendum 2026-09-01 — angle-form probes exist; source-built sister pin closes the angle-form half

The 2026-08-23 "angle form is loud, bracket form is silent" addendum above
referenced three probe files (`angle_source.sio`, `angle_literature.sio`,
`angle_input.sio`) that **did not exist on disk at the time** — the addendum
was a forward-pointing receipt of behaviour, not a record of files. This
addendum grounds that claim: the three probe files are now in
`docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19/`, and the
source-built sister pin exercises them.

### Files added (this branch)

| Probe | Form | Empirical outcome, current main |
|---|---|---|
| `angle_source.sio` | `Knowledge<f64, Source>` | **parse fail** — Source is an Ident; the angle-form component loop refuses it before the bracket-form epsilon sink can swallow it. |
| `angle_literature.sio` | `Knowledge<f64, Literature>` | **parse fail** — same path as Source. |
| `angle_input.sio` | `Knowledge<f64, Input>` | **check OK** — `Input` is a lexer keyword (PR #2229, commit 1adec5e731); the angle-form path resolves it to `AstProvInput` via the construction arm added by the same ruling. |
| `knowledge_angle_derived.sio` | `Knowledge<f64, Derived>` | **check OK** — pre-existing; `Derived` is a keyword since before the audit. |

### Source-built cross-check matrix (current main, 2026-09-01)

A fresh source-built Madaros was built against current main
(`bash scripts/ci/build_modular_madaros.sh /tmp/madaros-source-built-main`,
103192562-byte ELF, 2026-09-01). Cross-checked against the shipped
`./bin/souc` (which was rebuilt from a newer tree than
`/tmp/madaros-source-built`):

```
angle_source              shipped=parse failed   source-built-main=parse failed   AGREE
angle_literature          shipped=parse failed   source-built-main=parse failed   AGREE
angle_input               shipped=check: OK      source-built-main=check: OK      AGREE
knowledge_angle_derived   shipped=check: OK      source-built-main=check: OK      AGREE
intervention_angle_derived   shipped=check: OK   source-built-main=check: OK      AGREE
intervention_bracket_only    shipped=parse failed source-built-main=parse failed AGREE
intervention_bracket_derived shipped=parse failed source-built-main=parse failed AGREE
validated_bracket_derived    shipped=parse failed source-built-main=parse failed AGREE
```

Both ELFs agree on every angle-form Knowledge probe, on every wrapper probe,
and on the tripwire the previous addendum left open. The shipped ELF being
newer than the audit's `2026-08-23` source-built does not change the
agreement: the question is whether source-current and shipped-current agree,
and they do.

### Tripwire behaviour

`scripts/dev/knowledge_annotation_parser_coverage_source_built.sh` now
covers two halves:

1. The wrapper bracket closure (PR #2108) — `intervention_bracket_only`,
   `intervention_bracket_derived`, `validated_bracket_derived` must
   parse-fail; `intervention_angle_derived` must check OK.
2. The angle-form Knowledge closure (this addendum) —
   `knowledge_angle_derived` and `angle_input` must check OK;
   `angle_source` and `angle_literature` must parse-fail.

Verified against the *older* `/tmp/madaros-source-built` (built 2026-08-23,
before PR #2229), the pin **fails** on `angle_input` with the expected
diagnostic (`expected check OK, got parse failed`) — the tripwire is live.
Rebuilt against current main, the pin **passes**.

The source-built pin does not retest bracket-form Knowledge probes
(`source.sio`, `literature.sio`, `int_skip.sio`, `typo_ident.sio`,
`derived_eps.sio`). Those are exercised by the shipped pin's dynamic mode
and the Madaros Witness Gate; including them here would duplicate work
without adding a cross-check the shipped pin already performs.

### Why the previous addendum's `angle_input = parse fail` row is now stale

The 2026-08-23 addendum was measured before PR #2229 (commit 1adec5e731)
made `Input` a lexer keyword. The row at the time was correct. The
behaviour moved under #2229; the row did not, until the 2026-08-27
"Input" addendum above corrected the bracket-form case but left the
angle-form row untouched. This addendum is the matching correction for
the angle-form case, plus the receipts (probe files + source-built
cross-check).
