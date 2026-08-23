<!-- docs:meta
topic_id: repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.knowledge-annotation-parser-coverage-2026-08-19
-->

# Audit: `Knowledge<...>` annotation components — parser reachability vs. enum declaration

Date: 2026-08-19 · Scope: measurement only, `self-hosted/` untouched · Method: static reading of `self-hosted/parser/{ast,types,parser}.sio`, `self-hosted/check/epistemic.sio`, `self-hosted/bootstrap/bootstrap_v0.sio`, plus repo-wide grep and git history.

## Question

The parser reads `Knowledge<...>` annotation components from surface words and translates them into cases of enums declared in `self-hosted/parser/ast.sio`. For each component family: how many cases are declared, how many can the parser actually produce, what happens to the rest?

## Measurement

### Family 1 — `AstValidityKind` (ast.sio:441-445)

| Case | Declared | Parser branch | Surface word |
|---|---|---|---|
| `ValidDuration` | yes | types.sio:1079 | `Valid` |
| `ValidUntilTime` | yes | types.sio:1090 | `ValidUntil` |
| `ValidWhileCond` | yes | types.sio:1101 | `ValidWhile` |

**Declared: 3 · Reachable: 3 · Unreachable: 0.** All three words are lexed (`parser.sio:830-831`; `Valid` is a base word).

### Family 2 — `AstProvenanceKind` (ast.sio:452-459)

| Case | Declared | Parser branch | Surface word |
|---|---|---|---|
| `AstProvDerived` | yes | types.sio:1112 | `Derived` |
| `AstProvSource` | yes | **none** | **no such word** |
| `AstProvComputed` | yes | types.sio:1115 | `Computed` |
| `AstProvLiterature` | yes | **none** | **no such word** |
| `AstProvMeasured` | yes | types.sio:1118 | `Measured` |
| `AstProvInput` | yes | **none** | **no such word** |

**Declared: 6 · Reachable: 3 · Unreachable: 3** (`AstProvSource`, `AstProvLiterature`, `AstProvInput`). Verified by repo-wide grep: zero construction sites for these three anywhere under `self-hosted/parser/`; the lexer table (`parser.sio`, 302 return sites) contains no `Source`/`Literature`/`Input` word at all. The same 3-of-6 pattern is duplicated in `parse_epistemic_wrapper_type` (types.sio:1348-1393; used for `Intervention`/`Counterfactual`/`Validated` wrappers) and mirrored in the bootstrap seed (`bootstrap_v0.sio:2544-2549`, branches at 3924-4000) — the gap was copied forward, never closed.

### Family 3 — `KnowledgeConstraintKind` (ast.sio:461-468) and `KnowledgeConstraintValueKind` (ast.sio:470-475), `where {...}` constraints

All 6 constraint kinds (`SubclassOf, Eq, Ge, Gt, Le, Lt`) and all 4 value kinds (`None, Name, Int, Float`) are constructed in `parse_knowledge_constraint` (types.sio:879-950). **Declared: 6+4 · Reachable: 6+4 · Unreachable: 0.**

### Q4 — Unrecognized component

Silent consumption, no diagnostic. Two sinks:
1. Any `Ident` in component position is greedily eaten as an epsilon identifier ("Assume epsilon identifier", types.sio:1009-1010) — it never reaches the fallback.
2. Anything else hits `// Unknown component — skip` (types.sio:1119-1121): one `advance()`, no error, no warning. Same in the wrapper path (types.sio:1391-1392).

Consequence: a misspelled provenance word is invisible — it either becomes a bogus epsilon bound or vanishes.

### Q5 — Versioned `.sio` files using the annotations

- `tests/run-pass/covid_2020_kernel.sio:5-19` — `Knowledge[f64, ValidUntil("2020-03-31"), Derived]` (exercises `ValidUntilTime` + `AstProvDerived`).
- `tests/compile-fail/covid_2020_temporal_expiration.sio`, `covid_2020_temporal_alias_refusal.sio` — same two components.
- `self-hosted/test_knowledge.sio` T03/T06 — lexes `Derived`; builds the full `Knowledge[f64, eps < 0.05, Valid(...), Derived]` AST by hand (`AstProvDerived`).
- **No** versioned `.sio` file uses `Source`, `Literature`, `Input`, or `ValidWhile` annotations (grep across `tests/`, `examples/`, `ecosystem/`, `demo/`: zero hits). Bulk usage (`ecosystem/shared/epistemic_types.sio`, med/ PBPK tests) is bare `Knowledge[T]` or `where {...}` constraints.

## Controls

- **Positive** — a case the parser reaches, with branch and surface word: `ValidUntil` -> `AstValidityKind::ValidUntilTime` (types.sio:1088-1090), exercised end-to-end by `tests/run-pass/covid_2020_kernel.sio:5`. Also `Derived` -> `AstProvDerived` (types.sio:1110-1112).
- **Negative** — the three unreachable cases appear only at (a) their declaration (ast.sio:453,455,457), (b) the bootstrap enum mirror (bootstrap_v0.sio:2545,2547,2549), (c) checker match arms (check/epistemic.sio:250,256,262). Declarations and consumer matches are excluded from the reachable count; only word-driven construction sites in `self-hosted/parser/` count, and there are none.

## Downstream state

`check/epistemic.sio:248-268` (`provenance_from_ast`) matches **all six** cases and maps them onto runtime constants `PROVENANCE_KIND_SOURCE/LITERATURE/INPUT` (epistemic.sio:217-222); `provenance_is_external` (339-344) gives them semantics. But `provenance_new(PROVENANCE_KIND_SOURCE|LITERATURE|INPUT)` is called **only** from those three match arms — since the parser cannot produce the AST cases, the entire Source/Literature/Input pipeline, surface-to-runtime, is dead code.

## Git history

- All six provenance cases entered in **one commit**, `1dabedcbb2` (2026-02-13), alongside exactly the three parser branches that exist today. The other three never had branches: `git log -S ProvSource -- self-hosted/parser/types.sio` and `-S AstProvSource -- self-hosted/parser/types.sio` return **nothing** — never added, never removed.
- `02fd81876c` (2026-02-26, "[check,epistemic] Expand epistemic type system: provenance, validity, epsilon propagation, confidence subsumption") wired the checker for all six — touching only `check/epistemic.sio`, adding no surface words.
- `f9da2142f4` renamed `Prov*` -> `AstProv*` (pure rename, no case added/dropped).
- No TODO/FIXME, doc, or test mentions `Source`/`Literature`/`Input` words anywhere.

## Verdict

**The enum grew by anticipation; the surplus cases are cases nobody has yet wanted at the surface.** Evidence:

1. Birth pattern: the three unreachable cases were seeded in the enum creation commit together with only three branches — there was never a surface word for them, not even transiently. A parser "running behind" would show words planned, removed branches, or follow-up tickets; none exists.
2. Zero demand: no versioned `.sio` source, test, example, or doc uses or asks for `Source`/`Literature`/`Input`/`ValidWhile` components.
3. The only counter-signal — the checker wiring all six (02fd81876c) — is itself mechanical mirroring of the enum, and its runtime targets (`PROVENANCE_KIND_SOURCE/LITERATURE/INPUT`) have no other producer in the tree, confirming the pipeline is dead end-to-end rather than pending.
4. The silent-skip fallback shows no closing pressure was ever built into the surface.

What would flip or refine this: a design doc / issue specifying `Source`, `Literature`, `Input` as intended surface words (would indicate genuine lag), or a runtime path producing those provenance kinds from outside the AST (would indicate the AST cases are the redundant side). Neither exists in-tree today.

## Note on execution venue

Directive asked for runs via Slurm. Jobs 10392/10394 were submitted; the cluster is degraded (queue full of `launch failed requeued held` jobs, job FAILED with signal 53, `sacct` unreachable). The measurement is pure static text analysis over versioned files (no build, no execution of project code); the identical commands were run as reads on the pod. No project state was modified.

## Addendum 2026-08-23 — empirical Madaros discriminators; wrapper surface refined

Continuation of the same audit. Still no parser invention: no new surface words, no diagnostic, `self-hosted/` untouched. Two clocks, labeled:

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
