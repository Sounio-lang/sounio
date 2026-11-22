<!-- docs:meta
topic_id: repo.docs.spec.2026-10.10-research-scope-and-upstream-practice
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.10-research-scope-and-upstream-practice
-->

# Research scope and upstream reference practice

This edition was prepared by parallel source investigations and central
integration. Its purpose is a language reference that an implementer or
scientific reviewer can interrogate at the level of rules, representations,
call paths, witnesses, and unresolved decisions.

## 1. Repository coverage

`RESEARCH-001` — The research baseline is the commit and Git tree in
`edition.json`. A source observation is tied to that baseline. The active
remote development workspace, an older local worktree, and a later installed
artifact can contain different implementations; they cannot silently supply
facts about this edition.

The [coverage map](repository-coverage.tsv) assigns every tracked root entry
at the baseline and every immediate subdirectory of `self-hosted/`, `stdlib/`,
`tools/`, and `ecosystem/` to a primary reference chapter. The validator checks
that census against Git. These assignments identify the relevant semantic
and implementation interface. They do not claim that every file was read
line by line or that every library operation has a complete specification.

The deep source investigations concentrate on the production lexer/parser,
checker and effect/ownership paths, scientific lowering, numerical and
target emitters, resolvers, representative libraries, relevant formal
models, and the procedures that test those interfaces. Alternate implementations,
research code, generated data, papers, examples, and infrastructure are
classified according to their role and evidence boundary.

`RESEARCH-002` — An inspected directory is not a support claim. Historical
reports, static inventories, prototype modules, and generated receipts retain
their provenance. An API or language rule needs its actual call path and
corresponding witness. Each chapter's evidence table exposes the references
that carry its source observations and the remaining coverage gaps.

## 2. Independent research fronts

| Front | Main source questions |
|---|---|
| Lexical/syntax/modules | Which lexer/parser route is active? What bytes/tokens, contextual forms, precedence and newline predicates are accepted? How is the source graph resolved? |
| Types/ownership | Which type relations are actually used? How do expected types, calls, generics, trait obligations, loans, linear usage, and refinements reach decisions? |
| Effects/concurrency | Which names, masks/rows, function effects, mutation checks, handlers, and asynchronous operations have a reachable checked/runtime path? |
| Epistemic semantics | How do compiler scientific values, library records, variance/sensitivity channels, correlation, epsilon/confidence predicates, validity, and provenance interact? |
| Numerics/units/algebra | Which representations and operations are emitted, where are formats narrowed/refused, how do units and ordered algebraic operations behave? |
| Execution/targets | What does the public driver select and emit? Which memory, ABI, region, FFI, target, and GPU contracts are observable? |
| Central library/formal audit | Which APIs and tools constitute local package/support contracts? What do ontology, quantum, and Lean evidence establish? |
| Conformance integration | What does each runner actually execute and assert? Which source, target, artifact, and proof receipts would close the rules? |

The chapter investigations are followed by reviews across those boundaries.
For example, a parser rule for a binary operator must agree with effect and
execution descriptions; a confidence rule must not be borrowed from a
different library record; and an ABI observation must not imply a general
ownership theorem.

## 3. Upstream sources consulted

The following primary upstream references were consulted on 2026-10-16 for
specification organization and engineering practice. Their language rules
are not imported into Sounio.

| Reference | Practice used in this edition |
|---|---|
| [C++ working draft source](https://github.com/cplusplus/draft) | Distinguish language requirements, implementation behavior, diagnostics, and observable execution; review source and rendered changes. |
| [Rust Reference](https://doc.rust-lang.org/reference/) | Stable rule identifiers, topical chapters, grammar conventions, explicit feature status, and links between rules and tests. |
| [F# language specification](https://fsharp.org/specs/language-spec/) | Detailed treatment of lexical grammar, types, inference, expressions, and patterns; versioned design material. |
| [Koka reference](https://koka-lang.github.io/koka/doc/book.html) | Precise attention to effects, their typing, handlers, and implementation mechanisms. |
| [LLVM Language Reference](https://llvm.org/docs/LangRef.html) | Separate type/operation semantics, representation, target interfaces, and behavioral constraints. |
| [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) | Explicit separation of structural syntax, operation interfaces, verification, and extensible dialect boundaries. |

`RESEARCH-003` — Comparisons with upstream references establish editorial
and technical methods. They do not establish semantic equivalence,
performance superiority, standards-body acceptance, or independent validation
of Sounio. Claims about Sounio remain attached to its own contracts and
evidence.

## 4. Reproducible assets

| Asset | What it records |
|---|---|
| `edition.json` | Source identity, edition status, chapter/catalog paths, and scope. |
| `core-grammar.ebnf` | An explicit recognition profile and the contextual conditions needed beside the grammar. |
| `00-evidence.tsv` through `10-evidence.tsv` | Stable rules, standing, primary source/symbol, positive/negative witnesses, and gaps. |
| `repository-coverage.tsv` | Baseline repository census and primary chapter assignments. |
| `scripts/docs/validate_language_spec.mjs` | Structural validation of the edition, local links, rule catalogs, grammar, and repository census. |
| [Independent review receipt](17-preparation-and-review-receipt.md) | Findings, adjudications, provider outcomes, and the remaining validation boundaries. |
| [Bounded execution receipt](15-bounded-conformance-receipt.md) | Sixteen existing manifest cases executed in Slurm against the baseline's committed Linux Madaros binary; not a source rebuild or full conformance result. |

`RESEARCH-004` — A structural check is reported as a structural check.
Compiler execution, numerical verification, formal proof checking, remote
CI, review, and integration each have separate acceptance criteria. The
edition's preparation record must say which of those actually ran.

## 5. Maintenance

A later revision should retain stable identifiers where meaning is unchanged,
identify changed requirements, update source/witness mappings, and refresh the
affected executable evidence. The repository census is tied to this baseline;
rebasing the edition requires an explicit source identity update and a review
of changed compiler/library interfaces.

New language designs enter as candidate contracts with their semantic
assumptions and acceptance obligations. Existing source defects remain
implementation findings until repaired or resolved through an authorized
language decision. This preserves the founder's scientific intentions while
making the implementation accountable to precise interfaces.
