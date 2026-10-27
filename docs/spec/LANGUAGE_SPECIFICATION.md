<!-- docs:meta
topic_id: repo.docs.spec.language-specification
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.language-specification
-->

# The Sounio Language Specification

**Author:** Demetrios Chiuratto Agourakis

**Edition:** October 2026, source research edition

**Prepared:** 2026-10-16

**Source baseline:** `f5b2c5850be9f3d3995719ffcade44a6d851658a`

**Baseline commit date:** 2026-10-15T19:07:51Z

Sounio is a programming language for scientific and epistemic computation.
This reference describes program syntax and checking, the information carried
by scientific values, execution and target interfaces, and the evidence
required to establish their behavior.

The edition is recovered from the compiler, libraries, tests, formal models,
and approved project contracts. Its rules distinguish adopted semantic
contracts, editorial reporting conventions, source observations, target and
library interfaces, and open language decisions. This is a source-research
reference, not language-wide certification or an assertion that every rule is
implemented. Existing adopted contracts retain their authority when an
implementation fails to enforce them. Neither a rule identifier nor the word
*shall* establishes enforcement; library and target scope alone do not ratify
new language semantics.
The [foundations](2026-10/00-foundations-and-conventions.md) define those
categories and the common semantic notation. Each technical chapter provides
stable rule identifiers and a machine-readable source/witness table.
Referenced compiler and Lean tests require their own execution receipts.
Most chapter-catalog fixtures remain unrun by their source-research authors.
The separate [bounded conformance receipt](2026-10/15-bounded-conformance-receipt.md)
records the existing manifest's `total=16 passed=16 failed=0` result on Slurm
job `14573`, using the committed Linux x86-64 Madaros prebuilt with
`source-rebuilt=false`. It is evidence for that artifact, those inputs, and
those assertions, not validation of the compiler source, all scientific/formal
claims, or the whole reference. Structural documentation checks establish
different facts from this execution receipt.
The [preparation and review receipt](2026-10/17-preparation-and-review-receipt.md)
records the separate editorial/provider-review and preparation boundaries;
those activities do not certify all language contracts or scientific results.
The [lexical execution receipt](2026-10/18-lexical-execution-receipt.md) runs
125 probe programs against chapter 01's `LEX` rules on both engines and on a
Madaros built from source (2026-10-22). It records which rules execution
confirms or contradicts and does not change any rule's standing.

Navigation has two numbering systems: files `00` through `10` are the book's
subject chapters; the central thematic sections 1 through 17 retain the
earlier reference's organization and anchors. A central section number is
not a chapter filename or a promise of an additional chapter.

## Reading the reference

| Chapter | Technical scope |
|---|---|
| [Foundations and conventions](2026-10/00-foundations-and-conventions.md) | Authority, semantic judgments, outcomes, scientific invariants, and edition identity. |
| [Lexical structure, syntax, and modules](2026-10/01-lexical-syntax-and-modules.md) | Tokens, literals, precedence, declarations, patterns, statements, imports, and EBNF. |
| [Types, generics, and ownership](2026-10/02-types-generics-and-ownership.md) | Type identity/compatibility, inference, traits, refinements, ownership, and borrowing. |
| [Effects, handlers, and concurrency](2026-10/03-effects-handlers-and-concurrency.md) | Effect vocabulary, rows, subtyping, mutation, handlers, and concurrency interfaces. |
| [Knowledge, uncertainty, and provenance](2026-10/04-knowledge-uncertainty-and-provenance.md) | Scientific values, predicates, propagation, covariance, and evidence boundaries. |
| [Numerics, units, and algebra](2026-10/05-numerics-units-and-algebra.md) | Integer and floating semantics, wide formats, conversion, units, and ordered algebra. |
| [Execution, memory, and targets](2026-10/06-execution-memory-and-targets.md) | Compiler selection, IR paths, object representation, ABI, allocation, FFI, and targets. |
| [Libraries, packages, and tools](2026-10/07-library-packages-and-tools.md) | Library families, package resolution, host interfaces, CLI, and editor/agent tools. |
| [Ontology, quantum, and formal models](2026-10/08-ontology-quantum-and-formal-boundaries.md) | Ontology closure/checking, state simulation, circuit interfaces, and proof correspondence. |
| [Conformance and diagnostics](2026-10/09-conformance-diagnostics-and-compatibility.md) | Implementation profiles, test assertions, refusal, receipts, and compatibility. |
| [Research scope and repository map](2026-10/10-research-scope-and-upstream-practice.md) | Repository coverage, research method, and upstream reference practice. |

For a first program, use the [getting-started guide](../guide/getting-started.md).
The reference supports precise lookup and implementation work.

## 1. Introduction

A program has a syntactic structure, static obligations, and an execution
contract. Its central result can coexist with uncertainty, numerical error,
confidence, provenance, and resource effects. Each channel keeps the meaning
established by its language or library contract.

An implementation profile identifies the engine, source revision, compiler
artifact, target, library, CLI adapter, effective entrypoints, and stage
actually reached. Selection origin, engine family, artifact identity, and
adapter are separate fields; a suite's banner or availability variable alone
does not establish them. Madaros is the public default selected through
`bin/souc`; the legacy seed is separately selectable.

When an existing adopted contract settles the behavior, a conflicting
implementation result is a defect against that contract. Where no adopted
contract determines the behavior, alternatives remain an open language
decision requiring the appropriate ruling and evidence. Results from different
stages or profiles are distinguished before either conclusion. The August
frame and S17 retain their dated research and individual explicit rulings;
their historical measurements do not become current semantics. The
[foundations](2026-10/00-foundations-and-conventions.md) define the authority,
notation, and outcome categories used throughout the edition.

## 2. Lexical Structure

Lexical rules define how source bytes become tokens. Parsing interprets those
tokens in item, type, expression, statement, and pattern contexts. A name
permitted contextually need not be accepted in every position. Numeric
spelling and semantic numeric type are separate decisions.

The [syntax chapter](2026-10/01-lexical-syntax-and-modules.md) describes
identifier character handling, keyword classes, literals, escapes, punctuation,
newlines, token splitting, and malformed-input boundaries.

### 2.7 Comments

Comment recognition and termination are lexer properties. The syntax chapter
identifies the production lexer and helper routes, line/block comment
handling, and malformed-input evidence. A helper-lexer fixture certifies only
the route it actually reaches.

## 3. Types

A parsed type expression, a checker type entry, and a runtime representation
are distinct objects. Compatibility can depend on position: binding, argument,
return, field, generic substitution, or refinement obligation. Recognition of
a type name alone does not establish executable semantics for its operations.

The [type chapter](2026-10/02-types-generics-and-ownership.md) describes the
recovered domains and relations, primitive and compound types,
nominal/structural identity, arrays, tuples, references, functions, enums,
inference, and remaining checking boundaries.

### 3.7 Generic Types

Parsing a generic bound, recording a trait, selecting a substitution, and
enforcing an obligation are different operations. The type chapter traces
each through the current checker and specialization paths and identifies
supported cases and unresolved enforcement decisions.

### 3.9 Epistemic Types

Compiler `Knowledge<T>` has type-level and lowered scientific information.
Library epistemic records expose a distinct representation and API. The
[knowledge chapter](2026-10/04-knowledge-uncertainty-and-provenance.md)
describes construction, predicates, observation, conversion, propagation,
and refusal obligations for each surface.

## 4. Expressions

Expressions produce values and can have effects and resource obligations.
Precedence determines parsing; evaluation and scientific propagation determine
execution. Reassociation can change floating rounding, dependence, or
non-associative meaning even when a simpler algebraic model permits it.

The [syntax chapter](2026-10/01-lexical-syntax-and-modules.md) supplies
expression and pattern forms. The type, effect, numerical, and execution
chapters supply their corresponding rules and implementation boundaries.

### 4.7 Lambda Expressions

Function references and closures require a callable type, capture semantics,
resource accounting, and an ABI. The type and execution chapters identify
their implemented routes. The public-claim registry's `closures.lambdas` row
remains `stale_conflicting`; this edition's source observations do not promote
that public claim.

## 5. Statements

Bindings, assignments, expression statements, and control transfers affect
the environment and control flow. `let` and `var` express different binding
mutability. Loop control, return, block tails, and newline handling follow
their Sounio parser/checker rules.

The [syntax chapter](2026-10/01-lexical-syntax-and-modules.md) gives statement
forms and boundary conditions. The type, effect, and ownership chapters
describe their static obligations.

## 6. Declarations

Declarations introduce named entities whose visibility, type, and effect
information contribute to the checking environment. Collection, resolution,
body checking, specialization, and emission are distinct stages.

### 6.1 Function Declarations

A function signature supplies parameter/result types and effect information;
its body supplies control flow and a result. Checking an uncalled declaration
can interrogate fewer constraints than checking a call. The syntax and type
chapters identify the concrete paths and witness pairs.

### 6.2 Struct Declarations

A struct supplies field names and types. Construction, projection, cross-module
identity, by-value/reference parameters, aggregate returns, and scientific
fields must preserve their respective type and representation contracts.
The type, knowledge, and execution chapters give the detailed rules.

### 6.7 Modules

Modules organize declarations and importable source units. Filesystem,
package, and import names need not use identical spelling. The syntax
chapter gives source-graph rules; the
[package chapter](2026-10/07-library-packages-and-tools.md) gives the local
package mapping and missing-import acceptance criteria.

## 7. Effect System

An effect annotation contributes static information about a computation.
Name recognition, row inclusion, inference, masking, handler checking, and
runtime interpretation have distinct obligations. The
[effects chapter](2026-10/03-effects-handlers-and-concurrency.md) describes
the implemented vocabulary, representation, call/subtyping and mutation
rules, handler reachability, and concurrency interfaces.

## 8. Ownership and Borrowing

Ownership tracks permitted use of values and resources. Shared/exclusive
references and linear/affine values have different access and usage
obligations across branches, loops, calls, captures, and returns. The
[ownership chapter](2026-10/02-types-generics-and-ownership.md) traces active
checking and independent model modules, explicit releases, call-loan lifetime,
and remaining language decisions.

## 9. Units of Measure

Unit spelling, dimensional checking, numeric representation, and conversion
have separate contracts. The
[numerics and units chapter](2026-10/05-numerics-units-and-algebra.md) covers
declared/derived units, current checking paths, format interactions,
conversions, and engine-specific refusal/execution evidence.

## 10. Refinement Types

A refinement constrains values of an underlying type. Its parsed predicate,
static obligation, runtime assertion, and discharge result are different
objects. The [type chapter](2026-10/02-types-generics-and-ownership.md) and
[knowledge chapter](2026-10/04-knowledge-uncertainty-and-provenance.md)
describe solver boundaries and source-to-source/runtime guard routes.

## 11. GPU Programming

A GPU contract includes a supported kernel subset, target format, parameter
ABI, memory exchange, runtime service, device, and failure behavior.
PTX/MSL production and device execution have different observation contracts.
The [execution chapter](2026-10/06-execution-memory-and-targets.md) describes
the PTX, Metal, and Pireus routes; the numerical chapter gives precision
boundaries. Device claims require named hardware and execution receipts.

## 12. Standard Library

The library includes basic runtime interfaces, numerical/scientific packages,
domain models, and experimental facilities. Exported signatures, reachable
imports, effects, target requirements, and acceptance evidence define each
contract. The [library chapter](2026-10/07-library-packages-and-tools.md)
supplies the family map, package/tool interfaces, and distribution boundaries.

### 12.5 Epistemic Computing

Scientific APIs identify the meanings of uncertainty, confidence, dependence,
and provenance. The [knowledge chapter](2026-10/04-knowledge-uncertainty-and-provenance.md)
gives compiler/library contracts. The conformance chapter identifies the
existing GUM witnesses and the channels they actually assert.

Fixture names containing “GUM compliance” or “ISO budget” identify bounded
GUM-derived channel checks. They do not establish certification, metrological
traceability, or JCGM-wide conformance. The separate bounded prebuilt receipt
records only the manifest's stated assertions for those fixtures.

## 13. Execution and Memory

The [execution chapter](2026-10/06-execution-memory-and-targets.md) describes
active lowering, aggregate layout, calls/returns, stack/heap storage,
allocation lifetime, region interfaces, strings, foreign calls, failure,
and executable handling under named engine/target profiles.

## 14. Numerical Semantics

The [numerical chapter](2026-10/05-numerics-units-and-algebra.md) covers
integer widths/operations, floating formats/special values, literal
interpretation, conversions, wide arithmetic, units, and ordered algebra.
Mathematical exactness, finite representation, approximation error, and
physical measurement uncertainty remain distinct.

## 15. Ontology and Quantum Interfaces

The [ontology and quantum chapter](2026-10/08-ontology-quantum-and-formal-boundaries.md)
defines concept/role and closure interfaces, validation profiles, classical
statevector APIs, linear handles, uncertainty/sampling channels, circuit
serialization, device requests, and observation boundaries.

## 16. Diagnostics and Formal Models

A diagnostic establishes a particular refusal at a particular stage.
A formal theorem establishes its statement in a declared model and trusted
assumptions. The [conformance chapter](2026-10/09-conformance-diagnostics-and-compatibility.md)
gives diagnostic contracts; the
[formal-model chapter](2026-10/08-ontology-quantum-and-formal-boundaries.md)
gives theorem and runtime-correspondence boundaries.

## 17. Conformance

A conformance claim needs a settled rule, supported implementation profile,
and the accepted observations. A bounded test spine, evidence-path check,
bootstrap fixed point, and proof-model build are distinct results. The
[conformance chapter](2026-10/09-conformance-diagnostics-and-compatibility.md)
documents the actual test-runner behavior and required execution receipts.

Structural validation of this reference is available through:

```sh
node scripts/docs/validate_language_spec.mjs
```

## Appendix A: Grammar

The [EBNF recognition profile](2026-10/core-grammar.ebnf) accompanies the
syntax chapter. It states its covered subset and required contextual
predicates. Complete equivalence to the production parser requires additional
grammar-conformance work.

## Appendix B: Source and Witness Catalogs

The [edition manifest](2026-10/edition.json) lists chapters and their
`*-evidence.tsv` catalogs. Every row records rule standing, source path,
source symbol, positive/negative witness paths, and remaining gap.
The public-claim and spec-evidence matrices retain their own authority.

## Appendix C: Research Coverage and History

The [repository map](2026-10/repository-coverage.tsv) accounts for the
baseline's roots and major source families. The
[research method](2026-10/10-research-scope-and-upstream-practice.md) records
upstream reference sources and validation scope.

The [beta.6 text](../archived/spec/LANGUAGE_SPECIFICATION_BETA6.md) remains
available as historical lineage. Earlier studies under `docs/spec/` retain
their original dates, source identities, and evidence boundaries.
