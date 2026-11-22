<!-- docs:meta
topic_id: repo.docs.spec.2026-10.12-syntax-runtime-review
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.12-syntax-runtime-review
-->

# Syntax/runtime integration review

Review date: 2026-10-16. Compiler source baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.

Write scope: this review file only. The central reference, grammar, chapters,
catalogs, compiler, fixtures, governance files, and shared logs were not edited
for this review. The syntax chapter and grammar were prepared earlier by the
same agent; their inclusion here is a source-first self-audit, not a claim of
independent third-party review. The effects and execution chapters were audited
against source independently of their authors' conclusions.

No compiler, emitted program, GPU workload, numerical oracle, or proof build
was executed. Source-derived predictions below are not measured outcomes.
Fragments illustrate discriminating cases to prepare later; they are not new
compiled fixtures. Existing fixture names and directives are not pass receipts.

## 1. Disposition and severity

The reviewed effects and execution chapters do **not** currently promise
portable short-circuit evaluation, executable algebraic handlers, or a uniform
async runtime. Their qualifications are substantially consistent with the
inspected source and should be preserved. In particular, rewriting the handler
account back to an unconditional historical “success with silent erasure” would
make it less accurate.

The current material is suitable for continued integration as a **source
research edition**. Portable execution/handler conformance remains blocked by
the existing semantic and acceptance gaps. There are also concrete grammar
and stage-evidence corrections for the parent to integrate.

Severity in this report has two explicitly different scopes:

- **BLOCKER / promotion** prevents claiming the named runtime guarantee or
  promoting an open decision to a normative, implemented contract. It does not
  prohibit publishing an accurately qualified research edition.
- **MAJOR / correction** identifies a grammar, wording, or integration defect
  in the reviewed draft that should be repaired before treating it as an
  implementer-facing reference.

| Finding | Severity and scope | Principal locations | Disposition |
|---|---|---|---|
| `SR-B01` | BLOCKER / portable logical-execution promotion | Syntax `SYN-016`; execution `EXEC-010`, `EXEC-011`, `EXEC-040`; effects `EFF-003`, `EFF-008` | Keep decision open; connect effect checking to the unresolved dynamic sequencing boundary. |
| `SR-B02` | BLOCKER / operational handler promotion | Effects `EFF-022`–`EFF-024`; execution refusal/receipt contract | Preserve unsupported-lowering account; require clean build refusal and artifact-publication evidence. |
| `SR-M01` | MAJOR / grammar correction | `core-grammar.ebnf`: `impl-method`, `function-declaration`, `function-modifier` | The stated conservative subset admits modifiers rejected in impl bodies. |
| `SR-M02` | MAJOR / grammar and source-contract correction | `core-grammar.ebnf`: `return-expression`, `break-expression`, `range-expression`; syntax `SYN-012` | Enumerate the operand-start predicate and its omissions instead of implying arbitrary expression operands. |
| `SR-M03` | MAJOR / stage-evidence wording | Foundations `SPEC-005`; syntax `LEX-001`; effects handler acceptance terminology | A returned AST/program is not itself parser acceptance. Require explicit error-state and overflow checks. |
| `SR-M04` | MAJOR / integration completion, resolved on final path recheck | Central chapter-10 and repository-coverage links | Initially absent targets now exist; semantic and anchor checks remain separate. |

Finding labels are review identifiers, not additions to the language-rule
namespace or governance blocker registry. Suggested responsibility is the
parent integration lane for prose/grammar and the maintainer-designated
compiler lane for implementation changes. No compiler ownership was acquired.

## 2. BLOCKER findings and precise limits

### SR-B01 — Logical evaluation cannot be promoted to a portable contract

**Evidence class:** current-source control-flow inspection; no runtime receipt.
**Class:** unresolved language semantics plus target/lowering integration.
**Decision owner:** founder/maintainer ruling; implementation owner not assigned
by this report. **Acceptance gate:** route-specific dynamic and IR witnesses
under an explicitly chosen contract. **Next action:** retain open-decision
standing and prepare the discriminators below before selecting a repair.

The ordinary `Lowerer.lower_binary_expr_ref` in
`self-hosted/ir/lower.sio` calls `lower_opt_expr_ref` on the left operand,
threads its returned lowerer into a call on the right operand, and only then
emits the binary operation. Neither a conditional branch nor a logical-op test
suppresses the RHS before those calls. This supports a source-local eager
emission observation for ordinary logical expressions. Special wide-float
branches and unsupported-type refusals precede this path and must not be
included in a blanket “all binaries execute left-to-right” claim.

The in-place checker independently checks both operands in
`checker_check_binary_expr_inplace` and requires bool operands for `OpAnd` and
`OpOr` in `checker_check_binary_op_types_inplace`, both in
`self-hosted/check/check.sio`. Checking both branches and collecting their
possible effects is compatible with either eager or lazy runtime semantics:
an effect **set** can conservatively bound possible operations without encoding
their order or guaranteeing that every operation occurs. There is therefore
no demonstrated contradiction between eager emission and ordinary effect-set
inclusion. Do not “fix” that apparent contradiction by weakening the static
effect check or treating an annotation as an execution trace.

The separate native Metal AST renderer in
`self-hosted/gpu/native_metal.sio` maps `OpAnd` and `OpOr` to `&&` and `||` in
`nm_operator`, and `nm_expr` prints a parenthesized binary expression directly.
It does not mirror the native IR lowerer's explicit operand temporaries. This
is a concrete cross-route translation difference. No device run or external
MSL semantic verification was performed here, so it is not a measured target
disagreement. The current execution chapter correctly stops short of that claim.

**Required integration correction:** add an explicit reciprocal connection
between the effect chapter's set/inclusion rules and the syntax/execution open
decision: a checked `with IO`/`Mut` requirement does not establish dynamic order,
single evaluation, RHS suppression, or cross-target equivalence. Keep the
ordinary-lowering observations and the intended logical-operator decision
separate. `EXEC-011`, `EXEC-040`, and `SYN-016` already make the crucial
qualification; the parent should preserve it when summarizing the edition.

**Discriminators to prepare, not executed:**

| Question | Necessary observation | Important route limit |
|---|---|---|
| `false && rhs` and `true || rhs` | RHS entry/exit markers and an independently observed mutable counter, plus final bool | Choose nonconstant/opaque inputs or inspect IR so constant folding cannot make the test vacuous. |
| Other two logical truth cases | RHS occurs exactly once when required by the chosen semantics | A bool result alone cannot distinguish eager from lazy evaluation. |
| Operand/callee/argument order | Distinct trace markers for each phase; assert complete ordered output | `lower_call_expr_generic_ref`'s inline-callee path is narrower than all named/builtin calls. |
| Assignment place versus RHS | Observe receiver/index computations and mutation separately | Simple `x = value` does not exercise a compound lvalue. |
| Metal comparison | A supported kernel expression or a specific supported-subset refusal, plus actual target artifact and device receipt | `nm_expr` accepts only a narrow expression subset and recognizes only a specific thread-ID call; do not assume a CPU-side effectful helper call is admitted. |
| Optimization | Same source and identified compiler under the route's actual optimization selections; compare traces and IR | Value equality or a pretty-printed operator is insufficient. |

Existing `tests/ui/type/logical_and_not_bool.sio` checks operand typing, not
dynamic RHS behavior. The execution catalog correctly admits this limit for
`EXEC-040`. The specialized
`tests/run-pass/fo_product_rule_call_evaluated_once.sio` does not fill the
general sequencing gap either. No new passing case is claimed.

### SR-B02 — Parsed/checkable handlers are not operational handlers

**Evidence class:** parser/checker/lowering and publication-guard source
inspection; no end-to-end refusal result. **Class:** unsupported runtime feature
and incomplete acceptance evidence. **Owner:** parent for truthful wording;
compiler owner unassigned. **Acceptance gate:** real handler/control/refusal
fixtures through the selected public build path with error and artifact checks.
**Next action:** preserve the unsupported designation and obtain fresh scoped
refusal evidence before any handler-runtime promotion.

The source chain is explicit:

| Stage | Source and symbol | What it establishes |
|---|---|---|
| Recognition | `self-hosted/parser/exprs.sio`: `parse_handle_expr` | Builds `ExprHandle` with an ordinary body block and optional handler block. |
| Checking | `self-hosted/check/check.sio`: `check_handle_expr` | Temporarily adds a recognized built-in effect while checking the body, restores ambient effects, then checks the optional handler block. |
| Operations | `self-hosted/parser/exprs.sio`: `parse_perform_expr` | Consumes `perform` and retains the following ordinary expression, without an operation/continuation node. |
| Lowering | `self-hosted/ir/lower.sio`: `lower_expr_non_binary_ref` | No `ExprHandle` branch; unmatched forms call `report_error_at(15454)` and emit unit. |
| Error retention | `self-hosted/ir/lower.sio`: `report_error_at`, `lower_note_error`, `lower_hard_error_pending` | Records a lowering error and, when the compilation error scope is active, sets the compile error latch. |
| Publication boundary | `self-hosted/compiler/module_frontend.sio`: `lower_hard_error_pending` checks before native publication | Source contains a refusal mechanism; this is not a fresh observed exit/absence-of-artifact result. |

`EFF-024` accurately describes the error-marking fallback, and `EFF-023`
accurately distinguishes ambient authorization from operation interception.
There is no basis for correcting them to say that the body and handler run,
that an operation is resumed, or that a current compiler necessarily succeeds
and drops both blocks. A source occurrence count of `ExprHandle` is likewise
not a runtime-implementation test: unsupported-control-flow helpers also mention
that variant.

**Required correction at integration boundaries:** any short description of
“handler support” must name its stage. A valid AST and apparent checker success
must not be called executable support. Use “parsed form; bounded checker
handling; unsupported ordinary runtime lowering” until the intended semantics
and a genuine execution implementation exist. The lowerer emitting unit after
recording an error is recovery, not defined handler evaluation.

Existing controls are `tests/audit/handle_control.sio`,
`tests/audit/handle_io.sio`, `tests/audit/handle_unknown_effect.sio`, and the
known-failure `tests/run-pass/handler_discharge.sio`. The latter retains
historical comments predicting successful silent erasure; those comments must
not supply the expected present-day exit contract. A fresh acceptance exercise
must distinguish check success, unsupported build refusal, compiler crash,
missing dependency, and output publication. It must also verify that an old
output file cannot be mistaken for a newly accepted handler executable.

## 3. MAJOR corrections

### SR-M01 — The grammar admits rejected impl-method modifiers

`core-grammar.ebnf` defines `impl-method` through `function-declaration`, which
in turn allows `function-modifier = "kernel" | "const" | "async"`.
Consequently its supposedly conservative subset admits an impl body containing
`async fn method(...) { ... }`, `const fn method(...) { ... }`, or
`kernel fn method(...) { ... }`.

In `self-hosted/parser/items.sio`, `parse_impl_item` consumes optional method
visibility and requires the next token to be Fn before calling `parse_fn_item`.
It does not call the top-level `parse_item` modifier-consumption path. Any
other next token sets parser error state. The grammar therefore over-accepts
these method shapes relative to its declared current-source subset. A profile
disclaimer about omitting features does not justify including rejected ones.

**Correction for the parent:** factor a method production using an unmodified
function header/body, or explicitly exclude these modifiers in impl context.
Keep top-level modifier recognition separate. Do not change the compiler merely
to make a documentation grammar become true without a language decision.

**Acceptance:** a grammar-level positive ordinary impl method and modifier
negatives must agree with parser verdict/error state. No dedicated modifier
fixture was established here. `tests/run-pass/parser_public_impl_method.sio`
is an existing ordinary-method control, not proof of modifier acceptance.

### SR-M02 — Return, break, and range operands need an explicit starter boundary

The grammar writes optional general expressions after `return`/`break`, while
predicate P8 refers the implementer back to `Parser.at_expr_start`. The syntax
chapter describes this predicate but does not enumerate its mismatches with
the actual prefix parser. That reference prevents treating the grammar as a
complete CFG, but leaves a material implementer-facing gap.

`self-hosted/parser/parser.sio`: `at_expr_start` includes decimal IntLit,
FloatLit, Minus, Bang, reference/dereference starters, several blocks/control
forms, and contextual identifiers. It omits HexLit, BinLit, OctLit, Tilde,
Pipe/PipePipe closure starters, For, and Var. Some of these omitted starters
are accepted by `parse_prefix` in `self-hosted/parser/exprs.sio`; some literal
token kinds are not emitted by the active file lexer in the first place.
Those are different limitations and should be listed separately.

Therefore `return 0x10` does not attach the following HexLit as the return
operand at this parse decision; `return ~value` and a directly returned closure
have the same starter mismatch. This predicts a different AST partition or
later failure, not a verified runtime result. Parenthesizing the operand changes
its first token to LParen and is a useful discriminating control. The range
branch in `parse_expr_bp` also consults `at_expr_start`, so `start..0x10` has
the corresponding optional-endpoint ambiguity.

**Correction for the parent:** make the actual operand-start predicate explicit
for this source profile, flag the prefix/starter mismatch as a defect, and
distinguish it from the candidate language rule “any expression can follow
return.” A grammar implementation must not silently consume a value that the
source parser leaves as another statement. Conversely, the current omission
must not become an approved limitation of return values. Candidate witnesses
must compare AST shape and parser errors before asserting executable behavior.

This finding also applies at EOF/newline boundaries: a newline alone does not
disable a recognized return operand, while an omitted starter does. Tests must
vary both starter and separator independently.

### SR-M03 — A returned parse object does not establish syntax acceptance

Foundations `SPEC-005` says “A parse result establishes syntax acceptance.” Read
literally, this is too broad for this compiler's recovery interfaces. The
syntax chapter's P10 rule correctly says placeholder ASTs and reaching EOF are
not acceptance; the common stage definition should express that qualification
directly rather than rely on readers discovering it later.

`parse_items_loop` in `self-hosted/parser/mod.sio` continues after errors,
forces progress when necessary, records error state with
`parser_set_last_errors`, and still returns an item list. `load_module_file` in
`self-hosted/compiler/module_parse.sio` returns a Program after parsing; it
does not encode all parse errors in the Program's shape. The dependency closure
uses `parser_last_error_count` to classify parse failure. Likewise lexer
overflow is a separate refusal channel before parsing. These interfaces
demonstrate why nonempty AST, non-null Option, consumed bytes, or recovered EOF
cannot replace a successful parser verdict.

**Suggested correction:** “A successful parser verdict, including its error
state and lexer-capacity checks, establishes syntax acceptance on that route.”
Keep checking, lowering, emission, and execution as separate subsequent
obligations. Apply that wording when summarizing parsed handlers and async
forms as well as ordinary programs. This is not a request to redesign the
source API in this documentation change.

### SR-M04 — Initially unfinished central targets now resolve

The central `docs/spec/LANGUAGE_SPECIFICATION.md` was available during the
later review pass. Its local links to
`docs/spec/2026-10/10-research-scope-and-upstream-practice.md` and
`docs/spec/2026-10/repository-coverage.tsv` had no target at the checked snapshot.
The chapter-10 link appears more than once. The final read-only link recheck
found both targets present and no missing central-reference target. This
path-existence finding is therefore resolved at handoff; no sibling or central
file was edited by this reviewer to resolve it. It does not establish that the
target contents or their semantic coverage have been reviewed.

The six inspected index/chapter documents had 43 local link occurrences in the
check; the missing occurrences were confined to those two target paths. The
01/03/06 evidence catalogs had exact requested headers, existing primary
source/positive/negative paths, and their named symbol text present in the
selected source. This structural result is not an audit that each witness
actually discriminates its row's semantic claim.

**Acceptance:** preserve the successful final target-existence check and rerun
full link validation, including anchors, after all parent-owned files settle.
Do not substitute a different chapter or registry merely because a path happens
to exist. `scripts/docs/validate_language_spec.mjs` was absent during the
initial check and became available before handoff. It was not invoked in this
review, so no result from that validator is claimed.

## 4. EOF, Unicode, and newline audit: limits to retain

These points are substantially disclosed by the syntax and execution chapters.
They are review constraints, not newly discovered contradictions or requests
to replace their accurate qualifications.

| Boundary | Current-source evidence | Required reading and future discriminator |
|---|---|---|
| Actual EOF | `lexer/mod.sio`: `lex_source_to_globals`, `lexer_push_flat`; `parser/parser.sio`: `parser_at_eof` | EOF is a token at the source end; it must fit the token table. An Error token or an invalid byte must not be reclassified as an ordinary clean EOF. Inspect error state and both token interval endpoints. |
| Unterminated comment | Scalar block-comment loop in `lex_source_to_globals` | It can reach physical EOF with positive nesting depth without a dedicated comment error. The stricter closed-comment EBNF is a declared profile limit, not evidence of implemented refusal. |
| Unterminated string | Scalar string loop in `lex_source_to_globals` | It still emits StringLit at physical EOF and does not validate a closing quote. Reaching EOF with a string token cannot establish the grammar's closed-string contract. |
| EOF after complete source | `parse_items_loop` and parser error side channel | Trailing trivia, repeated top-level semicolons, recovered error tokens, and incomplete nested blocks need separate cases. Parse-object existence alone is insufficient. |
| Unicode names | `lexer/cursor.sio`: `is_ident_start`, `is_ident_continue`; scalar Error-token branch | ASCII name recognition and byte retention in strings/comments do not implement Unicode XID, normalization, or UTF-8 validation. Keep `unicode_ident.sio` as unrun rejection evidence, not a current exact diagnostic. |
| Runtime string representation | Execution `EXEC-016`; byte-string emitters and helpers described there | Byte indexing/NUL termination does not become scalar/grapheme indexing because source prose says “character.” Embedded-NUL behavior and source-encoding validity need their own contracts. |
| LF versus CR | Scalar whitespace branches and `had_newline_before` | LF changes line metadata, lone CR does not, and CRLF has one LF increment. Avoid an abstract newline rule that silently makes all three equivalent. |
| LF inside raw multiline string | Scalar string branch | Columns advance without line changes there. Later token line metadata can therefore disagree with physical source lines; newline-based parse decisions are affected, not just diagnostic appearance. Raw multiline source is excluded from the conservative grammar. |
| Newline and postfixes | `parse_expr_bp`, `parse_ident_expr`, `parse_dot_expr_box` | Calls/indexing and method arguments have LF guards; dot continuation has different handling. Whitespace-insensitive EBNF alone is insufficient. |
| Newline and tight prefix operators | `parse_expr_bp`; `PT_END`/`PT_START` comparisons | `*x` versus `* x`, and comments between the operator and operand, exercise byte adjacency separately from line separation. |
| Parenthesized block continuation | `parse_expr_bp`: `atom_parenthesized` | `(if ... { ... } else { ... })` can continue across LF differently from an ungrouped block-like atom. The known regression fixture is scoped to this distinction. |

The EOF/Unicode/newline witnesses already named in the chapters include
`tests/ui/lexer/unterminated_string.sio`,
`tests/ui/lexer/unterminated_block_comment.sio`,
`tests/ui/parser/unicode_ident.sio`,
`tests/ui/parser/double_semicolon.sio`, and
`tests/run-pass/paren_if_newline_operator_continuation.sio`. Several older
directives conflict with the current scalar lexer. They require classification
and fresh receipts, not automatic conversion into normative rejection rules.

## 5. Read-only review and acceptance procedure for integration

This is the recommended repeatable review design. Only the structural/source
inspection portions were performed here; future execution is explicitly separate.

1. **Freeze the review inputs.** Record HEAD, dirty paths, exact document and
   source hashes, and which parent writes are still in flight. Read applicable
   AGENTS guidance and the active coordination brief. Review named files without
   checking out a different branch, rebuilding, or changing shared state.
2. **Resolve the reference graph.** Check local Markdown targets relative to
   their containing documents; separately validate fragments, grammar links,
   catalog paths, and cited gate/fixture paths. Match rule definitions to catalog
   rows, not every incidental mention of an ID. Validate symbol declarations,
   not just substring matches, before claiming source-symbol correspondence.
3. **Parse the grammar as a document.** Check balanced comments, strings, special
   sequences and delimiters; unique definitions; defined references; production
   terminators; and explicit lexical priority. Then perform semantic counterexample
   review against item/type/pattern/expression dispatch. Structural checks alone
   cannot establish that the grammar is a conservative subset. SR-M01 is a
   concrete example of a structurally valid but semantically incorrect production.
4. **Trace three distinct paths.** For each claim, identify parser construction,
   active checker decisions, and selected lowering/publication/runtime behavior.
   Record when a modifier or operation node disappears. Trace the public resolver
   and target selector; a target implementation or library file's existence does
   not establish reachability.
5. **Audit effects as bounds, not traces.** Check unknown-name/overflow and
   closure/function-type contexts separately; then ask which dynamic observations
   are actually promised. Preserve the effects chapter's explicit local-variable,
   exclusive-reference, GPU exception, handler, inference, and capacity limits.
   Neither effect-set inclusion nor a “pure” label alone settles evaluation order.
6. **Classify existing witnesses by mechanism.** A bool type-error fixture is
   not a lazy-logical witness; a printing function named “handler” is not handler
   dispatch; a known-failure comment is not current behavior. Distinguish intended
   refusal from signals/timeouts/dependency errors and optimization refusal from
   program refusal. Preserve exact engine/target/mode scope.
7. **Design future executable discriminators.** Use positive controls, near-miss
   negatives, and observable side-effect markers/counts/identity. Prepare isolated
   output destinations and pin actual artifacts through the canonical resolver.
   For target comparisons, first prove that both routes admit the test shape;
   unsupported-shape refusal is a separate outcome. Numerical checks follow
   ADR009 and must not inherit Python numeric generators/oracles silently.
8. **Close only the evidenced claim.** A future successful check closes a checker
   obligation; a build closes emission; an asserted native/device execution closes
   that observation on that route. Promotion of the unresolved language rule also
   needs its maintainer decision. Rehash the documents at handoff and classify
   findings affected by concurrent changes as needing revalidation.

An independent lightweight recursive-descent check in this review parsed all
150 EBNF productions, including alternatives, concatenation, grouping, optional
and repeated forms, terminals, special sequences, and production terminators.
It reported no grammar-syntax errors, duplicate production definitions, or
undefined production references. This strengthens the structural check but
does not establish unambiguity, a conservative source subset, parser equivalence,
or executable conformance. The catalog/path/link checks were lightweight Node
text/filesystem inspection, not compiler tests or numerical verification.

## 6. Reviewed snapshot and checks

These SHA-256 values identify document bytes in the review's structural-check
snapshot. They are not compiler artifact identities or execution receipts.

| Path | SHA-256 |
|---|---|
| `docs/spec/LANGUAGE_SPECIFICATION.md` | `ee259eb9d6ee7574a5ec88e64209b6ef908af321c1a144271a24fe14c2216dbe` |
| `docs/spec/2026-10/00-foundations-and-conventions.md` | `e633d880551077d3c54458b4fecf628f7f14618c2917b90982ec720207dd3ccf` |
| `docs/spec/2026-10/01-lexical-syntax-and-modules.md` | `822719c8f8a7b60638761e09ec3d9972c91259e1dd3fb96a1eaac2fc20b1886e` |
| `docs/spec/2026-10/03-effects-handlers-and-concurrency.md` | `859ffac76f4458fdcf0c3ddfce65c9e0b1d5670f9510e45bcc64628cab528dd4` |
| `docs/spec/2026-10/06-execution-memory-and-targets.md` | `f129a3605d5753b812b09319784509aff85a23b2aa3faa00c3cee88b036f8130` |
| `docs/spec/2026-10/core-grammar.ebnf` | `5f7bf9e0ebe07750c2f7aa1a0e5118279a11f5b0f195c4ddb5738e2abd4198d2` |

At the hash comparison at `2026-10-16T00:34:54.398Z`, all six document inputs
still matched the snapshot above. New target files can appear without changing the index bytes; the
resolved SR-M04 finding records exactly that distinction.

Performed: read-only Git identity/status and coordination inspection; selected
chapter/index/grammar reads; targeted source/fixture inspection; local link
existence checks; 01/03/06 catalog header/path/symbol-text checks; independent
EBNF syntax/definition/reference checks; final snapshot-hash comparison; and
review-file format/path checks at handoff. Catalogs contained respectively 36, 46, and 53
rows at the structural snapshot. Symbol-text checks are weaker than a parser
for symbol definitions; semantic evidence coverage was assessed manually only
for the topics discussed here.

Not performed: complete audit of every rule in the sibling chapters, Markdown
anchor verification, grammar ambiguity/subset/equivalence testing, the parent's
validator, compiler rebuilding, current executable refusal or
stdout checks, device work, numeric recalculation, or Lean proof checking.
No source-backed result in this report should be read as one of those outcomes.
