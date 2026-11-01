<!-- docs:meta
topic_id: repo.docs.spec.2026-10.01-lexical-syntax-and-modules
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.01-lexical-syntax-and-modules
-->

# Lexical structure, core syntax, and modules

Source baseline: `f5b2c5850be9f3d3995719ffcade44a6d851658a`, inspected on
2026-10-16. Scope: the modular self-hosted frontend reached through Madaros;
bootstrap and backend differences are identified explicitly. Companions:
[core grammar](core-grammar.ebnf) and [rule evidence catalog](01-evidence.tsv).

## 1. Status, vocabulary, and reading method

This chapter is a source-derived specification draft, not a declaration of
conformance. A **source-observed** rule describes a control path present in the
inspected source. A **specified-contract** rule records an intended boundary
with a discriminating repository fixture or gate, without asserting that the
fixture passes. A **target-contract** is limited to a named implementation
path. An **open-decision** records a choice that this draft does not resolve.
Defective acceptance is reported as a defect, not promoted into language design.
No compiler, test suite, generated ceiling witness, or theorem prover was run
for this chapter. Every test and gate mentioned below is inspected, unrun
evidence. Fixture directives express expected behavior, not observed results.

`LEX-001` (**source-observed**, frontend identity). The principal file-input
route is `compiler::module_parse::load_module_file` →
`lexer::mod::lex_file_parse_items` → `lex_file_to_globals` →
`lex_source_to_globals` → `parser::mod::parse_items_preloaded`. Within
`lex_source_to_globals`, the scalar loop returns before an older Cursor-based
loop. Consequently, a feature in `scan_number`, `scan_string`,
`scan_char_literal`, or `scan_doc_comment_*` is not sufficient evidence that
file input recognizes it. `self-hosted/compiler/lexer.sio`,
`self-hosted/compiler/parser.sio`, `parser_min.sio`, `souc_v2/`, and
`lean_single.sio` are separate surfaces; they are not interchangeable
implementations of the route described here.

For implementers, **token spelling** means the original source byte interval;
**token kind** means the decoded `TokenKind`; **flat code** means the scalar
representation in `PT_KIND_CODE`. Those are distinct. The diagnostic field
`expected=` prints an enum cast while `actual=` prints a flat code in
`Parser::expect`; those numbers cannot safely be compared as a common token ABI.
`pt_read_kind`, `parser_peek`, and `parser_set_token_scalar` are the relevant
conversion boundaries. Source references here use paths and symbols, not
unstable line numbers.

## 2. Bytes, names, trivia, and token boundaries

`LEX-002` (**source-observed**, encoding and positions). Source is loaded as
bytes into `CURSOR_SOURCE`, an `i8` array. `is_alpha` recognizes ASCII letters;
`is_ident_start` adds `_`; `is_ident_continue` adds ASCII digits. Thus the
ordinary identifier shape is `[A-Za-z_][A-Za-z_0-9]*`. The inspected route does
not validate UTF-8, decode Unicode scalar values for identifiers, normalize
names, or implement Unicode XID properties. Non-ASCII bytes inside comments or
strings can be carried as bytes; this does not establish a Unicode character
type or a Unicode identifier contract. A BOM and non-ASCII whitespace have no
special treatment and fall through to Error-token emission. Token offsets are
byte offsets. Lines and columns begin at one; tabs increment columns by one,
and LF advances the line. Columns are not grapheme, display-cell, or UTF-16
positions. The string scanner's failure to update line metadata for embedded
LF is a separate defect described below.

`LEX-003` (**source-observed**, name limits). The active identifier loop stops
after 256 bytes, even when the next byte is still an identifier continuation.
An overlong word can therefore become adjacent identifier tokens rather than
a dedicated length refusal. `Name` and `StringList.head_buf` hold 384 bytes;
`Parser::current_name` copies the spelling from source and applies
`parser_user_box_rename`. The larger AST capacity does not make the lexer accept
384-byte identifiers as a single token. Some auxiliary paths have smaller
limits: `parser_name_at` clamps to 128, and the export pool in
`resolve/imports.sio` copies at most 128 name bytes. Portable maximum name
length, overflow diagnostics, and the public status of internal Box renaming
remain unresolved; do not specify silent splitting or truncation as intended
name semantics.

`LEX-004` (**source-observed**, whitespace and comments). Space, horizontal tab,
CR, and LF separate tokens. CR is whitespace, not a line increment; CRLF
therefore has one line increment, whereas a lone CR does not. `//` consumes
through the byte before LF. `/* ... */` supports nesting using a depth counter;
comment contents do not form tokens. In the scalar loop, `///`, `//!`, `/**`,
and `/*!` are consumed by the same branches as ordinary comments. Doc-comment
token kinds and dedicated scanners exist, but the active file-input route does
not emit them. A block comment reaching EOF with positive depth currently
continues without an unterminated-comment error. This is an acceptance gap,
not a rule that unterminated comments are valid. In particular,
`tests/ui/parser/nested_comment.sio` claims nesting is unsupported, contradicting
the active loop, while `tests/ui/lexer/unterminated_block_comment.sio` requests
a diagnostic absent from that loop.

`LEX-005` (**source-observed**, lexical keyword inventory). Keyword matching is
case-sensitive, byte-for-byte, in `lexer::tables::keyword_lookup`. The following
inventory lists spellings actually returned there, rather than inferring words
from enum variant names:

| Family | Spellings |
|---|---|
| Items and bindings | `module import use export fn let var mut const type struct enum trait impl pub unit algebra study ontology` |
| Control and paths | `if else match for while loop break continue return in as is where self Self super crate true false` |
| Effects, resources, concurrency | `effect handler handle with perform resume linear affine move copy drop ref dyn box unsafe extern static async await spawn kernel chan send recv dual session close` |
| Probabilistic and epistemic terms | `uncertain belief evidence confidence prior posterior likelihood entropy observe sample infer Knowledge Valid ValidUntil ValidWhile Derived Computed Measured Input` |
| Specialized type heads | `Forgettable ExactlyPrivate Editable CapabilityGated Composable Audited Revivable Interpretable Model Policy AcquisitionPolicy AcquisitionPlan RecoursePolicy RecoursePlan TransitionPolicy TransitionPlan MonitoringPolicy ObservedTransition RollbackCertificate AlternativePolicy AlternativeSet AlternativeOption DecisionPolicy DeferralPolicy Contest Robust Deferred Validated DPBudget Aleatoric Admissible DiffPrivate CausalEffect Intervention Counterfactual PotentialOutcome Proof Lemma Axiom` |
| Policy/item vocabulary | `policy acquisition_policy recourse_policy transition_policy transition_protocol monitoring_policy alternative_policy alternative_frontier decision_policy deferral_policy models validation audit require objective strategy step level scope contest on route max_steps rollback_route` |

A lexical keyword is not necessarily forbidden as a name. Conversely, an enum
variant does not establish an executable expression, declaration, or runtime
feature. `claim`, the two words `causal graph`, `relevant`, and `from` are
matched by parser spelling/lookahead in particular contexts, rather than by
dedicated lookup entries.

`LEX-006` (**source-observed**, contextual names). `expect_ident` accepts Ident
or any kind accepted by `tk_is_keyword`. Expression and pattern dispatch use
narrower, different sets. `tk_is_contextual_ident` currently includes `effect`,
`handler`, the lower-case policy families, `models`, `validation`, `audit`,
`on`, `level`, `scope`, `is`, and `study`. Expression dispatch additionally
accepts `Knowledge`, `close`, `unit`, `kernel`, and `var`; `handle` is a name
unless followed by `<`. Pattern dispatch separately admits `handle`, `unit`,
and `kernel`. A bare `var` in a binding pattern is a name before `=`, `:`,
`,`, `)`, `}`, `|`, or `=>`; otherwise `var name` is parsed as a mutable
binding pattern. `_` is decoded specially by `pt_read_kind` into Underscore
even though the scalar identifier scanner first recognizes it as a word.
Declaration acceptance therefore does not imply that a name is usable in every
expression. The contextual-keyword fixture still has an older comment excluding
`audit`; `tk_is_contextual_ident` now includes it. The source is authoritative.

`LEX-007` (**source-observed**, punctuators and maximal munch). The active
operator branch tests three-byte spellings before two-byte spellings before
one-byte spellings:

| Width | Active spellings |
|---|---|
| Three | `... ..= <<= >>=` |
| Two | `== != <= >= && || << >> += -= *= /= %= &= |= ^= -> => <- &! :: .. ++ +-` |
| One | `+ - * / % = < > ! & | ^ ~ ? ( ) { } [ ] ; : , . @ #` |

`+-` is the scalar PlusMinus spelling. A literal Unicode `±` is not recognized
by this byte dispatch. `++` maps to concatenation, not postfix increment.
Recognition of `?`, `...`, `<-`, `+-`, `<<=`, or `&=` does not imply an
expression production consumes it. Unknown bytes emit Error tokens and advance
one byte, rather than becoming identifier characters. A single quote currently
has no scalar character-literal branch.

`LEX-008` (**specified-contract**, capacity refusals). The file-input route
refuses source larger than 16,777,216 bytes and a token stream that cannot fit
its EOF token in the 2,097,152-entry table. `lexer_push_flat`,
`lexer_mark_token_overflow`, and `lex_file_parse_items` propagate the failure
as E229 instead of passing a truncated stream to the parser. These are
implementation limits, not abstract language size limits. The separate
`cursor_set_source` helper accepts a 2,097,152-byte input array and clamps its
length; that helper's boundary must not be substituted for the file-loader
boundary. `scripts/ci/token_table_ceiling_gate.sh` contains generated positive
prefix/overflow controls but uses Python generation; it was inspected only and
is not an ADR009-compatible execution recipe as written.

## 3. Literals: scanner acceptance versus value construction

`LEX-009` (**source-observed**, decimal and hexadecimal integers). The scalar
loop starts numeric scanning only on an ASCII digit. Decimal digits accumulate
into an `i64` token payload. `0x` and `0X` select hexadecimal accumulation.
An underscore is consumed inside a digit run only if the following byte is a
digit of that run's base. Thus `1_000` and `0x1_0` are single numeric tokens;
`1__0` and `1_` end the first token before the first underscore. No scalar
validation requires at least one hexadecimal digit after `0x`. No explicit
integer overflow check is present in this accumulator. Empty base prefixes and
oversized literals are defects/coverage gaps, not specified values or a promise
of modular arithmetic.

`LEX-010` (**source-observed**, decimal floating syntax). After a decimal run,
`.` joins the token only when followed by an ASCII digit. A decimal exponent
`e` or `E`, an optional sign, and at least one digit joins the same token;
underscores before digits are admitted within its digit runs. `1.0`, `1e3`,
and `1.0e-3` are float spellings; `1..3` leaves the range operator available.
`.5`, `1.`, `1e`, and `1e+` are not single float tokens. The last two split
rather than immediately producing an invalid-exponent diagnostic. A leading
minus is an expression operator, not part of a numeric token.

`LEX-011` (**source-observed**, extended numeric spellings and conversion).
The live scalar loop has no binary/octal-prefix or ordinary numeric-suffix
consumer: `0b101`, `0o17`, and `12i64` split into numeric and word tokens. The
older `scan_number`/`skip_numeric_suffix` and `parse_numeric` helpers support
more forms but are not reached by that loop. A blanket claim that all of their
literal grammar is implemented would be false.

C99-style hexadecimal floats are reconstructed by
`parser_hex_float_tail_ahead` and `parse_c99_hex_float_literal` from several
tokens. For example, `0x1.8p+0` need not be one FloatLit: the leading hex token,
fraction tokens, `p`/`P` exponent text, and exponent sign/digits are consumed by
the parser, and the original consumed byte interval is passed to
`f64_hex_literal_from_source` and `f128_literal_from_source`. Fused exponent
identifiers and glued suffix text have special handling; an identifier on the
following line is not consumed as a glued suffix. This is not permission to
attach arbitrary suffixes to decimal literals. Whitespace acceptance inside
these reconstructed forms needs explicit negative coverage; the companion
grammar describes a contiguous spelling and does not certify every token-tail
combination accepted by the implementation.

For decimal FloatLit, `Parser::current` uses `parser_parse_float_token`;
`lexer::numparse::num_parse_float` is not the authoritative AST value path.
The parser uses exact-power fast paths and `pfl_slow` outside their window.
The significand is accumulated in `i64` and excess digits are discarded before
that slow path. Accordingly, comments saying “correctly rounded” do not alone
establish correct rounding for arbitrary-length decimal input. Huge exponent
accumulation also precedes the final exponent clamp. The f128 literal parser
separately records limb identity and exactness from source. Acceptance and
lowering of `f128`/`f256`, and fidelity of their values, belong to the numeric
and execution chapters; older claims that both names are universally reserved
must not override the current literal and softfloat paths. No numeric values
were recalculated for this audit.

`LEX-012` (**source-observed**, strings and escape processing). The scalar
scanner begins a StringLit at `"`, skips a backslash and its following byte
together, and consumes the closing quote if present. It neither validates
escapes nor changes kind on EOF. It also advances columns, not lines, for raw
LF within a string. `parse_string_literal` stores the quoted source spelling
in `Expr.name`; it does not use the legacy scanner's decoded text.
`ir_normalize_string_literal_name` subsequently strips a matched quote pair
and expands `\n`, `\t`, `\r`, `\"`, and `\\`. Its fallback copies the byte
after any other backslash, dropping the slash. Consequently a `\q` diagnostic,
Unicode escape decoding, NUL escape decoding, and unterminated-string refusal
cannot be promised from this implementation. Acceptance of malformed strings
and lossy unknown escapes is a defect. The legacy char scanner and CharLit AST
branch do not make `'a'` executable syntax on the active scalar route.

`LEX-013` (**specified-contract**, string size honesty). `current_name` records
a string spelling longer than 384 bytes, including its quote bytes, in the
truncated-literal counters; `parse_items_loop` turns this into E258. The
intended boundary is refusal, not silently shortened output. For a normally
closed string, at most 382 spelling bytes fit between quotes; escape pairs
consume spelling capacity before decoding. This is a byte limit, not a count
of Unicode characters. `string_literal_truncation_gate.sh` names 127-, 128-,
129-, and 200-byte positive witnesses plus an oversize negative; imported
literal overflow has a separate gate. These scripts were not executed and
some use Python-based measurement, which needs an ADR009-compliant substitute
before a fresh numerical verification.

## 4. Expression construction and precedence

`SYN-001` (**source-observed**, precedence and associativity). Expressions use
`Parser::parse_expr_bp`. It parses a prefix/atom, then continues while the next
operator's binding power exceeds the current minimum. Standard binary RHS
parsing uses the operator's own power, producing left-associative trees. From
weakest to strongest:

| Power | Operators or operation | Construction |
|---|---|---|
| 2 | `||` | Binary logical-or node |
| 3 | `&&` | Binary logical-and node |
| 4 | `== !=` | Binary equality node |
| 5 | `< <= > >=` | Binary comparison node |
| 6 | `\|` | Bitwise-or node |
| 7 | `^` | Bitwise-xor node |
| 8 | `&` | Bitwise-and node |
| 9 | `<< >>` | Shift node |
| 10 | `.. ..=` | Range node, optional right endpoint |
| 11 | `+ - ++` | Add/subtract/concatenate node |
| 12 | `* / %` | Multiply/divide/remainder node |
| 13 | `as Type`; prefix `- ! ~ & &! *` | Cast; unary operand parsed with minimum 13 |
| Postfix | calls, indexing, `.field`, `.method(...)`, `audit name` | Handled directly before the power test |

This ordering makes ranges bind tighter than shifts and comparisons; it should
not be replaced by another language's familiar table. There is no exponentiation
operator `**`; `^` is xor in expressions and has a separate role in unit types.
`-x as T` becomes a cast of the unary expression under the strict power test.
Calls/indexing/dot are handled before that test, rather than solely by the
table's nominal power 14. Parentheses preserve the contained expression tree.
Comparison chaining parses as binary nesting, not mathematical chained
comparison semantics. The range branch is commented “non-associative” but does
not reject a second range at the same outer level; its actual loop permits
left nesting. Choosing rejection or assigning meaning to chained ranges is an
open language decision; no mathematical range chaining contract is specified.
The optional range endpoint additionally uses `Parser::at_expr_start` before
parsing at minimum binding power 10. That predicate omits some starters that
the prefix parser otherwise accepts, so endpoint recognition has the same
current-source defect described under `SYN-012`; an absent endpoint is not
evidence that the following token could never start an expression.

`SYN-002` (**specified-contract**, operator-position refusal). Assignment is
outside expression precedence. Only `= += -= *= /= %=` are recognized by
`parser_assign_op_code` and mapped by `assign_op_from_code`. Tokens such as
`&=` and `<<=` are lexed but do not acquire assignment semantics from this
parser. `!` and `~` are prefix-only: after an operand on the same line they
produce an explicit parse error, not a fallback addition. `token_is_binop`
guards the residual `token_to_binop` fallback. Rust-style `let mut` and type
`&mut T` set errors E040 and E041; tightly adjacent `ident!(...)` and
`path::ident!(...)` set E043. Recovery may consume them and form a placeholder
AST; recovery is not successful acceptance. Fixture diagnostic wording can
differ from the current source and needs fresh verification.

`SYN-003` (**source-observed**, newline-sensitive continuation). The lexer
discards LF tokens, but `had_newline_before` compares adjacent token line
metadata. Therefore the following distinctions survive trivia removal:

| Boundary | Parser treatment |
|---|---|
| `callee` then LF then `(` | Starts another expression; not a call |
| `base` then LF then `[` | Starts another expression; not an index |
| `.method` then LF then `(` | Field access followed by another expression |
| Dot after LF | Dot postfix is still considered; no matching LF prohibition |
| Unparenthesized `if`, `match`, `while`, `loop`, block, or `for` then LF then infix | Stops the block-like expression before that infix |
| Parenthesized block-like atom then LF then `+` | Can continue as a binary expression |
| Completed expression then LF then tight `*x`, `&x`, or `&!x` | Starts a prefix expression; tightness uses adjacent byte endpoints |
| Completed expression then LF then spaced `* x` or `& x` | May continue the previous binary expression |
| `!` or `~` after LF | Starts a new prefix expression, regardless of spacing |

Ordinary newline-led `+`/`-` after a non-block expression can continue it.
An operator before LF can take its operand on the next line. Comments can
affect this through line metadata and byte adjacency: a comment is not the
same as zero bytes between operator and operand. The EBNF must be read with
these predicates, not as a whitespace-insensitive complete recognition grammar.

`SYN-004` (**source-observed**, atoms, postfixes, and aggregate values). Atoms
include integer/float/string/bool literals, identifiers and `::` paths, grouped
expressions, tuples, arrays, blocks, the control forms below, closures, and
specialized forms dispatched by `parse_prefix`. `()` is an empty tuple value;
`(e)` groups; `(e,)` constructs a one-element tuple. Arrays accept `[]`,
`[e, ...]`, and `[value; count]`. Struct expressions use `Path { field: expr,
field_shorthand }`; shorthand creates an identifier expression. Ordinary
lists retain source order through reversal of parser cons-lists. Dot access
stores a Name, including a numeric tuple projection such as `.0`; parsing does
not validate every field/index spelling. Calls and methods admit optional
`::<T, U>` type arguments. Named call labels use `name: expr` or `name = expr`
when the label token is Ident; storing a label does not by itself specify
argument reordering, default arguments, or overload resolution.

The parser is more permissive than the conventional comma grammar in some
places: tuple continuation after its first comma, struct fields/literals,
enum variant loops, and match arms can consume repeated entries without a
comma when the next form parses unambiguously. The EBNF deliberately uses
commas consistently in those lists. Neither that conservative subset nor
accidental comma omission is a maintainer ruling about all separators.

## 5. Items, types, patterns, and blocks

`SYN-005` (**source-observed**, function declarations). The core shape is
`[pub...] [kernel] fn name [<parameters>] (parameters) [-> Type]
[with effects] Block`. An ordinary parameter is `name: Type`; `self` may omit
its annotation or have `self: Type`. There is no parsed `&self` receiver in
`parse_param`. At top-level item dispatch, `const fn` and `async fn` are
accepted by skipping the modifier;
`FnDef` retains `is_kernel` and `is_extern`, but no corresponding const/async
flag. This does not establish compile-time function evaluation or an async
function ABI. A post-body `with` is also consumed; if both effect lists occur,
the pre-body list is kept and the post-body list discarded. Duplicate lists
are an underspecified acceptance, not a sanctioned way to override effects.
`parse_type_param_header` consumes function parameter names/annotations without
retaining a function generic-parameter list in `FnDef`; specialization support
must be established from its own path rather than inferred from this syntax.

`SYN-006` (**source-observed**, data and implementation items). Structs have
named fields with optional bare `pub`; optional type parameters retain names,
while parsed `const` markers/annotations are not fully represented by
`TypeParamDef`. `linear` and `affine` set struct flags. Contextual `relevant`
before `struct` is consumed without a distinct relevance flag. Enums parse a
name followed directly by `{`, unit variants, named-field variants, and
tuple-payload variants;
the inspected `parse_enum_item` has no enum generic-header branch. `impl Type`
and `impl Trait for Type` parse method blocks; this is not evidence for
`impl<T>` headers. After optional method visibility, `parse_impl_item` requires
Fn directly and calls `parse_fn_item` with `is_kernel = false`. It does not
reuse top-level `parse_item` modifier dispatch. Accordingly `async fn`,
`const fn`, and `kernel fn` inside an impl set parser error state even though
those forms have top-level item routes. The core grammar therefore uses
`function-header, block` for an impl method, without `function-modifier`.
Whether any such method modifiers should be supported is an unresolved language
choice; the grammar correction does not grant them new semantics or change
the compiler. Traits parse a name and method signatures with optional
default bodies; `TraitItemDef.supertraits` is stored as None. Type aliases use
`type name = Type`, optionally followed by `;`, with no generic-header branch.
Top-level `let`/`const` and `var` are distinct item paths; global initializer
folding/deferment is implementation behavior outside a pure syntax grammar.
User-defined enum payload construction and payload subpatterns have E264
fail-closed branches in `check/check.sio`; the payload grammar must not be
advertised as general executable algebraic-data-type support. Built-in
Option-like constructor handling is a separate path.

`SYN-007` (**source-observed**, core types). `parse_type` recognizes named
paths with type arguments; `&T` and `&!T`; raw `*T`, `*const T`, and `*mut T`;
`[T]` and `[T; expression]`; unit/tuple/grouped types; `!`; `Self`; `_`;
prefix `?T` option shorthand; and `fn(T, U) [-> R] [with effects]`.
Raw-pointer `mut` is accepted and is distinct from the rejected `&mut` spelling.
`parse_optional_return_type` recognizes `->`; there is no inferred alternate
colon-return notation. The type parser accepts syntax before the checker
decides width, array-size constness, pointer legality, or instantiation.
`Parser::expect(Gt)` can split a `>>` token with `pending_gt`, preserving shift
tokens in expression contexts. Merely spacing generic closers differently
must not be used to hide this distinction.

`SYN-008` (**source-observed**, specialized type grammar boundaries). Unit
names can form `m/s`, products, parenthesized factors, and positive integer
powers through `parse_named_type`, `parse_unit_factor`, and
`parse_unit_pow_suffix`; primitive heads are excluded from this chain so that
`x as f64 / y` remains an arithmetic expression. Numeric generic heads such as
`f64<m/s>` use type arguments. `Knowledge` has a dedicated annotation parser,
and `IndepKnowledge<T,A,B | Z>` has head-specific arguments; neither is an
ordinary comma-only generic head. Dedicated epistemic wrapper heads, session
channel types, and policy types dispatch to their own branches. The core EBNF
excludes these extensions; their omission is a profile limit, not a claim that
they are absent.

Refinement parsing in `parse_refinement_type` is restricted: a binder/base,
one comparison with a decimal integer/float literal, and at most one additional
comparison joined by `&&` or `||` are represented directly. It is not a general
predicate-expression parser. Unknown operators use an unrecognized marker,
and the remaining token-skipping/recovery must not be interpreted as a proof
of the user's intended predicate. The companion grammar includes only a
conservative positive comparison subset. Logical soundness and admissibility
belong to the refinement chapter.

`SYN-009` (**source-observed**, pattern syntax and representation). Pattern
alternation uses `|`; atoms include `_`, booleans, integer/string patterns,
negative integer syntax, tuples, bindings, qualified enum variants, constructor
patterns, struct field patterns, and `var name`. A bare `None` is explicitly
PatEnum, rather than a binding that matches every value; other bare names
ordinarily bind. Constructor patterns may be bare `Some(x)` or path-qualified.
Tuple patterns always construct PatTuple, even `(p)` without a comma. Struct
patterns support `field` shorthand and `field: pattern`; no general rest-pattern
production is present. Negative-pattern parsing consumes the following token's
integer payload without a strict integer-kind check, a rejection-coverage gap.
Character and floating patterns are not generally provided by this dispatcher.

`SYN-010` (**source-observed**, destructuring boundary). `let pattern = expr`
is parsed, but Stmt stores one name, not an arbitrary pattern. Tuple and struct
lets desugar to a generated `__tupN` temporary and projection lets queued in
`TPLET_PENDING`. `tplet_emit_elems` and `tplet_emit_struct_fields` emit only
direct PatBinding children; nested subpatterns are skipped rather than
recursively destructured. Thus full recursive destructuring cannot be promised
from acceptance by `parse_pattern`. Wildcard/nonbinding lets and generated-name
hygiene also need semantics beyond the recognition grammar. The arity-mismatch
fixture expresses a checker boundary; this chapter does not claim that parser
desugaring alone enforces it.

`SYN-011` (**source-observed**, blocks and statements). Blocks contain let
bindings, var declarations, expression statements, and assignment statements.
`var name [: Type] [= expr]` permits omitted initialization syntactically;
definite initialization is a checker question. The parser does not require LF
or `;` between statements when the end of one expression and the beginning of
the next are structurally recognizable. `parse_block` consumes zero or more
semicolons after a statement; `1;;` is therefore accepted. Leading empty
statements are not equivalently skipped in that loop. Top-level item parsing
does skip semicolons before items. Local nested item declarations have no
`parse_stmt` branch. Assignment parsing stores an expression target, but does
not make every expression a valid lvalue.

`SYN-012` (**source-observed**, returns and block values). `parse_return_expr`
and `parse_break_expr` take an optional following expression whenever
`at_expr_start` succeeds, without a newline prohibition. `return` followed by
LF and `x` can therefore return `x`; use an explicit separator when no value
is intended. A recognized starter forces the operand parse at that decision;
the optional production is not permission to choose a bare return arbitrarily.
When the predicate is false, that routine attaches no operand and leaves the
next token for surrounding Pratt/statement parsing. The latter can form another
statement, interpret an infix operator, or report an error; a clean rejection
of the intended return value is not guaranteed.

The exact positive token-kind set in `self-hosted/parser/parser.sio` is:

| Starter family | Token kinds accepted by `at_expr_start` |
|---|---|
| Literal-like | `IntLit FloatLit StringLit CharLit True False` |
| Names handled explicitly | `Ident Knowledge Close Unit Kernel SelfLower` |
| Delimiters | `LParen LBracket LBrace` |
| Prefix operators | `Minus Bang Amp AmpBang Star` |
| Control | `If Match While Loop Return Break Continue` |
| Extensions | `ContestLower Handle Async Spawn Perform` |
| Contextual names | Every kind accepted by `tk_is_contextual_ident`, enumerated under `LEX-006` |

**Current defect, not approved design:** this predicate is narrower than
`parse_prefix`. It omits `HexLit`, `BinLit`, `OctLit`, `Tilde`, `Pipe`,
`PipePipe`, `For`, and `Var`, although the prefix dispatcher has branches
accepting those kinds. The active scalar lexer emits HexLit but not BinLit or
OctLit for base-prefixed numbers, so token availability and the starter mismatch
are separate facts. CharLit's presence in the positive set likewise does not
create a character-literal branch in that lexer.

Consequently `return 0x10` does not attach the hex token as an operand at this
decision, while `return (0x10)` starts with LParen and enters operand parsing.
`return ~value`, a directly returned closure, and a directly returned `for`
form have corresponding starter mismatches. The same predicate controls
`parse_break_expr` and the range branch in `parse_expr_bp`; for example,
`start..0x10` does not attach the HexLit endpoint there. Parenthesizing an
operand changes the starter decision, but does not establish static validity,
runtime support, or equivalence of the resulting program. These are
source-predicted boundaries, not executed fixture results.

**Unresolved candidate design:** align operand eligibility with the prefix
parser's expression starters, subject to the surrounding semantic context.
This is a candidate correction requiring a language decision and discriminating
AST/error witnesses; the present source-profile grammar documents the current
predicate instead of silently adopting that broader rule. The candidate does
not decide loop break-value legality or make every expression a valid range
endpoint. Dedicated direct/parenthesized and same-line/newline pairs remain
missing for the omitted starters.

Continue has no value parser. Blocks retain statements as an
ordered list and have no separate tail-expression field. Semicolons consumed
by `parse_block` are not retained as a “discard this value” marker. Thus Rust's
rule that a trailing semicolon turns a tail value into unit cannot simply be
imported: `{ 1 }` and `{ 1; }` have the same relevant statement structure.
`lower_block_ref` inspects the last statement and `lower_stmt_list_ref` tracks
its last register; empty blocks emit unit. Exact return-type compatibility and
control-flow reachability remain checker/lowering obligations. This chapter
records structure, not a guarantee that every backend implements all tails.

`SYN-013` (**source-observed**, core control forms). If uses an expression
condition, a block, and optional `else if`/`else` block. While uses a condition
and block; loop uses a block; for uses a single binding name, `in`, an iterator
expression, and block, not a general for-pattern or C-style three-clause loop.
Match has a scrutinee and arms `pattern [if guard] => body`; a body may be a
block, expression, or a parsed assignment wrapped as a block. Arms remain in
source order. If-let desugars to match with a wildcard else; while-let desugars
to loop/match with a failure break. Boolean typing, match exhaustiveness,
overlapping alternatives, and break-value compatibility are not grammar rules.
The parser sets `no_struct_lit` while reading conditions, iterators,
scrutinees, and guards to avoid confusing a following `{` with a struct value.
This flag propagates into recursive parsing; do not promise that parentheses
alone re-enable struct literals there without a witness.

`SYN-014` (**source-observed**, extension expressions). Closures accept
`|name [: Type], ...| [-> Type] [with effects] body` or `||` with the same
optional annotations; omitted parameter types are TypeInfer. `handle<...>`,
`perform`, `async`/`spawn` blocks, `contest`, and postfix `audit` have dedicated
parse paths. Parsed asynchronous blocks do not establish scheduling, suspension,
or await semantics; `await` has a token but no corresponding prefix branch in
the inspected dispatcher. Likewise recognizing handlers does not establish
effect-handler execution. The EBNF includes closures but leaves these other
extensions to their chapters. It excludes speculative postfix `?` and `!`.

`SYN-015` (**source-observed**, effects and foreign syntax). A core effect list
is `with Name [(integer)] (, Name [(integer)])*`. `parse_effect_payload` retains
a single nonnegative IntLit payload. Other balanced payload forms can be
consumed without retaining their meaning; a leading minus sets an error.
This is not a complete grammar for parameterized effects. `extern` accepts an
optional ABI string followed by either a bodyless function or a block of
function declarations, using reentrant extern-block state. `export` accepts
an optional ABI string and a function, or delegates to extern. These strings
are not a promise of arbitrary ABI support. Contextual `from "library"` is
also consumed by the parser; the inspected code advances before reading the
would-be library name, so its recorded identity needs a specific audit. The
core grammar excludes `from` pending that audit. Availability of native symbols,
FFI forwarding, and loader behavior are target-specific, outside syntax.

`SYN-016` (**open-decision**, evaluation order). Ordered AST lists and
left-associated binary trees establish grouping, not a universal order of
effects. `lower_stmt_list_ref` walks statements in source order; specialized
call, argument, aggregate, assignment, logical-operator, optimizer, and target
paths still need a joint contract for evaluation count and sequencing. In
particular this chapter does not infer short-circuit evaluation from the
spellings `&&`/`||`. More specifically, the ordinary
`lower_binary_expr_ref` path lowers the left operand and then the right operand
before emitting the operation; there is no logical-RHS laziness guard before
those two calls. `lower_expr_args_masked_borrow_ref` also visits argument-list
heads in source order. These are source observations about those paths, not
an approved eager-logical design or a measured guarantee that optimization
and all targets preserve that order. A lazy-logical contract would need a
compiler repair and side-effect witnesses. Required decisions/witnesses include callee versus argument
order; receiver/index versus assignment RHS; left versus right binary operands;
field initialization order; lazy logical RHS; and single evaluation of repeat
array values. `fo_product_rule_call_evaluated_once.sio` is a useful scoped
witness for a specialized call path, not evidence for all these cases.
Candidate future wording should preserve source grouping and require explicit
sequencing tests before assigning observable effect order across backends.

## 6. Imports, module identity, and visibility

`MOD-001` (**source-observed**, item shapes). `module path`, `use path`, and
`import path` are item forms; import delegates to `parse_use_item`. Paths use
`::`. Use/import supports a whole path, `path::*`, or `path::{name, ...}`.
The selected list is flat names, not recursive import trees. No alias `as`
branch is present. A semicolon is optional. A module declaration consumes its
path structurally and produces an unnamed ItemUnit; it does not establish the
loaded Program's module identity. Keyword path segments are allowed after
`::`; module declaration parsing explicitly admits keywords at segment
positions. There is no inline `module name { items }` namespace-body grammar
here. The broad path parser's permissive first-token consumption is recovery/
validation behavior, not a reason to admit punctuation as module names.

`MOD-002` (**source-observed**, filesystem search). AST import paths become
slash-separated `.sio` relative paths through
`module_frontend_ast_import_relpath`. Leading `super/` components move the base
directory upward before search. For a resulting relative path, the search in
`module_frontend_resolve_import_relpath` is ordered:

1. Importing file's directory.
2. Successive ancestor directories, bounded by eight iterations.
3. The relative path as supplied, against the process working directory.
4. `SOUNIO_STDLIB_PATH`, when nonempty.
5. Working-directory `stdlib/`.
6. Working-directory `self-hosted/`.
7. Working-directory `packages/`: first namespace component maps underscores
   to hyphens; a lone package uses `src/lib.sio`, a longer path uses `src/...`.
8. Directory-module roots, only after every file candidate above has missed,
   so a path that names a `.sio` file keeps that file: `X.sio` is retried as
   `X/mod.sio` against the importing file's directory (single-segment paths
   only), the working directory, `SOUNIO_STDLIB_PATH` and `stdlib/` (the two
   stdlib roots also as `X/lib.sio`, after `X/mod.sio`), `self-hosted/`, and
   the package path.

The first existing file wins. Failure returns the initial local candidate for
diagnosis. This is environment-sensitive source behavior, not a hermetic package
resolution standard. Step 8 is the language decision for directory modules:
`use json::*` loads `stdlib/json/mod.sio`, and a directory with only `lib.sio`
(`stdlib/cmp/`) loads that, as `lean_single` always has (its import loader
tries `X.sio`, `X/mod.sio`, `X/lib.sio` for stdlib roots). Madaros used to try
`/mod.sio` only on the package candidate, so every stdlib directory module was
an unresolved import (`tests/run-pass/use_stdlib_directory_module_root.sio`).
The one ordering difference: `lean_single` tries a directory root right after
the same location's file, Madaros only after all file candidates; the two pick
different files only when `X.sio` exists at a later candidate and `X/mod.sio`
or `X/lib.sio` at an earlier one. `crate`/`self` keyword acceptance in paths
does not establish Rust-like root rewrites in this search.

`MOD-003` (**source-observed**, named-item fallback). `use a::b::f` is ambiguous
between a module file and a named item. `resolve_import_file_path` first tries
the full file path, then drops the terminal segment and retries the parent
file. A full module hit wins. On parent success the frontend can record the
terminal as a named item for checker/lowerer use; it first avoids recording
that function-oriented marker for a terminal detected as a struct/enum type.
The named-item registry is keyed by path text globally, not by importing
directory/resolution context. Its source comment explicitly records the
remaining ambiguity when the same text resolves differently from two callers.
This is a known residual defect; the specification does not license context
leakage or silent renaming of imported types.

`MOD-004` (**source-observed**, closure and resolution phases).
`extract_imports_from_ast` collects ItemUse paths, with a 128-import capacity
and an overflow flag. `module_frontend_collect_ast_closure_into` records at
most 256 nodes, 512 edges, and 256 unresolved imports, setting saturation or
parse failure explicitly. It returns false on those conditions or unresolved
imports. A visited-path check prevents revisiting the same path string; that
does not define canonical realpath/symlink identity, cycle legality, initialization
order, or a topological module execution order. Separate source-scanning import
paths remain in the frontend and must not be treated as automatically equivalent
to AST extraction. The broader registry has different capacities; the smallest
active stage is the effective limit, not an aggregate maximum.

In `resolve_modules`, exports are collected before module bodies are resolved;
the resolver collects declarations before resolving references. `import_use`
defines the terminal module name for whole-module imports, imports public
exports for a glob, and validates requested exports for selective imports.
Compiler driver/checker/lowering paths additionally implement contextual
qualified-name lookup and private-function identity. The resolver algorithm's
presence and import graph loading alone do not prove every compile/run path
uses identical scope rules or that re-exported names are transitively visible.

`MOD-005` (**source-observed**, file identity). The loader actually used by
the frontend takes `file_path_to_module_path` from `compiler/module_parse.sio`,
not the duplicate implementation in `module_loader.sio`. It derives one
AstPath segment from the last parent-directory component plus file stem
(when available), using a bounded byte buffer. A textual `module` declaration
does not replace this identity. This improves on basename-only collisions but
does not constitute a canonical, arbitrarily deep namespace: paths ending in
the same parent/stem can still collide, and spelling/symlink normalization is
not established. A universal path-identity rule remains unresolved; this draft
does not make the current collision class a language feature.

`MOD-006` (**target-contract**, current restricted visibility).
`visibility_allows_access` in the modular checker implements:

| Visibility | Current checker predicate |
|---|---|
| Private/default | Defining and current module identities compare equal |
| `pub` | Allowed |
| `pub(crate)` | Allowed among all loaded modules; no distinct crate boundary |
| `pub(super)` | Currently treated as `pub(crate)` without a true parent hierarchy |
| `pub(path::to)` | Stored path is a prefix of current module identity |

The parser spells the last form `pub(path::to)`, not Rust's `pub(in path::to)`.
Struct fields store a separate bare-public flag. Cross-module private
function/type/field rejection has dedicated fixtures and gates, including
same-basename controls and generic-specialization cases. Their existence is
not a fresh result. `pub(super)`'s permissiveness and prefix checks against a
single synthetic file segment are implementation limitations; they do not
settle the intended namespace/security model. No package-private or transitive
public re-export guarantee is inferred from these predicates.

`MOD-007` (**open-decision**, module contract still to ratify). A complete
module standard needs canonical module identity, actual crate/package roots,
symbol alias/re-export rules, duplicate-definition and ambiguity rules,
missing-item diagnostics independent of file loading, cycle/initialization
semantics, reproducible search roots, and consistent enforcement through
specialization and target lowering. Existing private-symbol and ambiguity
fixtures are starting constraints, not authority to silently broaden access.
The current draft preserves source-observed rules and names unresolved choices
without selecting a divergent module design.

## 7. Recognition profile and grammar coverage

The companion EBNF is **CORE-MADAROS-2026-10**, a conservative reconstruction
for ordinary ASCII source and the audited core parse shapes. It is not generated
from the parser, machine-proved equivalent, a complete accepted-language grammar,
or a claim of executable conformance. It uses explicit separators even where
the parser's loops accept omission. It requires closed comments/strings and
known escapes rather than canonizing malformed-input acceptance. It excludes
keyword-as-name relaxation from its ordinary `identifier`, while documenting
path-segment exceptions. It also excludes specialized research DSLs and their
annotations, whose parsers remain part of the actual source.

To implement the core parser, retain source spelling and location metadata,
lex the maximal-munch tokens, implement the power table with the newline and
adjacency predicates, parse items/types/patterns separately, track synthetic
generic closers, preserve list order, desugar the explicitly bounded patterns,
and carry parse errors independently from recovery ASTs. Apply static checking
and target capability checks after recognition. Merely recognizing this EBNF
must not be advertised as implementing Sounio in full.

The following representation map is relevant when implementing an AST consumer
or replacing a parser component. These are source structures, not a stable
serialization or binary ABI:

| Repository-relative source | Symbols | Information retained or transformed |
|---|---|---|
| `self-hosted/lexer/span.sio` | `Span` | File identifier and source interval; location metadata is separate from semantic type identity |
| `self-hosted/lexer/token.sio` | `TokenKind`, `Token` | Token category, position, spelling buffer, integer/float payload fields; not interchangeable with flat scalar codes |
| `self-hosted/parser/parser.sio` | `Parser`, `PT_KIND_CODE`, `PT_START`, `PT_END`, `PT_LINE`, `PT_COL`, `PT_INT_VALUE` | Shared token tables, cursor state, error state, `pending_gt`, and struct-literal suppression |
| `self-hosted/parser/ast.sio` | `Name`, `StringList`, `AstPath`, `Program` | Byte names, linked path components/count, file module identity, and item list |
| `self-hosted/parser/ast.sio` | `Item`, `ItemKind`, `FnDef`, `StructDef`, `EnumDef`, `TypeParamDef` | Tagged item payloads; function bodies and retained struct generic names; consumed syntax is not always represented |
| `self-hosted/parser/ast.sio` | `Expr`, `ExprKind`, `ExprList`, `Block`, `Stmt`, `StmtKind` | Tagged expression payloads, ordered lists, block statement list, no distinct tail-semicolon flag |
| `self-hosted/parser/ast.sio` | `TypeExpr`, `TypeExprKind`, `Pattern`, `PatternKind`, `Visibility` | Type/pattern/visibility syntax; static validity is a later obligation |
| `self-hosted/parser/stmts.sio` | `TPLET_PENDING`, `TPLET_COUNTER`, `tplet_emit_elems` | Extra projection bindings produced by destructuring; not an arbitrary retained let-pattern tree |
| `self-hosted/parser/mod.sio` | `parse_items_loop`, `parse_items_preloaded` | Top-level recovery/error propagation; claims and causal graphs are excluded from the returned ordinary item stream after their metadata parsing |
| `self-hosted/compiler/module_parse.sio` | `ImportList`, `load_module_file`, `file_path_to_module_path` | Import extraction and file-derived identity used by the active frontend |
| `self-hosted/compiler/module_frontend.sio` | `ModuleClosure`, `module_frontend_collect_ast_closure_into` | Loaded dependency paths, edges, saturation, unresolved imports, and parse failure |
| `self-hosted/resolve/resolve.sio` | `Resolver`, `import_use`, `import_use_items` | Declaration collection, module/glob/selective import handling, and imported symbol bindings |
| `self-hosted/check/check.sio` | `visibility_allows_access` | Current modular-checker access predicate; not a complete hierarchical namespace model |
| `self-hosted/ir/lower.sio` | `lower_binary_expr_ref`, `lower_expr_args_masked_borrow_ref`, `lower_stmt_list_ref` | Ordinary operand/argument/statement lowering order; no universal sequencing or lazy-logical contract inferred |

| EBNF group | Implementation anchor | Coverage/status |
|---|---|---|
| identifier, trivia, integer/float/string | `lexer/mod.sio`: `lex_source_to_globals`; `lexer/cursor.sio`: classifiers | Conservative lexical subset; real malformed-input gaps documented above |
| hexadecimal-float | `parser/exprs.sio`: `parse_c99_hex_float_literal` | Contiguous spelling in grammar; implementation uses token-tail reconstruction |
| translation-unit, item | `parser/mod.sio`: `parse_items_loop`; `parser/items.sio`: `parse_item` | Extension item families intentionally excluded |
| function, struct, enum, impl, trait, alias, globals | `parser/items.sio`: corresponding `parse_*_item` methods | Syntax only; generic/trait semantic completeness not implied |
| type, type-arguments, refinement-type | `parser/types.sio`: `parse_type`, `parse_type_args`, `parse_refinement_type` | Conservative named/core/refinement subset; specialized heads excluded |
| pattern | `parser/patterns.sio`: `parse_pattern`, `parse_pattern_atom` | Character/float/rest patterns excluded; let desugaring is shallower than recognition |
| block, statement, assignment | `parser/stmts.sio`: `parse_block`, `parse_stmt`, `parse_expr_or_assign_stmt` | Semicolons are separators, not retained value-discard markers |
| expression precedence and postfix | `parser/exprs.sio`: `parse_expr_bp`, `parse_unary`, `parse_dot_expr_box` | Requires textual location predicates; a plain CFG is insufficient |
| if, match, while, for, return, break, closure | `parser/exprs.sio`: matching `parse_*` methods | Parser construction only; sequencing and backend capability remain separate |
| imports, module, visibility | `parser/items.sio`: `parse_use_item`, `parse_module_item`, `parse_pub_visibility` | Filesystem resolution and access checking are not EBNF productions |
| effects, extern/export | `parser/types.sio`: `parse_effect_list`; `parser/items.sio`: `parse_extern_fn_item` | Simple retained payload subset; ABI availability not implied |
| specialized extensions absent from EBNF | `parser/items.sio`: policy/study/ontology/algebra/session/claim/causal parsers; `parser/types.sio`: wrapper parsers | Deliberate grammar limit; no unsupported claim inferred |

## 8. Fixture and gate evidence map

All paths are repository-relative. “Positive” and “negative” identify expected
roles, not results. The TSV gives one row for every rule definition; this table
adds the boundary each fixture or gate discriminates.

| Area | Positive or control | Negative/boundary | What remains to verify |
|---|---|---|---|
| Basic item syntax | `examples/hello.sio`; `tests/run-pass/parser_card_a_regression.sio` | `tests/ui/parser/fn_no_name.sio`; `tests/ui/parser/missing_brace.sio` | Full parser execution, expected diagnostics, and no-progress recovery |
| Names and contextual keywords | `tests/run-pass/contextual_keyword_idents.sio` | `tests/ui/parser/unicode_ident.sio` | Overlong names, BOM, keyword declaration/use symmetry, Unicode byte diagnostics |
| Comments | `self-hosted/test_lexer.sio` | `tests/ui/lexer/unterminated_block_comment.sio` | Top-level test helper may use another lexer; active nesting contradicts the old nested-comment negative |
| Float syntax and separators | `tests/run-pass/f128_v0e510_language_surface_closure.sio` | `tests/compile-fail/f128_inexact_literal_refused_on_lean_single.sio` | Separate Madaros/seed contracts; malformed exponents/prefixes; arbitrary long decimal rounding |
| Float rounding boundary | `tests/run-pass/float_literal_subnormal_half_ulp_2769.sio` | `tests/oracles/float_literal_subnormal_half_ulp_2769.cpp` | C++ oracle is a reference source, not an executed negative fixture; do not count it as a pass |
| Strings | `tests/run-pass/string_literal_len200_print.sio` | `tests/compile-fail/string_literal_oversize_e258.sio` | `scripts/ci/string_literal_truncation_gate.sh`; imported overflow gate; existing Python measurement requires ADR009-compliant replacement before execution |
| Malformed strings | No established current positive/negative pair | `tests/ui/lexer/invalid_escape.sio`; `tests/ui/lexer/unterminated_string.sio` | Source contradicts intended rejection; do not adopt fixture expectations as observations |
| Newline expression boundary | `tests/run-pass/paren_if_newline_operator_continuation.sio`; `tests/run-pass/prefix_bang_tilde_line_start_2249.sio` | `tests/compile-fail/infix_bang_refused_issue_2249.sio`; `tests/compile-fail/infix_tilde_refused_issue_2249.sio` | Calls/indexing/methods, comments, prefix spacing, and nested grouped blocks |
| Rust syntax rejection | `tests/run-pass/parser_card_a_raw_pointer_types.sio` | `tests/compile-fail/parser_rust_let_mut.sio`; `tests/compile-fail/parser_rust_mut_ref.sio`; `tests/compile-fail/parser_rust_macro_path_qualified.sio` | Exact diagnostic text, not filename matches; old macro comments preserve superseded spaced-bang behavior |
| Destructuring | `tests/run-pass/tuple_destructure_let.sio`; `tests/run-pass/parser_card_a_misc_patterns.sio` | `tests/compile-fail/tuple_destructure_arity_mismatch.sio` | Recursive subpattern refusal/support, temporary hygiene, and evaluation count |
| Control and closures | `tests/run-pass/if_let_desugar_forms.sio`; `tests/run-pass/closure_higher_order.sio` | `tests/compile-fail/madaros_closure_inferred_numeric_rejects_bool.sio` | Guards, exhaustiveness, loop break values, closure capability limits |
| Imports | `tests/conformance/import_public.sio` | `tests/ui/module/use_nonexistent.sio` | Manifest `modules-import-check` is check-only and does not validate runtime or all resolver search roots |
| Named path disambiguation | `tests/run-pass/madaros_module_qualified_call_named_item_module_collision.sio` | `tests/compile-fail/madaros_module_name_ambiguous_import.sio` | Context-sensitive named-item registry key; same textual path from two importing directories |
| Private access | `tests/run-pass/capability_public_api_cross_module_1386.sio` | `tests/compile-fail/private_field_read_cross_module_1386.sio`; `tests/compile-fail/madaros_module_qualified_call_private_fn.sio` | `scripts/ci/madaros_kl4_field_visibility_gate.sh`; specialized generic controls and target consistency |
| Module identity | `tests/multimodule/visibility_basename_collision_main.sio` | `tests/multimodule/visibility_fn_private_main.sio` | Deep same-parent/stem collision and path normalization are still missing contract coverage |
| Capacity | `scripts/ci/token_table_ceiling_gate.sh` | Same gate's source/token overflow witnesses | E229 clean refusal; generator is inspected only, not executed |

The bounded serious-language spine is
`tests/conformance/manifest.v1.tsv` with
`scripts/ci/serious_language_conformance_gate.sh`. Its documented scope is
bounded behavior, not whole-language certification. Broader fixture execution
routes through `scripts/run_sio_test_suite.sh`, which delegates to
`scripts/dev/run_sio_test_suite.sh`. No gate is newly closed by writing this
chapter, and no historical receipt is treated as a current-source result.

## 9. Outstanding acceptance work

Before this chapter can become a normative conformance chapter, maintainers
need decisions and current-source witnesses for: source encoding/Unicode;
name length and hygiene; malformed literal refusal; binary/octal and numeric
suffix support; arbitrary long decimal rounding; chained ranges/comparisons;
comma omission; tail-semicolon semantics; duplicate effects; recursive pattern
destructuring; effect sequencing/short circuiting; canonical module identity;
crate/restricted visibility; alias/re-export/cycle/initialization semantics;
and reproducible search roots. Each choice needs a positive case and a
discriminating negative or boundary case on the engines/targets it claims.

Implementation defects uncovered here remain evidence gaps for this
documentation-only write set. Compiler repairs, governance registration,
catalog integration, heavy validation, and language rulings are outside this
chapter's edit authority. This chapter intentionally makes none of those
completion claims.
