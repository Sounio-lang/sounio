# Sounio Grammar Reference

This document provides the formal grammar of the Sounio programming language in Extended Backus-Naur Form (EBNF).

## Notation

| Symbol | Meaning |
|--------|---------|
| `::=` | Definition |
| `\|` | Alternative |
| `( )` | Grouping |
| `[ ]` | Optional (0 or 1) |
| `{ }` | Repetition (0 or more) |
| `" "` | Terminal (literal token) |
| `< >` | Non-terminal |
| `(* *)` | Comment |

---

## Lexical Grammar

### Whitespace and Comments

```ebnf
<whitespace>     ::= " " | "\t" | "\r" | "\n" | "\f"
<line-comment>   ::= "//" { <any-char-except-newline> }
<block-comment>  ::= "/*" { <any-char> } "*/"
<doc-comment>    ::= "///" { <any-char-except-newline> }
                   | "//!" { <any-char-except-newline> }
                   | "/**" { <any-char> } "*/"
                   | "/*!" { <any-char> } "*/"
```

### Identifiers

```ebnf
<identifier>     ::= <ident-start> { <ident-continue> }
<ident-start>    ::= <letter> | "_"
<ident-continue> ::= <letter> | <digit> | "_"
<letter>         ::= "a".."z" | "A".."Z"
<digit>          ::= "0".."9"
```

### Literals

```ebnf
<literal>        ::= <int-literal>
                   | <float-literal>
                   | <string-literal>
                   | <char-literal>
                   | <bool-literal>
                   | <unit-literal>

<int-literal>    ::= <dec-literal>
                   | <hex-literal>
                   | <bin-literal>
                   | <oct-literal>

<dec-literal>    ::= <digit> { <digit> | "_" }
<hex-literal>    ::= "0x" <hex-digit> { <hex-digit> | "_" }
<bin-literal>    ::= "0b" <bin-digit> { <bin-digit> | "_" }
<oct-literal>    ::= "0o" <oct-digit> { <oct-digit> | "_" }

<hex-digit>      ::= <digit> | "a".."f" | "A".."F"
<bin-digit>      ::= "0" | "1"
<oct-digit>      ::= "0".."7"

<float-literal>  ::= <dec-literal> "." <dec-literal> [ <exponent> ]
                   | <dec-literal> <exponent>

<exponent>       ::= ( "e" | "E" ) [ "+" | "-" ] <dec-literal>

<string-literal> ::= '"' { <string-char> } '"'
<string-char>    ::= <any-char-except-quote-or-backslash>
                   | <escape-sequence>

<char-literal>   ::= "'" <char-char> "'"
<char-char>      ::= <any-char-except-quote-or-backslash>
                   | <escape-sequence>

<escape-sequence> ::= "\\" | "\'" | '\"' | "\n" | "\r" | "\t"
                    | "\0" | "\x" <hex-digit> <hex-digit>
                    | "\u{" <hex-digit> { <hex-digit> } "}"

<bool-literal>   ::= "true" | "false"

<unit-literal>   ::= <int-literal> "_" <unit-suffix>
                   | <float-literal> "_" <unit-suffix>

<unit-suffix>    ::= <identifier> { "/" <identifier> }
```

---

## Module Level

### Source File

```ebnf
<source-file>    ::= [ <module-decl> ] { <item> }

<module-decl>    ::= "module" <path> [ ";" ]
```

### Items

```ebnf
<item>           ::= [ <visibility> ] <item-kind>

<item-kind>      ::= <fn-item>
                   | <struct-item>
                   | <enum-item>
                   | <trait-item>
                   | <impl-item>
                   | <type-alias>
                   | <const-item>
                   | <static-item>
                   | <import-item>
                   | <effect-item>
                   | <handler-item>
                   | <ode-item>
                   | <pde-item>
                   | <causal-item>
                   | <ontology-item>
                   | <extern-block>

<visibility>     ::= "pub"
```

---

## Functions

```ebnf
<fn-item>        ::= [ "async" ] [ "kernel" ] "fn" <identifier>
                     [ <generic-params> ]
                     "(" [ <fn-params> ] ")"
                     [ "->" <type> ]
                     [ <effect-clause> ]
                     [ <where-clause> ]
                     <block>

<fn-params>      ::= <fn-param> { "," <fn-param> } [ "," ]

<fn-param>       ::= <pattern> ":" <type>

<effect-clause>  ::= "with" <effect-list>

<effect-list>    ::= <type> { "," <type> }
```

---

## Types

### Type Expressions

```ebnf
<type>           ::= <path-type>
                   | <reference-type>
                   | <pointer-type>
                   | <array-type>
                   | <slice-type>
                   | <tuple-type>
                   | <fn-type>
                   | <never-type>
                   | <unit-type>
                   | <infer-type>
                   | <refinement-type>

<path-type>      ::= <path> [ <generic-args> ]

<reference-type> ::= "&" [ "!" ] <type>

<pointer-type>   ::= "*" ( "const" | "mut" ) <type>

<array-type>     ::= "[" <type> ";" <expr> "]"

<slice-type>     ::= "[" <type> "]"

<tuple-type>     ::= "(" [ <type> { "," <type> } [ "," ] ] ")"

<fn-type>        ::= "fn" "(" [ <type-list> ] ")" [ "->" <type> ]
                     [ <effect-clause> ]

<type-list>      ::= <type> { "," <type> } [ "," ]

<never-type>     ::= "!"

<unit-type>      ::= "(" ")"

<infer-type>     ::= "_"

<refinement-type> ::= "{" <identifier> ":" <type> "|" <expr> "}"
```

### Generic Parameters

```ebnf
<generic-params> ::= "<" <generic-param> { "," <generic-param> } [ "," ] ">"

<generic-param>  ::= <identifier> [ ":" <type-bounds> ]
                   | <const-param>

<type-bounds>    ::= <type> { "+" <type> }

<const-param>    ::= "const" <identifier> ":" <type>

<generic-args>   ::= "<" <generic-arg> { "," <generic-arg> } [ "," ] ">"

<generic-arg>    ::= <type>
                   | <literal>
                   | "{" <expr> "}"

<where-clause>   ::= "where" <where-pred> { "," <where-pred> } [ "," ]

<where-pred>     ::= <type> ":" <type-bounds>
```

---

## Structs and Enums

### Struct Definitions

```ebnf
<struct-item>    ::= [ "linear" | "affine" ] "struct" <identifier>
                     [ <generic-params> ]
                     ( <struct-body> | ";" )

<struct-body>    ::= "{" [ <struct-fields> ] "}"

<struct-fields>  ::= <struct-field> { "," <struct-field> } [ "," ]

<struct-field>   ::= [ <visibility> ] <identifier> ":" <type>
                     [ "=" <expr> ]
```

### Enum Definitions

```ebnf
<enum-item>      ::= "enum" <identifier>
                     [ <generic-params> ]
                     "{" [ <enum-variants> ] "}"

<enum-variants>  ::= <enum-variant> { "," <enum-variant> } [ "," ]

<enum-variant>   ::= <identifier> [ <enum-variant-data> ]

<enum-variant-data> ::= "(" <type-list> ")"
                      | "{" <struct-fields> "}"
```

---

## Traits and Implementations

### Trait Definitions

```ebnf
<trait-item>     ::= "trait" <identifier>
                     [ <generic-params> ]
                     [ ":" <type-bounds> ]
                     [ <where-clause> ]
                     "{" { <trait-member> } "}"

<trait-member>   ::= <fn-sig> ";"
                   | <fn-item>
                   | <type-alias>
                   | <const-item>
```

### Implementations

```ebnf
<impl-item>      ::= "impl" [ <generic-params> ]
                     [ <type> "for" ] <type>
                     [ <where-clause> ]
                     "{" { <impl-member> } "}"

<impl-member>    ::= [ <visibility> ] <fn-item>
                   | <type-alias>
                   | <const-item>
```

---

## Effects and Handlers

### Effect Definitions

```ebnf
<effect-item>    ::= "effect" <identifier>
                     [ <generic-params> ]
                     "{" { <effect-op> } "}"

<effect-op>      ::= "fn" <identifier>
                     "(" [ <fn-params> ] ")"
                     [ "->" <type> ] ";"
```

### Handler Definitions

```ebnf
<handler-item>   ::= "handler" <identifier>
                     [ <generic-params> ]
                     "for" <type>
                     "{" { <handler-clause> } "}"

<handler-clause> ::= <identifier> "(" [ <pattern-list> ] ")"
                     "=>" <expr> ","
                   | "return" "(" <pattern> ")" "=>" <expr> ","
```

---

## Statements

```ebnf
<statement>      ::= <let-stmt>
                   | <expr-stmt>
                   | <item>

<let-stmt>       ::= ( "let" | "var" ) <pattern>
                     [ ":" <type> ]
                     [ "=" <expr> ] ";"

<expr-stmt>      ::= <expr-without-block> ";"
                   | <expr-with-block>
```

---

## Expressions

### Expression Categories

```ebnf
<expr>           ::= <expr-without-block>
                   | <expr-with-block>

<expr-without-block> ::= <literal-expr>
                       | <path-expr>
                       | <operator-expr>
                       | <call-expr>
                       | <method-expr>
                       | <field-expr>
                       | <index-expr>
                       | <range-expr>
                       | <cast-expr>
                       | <closure-expr>
                       | <return-expr>
                       | <break-expr>
                       | <continue-expr>
                       | <array-expr>
                       | <tuple-expr>
                       | <struct-expr>
                       | <grouped-expr>
                       | <await-expr>
                       | <perform-expr>

<expr-with-block>    ::= <block>
                       | <if-expr>
                       | <match-expr>
                       | <loop-expr>
                       | <while-expr>
                       | <for-expr>
                       | <handle-expr>
                       | <async-block>
                       | <spawn-block>
```

### Primary Expressions

```ebnf
<literal-expr>   ::= <literal>

<path-expr>      ::= <path>

<grouped-expr>   ::= "(" <expr> ")"

<array-expr>     ::= "[" [ <expr> { "," <expr> } [ "," ] ] "]"
                   | "[" <expr> ";" <expr> "]"

<tuple-expr>     ::= "(" [ <expr> "," [ <expr> { "," <expr> } ] ] [ "," ] ")"

<struct-expr>    ::= <path> "{" [ <struct-expr-fields> ] "}"

<struct-expr-fields> ::= <struct-expr-field> { "," <struct-expr-field> } [ "," ]

<struct-expr-field>  ::= <identifier> [ ":" <expr> ]
                       | ".." <expr>
```

### Operator Expressions

```ebnf
<operator-expr>  ::= <unary-expr>
                   | <binary-expr>

<unary-expr>     ::= <unary-op> <expr>

<unary-op>       ::= "-" | "!" | "~" | "&" | "&!" | "*"

<binary-expr>    ::= <expr> <binary-op> <expr>

<binary-op>      ::= "+" | "-" | "*" | "/" | "%"
                   | "&" | "|" | "^" | "<<" | ">>"
                   | "&&" | "||"
                   | "==" | "!=" | "<" | ">" | "<=" | ">="
                   | "=" | "+=" | "-=" | "*=" | "/=" | "%="
                   | "&=" | "|=" | "^=" | "<<=" | ">>="
                   | "++" | "+-"
```

### Call and Access Expressions

```ebnf
<call-expr>      ::= <expr> "(" [ <call-args> ] ")"

<call-args>      ::= <expr> { "," <expr> } [ "," ]

<method-expr>    ::= <expr> "." <identifier> [ <generic-args> ]
                     "(" [ <call-args> ] ")"

<field-expr>     ::= <expr> "." <identifier>
                   | <expr> "." <int-literal>

<index-expr>     ::= <expr> "[" <expr> "]"
                   | <expr> "[" [ <expr> ] ".." [ <expr> ] "]"
                   | <expr> "[" [ <expr> ] "..=" <expr> "]"
```

### Control Flow Expressions

```ebnf
<if-expr>        ::= "if" <expr> <block>
                     { "else" "if" <expr> <block> }
                     [ "else" <block> ]

<match-expr>     ::= "match" <expr> "{" { <match-arm> } "}"

<match-arm>      ::= <pattern> [ "if" <expr> ] "=>" <expr> ","

<loop-expr>      ::= "loop" <block>

<while-expr>     ::= "while" <expr> <block>

<for-expr>       ::= "for" <pattern> "in" <expr> <block>
```

### Range Expressions

```ebnf
<range-expr>     ::= [ <expr> ] ".." [ <expr> ]
                   | [ <expr> ] "..=" <expr>
```

### Other Expressions

```ebnf
<cast-expr>      ::= <expr> "as" <type>

<closure-expr>   ::= [ "move" ] "|" [ <closure-params> ] "|"
                     [ "->" <type> ]
                     ( <expr> | <block> )

<closure-params> ::= <closure-param> { "," <closure-param> } [ "," ]

<closure-param>  ::= <pattern> [ ":" <type> ]

<return-expr>    ::= "return" [ <expr> ]

<break-expr>     ::= "break" [ <expr> ]

<continue-expr>  ::= "continue"

<await-expr>     ::= <expr> "." "await"

<perform-expr>   ::= "perform" <path> "." <identifier>
                     "(" [ <call-args> ] ")"

<async-block>    ::= "async" <block>

<spawn-block>    ::= "spawn" <block>

<handle-expr>    ::= "handle" <block> "with" <handler-instance>
                     { "with" <handler-instance> }

<handler-instance> ::= <path> "{" [ <struct-expr-fields> ] "}"
```

### Blocks

```ebnf
<block>          ::= "{" { <statement> } [ <expr> ] "}"
```

---

## Patterns

```ebnf
<pattern>        ::= <literal-pattern>
                   | <ident-pattern>
                   | <wildcard-pattern>
                   | <rest-pattern>
                   | <ref-pattern>
                   | <struct-pattern>
                   | <tuple-struct-pattern>
                   | <tuple-pattern>
                   | <slice-pattern>
                   | <or-pattern>
                   | <range-pattern>
                   | <grouped-pattern>

<literal-pattern>    ::= <literal>

<ident-pattern>      ::= [ "mut" ] <identifier> [ "@" <pattern> ]

<wildcard-pattern>   ::= "_"

<rest-pattern>       ::= ".."

<ref-pattern>        ::= "&" [ "!" ] <pattern>

<struct-pattern>     ::= <path> "{" [ <field-patterns> ] "}"

<field-patterns>     ::= <field-pattern> { "," <field-pattern> } [ "," ]

<field-pattern>      ::= <identifier> [ ":" <pattern> ]
                       | ".."

<tuple-struct-pattern> ::= <path> "(" [ <pattern-list> ] ")"

<tuple-pattern>      ::= "(" [ <pattern-list> ] ")"

<pattern-list>       ::= <pattern> { "," <pattern> } [ "," ]

<slice-pattern>      ::= "[" [ <pattern-list> ] "]"

<or-pattern>         ::= <pattern> "|" <pattern>

<range-pattern>      ::= <literal> ".." <literal>
                       | <literal> "..=" <literal>

<grouped-pattern>    ::= "(" <pattern> ")"
```

---

## Paths

```ebnf
<path>           ::= [ "::" ] <path-segment> { "::" <path-segment> }

<path-segment>   ::= <identifier> [ <generic-args> ]
                   | "self"
                   | "Self"
                   | "super"
                   | "crate"
```

---

## Scientific DSL Items

### ODE Systems

```ebnf
<ode-item>       ::= "ode" <identifier>
                     [ <generic-params> ]
                     "{" { <ode-section> } "}"

<ode-section>    ::= "state" "{" <ode-vars> "}"
                   | "params" "{" <ode-vars> "}"
                   | "equations" "{" <ode-eqns> "}"
                   | "initial" "{" <ode-inits> "}"

<ode-vars>       ::= <identifier> ":" <type> { "," <identifier> ":" <type> } [ "," ]

<ode-eqns>       ::= <ode-eqn> { "," <ode-eqn> } [ "," ]

<ode-eqn>        ::= <deriv-expr> "=" <expr>

<deriv-expr>     ::= "d" <identifier> "/" "d" <identifier>

<ode-inits>      ::= <identifier> "=" <expr> { "," <identifier> "=" <expr> } [ "," ]
```

### PDE Systems

```ebnf
<pde-item>       ::= "pde" <identifier>
                     [ <generic-params> ]
                     "{" { <pde-section> } "}"

<pde-section>    ::= <ode-section>
                   | "domain" "{" <domain-specs> "}"
                   | "boundary" "{" <boundary-specs> "}"

<domain-specs>   ::= <domain-spec> { "," <domain-spec> } [ "," ]

<domain-spec>    ::= <identifier> ":" <expr> ".." <expr>

<boundary-specs> ::= <boundary-spec> { "," <boundary-spec> } [ "," ]

<boundary-spec>  ::= <expr> "=" <expr>
```

### Causal Models

```ebnf
<causal-item>    ::= "causal" <identifier>
                     [ <generic-params> ]
                     "{" { <causal-section> } "}"

<causal-section> ::= "nodes" "{" <causal-nodes> "}"
                   | "edges" "{" <causal-edges> "}"
                   | "equations" "{" <causal-eqns> "}"

<causal-nodes>   ::= <identifier> ":" <type> { "," <identifier> ":" <type> } [ "," ]

<causal-edges>   ::= <causal-edge> { "," <causal-edge> } [ "," ]

<causal-edge>    ::= <identifier> "->" <identifier>

<causal-eqns>    ::= <causal-eqn> { "," <causal-eqn> } [ "," ]

<causal-eqn>     ::= <identifier> "=" <expr>
```

---

## Imports and Exports

```ebnf
<import-item>    ::= "import" <import-tree> [ ";" ]
                   | "use" <import-tree> [ ";" ]

<import-tree>    ::= <path>
                   | <path> "::" "*"
                   | <path> "::" "{" <import-items> "}"

<import-items>   ::= <import-item-spec> { "," <import-item-spec> } [ "," ]

<import-item-spec> ::= <identifier> [ "as" <identifier> ]
                     | <import-tree>
```

---

## External Declarations

```ebnf
<extern-block>   ::= "extern" [ <abi> ] "{" { <extern-item> } "}"

<abi>            ::= <string-literal>

<extern-item>    ::= "fn" <identifier>
                     "(" [ <fn-params> ] ")"
                     [ "->" <type> ] ";"
                   | "static" [ "mut" ] <identifier> ":" <type> ";"
```

---

## Constants and Statics

```ebnf
<const-item>     ::= "const" <identifier> [ ":" <type> ] "=" <expr> ";"

<static-item>    ::= "static" [ "mut" ] <identifier> ":" <type>
                     [ "=" <expr> ] ";"
```

---

## Type Aliases

```ebnf
<type-alias>     ::= "type" <identifier>
                     [ <generic-params> ]
                     [ "=" <type> ] ";"
```

---

## Ontology Declarations

```ebnf
<ontology-item>  ::= "ontology" <string-literal>
                     "from" <string-literal>
                     [ "align" <align-spec> ]
                     ";"

<align-spec>     ::= <path> "with" <path>
```

---

## Attribute Syntax

```ebnf
<outer-attribute> ::= "#" "[" <attr-content> "]"

<inner-attribute> ::= "#" "!" "[" <attr-content> "]"

<attr-content>    ::= <identifier> [ "=" <literal> ]
                    | <identifier> "(" [ <attr-args> ] ")"

<attr-args>       ::= <attr-arg> { "," <attr-arg> } [ "," ]

<attr-arg>        ::= <literal>
                    | <identifier>
                    | <identifier> "=" <literal>
```

---

## Reserved Words

The following identifiers are reserved and cannot be used as user-defined names:

```
abstract   align      as         assert     assume     async
await      boundary   break      causal     const      continue
copy       counterfactual  device     distance   do         domain
drop       dual       edges      effect     else       enum
equations  ensures    export     extern     false      fn
for        from       gpu        grad       handle     handler
hessian    if         impl       import     in         infer
initial    Input      invariant  jacobian   kernel     Knowledge
let        linear     Literature loop       match      Measured
module     move       mut        nodes      observe    ode
ontology   params     pde        perform    proof      pub
query      quat       requires   resume     return     sample
Self       self       shared     Source     spawn      state
static     struct     threshold  tile       trait      true
type       unsafe     use        var        vec2       vec3
vec4       mat2       mat3       mat4       where      while
with       Computed   Derived    OntologyTerm  Quantity   Tensor
Valid      ValidUntil ValidWhile compat
```

---

## Grammar Notes

### Semicolon Insertion

Sounio uses explicit semicolons but allows them to be omitted in certain contexts:

1. After the last expression in a block (implicit return)
2. After block expressions (`if`, `match`, `loop`, etc.)
3. Before closing braces

### Expression vs Statement

An expression can be used as a statement by adding a semicolon. The final expression in a block without a semicolon becomes the block's value.

### Precedence Parsing

Binary operators are parsed using Pratt parsing with the precedence table defined in the [Operators Reference](./operators.md).
