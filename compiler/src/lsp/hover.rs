//! Hover information provider
//!
//! Provides type information and documentation on hover.

use tower_lsp::lsp_types::*;

use crate::resolve::{DefKind, SymbolTable};

/// Provider for hover information
pub struct HoverProvider;

impl HoverProvider {
    /// Create a new hover provider
    pub fn new() -> Self {
        Self
    }

    /// Get hover information for a symbol
    pub fn hover_for_symbol(
        &self,
        word: &str,
        range: Range,
        symbols: &SymbolTable,
    ) -> Option<Hover> {
        // Look up in value namespace
        if let Some(def_id) = symbols.lookup(word) {
            if let Some(symbol) = symbols.get(def_id) {
                let contents = self.format_symbol_hover(symbol);
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: contents,
                    }),
                    range: Some(range),
                });
            }
        }

        // Look up in type namespace
        if let Some(def_id) = symbols.lookup_type(word) {
            if let Some(symbol) = symbols.get(def_id) {
                let contents = self.format_symbol_hover(symbol);
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: contents,
                    }),
                    range: Some(range),
                });
            }
        }

        None
    }

    /// Get hover information for a keyword
    pub fn hover_for_keyword(&self, word: &str, range: Range) -> Option<Hover> {
        let info = self.keyword_info(word)?;

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: info,
            }),
            range: Some(range),
        })
    }

    /// Format hover content for a symbol
    fn format_symbol_hover(&self, symbol: &crate::resolve::Symbol) -> String {
        let mut content = String::new();

        // Type signature
        content.push_str("```d\n");
        content.push_str(&self.format_def_kind(&symbol.kind, &symbol.name));
        content.push_str("\n```\n");

        // Kind description
        let kind_desc = match &symbol.kind {
            DefKind::Function => "Function",
            DefKind::Variable { mutable } => {
                if *mutable {
                    "Mutable variable"
                } else {
                    "Immutable variable"
                }
            }
            DefKind::Parameter { mutable } => {
                if *mutable {
                    "Mutable parameter"
                } else {
                    "Parameter"
                }
            }
            DefKind::Struct {
                is_linear,
                is_affine,
            } => {
                if *is_linear {
                    "Linear struct (must be consumed exactly once)"
                } else if *is_affine {
                    "Affine struct (can be consumed at most once)"
                } else {
                    "Struct"
                }
            }
            DefKind::Enum {
                is_linear,
                is_affine,
            } => {
                if *is_linear {
                    "Linear enum"
                } else if *is_affine {
                    "Affine enum"
                } else {
                    "Enum"
                }
            }
            DefKind::Variant => "Enum variant",
            DefKind::TypeAlias => "Type alias",
            DefKind::TypeParam => "Type parameter",
            DefKind::Effect => "Effect",
            DefKind::EffectOp => "Effect operation",
            DefKind::Const => "Constant",
            DefKind::Module => "Module",
            DefKind::Trait => "Trait",
            DefKind::Field => "Field",
            DefKind::Kernel => "GPU kernel function",
            DefKind::BuiltinType => "Built-in type",
        };

        content.push_str(&format!("\n*{}*\n", kind_desc));

        content
    }

    /// Format definition kind as code
    fn format_def_kind(&self, kind: &DefKind, name: &str) -> String {
        match kind {
            DefKind::Function => format!("fn {}(...)", name),
            DefKind::Variable { mutable } => {
                if *mutable {
                    format!("var {}", name)
                } else {
                    format!("let {}", name)
                }
            }
            DefKind::Parameter { .. } => format!("{}: T", name),
            DefKind::Struct {
                is_linear,
                is_affine,
            } => {
                let modifier = if *is_linear {
                    "linear "
                } else if *is_affine {
                    "affine "
                } else {
                    ""
                };
                format!("{}struct {}", modifier, name)
            }
            DefKind::Enum {
                is_linear,
                is_affine,
            } => {
                let modifier = if *is_linear {
                    "linear "
                } else if *is_affine {
                    "affine "
                } else {
                    ""
                };
                format!("{}enum {}", modifier, name)
            }
            DefKind::Variant => format!("{}(...)", name),
            DefKind::TypeAlias => format!("type {} = ...", name),
            DefKind::TypeParam => format!("<{}>", name),
            DefKind::Effect => format!("effect {}", name),
            DefKind::EffectOp => format!("{}(...)", name),
            DefKind::Const => format!("const {}", name),
            DefKind::Module => format!("module {}", name),
            DefKind::Trait => format!("trait {}", name),
            DefKind::Field => format!("{}: T", name),
            DefKind::Kernel => format!("kernel fn {}(...)", name),
            DefKind::BuiltinType => name.to_string(),
        }
    }

    /// Get keyword documentation
    fn keyword_info(&self, word: &str) -> Option<String> {
        let info = match word {
            "fn" => {
                r#"**fn** — Function declaration

```d
fn name(params) -> ReturnType with Effects {
    body
}
```

Functions can have:
- Generic type parameters
- Effect annotations with `with`
- Linear/affine ownership requirements"#
            }

            "let" => {
                r#"**let** — Immutable variable binding

```d
let x: Type = value
let y = inferred_value
```

Variables bound with `let` cannot be reassigned."#
            }

            "mut" => {
                r#"**mut** — Mutable modifier

```d
let mut x = 5
x = 10  // OK

fn foo(mut param: T) { ... }
&mut reference
```

Allows mutation of variables, parameters, and references."#
            }

            "const" => {
                r#"**const** — Compile-time constant

```d
const PI: f64 = 3.14159
const MAX_SIZE: usize = 1024
```

Constants are evaluated at compile time."#
            }

            "struct" => {
                r#"**struct** — Structure definition

```d
struct Name {
    field1: Type1,
    field2: Type2,
}
```

Can be marked as `linear` or `affine` for resource tracking."#
            }

            "enum" => {
                r#"**enum** — Enumeration definition

```d
enum Name {
    Variant1,
    Variant2(Type),
    Variant3 { field: Type },
}
```

Enums support unit, tuple, and struct variants."#
            }

            "linear" => {
                r#"**linear** — Linear type modifier

```d
linear struct FileHandle {
    fd: i32,
}
```

Linear types must be consumed exactly once. They cannot be:
- Dropped implicitly
- Duplicated
- Ignored

This ensures resources are properly cleaned up."#
            }

            "affine" => {
                r#"**affine** — Affine type modifier

```d
affine struct Buffer {
    ptr: *mut u8,
    len: usize,
}
```

Affine types can be consumed at most once. They can be:
- Dropped implicitly
- NOT duplicated

Use for resources that may or may not need cleanup."#
            }

            "with" => {
                r#"**with** — Effect annotation

```d
fn read_file(path: string) -> string with IO {
    // ... can perform IO
}

fn pure_function(x: i32) -> i32 {
    // No effects allowed
}
```

Effects: `IO`, `Mut`, `Alloc`, `Panic`, `Async`, `GPU`, `Prob`, `Div`"#
            }

            "kernel" => {
                r#"**kernel** — GPU kernel function

```d
kernel fn matmul(
    a: &[f32],
    b: &[f32],
    c: &![f32],
) {
    let idx = thread_idx()
    // ... parallel computation
}
```

Kernel functions run on the GPU with massive parallelism."#
            }

            "effect" => {
                r#"**effect** — Effect definition

```d
effect State<T> {
    get() -> T,
    put(value: T) -> (),
}
```

Effects define operations that can be handled by effect handlers."#
            }

            "handler" => {
                r#"**handler** — Effect handler

```d
handler StateHandler<T> for State<T> {
    get() => resume(self.value),
    put(v) => {
        self.value = v
        resume(())
    },
}
```

Handlers provide implementations for effect operations."#
            }

            "handle" => {
                r#"**handle** — Apply effect handler

```d
let result = handle computation() with StateHandler::new(0)
```

Applies an effect handler to a computation."#
            }

            "perform" => {
                r#"**perform** — Perform effect operation

```d
fn increment() with State<i32> {
    let x = perform State::get()
    perform State::put(x + 1)
}
```

Invokes an effect operation."#
            }

            "sample" => {
                r#"**sample** — Sample from distribution

```d
fn monte_carlo() with Prob {
    let x = sample(Normal(0.0, 1.0))
    let y = sample(Uniform(0.0, 1.0))
    // ...
}
```

Samples a value from a probability distribution."#
            }

            "observe" => {
                r#"**observe** — Condition on observation

```d
fn bayesian_inference(data: f64) with Prob {
    let mu = sample(Normal(0.0, 10.0))
    observe(Normal(mu, 1.0), data)
    mu
}
```

Conditions the probabilistic model on observed data."#
            }

            "if" => {
                r#"**if** — Conditional expression

```d
if condition {
    // then branch
} else if other_condition {
    // else-if branch
} else {
    // else branch
}
```

Branches must have compatible types when used as expressions."#
            }

            "match" => {
                r#"**match** — Pattern matching

```d
match value {
    Pattern1 => expr1,
    Pattern2(x) => expr2,
    Pattern3 { field } => expr3,
    _ => default,
}
```

Exhaustive pattern matching with destructuring."#
            }

            "for" => {
                r#"**for** — For loop

```d
for item in collection {
    // use item
}

for i in 0..10 {
    // iterate 0 to 9
}
```

Iterates over collections or ranges."#
            }

            "while" => {
                r#"**while** — While loop

```d
while condition {
    // loop body
}
```

Loops while condition is true."#
            }

            "loop" => {
                r#"**loop** — Infinite loop

```d
loop {
    if done {
        break result
    }
}
```

Use `break` to exit with a value."#
            }

            "return" => {
                r#"**return** — Return from function

```d
fn foo() -> i32 {
    return 42
}
```

Early return from a function."#
            }

            "break" => {
                r#"**break** — Break out of loop

```d
loop {
    if done {
        break value  // with value
    }
}

while true {
    break  // without value
}
```"#
            }

            "continue" => {
                r#"**continue** — Continue to next iteration

```d
for i in 0..10 {
    if skip(i) {
        continue
    }
    process(i)
}
```"#
            }

            "true" | "false" => {
                r#"**bool** — Boolean literal

```d
let a: bool = true
let b: bool = false
```"#
            }

            "type" => {
                r#"**type** — Type alias

```d
type UserId = u64
type Result<T> = Option<T, Error>
```

Creates an alias for an existing type."#
            }

            "trait" => {
                r#"**trait** — Trait definition

```d
trait Display {
    fn fmt(&self) -> string
}

impl Display for MyType {
    fn fmt(&self) -> string { ... }
}
```"#
            }

            "impl" => {
                r#"**impl** — Implementation block

```d
impl MyType {
    fn new() -> Self { ... }
    fn method(&self) { ... }
}

impl Trait for MyType {
    fn trait_method(&self) { ... }
}
```"#
            }

            "pub" => {
                r#"**pub** — Public visibility

```d
pub fn public_function() { ... }
pub struct PublicStruct { ... }
```

Makes items visible outside the module."#
            }

            "async" => {
                r#"**async** — Asynchronous function

```d
async fn fetch(url: string) -> Response with IO, Async {
    // ... async implementation
}
```"#
            }

            "await" => {
                r#"**await** — Await async result

```d
async fn main() with Async {
    let response = await fetch("http://example.com")
}
```"#
            }

            _ => return None,
        };

        Some(info.to_string())
    }
}

impl Default for HoverProvider {
    fn default() -> Self {
        Self::new()
    }
}
