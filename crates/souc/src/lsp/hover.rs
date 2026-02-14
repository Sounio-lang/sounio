//! Hover information provider
//!
//! Provides type information and documentation on hover.
//!
//! # Tensor Shape Information
//!
//! For tensor types, hover provides detailed shape information:
//! - Element type (f32, f64, etc.)
//! - Shape dimensions (fixed, named, or dynamic)
//! - Total element count (when computable)
//! - Shape compatibility warnings for operations

use tower_lsp::lsp_types::*;

use crate::hir::{HirTensorDim, HirType};
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
            DefKind::BuiltinFunction => "Built-in function",
            DefKind::EffectParam => "Effect parameter",
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
            DefKind::BuiltinFunction => format!("fn {}(...)", name),
            DefKind::EffectParam => format!("with {}", name),
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

            "var" => {
                r#"**var** — Mutable variable binding

```d
var x = 5
x = 10  // OK: x is mutable
```

Variables bound with `var` can be reassigned. Equivalent to mutable bindings."#
            }

            "Knowledge" => {
                r#"**Knowledge<T>** — Epistemic type

```d
let m: Knowledge<f64> = measure(500.0, uncertainty: 2.5)
let derived = m + Knowledge::new(100.0, epsilon: 1.0)
```

Wraps a value `T` with epistemic metadata:
- **Confidence** — How certain is this value? (0.0 to 1.0)
- **Uncertainty (epsilon)** — GUM-compliant uncertainty propagation
- **Provenance** — Where did this knowledge come from?
- **Temporal bounds** — When is this knowledge valid?

Arithmetic operations automatically propagate uncertainty via GUM rules."#
            }

            "CausalKnowledge" => {
                r#"**CausalKnowledge<T>** — Causal epistemic type (Level 2)

```d
let effect: CausalKnowledge<f64> = model.do_intervention(dose: 100.0)
```

Extends `Knowledge<T>` with causal structure:
- **Causal graph** — DAG encoding causal relationships
- **Identifiability** — Verified at compile time via type system
- **do-operator** — Interventional queries (Pearl's do-calculus)
- **Uncertainty** — Includes both epistemic and causal uncertainty

The type system verifies identifiability at compile time:
- Identifiable queries type-check
- Non-identifiable queries produce compile errors"#
            }

            "do" => {
                r#"**do** — Causal intervention operator

```d
let result = do(graph, treatment = value)
```

Pearl's do-operator for interventional queries:
- `P(Y | do(X = x))` — What happens when we *set* X to x?
- Different from `P(Y | X = x)` — What we *observe* when X is x

The type system verifies that the interventional query is identifiable
from observational data before allowing the computation."#
            }

            "counterfactual" => {
                r#"**counterfactual** — Counterfactual query (Level 3)

```d
let cf = counterfactual(model, X_x | X = x', Y = y)
```

Level 3 of Pearl's causal hierarchy:
- What *would have happened* if X had been x, given that we observed X=x' and Y=y?
- Requires full structural causal model (not just DAG)
- Most powerful but most demanding form of causal reasoning"#
            }

            "measure" => {
                r#"**measure** — Create epistemic knowledge from measurement

```d
let temp: Knowledge<f64> = measure(37.5, uncertainty: 0.2)
let dose: Knowledge<mg> = measure(500.0, confidence: 0.95)
```

Creates a `Knowledge<T>` value with:
- The measured value
- Measurement uncertainty (epsilon)
- Confidence level
- Source provenance (instrument, protocol, timestamp)"#
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

    // ========================================================================
    // Tensor Shape Hover Information
    // ========================================================================

    /// Get hover information for a tensor type
    pub fn hover_for_tensor(&self, hir_type: &HirType, range: Range) -> Option<Hover> {
        let content = self.format_tensor_hover(hir_type)?;
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: Some(range),
        })
    }

    /// Format detailed hover content for tensor types
    pub fn format_tensor_hover(&self, ty: &HirType) -> Option<String> {
        match ty {
            HirType::Tensor { element, dims } => {
                let elem_str = element.format_short();
                let dims_short = dims
                    .iter()
                    .map(|d| d.format_short())
                    .collect::<Vec<_>>()
                    .join(", ");
                let dims_detailed = dims
                    .iter()
                    .enumerate()
                    .map(|(i, d)| format!("  - dim[{}]: {}", i, d.format_display()))
                    .collect::<Vec<_>>()
                    .join("\n");

                let total = HirTensorDim::compute_total_elements(dims);
                let total_str = total
                    .map(|n| format!("{}", n))
                    .unwrap_or_else(|| "dynamic (computed at runtime)".to_string());

                let memory_info = total.map(|n| {
                    let elem_size = self.element_size_bytes(&element);
                    let total_bytes = n * elem_size;
                    self.format_memory_size(total_bytes)
                });
                let memory_str = memory_info
                    .map(|s| format!("\n- Memory: ~{}", s))
                    .unwrap_or_default();

                Some(format!(
                    "**Tensor**\n\n\
                     ```d\n\
                     Tensor[{}, ({})]\n\
                     ```\n\n\
                     **Properties**\n\
                     - Element type: `{}`\n\
                     - Shape: `[{}]`\n\
                     - Rank: {}\n\
                     - Total elements: {}{}\n\n\
                     **Dimensions**\n\
                     {}",
                    elem_str,
                    dims_short,
                    elem_str,
                    dims_short,
                    dims.len(),
                    total_str,
                    memory_str,
                    dims_detailed
                ))
            }
            HirType::Mat2 | HirType::Mat3 | HirType::Mat4 => {
                let dim = match ty {
                    HirType::Mat2 => 2,
                    HirType::Mat3 => 3,
                    HirType::Mat4 => 4,
                    _ => unreachable!(),
                };
                Some(format!(
                    "**Matrix**\n\n\
                     ```d\n\
                     mat{}\n\
                     ```\n\n\
                     **Properties**\n\
                     - Element type: `f32`\n\
                     - Shape: `{}x{}` (column-major)\n\
                     - Total elements: {}\n\
                     - Memory: {} bytes\n\n\
                     **Operations**\n\
                     - `*` — Matrix multiplication\n\
                     - `+`, `-` — Element-wise operations\n\
                     - `.transpose()` — Transpose\n\
                     - `.inverse()` — Inverse (if invertible)\n\
                     - `.determinant()` — Determinant",
                    dim,
                    dim,
                    dim,
                    dim * dim,
                    dim * dim * 4
                ))
            }
            HirType::Vec2 | HirType::Vec3 | HirType::Vec4 => {
                let dim = match ty {
                    HirType::Vec2 => 2,
                    HirType::Vec3 => 3,
                    HirType::Vec4 => 4,
                    _ => unreachable!(),
                };
                let components = match dim {
                    2 => "x, y",
                    3 => "x, y, z",
                    4 => "x, y, z, w",
                    _ => "",
                };
                Some(format!(
                    "**Vector**\n\n\
                     ```d\n\
                     vec{}\n\
                     ```\n\n\
                     **Properties**\n\
                     - Element type: `f32`\n\
                     - Dimension: {}\n\
                     - Components: `{}`\n\
                     - Memory: {} bytes\n\n\
                     **Operations**\n\
                     - `.dot(other)` — Dot product\n\
                     - `.cross(other)` — Cross product (vec3 only)\n\
                     - `.length()` — Euclidean length\n\
                     - `.normalize()` — Unit vector\n\
                     - `+`, `-`, `*`, `/` — Component-wise operations",
                    dim,
                    dim,
                    components,
                    dim * 4
                ))
            }
            HirType::Quat => Some(
                "**Quaternion**\n\n\
                 ```d\n\
                 quat\n\
                 ```\n\n\
                 **Properties**\n\
                 - Components: `x, y, z, w` (f32)\n\
                 - Memory: 16 bytes\n\n\
                 **Operations**\n\
                 - `*` — Quaternion multiplication (rotation composition)\n\
                 - `.normalize()` — Unit quaternion\n\
                 - `.inverse()` — Inverse rotation\n\
                 - `.to_matrix()` — Convert to rotation matrix\n\
                 - `.slerp(other, t)` — Spherical interpolation"
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// Get element size in bytes for a type
    fn element_size_bytes(&self, ty: &HirType) -> usize {
        match ty {
            HirType::I8 | HirType::U8 | HirType::Bool => 1,
            HirType::I16 | HirType::U16 => 2,
            HirType::I32 | HirType::U32 | HirType::F32 => 4,
            HirType::I64 | HirType::U64 | HirType::F64 => 8,
            HirType::I128 | HirType::U128 => 16,
            _ => 4, // Default to 4 bytes
        }
    }

    /// Format memory size for display
    fn format_memory_size(&self, bytes: usize) -> String {
        if bytes < 1024 {
            format!("{} bytes", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Get hover information for tensor operations
    pub fn hover_for_tensor_operation(&self, op_name: &str, range: Range) -> Option<Hover> {
        let content = self.tensor_operation_info(op_name)?;
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: Some(range),
        })
    }

    /// Get documentation for tensor operations
    fn tensor_operation_info(&self, op: &str) -> Option<String> {
        let info = match op {
            "matmul" | "mm" => {
                r#"**matmul** — Matrix multiplication

```d
fn matmul(a: Tensor[T, (M, K)], b: Tensor[T, (K, N)]) -> Tensor[T, (M, N)]
```

Computes matrix product `C = A @ B` where:
- `A` has shape `(M, K)`
- `B` has shape `(K, N)`
- Result has shape `(M, N)`

**Shape Rules**
- Inner dimensions must match: `A.shape[-1] == B.shape[-2]`
- Supports batched matmul when shapes have more than 2 dimensions

**Performance**
Uses optimized BLAS/cuBLAS when available."#
            }
            "bmm" => {
                r#"**bmm** — Batched matrix multiplication

```d
fn bmm(a: Tensor[T, (B, M, K)], b: Tensor[T, (B, K, N)]) -> Tensor[T, (B, M, N)]
```

Performs batch-wise matrix multiplication where:
- `B` is the batch dimension
- Each batch computes `C[i] = A[i] @ B[i]`"#
            }
            "transpose" | "t" => {
                r#"**transpose** — Swap tensor dimensions

```d
fn transpose(t: Tensor[T, (M, N)]) -> Tensor[T, (N, M)]
fn transpose(t: Tensor[T, dims], dim0: int, dim1: int) -> Tensor[T, dims']
```

Swaps the specified dimensions. For 2D tensors, swaps rows and columns."#
            }
            "reshape" | "view" => {
                r#"**reshape** — Change tensor shape

```d
fn reshape(t: Tensor[T, dims], new_shape: (int...)) -> Tensor[T, new_shape]
```

Changes the shape of a tensor without copying data (when possible).

**Requirements**
- Total elements must remain the same
- One dimension can be `-1` to infer from others"#
            }
            "squeeze" => {
                r#"**squeeze** — Remove size-1 dimensions

```d
fn squeeze(t: Tensor[T, dims]) -> Tensor[T, dims']
fn squeeze(t: Tensor[T, dims], dim: int) -> Tensor[T, dims']
```

Removes all dimensions of size 1, or just the specified dimension."#
            }
            "unsqueeze" => {
                r#"**unsqueeze** — Add size-1 dimension

```d
fn unsqueeze(t: Tensor[T, dims], dim: int) -> Tensor[T, dims']
```

Adds a dimension of size 1 at the specified position."#
            }
            "broadcast" | "expand" => {
                r#"**broadcast** — Expand tensor to larger shape

```d
fn broadcast(t: Tensor[T, dims], target: (int...)) -> Tensor[T, target]
```

Expands tensor to match target shape using broadcasting rules.

**Broadcasting Rules**
- Dimensions are aligned from the right
- Size-1 dimensions can be expanded
- Missing dimensions are added as size-1"#
            }
            "sum" => {
                r#"**SUM** — Reduction operation

```d
fn sum(t: Tensor[T, dims]) -> T
fn sum(t: Tensor[T, dims], dim: int, keepdim: bool = false) -> Tensor[T, dims']
```

Reduces tensor along specified dimension(s).

**Parameters**
- `dim`: Dimension to reduce (optional, reduces all if not specified)
- `keepdim`: If true, reduced dimension becomes size 1"#
            }
            "mean" => {
                r#"**MEAN** — Reduction operation

```d
fn mean(t: Tensor[T, dims]) -> T
fn mean(t: Tensor[T, dims], dim: int, keepdim: bool = false) -> Tensor[T, dims']
```

Reduces tensor along specified dimension(s).

**Parameters**
- `dim`: Dimension to reduce (optional, reduces all if not specified)
- `keepdim`: If true, reduced dimension becomes size 1"#
            }
            "max" => {
                r#"**MAX** — Reduction operation

```d
fn max(t: Tensor[T, dims]) -> T
fn max(t: Tensor[T, dims], dim: int, keepdim: bool = false) -> Tensor[T, dims']
```

Reduces tensor along specified dimension(s).

**Parameters**
- `dim`: Dimension to reduce (optional, reduces all if not specified)
- `keepdim`: If true, reduced dimension becomes size 1"#
            }
            "min" => {
                r#"**MIN** — Reduction operation

```d
fn min(t: Tensor[T, dims]) -> T
fn min(t: Tensor[T, dims], dim: int, keepdim: bool = false) -> Tensor[T, dims']
```

Reduces tensor along specified dimension(s).

**Parameters**
- `dim`: Dimension to reduce (optional, reduces all if not specified)
- `keepdim`: If true, reduced dimension becomes size 1"#
            }
            "zeros" => {
                r#"**zeros** — Create filled tensor

```d
fn zeros(shape: (int...)) -> Tensor[f32, shape]
fn zeros<T>(shape: (int...), dtype: Type[T]) -> Tensor[T, shape]
```

Creates a tensor filled with 0s of the specified shape."#
            }
            "ones" => {
                r#"**ones** — Create filled tensor

```d
fn ones(shape: (int...)) -> Tensor[f32, shape]
fn ones<T>(shape: (int...), dtype: Type[T]) -> Tensor[T, shape]
```

Creates a tensor filled with 1s of the specified shape."#
            }
            "rand" => {
                r#"**rand** — Create random tensor

```d
fn rand(shape: (int...)) -> Tensor[f32, shape]
```

Creates a tensor with random values from uniform [0, 1) distribution."#
            }
            "randn" => {
                r#"**randn** — Create random tensor

```d
fn randn(shape: (int...)) -> Tensor[f32, shape]
```

Creates a tensor with random values from standard normal (mean=0, std=1) distribution."#
            }
            "flatten" => {
                r#"**flatten** — Flatten tensor to 1D

```d
fn flatten(t: Tensor[T, dims]) -> Tensor[T, (N,)]
fn flatten(t: Tensor[T, dims], start_dim: int, end_dim: int) -> Tensor[T, dims']
```

Flattens tensor into 1D or flattens a range of dimensions."#
            }
            _ => return None,
        };

        Some(info.to_string())
    }

    /// Get hover for tensor shape compatibility
    pub fn hover_for_shape_check(
        &self,
        shape_a: &[HirTensorDim],
        shape_b: &[HirTensorDim],
        operation: &str,
        range: Range,
    ) -> Option<Hover> {
        let (compatible, message) = HirTensorDim::shapes_compatible(shape_a, shape_b);

        let shape_a_str = shape_a
            .iter()
            .map(|d| d.format_short())
            .collect::<Vec<_>>()
            .join(", ");
        let shape_b_str = shape_b
            .iter()
            .map(|d| d.format_short())
            .collect::<Vec<_>>()
            .join(", ");

        let status = if compatible {
            "Compatible"
        } else {
            "Incompatible"
        };
        let icon = if compatible { "OK" } else { "Warning" };

        let content = format!(
            "**Shape Check: {}**\n\n\
             **Operation**: `{}`\n\n\
             | Operand | Shape |\n\
             |---------|-------|\n\
             | Left | `[{}]` |\n\
             | Right | `[{}]` |\n\n\
             **Status**: {} ({})",
            icon, operation, shape_a_str, shape_b_str, status, message
        );

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: Some(range),
        })
    }

    /// Get hover for matmul shape inference
    pub fn hover_for_matmul(
        &self,
        shape_a: &[HirTensorDim],
        shape_b: &[HirTensorDim],
        range: Range,
    ) -> Option<Hover> {
        let (compatible, message, result_shape) = HirTensorDim::matmul_compatible(shape_a, shape_b);

        let shape_a_str = shape_a
            .iter()
            .map(|d| d.format_short())
            .collect::<Vec<_>>()
            .join(", ");
        let shape_b_str = shape_b
            .iter()
            .map(|d| d.format_short())
            .collect::<Vec<_>>()
            .join(", ");

        let result_str = result_shape
            .as_ref()
            .map(|s| {
                s.iter()
                    .map(|d| d.format_short())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "N/A".to_string());

        let status = if compatible { "OK" } else { "Error" };
        let status_msg = if compatible { &message } else { &message };

        let content = format!(
            "**Matrix Multiplication**\n\n\
             ```\n\
             [{shape_a}] @ [{shape_b}] -> [{result}]\n\
             ```\n\n\
             | Shape | Value |\n\
             |-------|-------|\n\
             | A | `[{shape_a}]` |\n\
             | B | `[{shape_b}]` |\n\
             | Result | `[{result}]` |\n\n\
             **Status**: {status}\n\
             {status_msg}",
            shape_a = shape_a_str,
            shape_b = shape_b_str,
            result = result_str,
            status = status,
            status_msg = status_msg
        );

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: Some(range),
        })
    }
}

impl Default for HoverProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_hover_fixed_dims() {
        let provider = HoverProvider::new();
        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F32),
            dims: vec![
                HirTensorDim::Fixed(32),
                HirTensorDim::Fixed(64),
                HirTensorDim::Fixed(128),
            ],
        };

        let content = provider.format_tensor_hover(&tensor_type);
        assert!(content.is_some());

        let content = content.unwrap();
        assert!(content.contains("Tensor"));
        assert!(content.contains("f32"));
        assert!(content.contains("32"));
        assert!(content.contains("64"));
        assert!(content.contains("128"));
        assert!(content.contains("Rank: 3"));
    }

    #[test]
    fn test_tensor_hover_named_dims() {
        let provider = HoverProvider::new();
        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F64),
            dims: vec![
                HirTensorDim::Named("batch".to_string()),
                HirTensorDim::Named("features".to_string()),
            ],
        };

        let content = provider.format_tensor_hover(&tensor_type);
        assert!(content.is_some());

        let content = content.unwrap();
        assert!(content.contains("batch"));
        assert!(content.contains("features"));
        assert!(content.contains("dynamic"));
    }

    #[test]
    fn test_matrix_hover() {
        let provider = HoverProvider::new();
        let content = provider.format_tensor_hover(&HirType::Mat4);
        assert!(content.is_some());

        let content = content.unwrap();
        assert!(content.contains("Matrix"));
        assert!(content.contains("4x4"));
        assert!(content.contains("16"));
        assert!(content.contains("transpose"));
    }

    #[test]
    fn test_vector_hover() {
        let provider = HoverProvider::new();
        let content = provider.format_tensor_hover(&HirType::Vec3);
        assert!(content.is_some());

        let content = content.unwrap();
        assert!(content.contains("Vector"));
        assert!(content.contains("3"));
        assert!(content.contains("x, y, z"));
        assert!(content.contains("dot"));
        assert!(content.contains("cross"));
    }

    #[test]
    fn test_quaternion_hover() {
        let provider = HoverProvider::new();
        let content = provider.format_tensor_hover(&HirType::Quat);
        assert!(content.is_some());

        let content = content.unwrap();
        assert!(content.contains("Quaternion"));
        assert!(content.contains("slerp"));
    }

    #[test]
    fn test_tensor_operation_info() {
        let provider = HoverProvider::new();

        let matmul_info = provider.tensor_operation_info("matmul");
        assert!(matmul_info.is_some());
        assert!(matmul_info.unwrap().contains("Matrix multiplication"));

        let reshape_info = provider.tensor_operation_info("reshape");
        assert!(reshape_info.is_some());
        assert!(reshape_info.unwrap().contains("Change tensor shape"));
    }

    #[test]
    fn test_format_memory_size() {
        let provider = HoverProvider::new();

        assert_eq!(provider.format_memory_size(512), "512 bytes");
        assert_eq!(provider.format_memory_size(1024), "1.0 KB");
        assert_eq!(provider.format_memory_size(1024 * 1024), "1.0 MB");
        assert_eq!(provider.format_memory_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_matmul_hover_compatible() {
        let provider = HoverProvider::new();
        let range = Range::default();

        let shape_a = vec![HirTensorDim::Fixed(32), HirTensorDim::Fixed(64)];
        let shape_b = vec![HirTensorDim::Fixed(64), HirTensorDim::Fixed(128)];

        let hover = provider.hover_for_matmul(&shape_a, &shape_b, range);
        assert!(hover.is_some());

        if let HoverContents::Markup(content) = hover.unwrap().contents {
            assert!(content.value.contains("32"));
            assert!(content.value.contains("64"));
            assert!(content.value.contains("128"));
            assert!(content.value.contains("OK"));
        }
    }

    #[test]
    fn test_matmul_hover_incompatible() {
        let provider = HoverProvider::new();
        let range = Range::default();

        let shape_a = vec![HirTensorDim::Fixed(32), HirTensorDim::Fixed(64)];
        let shape_b = vec![HirTensorDim::Fixed(128), HirTensorDim::Fixed(256)]; // Wrong inner dim

        let hover = provider.hover_for_matmul(&shape_a, &shape_b, range);
        assert!(hover.is_some());

        if let HoverContents::Markup(content) = hover.unwrap().contents {
            assert!(content.value.contains("Error"));
        }
    }

    #[test]
    fn test_non_tensor_returns_none() {
        let provider = HoverProvider::new();
        let content = provider.format_tensor_hover(&HirType::I32);
        assert!(content.is_none());
    }
}
