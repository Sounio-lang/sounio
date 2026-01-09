# Effect System and Checking

Sounio features a full algebraic effect system with row polymorphism. Effects track computational side effects and enable modular, composable effect handling.

## Overview

The effect system is implemented across several modules:

- `compiler/src/effects/` - Core effect types and handlers
- `compiler/src/types/effects.rs` - Effect type representation
- `compiler/src/check/` - Effect checking during type checking

## Built-in Effects

Sounio provides several built-in effects:

| Effect | Description |
|--------|-------------|
| `IO` | Input/output operations |
| `Mut` | Mutable state |
| `Alloc` | Memory allocation |
| `Panic` | Exceptions and panics |
| `Async` | Asynchronous operations |
| `GPU` | GPU computation |
| `Prob` | Probabilistic computation |
| `Div` | Division (potential division by zero) |
| `Network` | Network operations |
| `Sensor` | Sensor data acquisition |
| `Exn` | Exceptions |

## Effect Annotations

Functions declare their effects using the `with` keyword:

```sio
fn read_file(path: string) -> string with IO, Panic {
    // Can perform IO and may panic
}

fn simulate(params: Params) -> f64 with Prob, Alloc {
    // Can sample from distributions and allocate memory
}

fn pure_function(x: i32) -> i32 {
    // No effects - implicitly pure
}
```

## Effect Row Polymorphism

Functions can be polymorphic over their effects using effect parameters:

```sio
// The function is polymorphic over effect E
fn map<T, U, effect E>(f: fn(T) -> U with E, xs: [T]) -> [U] with E {
    // Propagates whatever effects f has
}
```

This is implemented using effect variables:

```rust
pub enum GenericParam {
    Type { name: String, bounds: Vec<Path>, default: Option<TypeExpr> },
    Const { name: String, ty: TypeExpr },
    Effect { name: String },  // Effect parameter for row polymorphism
}

pub struct EffectRow {
    pub effects: Vec<String>,
    pub row_var: Option<String>,  // Effect variable for polymorphism
    pub is_open: bool,
}
```

## Effect Inference

During type checking, effects are inferred from function bodies:

```rust
pub struct EffectInference {
    /// Known effect definitions
    effect_defs: HashMap<String, EffectDef>,

    /// Inferred effects for current function
    current_effects: EffectSet,

    /// Effect variable substitutions
    effect_subst: HashMap<EffectVar, EffectSet>,
}

impl EffectInference {
    /// Infer effects from an expression
    pub fn infer_effects(&mut self, expr: &HirExpr) -> EffectSet {
        match expr {
            HirExpr::Call { callee, .. } => {
                // Get effects from called function
                self.get_call_effects(callee)
            }

            HirExpr::Perform { effect, .. } => {
                // Performing an effect adds it to the set
                EffectSet::singleton(effect)
            }

            HirExpr::Handle { inner, handler, .. } => {
                // Handler masks the handled effect
                let inner_effects = self.infer_effects(inner);
                let handled = self.handler_effect(handler);
                inner_effects.subtract(&[handled])
            }

            // ...
        }
    }
}
```

## Effect Set Operations

Effect sets support standard set operations:

```rust
pub struct EffectSet {
    pub effects: HashSet<String>,
}

impl EffectSet {
    pub fn new() -> Self
    pub fn singleton(effect: &str) -> Self
    pub fn from_effects(effects: &[&str]) -> Self

    pub fn insert(&mut self, effect: &str)
    pub fn contains(&self, effect: &str) -> bool
    pub fn is_empty(&self) -> bool

    pub fn union(&self, other: &EffectSet) -> EffectSet
    pub fn subtract(&self, effects: &[String]) -> EffectSet
    pub fn is_subset_of(&self, other: &EffectSet) -> bool
}
```

## Effect Checking

The type checker verifies that:

1. **Declared effects match inferred effects**: A function's body must not perform effects beyond what's declared.

2. **Effect subtyping**: A function with fewer effects can be used where more effects are expected (effects are contravariant in function arguments).

3. **Effect handling**: Handled effects are removed from the visible effect set.

```rust
impl TypeChecker {
    fn check_function_effects(
        &mut self,
        fn_def: &FnDef,
        inferred: &EffectSet,
    ) -> Result<()> {
        let declared = self.declared_effect_set(&fn_def.effects);

        // Check that all inferred effects are declared
        for effect in &inferred.effects {
            if !declared.contains(effect) && !self.masked_effects.contains(effect) {
                self.error(format!(
                    "Function performs effect '{}' but does not declare it",
                    effect
                ), fn_def.span);
            }
        }

        // Warn about over-declared effects
        for effect in &declared.effects {
            if !inferred.effects.contains(effect) {
                self.warnings.push(format!(
                    "Function declares effect '{}' but never performs it",
                    effect
                ));
            }
        }

        Ok(())
    }
}
```

## Effect Handlers

Effects are handled using the `handle` expression:

```sio
handle computation() with IOHandler
```

Handler definitions implement effect operations:

```sio
handler IOHandler for IO {
    fn print(msg: string) -> () {
        // Implementation
        resume()
    }

    fn read_line() -> string {
        // Implementation
        resume(line)
    }
}
```

### Handler Registration

During type checking, handlers are registered:

```rust
impl TypeChecker {
    fn register_handler(&mut self, handler_name: String, effect_name: String) {
        self.handler_effects.insert(handler_name, effect_name);
    }

    fn lookup_handler_effect(&self, handler_name: &str) -> Option<String> {
        // Check explicit registrations
        if let Some(effect) = self.handler_effects.get(handler_name) {
            return Some(effect.clone());
        }

        // Check naming convention: XHandler -> X
        if handler_name.ends_with("Handler") {
            let effect_name = &handler_name[..handler_name.len() - 7];
            if self.effects.lookup_effect(effect_name).is_some() {
                return Some(effect_name.to_string());
            }
        }

        None
    }
}
```

## Effect Masking

When an effect is handled, it is masked from the function's external signature:

```rust
impl TypeChecker {
    /// Effects masked by handlers in current context
    masked_effects: EffectSet,

    fn check_handle_expr(&mut self, inner: &Expr, handler: &Path) -> Result<...> {
        // Determine which effect is handled
        let effect = self.lookup_handler_effect(&handler.segments[0])?;

        // Add to masked effects for the inner expression
        self.masked_effects.insert(&effect);

        let result = self.check_expr(inner)?;

        Ok(result)
    }

    /// Compute visible effects after masking
    fn residual_effects(&self, inferred: &EffectSet) -> EffectSet {
        let masked_list: Vec<String> = self.masked_effects.effects.iter().cloned().collect();
        inferred.subtract(&masked_list)
    }
}
```

## Effect Variables and Unification

Effect variables are unified during type checking:

```rust
impl TypeChecker {
    fn fresh_effect_var(&mut self) -> EffectVar {
        let var = EffectVar::new(self.next_effect_var);
        self.next_effect_var += 1;
        var
    }

    fn register_effect_param(&mut self, name: &str) -> EffectVar {
        let var = self.fresh_effect_var();
        self.effect_params.insert(name.to_string(), var);
        var
    }

    fn unify_effects(
        &mut self,
        expected: &EffectSet,
        actual: &EffectSet,
    ) -> Result<()> {
        // Handle effect variables
        if let Some(var) = actual.row_var() {
            // Bind variable to remaining effects
            let remaining = expected.subtract(&actual.concrete_effects());
            self.bind_effect_var(var, remaining);
            return Ok(());
        }

        // Check subset relationship
        if actual.is_subset_of(expected) {
            Ok(())
        } else {
            Err(...)
        }
    }
}
```

## Continuation-Based Handlers

For full algebraic effect support, the compiler includes continuation infrastructure:

```rust
pub mod continuation {
    pub struct CapturedContinuation {
        pub id: ContinuationId,
        pub resume_point: ResumePoint,
        pub captured_env: Environment,
    }

    pub struct ContinuationStore {
        continuations: HashMap<ContinuationId, CapturedContinuation>,
    }
}
```

## Runtime Handler Support

The effect system includes runtime support for:

- **Direct handlers**: Specialized runtime functions for common effects
- **Registry-based dispatch**: Configurable handlers via `HandlerRegistry`
- **JIT support**: Effect handling in Cranelift-compiled code

```rust
pub trait Handler<E> {
    type Output;
    fn handle(&self, effect: E) -> Self::Output;
}

pub struct HandlerRegistry {
    io_handler: Box<dyn IOHandler>,
    prob_handler: Box<dyn ProbHandler>,
    // ...
}

impl HandlerRegistry {
    pub fn with_defaults() -> Self { ... }
    pub fn with_io(self, handler: impl IOHandler) -> Self { ... }
    pub fn with_prob(self, handler: impl ProbHandler) -> Self { ... }
}
```

## Files

- `compiler/src/effects/mod.rs` - Effect system re-exports
- `compiler/src/effects/continuation.rs` - Continuation capture/resume
- `compiler/src/effects/inference.rs` - Effect inference
- `compiler/src/effects/handlers.rs` - Built-in handler implementations
- `compiler/src/effects/handler_capability.rs` - Handler trait
- `compiler/src/effects/epistemic_effects.rs` - Epistemic effect tracking
- `compiler/src/effects/jit_resume.rs` - JIT continuation support
- `compiler/src/types/effects.rs` - Effect types and sets
