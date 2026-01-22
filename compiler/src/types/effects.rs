//! Algebraic effect system
//!
//! D has full algebraic effects with handlers, inspired by Koka/Eff.
//! Effects allow modular handling of side effects like IO, state, exceptions, etc.

use super::core::{Effect, EffectSet, EffectVar, Type, TypeVar};

/// Effect definition
#[derive(Debug, Clone)]
pub struct EffectDef {
    pub name: String,
    pub operations: Vec<EffectOperation>,
}

impl EffectDef {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            operations: Vec::new(),
        }
    }

    pub fn with_op(mut self, op: EffectOperation) -> Self {
        self.operations.push(op);
        self
    }
}

/// Effect operation signature
#[derive(Debug, Clone)]
pub struct EffectOperation {
    pub name: String,
    pub params: Vec<Type>,
    pub return_type: Type,
}

impl EffectOperation {
    pub fn new(name: &str, params: Vec<Type>, return_type: Type) -> Self {
        Self {
            name: name.to_string(),
            params,
            return_type,
        }
    }
}

/// Effect handler
#[derive(Debug, Clone)]
pub struct EffectHandler {
    pub effect: Effect,
    pub cases: Vec<HandlerCase>,
    pub return_clause: Option<Box<HandlerCase>>,
}

impl EffectHandler {
    pub fn new(effect: Effect) -> Self {
        Self {
            effect,
            cases: Vec::new(),
            return_clause: None,
        }
    }

    pub fn with_case(mut self, case: HandlerCase) -> Self {
        self.cases.push(case);
        self
    }

    pub fn with_return(mut self, ret: HandlerCase) -> Self {
        self.return_clause = Some(Box::new(ret));
        self
    }
}

/// Handler case for a specific operation
#[derive(Debug, Clone)]
pub struct HandlerCase {
    pub operation: String,
    pub params: Vec<String>,
    pub resume_param: Option<String>,
    pub body: HandlerBody,
}

/// Handler body (placeholder for actual expression)
#[derive(Debug, Clone)]
pub enum HandlerBody {
    /// Resume with a value
    Resume(Box<HandlerBody>),
    /// Return a value
    Return(Box<HandlerBody>),
    /// Placeholder for actual expression
    Expr,
}

/// Effect inference context
pub struct EffectInference {
    /// Effect variables
    vars: Vec<EffectVar>,
    /// Constraints: effect1 ⊆ effect2
    constraints: Vec<(EffectSet, EffectSet)>,
    /// Known effect definitions
    definitions: Vec<EffectDef>,
}

impl EffectInference {
    pub fn new() -> Self {
        let mut ctx = Self {
            vars: Vec::new(),
            constraints: Vec::new(),
            definitions: Vec::new(),
        };

        // Register built-in effects
        ctx.register_builtin_effects();
        ctx
    }

    fn register_builtin_effects(&mut self) {
        // IO effect
        self.definitions.push(
            EffectDef::new("IO")
                .with_op(EffectOperation::new(
                    "print",
                    vec![Type::String],
                    Type::Unit,
                ))
                .with_op(EffectOperation::new("read_line", vec![], Type::String))
                .with_op(EffectOperation::new(
                    "read_file",
                    vec![Type::String],
                    Type::String,
                ))
                .with_op(EffectOperation::new(
                    "write_file",
                    vec![Type::String, Type::String],
                    Type::Unit,
                )),
        );

        // Mut effect (mutable state)
        // Provides a named state store for mutable computations
        self.definitions.push(
            EffectDef::new("Mut")
                .with_op(EffectOperation::new(
                    "get",
                    vec![Type::String], // state name
                    Type::F64,
                ))
                .with_op(EffectOperation::new(
                    "set",
                    vec![Type::String, Type::F64], // name, value
                    Type::Unit,
                ))
                .with_op(EffectOperation::new(
                    "modify",
                    vec![Type::String, Type::F64], // name, delta
                    Type::F64,                     // returns new value
                ))
                .with_op(EffectOperation::new("clear", vec![], Type::Unit)),
        );

        // Alloc effect (memory allocation)
        // Provides tracked memory allocation for safe memory management
        self.definitions.push(
            EffectDef::new("Alloc")
                .with_op(EffectOperation::new(
                    "alloc",
                    vec![Type::I64], // size in bytes
                    Type::I64,       // pointer
                ))
                .with_op(EffectOperation::new(
                    "alloc_array",
                    vec![Type::I64, Type::I64], // count, elem_size
                    Type::I64,                  // pointer
                ))
                .with_op(EffectOperation::new(
                    "dealloc",
                    vec![Type::I64], // pointer
                    Type::Unit,
                ))
                .with_op(EffectOperation::new(
                    "realloc",
                    vec![Type::I64, Type::I64], // pointer, new_size
                    Type::I64,                  // new pointer
                ))
                .with_op(EffectOperation::new(
                    "size_of",
                    vec![Type::I64], // pointer
                    Type::I64,       // size
                ))
                .with_op(EffectOperation::new("clear", vec![], Type::Unit))
                .with_op(EffectOperation::new("total_allocated", vec![], Type::I64)),
        );

        // Exception effect
        self.definitions
            .push(EffectDef::new("Exn").with_op(EffectOperation::new(
                "throw",
                vec![Type::String],
                Type::Never,
            )));

        // Async effect
        self.definitions.push(
            EffectDef::new("Async")
                .with_op(EffectOperation::new(
                    "await",
                    vec![Type::Var(TypeVar(0))], // Generic over T
                    Type::Var(TypeVar(0)),
                ))
                .with_op(EffectOperation::new(
                    "spawn",
                    vec![Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Var(TypeVar(0))),
                        effects: EffectSet::new(),
                        abi: None,
                    }],
                    Type::Named {
                        name: "Future".to_string(),
                        args: vec![Type::Var(TypeVar(0))],
                    },
                )),
        );

        // Probabilistic effect
        self.definitions.push(
            EffectDef::new("Prob")
                .with_op(EffectOperation::new(
                    "sample",
                    vec![Type::Named {
                        name: "Distribution".to_string(),
                        args: vec![Type::Var(TypeVar(0))],
                    }],
                    Type::Var(TypeVar(0)),
                ))
                .with_op(EffectOperation::new(
                    "observe",
                    vec![
                        Type::Named {
                            name: "Distribution".to_string(),
                            args: vec![Type::Var(TypeVar(0))],
                        },
                        Type::Var(TypeVar(0)),
                    ],
                    Type::Unit,
                )),
        );

        // GPU effect
        self.definitions.push(
            EffectDef::new("GPU")
                .with_op(EffectOperation::new(
                    "launch",
                    vec![
                        Type::Named {
                            name: "Kernel".to_string(),
                            args: vec![],
                        },
                        Type::Tuple(vec![Type::U32, Type::U32, Type::U32]), // grid dims
                        Type::Tuple(vec![Type::U32, Type::U32, Type::U32]), // block dims
                    ],
                    Type::Unit,
                ))
                .with_op(EffectOperation::new("sync", vec![], Type::Unit)),
        );

        // Epistemic effect - tracks operations that affect confidence/provenance
        //
        // The Epistemic effect enables:
        // - Compile-time tracking of uncertainty propagation
        // - Effect handlers for epistemic firewalls (confidence boundaries)
        // - Safe composition of epistemic computations
        // - Integration with the Knowledge[τ, ε, δ, Φ] type system
        self.definitions.push(
            EffectDef::new("Epistemic")
                // Degrade confidence by a factor
                .with_op(EffectOperation::new(
                    "degrade",
                    vec![Type::F64], // degradation factor
                    Type::Unit,
                ))
                // Assert minimum confidence (may fail at runtime)
                .with_op(EffectOperation::new(
                    "assert_confidence",
                    vec![Type::F64], // minimum required confidence
                    Type::Unit,
                ))
                // Enter a firewall with specific mode
                .with_op(EffectOperation::new(
                    "firewall",
                    vec![Type::Named {
                        name: "FirewallConfig".to_string(),
                        args: vec![],
                    }],
                    Type::Unit,
                ))
                // Record provenance transformation
                .with_op(EffectOperation::new(
                    "record_provenance",
                    vec![Type::String], // transformation name
                    Type::Unit,
                ))
                // Switch uncertainty model
                .with_op(EffectOperation::new(
                    "switch_model",
                    vec![Type::Named {
                        name: "UncertaintyModel".to_string(),
                        args: vec![],
                    }],
                    Type::Unit,
                ))
                // Merge knowledge from multiple sources
                .with_op(EffectOperation::new(
                    "merge",
                    vec![Type::Named {
                        name: "MergeStrategy".to_string(),
                        args: vec![],
                    }],
                    Type::Unit,
                )),
        );

        // Div effect - division that may fail
        self.definitions
            .push(EffectDef::new("Div").with_op(EffectOperation::new(
                "div",
                vec![Type::F64, Type::F64],
                Type::F64,
            )));

        // FFI effect - foreign function interface operations
        //
        // The FFI effect tracks operations that interact with foreign code:
        // - Calling extern "C" functions
        // - Dereferencing raw pointers
        // - Memory operations on foreign data
        //
        // This provides safety guarantees by ensuring FFI operations are:
        // - Explicitly marked in function signatures (with FFI)
        // - Tracked through the type system
        // - Can be handled at module boundaries
        self.definitions.push(
            EffectDef::new("FFI")
                // Call an external function
                .with_op(EffectOperation::new(
                    "call_extern",
                    vec![Type::String], // function name
                    Type::Var(TypeVar(0)),
                ))
                // Read from a raw pointer
                .with_op(EffectOperation::new(
                    "ptr_read",
                    vec![Type::I64], // pointer
                    Type::Var(TypeVar(0)),
                ))
                // Write to a raw pointer
                .with_op(EffectOperation::new(
                    "ptr_write",
                    vec![Type::I64, Type::Var(TypeVar(0))], // pointer, value
                    Type::Unit,
                ))
                // Get pointer address
                .with_op(EffectOperation::new(
                    "addr_of",
                    vec![Type::Var(TypeVar(0))], // reference
                    Type::I64,                   // pointer
                ))
                // Offset a pointer
                .with_op(EffectOperation::new(
                    "ptr_offset",
                    vec![Type::I64, Type::I64], // pointer, offset
                    Type::I64,                  // new pointer
                )),
        );
    }

    /// Create fresh effect variable
    pub fn fresh_var(&mut self) -> EffectVar {
        let v = EffectVar::new(self.vars.len() as u32);
        self.vars.push(v);
        v
    }

    /// Add constraint: e1 ⊆ e2 (e1 is a subset of e2)
    pub fn add_constraint(&mut self, e1: EffectSet, e2: EffectSet) {
        self.constraints.push((e1, e2));
    }

    /// Look up an effect definition
    pub fn lookup_effect(&self, name: &str) -> Option<&EffectDef> {
        self.definitions.iter().find(|e| e.name == name)
    }

    /// Check if an effect operation exists
    pub fn lookup_operation(&self, effect: &str, op: &str) -> Option<&EffectOperation> {
        self.lookup_effect(effect)
            .and_then(|e| e.operations.iter().find(|o| o.name == op))
    }

    /// Solve constraints and return solutions
    pub fn solve(&self) -> Result<Vec<(EffectVar, EffectSet)>, String> {
        // Simple constraint solving: iterate until fixed point
        let mut solutions: std::collections::HashMap<EffectVar, EffectSet> =
            std::collections::HashMap::new();

        // Initialize all variables to empty
        for var in &self.vars {
            solutions.insert(*var, EffectSet::new());
        }

        // Iterate until no changes
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for (e1, e2) in &self.constraints {
                // Get current effect sets
                let s1 = self.resolve_effect_set(e1, &solutions);
                let s2 = self.resolve_effect_set(e2, &solutions);

                // e1 ⊆ e2 means all effects in e1 must be in e2
                // For each var in e2, add all effects from s1
                for var in &e2.effect_vars {
                    if let Some(current) = solutions.get_mut(var) {
                        let old_size = current.effects.len();
                        current.effects.extend(s1.effects.iter().cloned());
                        if current.effects.len() != old_size {
                            changed = true;
                        }
                    }
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            return Err("Effect constraint solving did not converge".to_string());
        }

        Ok(solutions.into_iter().collect())
    }

    fn resolve_effect_set(
        &self,
        set: &EffectSet,
        solutions: &std::collections::HashMap<EffectVar, EffectSet>,
    ) -> EffectSet {
        let mut result = EffectSet::new();
        result.effects = set.effects.clone();

        for var in &set.effect_vars {
            if let Some(resolved) = solutions.get(var) {
                result.effects.extend(resolved.effects.iter().cloned());
            }
        }

        result
    }
}

impl Default for EffectInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if effects are handled
pub fn effects_handled(required: &EffectSet, available: &EffectSet) -> bool {
    required.effects.is_subset(&available.effects)
}

/// Compute the residual effects after handling
pub fn residual_effects(effects: &EffectSet, handled: &[String]) -> EffectSet {
    let mut result = effects.clone();
    for h in handled {
        result.effects.remove(h);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_inference() {
        let mut ctx = EffectInference::new();
        let v1 = ctx.fresh_var();
        let v2 = ctx.fresh_var();

        let mut e1 = EffectSet::new();
        e1.add(Effect::io());
        e1.add_var(v1);

        let mut e2 = EffectSet::new();
        e2.add_var(v2);

        ctx.add_constraint(e1, e2);

        let solutions = ctx.solve().unwrap();
        let v2_solution = solutions.iter().find(|(v, _)| *v == v2).map(|(_, e)| e);
        assert!(v2_solution.is_some());
        assert!(v2_solution.unwrap().contains("IO"));
    }

    #[test]
    fn test_builtin_effects() {
        let ctx = EffectInference::new();

        assert!(ctx.lookup_effect("IO").is_some());
        assert!(ctx.lookup_effect("Prob").is_some());
        assert!(ctx.lookup_effect("GPU").is_some());

        let print_op = ctx.lookup_operation("IO", "print");
        assert!(print_op.is_some());
        assert_eq!(print_op.unwrap().params.len(), 1);
    }

    #[test]
    fn test_residual_effects() {
        let mut effects = EffectSet::new();
        effects.add(Effect::io());
        effects.add(Effect::mut_effect());
        effects.add(Effect::alloc());

        let residual = residual_effects(&effects, &["IO".to_string()]);
        assert!(!residual.contains("IO"));
        assert!(residual.contains("Mut"));
        assert!(residual.contains("Alloc"));
    }
}
