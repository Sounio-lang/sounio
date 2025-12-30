//! Effect handler dispatch for interpreter
//!
//! Implements algebraic effect semantics using a handler stack.
//! This module provides the runtime mechanism for dispatching effect
//! operations (Prob, Causal, IO, etc.) to their registered handlers.
//!
//! # Algebraic Effects Overview
//!
//! Effects in Sounio are first-class: operations like `sample`, `observe`,
//! `do` are effect operations that suspend computation and delegate to
//! a handler. Handlers define how to interpret these operations.
//!
//! # Example
//!
//! ```text
//! // Sounio code:
//! fn coin_model() with Prob {
//!     let p = sample(Beta(1.0, 1.0));  // Prob effect operation
//!     observe(Bernoulli(p), true);      // Prob effect operation
//!     p
//! }
//!
//! // Run with MH handler:
//! handle coin_model() with mh_handler { ... }
//! ```

use std::collections::HashMap;
use std::fmt;

use thiserror::Error;

use super::value::{Distribution, Value};
use crate::runtime::causal;
use crate::runtime::prob::{self, ProbContext, Rng};

/// Errors that can occur during effect dispatch
#[derive(Error, Debug)]
pub enum EffectError {
    /// No handler found for the requested effect
    #[error("unhandled effect `{effect}`: no handler for operation `{operation}`")]
    UnhandledEffect { effect: String, operation: String },

    /// Handler returned an error
    #[error("handler error for `{effect}.{operation}`: {message}")]
    HandlerError {
        effect: String,
        operation: String,
        message: String,
    },

    /// Invalid arguments passed to effect operation
    #[error("invalid arguments for `{effect}.{operation}`: expected {expected}, got {got}")]
    InvalidArguments {
        effect: String,
        operation: String,
        expected: usize,
        got: usize,
    },

    /// Type mismatch in effect operation
    #[error("type mismatch in `{effect}.{operation}`: {message}")]
    TypeMismatch {
        effect: String,
        operation: String,
        message: String,
    },
}

/// Effect kinds supported by the interpreter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// Probabilistic programming effect
    Prob,
    /// Causal inference effect
    Causal,
    /// Input/Output effect
    IO,
    /// Mutable state effect
    Mut,
    /// Memory allocation effect
    Alloc,
    /// Exception effect
    Exn,
    /// Async effect
    Async,
    /// GPU effect
    GPU,
    /// Epistemic (knowledge tracking) effect
    Epistemic,
    /// Division effect (may fail)
    Div,
}

impl EffectKind {
    /// Parse effect kind from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Prob" => Some(EffectKind::Prob),
            "Causal" => Some(EffectKind::Causal),
            "IO" => Some(EffectKind::IO),
            "Mut" => Some(EffectKind::Mut),
            "Alloc" => Some(EffectKind::Alloc),
            "Exn" => Some(EffectKind::Exn),
            "Async" => Some(EffectKind::Async),
            "GPU" => Some(EffectKind::GPU),
            "Epistemic" => Some(EffectKind::Epistemic),
            "Div" => Some(EffectKind::Div),
            _ => None,
        }
    }

    /// Get the string name of this effect
    pub fn as_str(&self) -> &'static str {
        match self {
            EffectKind::Prob => "Prob",
            EffectKind::Causal => "Causal",
            EffectKind::IO => "IO",
            EffectKind::Mut => "Mut",
            EffectKind::Alloc => "Alloc",
            EffectKind::Exn => "Exn",
            EffectKind::Async => "Async",
            EffectKind::GPU => "GPU",
            EffectKind::Epistemic => "Epistemic",
            EffectKind::Div => "Div",
        }
    }
}

impl fmt::Display for EffectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single handler case for an effect operation
pub struct HandlerCase {
    /// Operation name (e.g., "sample", "observe")
    pub operation: String,
    /// Handler function: takes arguments, returns result
    pub handler_fn: Box<dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError>>,
}

impl fmt::Debug for HandlerCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerCase")
            .field("operation", &self.operation)
            .finish()
    }
}

/// State shared across handler invocations
pub struct HandlerState {
    /// Random number generator for probabilistic operations
    pub rng: Rng,
    /// Probabilistic programming context
    pub prob_ctx: ProbContext,
    /// Causal model state (DAG name -> DAG)
    pub causal_models: HashMap<String, causal::DAG>,
    /// Intervention stack for causal operations
    pub interventions: Vec<(String, Value)>,
    /// Observation log for probabilistic conditioning
    pub observations: Vec<(String, Value)>,
    /// Custom state for user-defined handlers
    pub custom: HashMap<String, Value>,
}

impl HandlerState {
    /// Create new handler state with default seed
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    /// Create new handler state with specified seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            prob_ctx: ProbContext::new(seed),
            causal_models: HashMap::new(),
            interventions: Vec::new(),
            observations: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Reset state for a new execution
    pub fn reset(&mut self) {
        self.prob_ctx.reset();
        self.interventions.clear();
        self.observations.clear();
    }
}

impl Default for HandlerState {
    fn default() -> Self {
        Self::new()
    }
}

/// An effect handler with cases for each operation
pub struct EffectHandler {
    /// Which effect this handler handles
    pub effect: EffectKind,
    /// Handler cases for each operation
    pub cases: Vec<HandlerCase>,
    /// Handler name (for debugging/display)
    pub name: String,
}

impl EffectHandler {
    /// Create a new effect handler
    pub fn new(effect: EffectKind, name: &str) -> Self {
        Self {
            effect,
            cases: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Add a handler case
    pub fn with_case<F>(mut self, operation: &str, handler: F) -> Self
    where
        F: Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> + 'static,
    {
        self.cases.push(HandlerCase {
            operation: operation.to_string(),
            handler_fn: Box::new(handler),
        });
        self
    }

    /// Find a handler case for an operation
    pub fn find_case(&self, operation: &str) -> Option<&HandlerCase> {
        self.cases.iter().find(|c| c.operation == operation)
    }
}

impl fmt::Debug for EffectHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectHandler")
            .field("effect", &self.effect)
            .field("name", &self.name)
            .field("cases", &self.cases.iter().map(|c| &c.operation).collect::<Vec<_>>())
            .finish()
    }
}

/// Effect handler context managing a stack of handlers
pub struct EffectContext {
    /// Stack of active handlers (most recent on top)
    handler_stack: Vec<EffectHandler>,
    /// Shared state across handlers
    pub state: HandlerState,
}

impl EffectContext {
    /// Create a new effect context with default handlers
    pub fn new() -> Self {
        let mut ctx = Self {
            handler_stack: Vec::new(),
            state: HandlerState::new(),
        };

        // Install default handlers for Prob and Causal effects
        ctx.push_handler(default_prob_handler());
        ctx.push_handler(default_causal_handler());

        ctx
    }

    /// Create a new effect context with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        let mut ctx = Self {
            handler_stack: Vec::new(),
            state: HandlerState::with_seed(seed),
        };

        ctx.push_handler(default_prob_handler());
        ctx.push_handler(default_causal_handler());

        ctx
    }

    /// Push a handler onto the stack
    pub fn push_handler(&mut self, handler: EffectHandler) {
        self.handler_stack.push(handler);
    }

    /// Pop a handler from the stack
    pub fn pop_handler(&mut self) -> Option<EffectHandler> {
        self.handler_stack.pop()
    }

    /// Find a handler for the given effect and operation
    fn find_handler(&self, effect: EffectKind, operation: &str) -> Option<&HandlerCase> {
        // Search from top of stack (most recently pushed) to bottom
        for handler in self.handler_stack.iter().rev() {
            if handler.effect == effect {
                if let Some(case) = handler.find_case(operation) {
                    return Some(case);
                }
            }
        }
        None
    }

    /// Dispatch an effect operation to the appropriate handler
    pub fn dispatch(
        &mut self,
        effect: EffectKind,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        // Find the handler index first
        let handler_idx = self
            .handler_stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, handler)| {
                if handler.effect == effect && handler.find_case(operation).is_some() {
                    Some(idx)
                } else {
                    None
                }
            })
            .ok_or_else(|| EffectError::UnhandledEffect {
                effect: effect.to_string(),
                operation: operation.to_string(),
            })?;

        // Get pointer to handler function to avoid holding borrow on handler_stack
        // SAFETY: We know the index is valid and we don't modify handler_stack
        // during the call
        let handler_fn_ptr: *const dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> = {
            let handler = &self.handler_stack[handler_idx];
            let case = handler.find_case(operation).unwrap();
            &*case.handler_fn as *const _
        };

        // Now call with mutable state - the handler_stack borrow is released
        // SAFETY: handler_fn_ptr is valid for the duration of this call
        unsafe { (*handler_fn_ptr)(&args, &mut self.state) }
    }

    /// Dispatch using effect name as string
    pub fn dispatch_by_name(
        &mut self,
        effect_name: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        let effect = EffectKind::from_str(effect_name).ok_or_else(|| {
            EffectError::UnhandledEffect {
                effect: effect_name.to_string(),
                operation: operation.to_string(),
            }
        })?;
        self.dispatch(effect, operation, args)
    }

    /// Convenience method for Prob.sample
    pub fn sample(&mut self, distribution: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Prob, "sample", vec![distribution])
    }

    /// Convenience method for Prob.observe
    pub fn observe(&mut self, distribution: Value, value: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Prob, "observe", vec![distribution, value])
    }

    /// Convenience method for Causal.do
    pub fn do_intervention(&mut self, variable: Value, value: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Causal, "do", vec![variable, value])
    }

    /// Convenience method for Causal.counterfactual
    pub fn counterfactual(
        &mut self,
        factual: Value,
        intervention: Value,
        query: Value,
    ) -> Result<Value, EffectError> {
        self.dispatch(
            EffectKind::Causal,
            "counterfactual",
            vec![factual, intervention, query],
        )
    }
}

impl Default for EffectContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for EffectContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectContext")
            .field("handler_stack_size", &self.handler_stack.len())
            .field(
                "handlers",
                &self
                    .handler_stack
                    .iter()
                    .map(|h| format!("{}:{}", h.effect, h.name))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

// =============================================================================
// Default Handlers
// =============================================================================

/// Create the default handler for the Prob effect
pub fn default_prob_handler() -> EffectHandler {
    EffectHandler::new(EffectKind::Prob, "default_prob")
        .with_case("sample", |args, state| {
            // Sample from a distribution
            if args.is_empty() {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "sample".to_string(),
                    expected: 1,
                    got: 0,
                });
            }

            let dist = &args[0];
            match dist {
                Value::Distribution(d) => {
                    let value = sample_from_distribution(d, &mut state.rng);
                    Ok(Value::Float(value))
                }
                Value::Struct { name, fields } => {
                    // Handle distribution structs (e.g., Normal { mean: 0.0, std: 1.0 })
                    let prob_dist = value_to_prob_distribution(name, fields)?;
                    let value = prob_dist.sample(&mut state.rng);
                    Ok(Value::Float(value))
                }
                _ => Err(EffectError::TypeMismatch {
                    effect: "Prob".to_string(),
                    operation: "sample".to_string(),
                    message: format!("expected Distribution, got {:?}", dist.type_name()),
                }),
            }
        })
        .with_case("observe", |args, state| {
            // Observe (condition on) a value
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "observe".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let dist = &args[0];
            let observed_value = &args[1];

            let obs_float = match observed_value {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Prob".to_string(),
                        operation: "observe".to_string(),
                        message: "observed value must be numeric".to_string(),
                    });
                }
            };

            match dist {
                Value::Distribution(d) => {
                    let prob_dist = interpreter_dist_to_prob(d);
                    state.prob_ctx.observe(&prob_dist, obs_float);
                    Ok(Value::Unit)
                }
                Value::Struct { name, fields } => {
                    let prob_dist = value_to_prob_distribution(name, fields)?;
                    state.prob_ctx.observe(&prob_dist, obs_float);
                    Ok(Value::Unit)
                }
                _ => Err(EffectError::TypeMismatch {
                    effect: "Prob".to_string(),
                    operation: "observe".to_string(),
                    message: format!("expected Distribution, got {:?}", dist.type_name()),
                }),
            }
        })
        .with_case("score", |args, state| {
            // Add log probability to the trace
            if args.is_empty() {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "score".to_string(),
                    expected: 1,
                    got: 0,
                });
            }

            let log_prob = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Prob".to_string(),
                        operation: "score".to_string(),
                        message: "log_prob must be numeric".to_string(),
                    });
                }
            };

            // Add to log probability in context
            // (ProbContext tracks this internally)
            state.prob_ctx.observe(
                &prob::Distribution::Normal { mean: 0.0, std: 1.0 },
                0.0, // Dummy observation
            );
            // Note: Full implementation would directly add log_prob

            Ok(Value::Float(log_prob))
        })
}

/// Create the default handler for the Causal effect
pub fn default_causal_handler() -> EffectHandler {
    EffectHandler::new(EffectKind::Causal, "default_causal")
        .with_case("do", |args, state| {
            // do(variable, value) - intervene on a variable
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "do".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let variable = match &args[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Causal".to_string(),
                        operation: "do".to_string(),
                        message: "variable must be a string".to_string(),
                    });
                }
            };

            let value = args[1].clone();

            // Record intervention
            state.interventions.push((variable.clone(), value.clone()));

            // Return the intervention value
            Ok(value)
        })
        .with_case("counterfactual", |args, state| {
            // counterfactual(factual, intervention, query)
            // Three-step process: Abduction, Action, Prediction
            if args.len() < 3 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "counterfactual".to_string(),
                    expected: 3,
                    got: args.len(),
                });
            }

            // For the skeleton, return the query value directly
            // Full implementation would:
            // 1. Abduction: infer latent variables from factual
            // 2. Action: apply intervention
            // 3. Prediction: compute outcome under modified model
            let query = args[2].clone();
            Ok(query)
        })
        .with_case("observe_causal", |args, state| {
            // Observe a causal variable (for abduction)
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "observe_causal".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let variable = match &args[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Causal".to_string(),
                        operation: "observe_causal".to_string(),
                        message: "variable must be a string".to_string(),
                    });
                }
            };

            let value = args[1].clone();
            state.observations.push((variable, value.clone()));

            Ok(value)
        })
        .with_case("query", |args, _state| {
            // P(target | given, do(interventions))
            // Simplified: return target expression value
            if args.is_empty() {
                return Ok(Value::Unit);
            }
            Ok(args[0].clone())
        })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Sample from an interpreter Distribution value
fn sample_from_distribution(dist: &Distribution, rng: &mut Rng) -> f64 {
    match dist {
        Distribution::Normal { mean, std } => mean + std * rng.next_normal(),
        Distribution::Uniform { a, b } => a + (b - a) * rng.next_f64(),
        Distribution::Beta { alpha, beta } => {
            // Use runtime prob module
            let prob_dist = prob::Distribution::Beta {
                alpha: *alpha,
                beta: *beta,
            };
            prob_dist.sample(rng)
        }
        Distribution::Exponential { lambda } => {
            let prob_dist = prob::Distribution::Exponential { rate: *lambda };
            prob_dist.sample(rng)
        }
        Distribution::Categorical { probs } => {
            let prob_dist = prob::Distribution::Categorical {
                probs: probs.clone(),
            };
            prob_dist.sample(rng)
        }
    }
}

/// Convert interpreter Distribution to runtime prob Distribution
fn interpreter_dist_to_prob(dist: &Distribution) -> prob::Distribution {
    match dist {
        Distribution::Normal { mean, std } => prob::Distribution::Normal {
            mean: *mean,
            std: *std,
        },
        Distribution::Uniform { a, b } => prob::Distribution::Uniform {
            low: *a,
            high: *b,
        },
        Distribution::Beta { alpha, beta } => prob::Distribution::Beta {
            alpha: *alpha,
            beta: *beta,
        },
        Distribution::Exponential { lambda } => prob::Distribution::Exponential { rate: *lambda },
        Distribution::Categorical { probs } => prob::Distribution::Categorical {
            probs: probs.clone(),
        },
    }
}

/// Convert a Value struct to a prob::Distribution
fn value_to_prob_distribution(
    name: &str,
    fields: &HashMap<String, Value>,
) -> Result<prob::Distribution, EffectError> {
    let get_float = |field: &str| -> Result<f64, EffectError> {
        fields
            .get(field)
            .and_then(|v| match v {
                Value::Float(f) => Some(*f),
                Value::Int(i) => Some(*i as f64),
                _ => None,
            })
            .ok_or_else(|| EffectError::TypeMismatch {
                effect: "Prob".to_string(),
                operation: "sample".to_string(),
                message: format!("missing or invalid field `{}`", field),
            })
    };

    match name {
        "Normal" => Ok(prob::Distribution::Normal {
            mean: get_float("mean")?,
            std: get_float("std").or_else(|_| get_float("stddev"))?,
        }),
        "Uniform" => Ok(prob::Distribution::Uniform {
            low: get_float("low").or_else(|_| get_float("a"))?,
            high: get_float("high").or_else(|_| get_float("b"))?,
        }),
        "Beta" => Ok(prob::Distribution::Beta {
            alpha: get_float("alpha")?,
            beta: get_float("beta")?,
        }),
        "Gamma" => Ok(prob::Distribution::Gamma {
            shape: get_float("shape")?,
            rate: get_float("rate")?,
        }),
        "Exponential" => Ok(prob::Distribution::Exponential {
            rate: get_float("rate").or_else(|_| get_float("lambda"))?,
        }),
        "Bernoulli" => Ok(prob::Distribution::Bernoulli {
            p: get_float("p")?,
        }),
        "Poisson" => Ok(prob::Distribution::Poisson {
            lambda: get_float("lambda")?,
        }),
        _ => Err(EffectError::TypeMismatch {
            effect: "Prob".to_string(),
            operation: "sample".to_string(),
            message: format!("unknown distribution type `{}`", name),
        }),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_context_creation() {
        let ctx = EffectContext::new();
        assert!(ctx.handler_stack.len() >= 2); // Prob and Causal handlers
    }

    #[test]
    fn test_effect_kind_from_str() {
        assert_eq!(EffectKind::from_str("Prob"), Some(EffectKind::Prob));
        assert_eq!(EffectKind::from_str("Causal"), Some(EffectKind::Causal));
        assert_eq!(EffectKind::from_str("IO"), Some(EffectKind::IO));
        assert_eq!(EffectKind::from_str("Unknown"), None);
    }

    #[test]
    fn test_prob_sample_dispatch() {
        let mut ctx = EffectContext::with_seed(42);

        // Create a Normal distribution value
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        // Sample should succeed
        let result = ctx.sample(dist);
        assert!(result.is_ok());

        if let Ok(Value::Float(v)) = result {
            // Value should be reasonable for N(0, 1)
            assert!(v > -10.0 && v < 10.0);
        } else {
            panic!("Expected Float value from sample");
        }
    }

    #[test]
    fn test_prob_observe_dispatch() {
        let mut ctx = EffectContext::with_seed(42);

        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.observe(dist, Value::Float(0.5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_do_dispatch() {
        let mut ctx = EffectContext::new();

        let result = ctx.do_intervention(
            Value::String("X".to_string()),
            Value::Float(1.0),
        );

        assert!(result.is_ok());
        assert_eq!(ctx.state.interventions.len(), 1);
        assert_eq!(ctx.state.interventions[0].0, "X");
    }

    #[test]
    fn test_unhandled_effect() {
        let mut ctx = EffectContext::new();

        // Try to dispatch an operation that doesn't exist
        let result = ctx.dispatch(EffectKind::Prob, "nonexistent_op", vec![]);
        assert!(result.is_err());

        if let Err(EffectError::UnhandledEffect { effect, operation }) = result {
            assert_eq!(effect, "Prob");
            assert_eq!(operation, "nonexistent_op");
        }
    }

    #[test]
    fn test_handler_stack() {
        let mut ctx = EffectContext::new();
        let initial_count = ctx.handler_stack.len();

        // Push a custom handler
        let custom_handler = EffectHandler::new(EffectKind::Prob, "custom_prob")
            .with_case("sample", |_args, _state| Ok(Value::Float(42.0)));

        ctx.push_handler(custom_handler);
        assert_eq!(ctx.handler_stack.len(), initial_count + 1);

        // Custom handler should take precedence
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.sample(dist);
        assert!(matches!(result, Ok(Value::Float(v)) if (v - 42.0).abs() < 0.001));

        // Pop and verify default handler is used again
        ctx.pop_handler();
        assert_eq!(ctx.handler_stack.len(), initial_count);
    }

    #[test]
    fn test_dispatch_by_name() {
        let mut ctx = EffectContext::with_seed(123);

        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(5.0));
        fields.insert("std".to_string(), Value::Float(0.1));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.dispatch_by_name("Prob", "sample", vec![dist]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_handler_state_reset() {
        let mut state = HandlerState::new();

        state.interventions.push(("X".to_string(), Value::Float(1.0)));
        state.observations.push(("Y".to_string(), Value::Float(2.0)));

        assert!(!state.interventions.is_empty());
        assert!(!state.observations.is_empty());

        state.reset();

        assert!(state.interventions.is_empty());
        assert!(state.observations.is_empty());
    }

    #[test]
    fn test_distribution_conversion() {
        // Test value_to_prob_distribution for various distribution types
        let mut fields = HashMap::new();
        fields.insert("alpha".to_string(), Value::Float(2.0));
        fields.insert("beta".to_string(), Value::Float(5.0));

        let result = value_to_prob_distribution("Beta", &fields);
        assert!(result.is_ok());

        if let Ok(prob::Distribution::Beta { alpha, beta }) = result {
            assert!((alpha - 2.0).abs() < 0.001);
            assert!((beta - 5.0).abs() < 0.001);
        }
    }
}
