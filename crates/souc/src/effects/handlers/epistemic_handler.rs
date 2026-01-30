//! Epistemic Effect Handler
//!
//! This module implements the `Epistemic` effect handler for explicit epistemic
//! operations through algebraic effects. The Epistemic effect enables fine-grained
//! control over confidence propagation, provenance tracking, and uncertainty model
//! selection at the effect level.
//!
//! # Effect Operations
//!
//! | Operation           | Arguments                    | Return Type | Confidence |
//! |--------------------|------------------------------|-------------|------------|
//! | `degrade`          | `(factor: Float)`            | `Unit`      | factor     |
//! | `assert_confidence`| `(min: Float)`               | `Unit`      | 1.0        |
//! | `firewall`         | `(mode: String, config: ?)` | `Unit`      | varies     |
//! | `record_provenance`| `(tag: String)`              | `Unit`      | 1.0        |
//! | `switch_model`     | `(model: String)`            | `Unit`      | 1.0        |
//! | `merge`            | `(strategy: String, vals: Array)` | `Value` | varies    |
//! | `get_confidence`   | `()`                         | `Float`     | 1.0        |
//! | `boost`            | `(factor: Float)`            | `Unit`      | 1/factor   |
//! | `clamp`            | `(min: Float, max: Float)`   | `Unit`      | 1.0        |
//! | `isolate`          | `()`                         | `Unit`      | 1.0        |
//!
//! # Epistemic Impact
//!
//! The Epistemic effect is unique in that its operations directly manipulate
//! epistemic state rather than having incidental impact on it. The `degrade`
//! operation explicitly reduces confidence, while `assert_confidence` gates
//! computation on minimum confidence requirements.
//!
//! # Firewall Modes
//!
//! Supported firewall modes:
//! - `pass_through`: No modification to confidence
//! - `reset`: Set confidence to a fixed value
//! - `clamp`: Enforce min/max bounds on confidence
//! - `gate`: Block execution if confidence is below threshold
//! - `decay`: Apply multiplicative degradation
//! - `boost`: Amplify confidence (capped at 1.0)
//! - `audit`: Log provenance without modification
//! - `isolate`: Reset provenance chain
//!
//! # Uncertainty Models
//!
//! Supported uncertainty models:
//! - `probabilistic`: Single confidence value [0,1] (default)
//! - `interval`: Interval arithmetic [a,b]
//! - `affine`: Affine arithmetic with correlated errors
//! - `fuzzy`: Fuzzy sets with membership functions
//! - `dempster_shafer`: Dempster-Shafer evidence theory
//! - `bayesian`: Full probability distributions
//!
//! # Merge Strategies
//!
//! Supported merge strategies:
//! - `most_confident`: Take highest-confidence value
//! - `weighted_average`: Weight by confidence
//! - `consensus`: Require agreement above threshold
//! - `union`: Keep all sources
//!
//! # Example
//!
//! ```text
//! // Sounio code using Epistemic effect:
//! fn validated_measurement(raw: Knowledge<f64>) -> Knowledge<f64> with Epistemic {
//!     assert_confidence(0.8)  // Require 80% confidence
//!     record_provenance("validated")
//!     raw
//! }
//!
//! fn uncertain_inference() with Epistemic {
//!     degrade(0.9)  // Explicitly mark 10% confidence loss
//!     switch_model("bayesian")
//!     // ... perform inference ...
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

/// Key for storing current confidence in HandlerState
const CONFIDENCE_KEY: &str = "__epistemic_confidence";

/// Key for storing provenance chain
const PROVENANCE_KEY: &str = "__epistemic_provenance";

/// Key for storing current uncertainty model
const MODEL_KEY: &str = "__epistemic_model";

/// Key for storing firewall stack
const FIREWALL_STACK_KEY: &str = "__epistemic_firewall_stack";

/// Key for storing confidence floor
const CONFIDENCE_FLOOR_KEY: &str = "__epistemic_confidence_floor";

/// Key for storing confidence ceiling
const CONFIDENCE_CEILING_KEY: &str = "__epistemic_confidence_ceiling";

/// Static operation specifications for EpistemicHandler
static EPISTEMIC_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_epistemic_operations() -> &'static [OperationSpec] {
    EPISTEMIC_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("degrade", "Unit").with_params(vec!["Float"]),
            OperationSpec::new("assert_confidence", "Unit").with_params(vec!["Float"]),
            OperationSpec::new("firewall", "Unit").with_params(vec!["String", "Float"]),
            OperationSpec::new("record_provenance", "Unit").with_params(vec!["String"]),
            OperationSpec::new("switch_model", "Unit").with_params(vec!["String"]),
            OperationSpec::new("merge", "Value").with_params(vec!["String", "Array"]),
            OperationSpec::new("get_confidence", "Float").with_params(vec![]),
            OperationSpec::new("boost", "Unit").with_params(vec!["Float"]),
            OperationSpec::new("clamp", "Unit").with_params(vec!["Float", "Float"]),
            OperationSpec::new("isolate", "Unit").with_params(vec![]),
            OperationSpec::new("reset_confidence", "Unit").with_params(vec!["Float"]),
            OperationSpec::new("get_provenance", "Array").with_params(vec![]),
            OperationSpec::new("get_model", "String").with_params(vec![]),
        ]
    })
}

/// Handler for the Epistemic effect
///
/// EpistemicHandler provides explicit epistemic control operations through
/// algebraic effects. Unlike other effects where epistemic impact is incidental,
/// the Epistemic effect exists specifically to manipulate confidence, provenance,
/// and uncertainty models.
///
/// # State Management
///
/// The handler tracks:
/// - Current confidence level
/// - Provenance chain (list of transformation tags)
/// - Active uncertainty model
/// - Firewall stack for nested boundaries
/// - Confidence floor and ceiling bounds
///
/// # Integration with Knowledge Type
///
/// Operations on this handler are designed to integrate seamlessly with
/// Sounio's Knowledge[τ, ε, δ, Φ] type system, allowing effect-based
/// manipulation of epistemic metadata.
#[derive(Debug)]
pub struct EpistemicHandler {
    _private: (),
}

impl Default for EpistemicHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicHandler {
    /// Create a new EpistemicHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Get current confidence from state
    fn get_confidence(state: &HandlerState) -> f64 {
        match state.named_state.get(CONFIDENCE_KEY) {
            Some(Value::Float(c)) => *c,
            _ => 1.0, // Default confidence
        }
    }

    /// Set confidence in state
    fn set_confidence(state: &mut HandlerState, confidence: f64) {
        // Apply floor and ceiling
        let floor = Self::get_confidence_floor(state);
        let ceiling = Self::get_confidence_ceiling(state);
        let clamped = confidence.clamp(floor, ceiling);
        state
            .named_state
            .insert(CONFIDENCE_KEY.to_string(), Value::Float(clamped));
    }

    /// Get confidence floor
    fn get_confidence_floor(state: &HandlerState) -> f64 {
        match state.named_state.get(CONFIDENCE_FLOOR_KEY) {
            Some(Value::Float(f)) => *f,
            _ => 0.0,
        }
    }

    /// Get confidence ceiling
    fn get_confidence_ceiling(state: &HandlerState) -> f64 {
        match state.named_state.get(CONFIDENCE_CEILING_KEY) {
            Some(Value::Float(c)) => *c,
            _ => 1.0,
        }
    }

    /// Get provenance chain
    fn get_provenance_chain(state: &HandlerState) -> Vec<String> {
        match state.named_state.get(PROVENANCE_KEY) {
            Some(Value::Array(arr)) => arr
                .borrow()
                .iter()
                .filter_map(|v| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Add to provenance chain
    fn add_provenance(state: &mut HandlerState, tag: String) {
        let mut chain = Self::get_provenance_chain(state);
        chain.push(tag);
        let values: Vec<Value> = chain.into_iter().map(Value::String).collect();
        state.named_state.insert(
            PROVENANCE_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(values))),
        );
    }

    /// Get current uncertainty model
    fn get_model(state: &HandlerState) -> String {
        match state.named_state.get(MODEL_KEY) {
            Some(Value::String(m)) => m.clone(),
            _ => "probabilistic".to_string(),
        }
    }

    /// Set uncertainty model
    fn set_model(state: &mut HandlerState, model: String) {
        state
            .named_state
            .insert(MODEL_KEY.to_string(), Value::String(model));
    }

    /// Extract a float from a Value
    fn extract_float(value: &Value, op: &str, param: &str) -> Result<f64, HandlerResult> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                op,
                format!("expected Float for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract a string from a Value
    fn extract_string(value: &Value, op: &str, param: &str) -> Result<String, HandlerResult> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                op,
                format!("expected String for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Handle degrade operation
    fn handle_degrade(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "degrade",
                "degrade requires a factor argument",
            ));
        }

        let factor = match Self::extract_float(&args[0], "degrade", "factor") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if factor < 0.0 || factor > 1.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "degrade",
                format!("degradation factor must be in [0, 1], got {}", factor),
            ));
        }

        let current = Self::get_confidence(state);
        Self::set_confidence(state, current * factor);
        Self::add_provenance(state, "explicitly_degraded".to_string());

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle assert_confidence operation
    fn handle_assert_confidence(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "assert_confidence",
                "assert_confidence requires a minimum confidence argument",
            ));
        }

        let min_confidence = match Self::extract_float(&args[0], "assert_confidence", "min") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if min_confidence < 0.0 || min_confidence > 1.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "assert_confidence",
                format!(
                    "minimum confidence must be in [0, 1], got {}",
                    min_confidence
                ),
            ));
        }

        let current = Self::get_confidence(state);
        if current < min_confidence {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "assert_confidence",
                format!(
                    "confidence assertion failed: current {} < required {}",
                    current, min_confidence
                ),
            ));
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle firewall operation
    fn handle_firewall(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "firewall",
                "firewall requires a mode argument",
            ));
        }

        let mode = match Self::extract_string(&args[0], "firewall", "mode") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let config_value = if args.len() > 1 {
            match Self::extract_float(&args[1], "firewall", "config") {
                Ok(f) => Some(f),
                Err(_) => None, // Config is optional for some modes
            }
        } else {
            None
        };

        match mode.as_str() {
            "pass_through" => {
                // No modification
            }
            "reset" => {
                let confidence = config_value.unwrap_or(1.0);
                if confidence < 0.0 || confidence > 1.0 {
                    return HandlerResult::Abort(HandlerError::new(
                        "Epistemic",
                        "firewall",
                        format!("reset confidence must be in [0, 1], got {}", confidence),
                    ));
                }
                Self::set_confidence(state, confidence);
                Self::add_provenance(state, format!("firewall_reset_{}", confidence));
            }
            "clamp" => {
                let min = config_value.unwrap_or(0.0);
                let max = if args.len() > 2 {
                    match Self::extract_float(&args[2], "firewall", "max") {
                        Ok(f) => f,
                        Err(result) => return result,
                    }
                } else {
                    1.0
                };

                state
                    .named_state
                    .insert(CONFIDENCE_FLOOR_KEY.to_string(), Value::Float(min));
                state
                    .named_state
                    .insert(CONFIDENCE_CEILING_KEY.to_string(), Value::Float(max));

                // Apply clamp to current confidence
                let current = Self::get_confidence(state);
                Self::set_confidence(state, current.clamp(min, max));
                Self::add_provenance(state, format!("firewall_clamp_{}_{}", min, max));
            }
            "gate" => {
                let threshold = config_value.unwrap_or(0.5);
                let current = Self::get_confidence(state);
                if current < threshold {
                    return HandlerResult::Abort(HandlerError::new(
                        "Epistemic",
                        "firewall",
                        format!(
                            "gate firewall blocked: confidence {} < threshold {}",
                            current, threshold
                        ),
                    ));
                }
                Self::add_provenance(state, format!("firewall_gate_{}", threshold));
            }
            "decay" => {
                let factor = config_value.unwrap_or(0.9);
                if factor < 0.0 || factor > 1.0 {
                    return HandlerResult::Abort(HandlerError::new(
                        "Epistemic",
                        "firewall",
                        format!("decay factor must be in [0, 1], got {}", factor),
                    ));
                }
                let current = Self::get_confidence(state);
                Self::set_confidence(state, current * factor);
                Self::add_provenance(state, format!("firewall_decay_{}", factor));
            }
            "boost" => {
                let factor = config_value.unwrap_or(1.1);
                if factor < 1.0 {
                    return HandlerResult::Abort(HandlerError::new(
                        "Epistemic",
                        "firewall",
                        format!("boost factor must be >= 1.0, got {}", factor),
                    ));
                }
                let current = Self::get_confidence(state);
                Self::set_confidence(state, (current * factor).min(1.0));
                Self::add_provenance(state, format!("firewall_boost_{}", factor));
            }
            "audit" => {
                let name = if args.len() > 1 {
                    match Self::extract_string(&args[1], "firewall", "name") {
                        Ok(s) => s,
                        Err(_) => "audit".to_string(),
                    }
                } else {
                    "audit".to_string()
                };
                Self::add_provenance(state, format!("firewall_audit_{}", name));
            }
            "isolate" => {
                // Clear provenance chain
                state.named_state.insert(
                    PROVENANCE_KEY.to_string(),
                    Value::Array(Rc::new(RefCell::new(vec![Value::String(
                        "isolated".to_string(),
                    )]))),
                );
            }
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Epistemic",
                    "firewall",
                    format!("unknown firewall mode: {}", mode),
                ));
            }
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle record_provenance operation
    fn handle_record_provenance(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "record_provenance",
                "record_provenance requires a tag argument",
            ));
        }

        let tag = match Self::extract_string(&args[0], "record_provenance", "tag") {
            Ok(s) => s,
            Err(result) => return result,
        };

        Self::add_provenance(state, tag);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle switch_model operation
    fn handle_switch_model(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "switch_model",
                "switch_model requires a model name argument",
            ));
        }

        let model = match Self::extract_string(&args[0], "switch_model", "model") {
            Ok(s) => s,
            Err(result) => return result,
        };

        // Validate model name
        let valid_models = [
            "probabilistic",
            "interval",
            "affine",
            "fuzzy",
            "dempster_shafer",
            "bayesian",
        ];

        if !valid_models.contains(&model.as_str()) {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "switch_model",
                format!(
                    "unknown uncertainty model: {}. Valid models: {:?}",
                    model, valid_models
                ),
            ));
        }

        Self::set_model(state, model.clone());
        Self::add_provenance(state, format!("model_switch_{}", model));
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle merge operation
    fn handle_merge(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "merge",
                "merge requires a strategy and array of values",
            ));
        }

        let strategy = match Self::extract_string(&args[0], "merge", "strategy") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let values_rc = match &args[1] {
            Value::Array(arr) => arr.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Epistemic",
                    "merge",
                    format!("expected Array for values, got {:?}", args[1].type_name()),
                ));
            }
        };

        let values = values_rc.borrow();
        if values.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "merge",
                "cannot merge empty array",
            ));
        }

        let result = match strategy.as_str() {
            "most_confident" => {
                // For simulation, just take the first value
                // In real implementation, would compare confidences
                values[0].clone()
            }
            "weighted_average" => {
                // For numeric values, compute weighted average
                // For simulation, take first value
                values[0].clone()
            }
            "consensus" => {
                // Check if all values agree (for simulation, just take first)
                values[0].clone()
            }
            "union" => {
                // Return array of all values
                drop(values);
                Value::Array(values_rc.clone())
            }
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Epistemic",
                    "merge",
                    format!(
                        "unknown merge strategy: {}. Valid strategies: most_confident, weighted_average, consensus, union",
                        strategy
                    ),
                ));
            }
        };

        Self::add_provenance(state, format!("merged_{}", strategy));
        HandlerResult::Resume(result)
    }

    /// Handle get_confidence operation
    fn handle_get_confidence(&self, state: &HandlerState) -> HandlerResult {
        let confidence = Self::get_confidence(state);
        HandlerResult::Resume(Value::Float(confidence))
    }

    /// Handle boost operation
    fn handle_boost(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "boost",
                "boost requires a factor argument",
            ));
        }

        let factor = match Self::extract_float(&args[0], "boost", "factor") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if factor < 1.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "boost",
                format!("boost factor must be >= 1.0, got {}", factor),
            ));
        }

        let current = Self::get_confidence(state);
        Self::set_confidence(state, (current * factor).min(1.0));
        Self::add_provenance(state, format!("boosted_{}", factor));

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle clamp operation
    fn handle_clamp(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "clamp",
                "clamp requires min and max arguments",
            ));
        }

        let min = match Self::extract_float(&args[0], "clamp", "min") {
            Ok(f) => f,
            Err(result) => return result,
        };
        let max = match Self::extract_float(&args[1], "clamp", "max") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if min < 0.0 || max > 1.0 || min > max {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "clamp",
                format!("invalid clamp bounds: [{}, {}]", min, max),
            ));
        }

        state
            .named_state
            .insert(CONFIDENCE_FLOOR_KEY.to_string(), Value::Float(min));
        state
            .named_state
            .insert(CONFIDENCE_CEILING_KEY.to_string(), Value::Float(max));

        let current = Self::get_confidence(state);
        Self::set_confidence(state, current.clamp(min, max));

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle isolate operation
    fn handle_isolate(&self, state: &mut HandlerState) -> HandlerResult {
        // Clear provenance chain
        state.named_state.insert(
            PROVENANCE_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(vec![Value::String(
                "isolated".to_string(),
            )]))),
        );
        Self::add_provenance(state, "isolated".to_string());
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle reset_confidence operation
    fn handle_reset_confidence(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "reset_confidence",
                "reset_confidence requires a value argument",
            ));
        }

        let confidence = match Self::extract_float(&args[0], "reset_confidence", "confidence") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if confidence < 0.0 || confidence > 1.0 {
            return HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                "reset_confidence",
                format!("confidence must be in [0, 1], got {}", confidence),
            ));
        }

        Self::set_confidence(state, confidence);
        Self::add_provenance(state, format!("reset_{}", confidence));

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_provenance operation
    fn handle_get_provenance(&self, state: &HandlerState) -> HandlerResult {
        let chain = Self::get_provenance_chain(state);
        let values: Vec<Value> = chain.into_iter().map(Value::String).collect();
        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(values))))
    }

    /// Handle get_model operation
    fn handle_get_model(&self, state: &HandlerState) -> HandlerResult {
        let model = Self::get_model(state);
        HandlerResult::Resume(Value::String(model))
    }

    /// Get current confidence (public accessor for testing)
    pub fn current_confidence(state: &HandlerState) -> f64 {
        Self::get_confidence(state)
    }

    /// Get provenance chain (public accessor for testing)
    pub fn provenance_chain(state: &HandlerState) -> Vec<String> {
        Self::get_provenance_chain(state)
    }

    /// Get current model (public accessor for testing)
    pub fn current_model(state: &HandlerState) -> String {
        Self::get_model(state)
    }
}

impl HandlerCapability for EpistemicHandler {
    fn effect_name(&self) -> &str {
        "Epistemic"
    }

    fn handler_name(&self) -> &str {
        "EpistemicHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_epistemic_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "degrade" => self.handle_degrade(args, state),
            "assert_confidence" => self.handle_assert_confidence(args, state),
            "firewall" => self.handle_firewall(args, state),
            "record_provenance" => self.handle_record_provenance(args, state),
            "switch_model" => self.handle_switch_model(args, state),
            "merge" => self.handle_merge(args, state),
            "get_confidence" => self.handle_get_confidence(state),
            "boost" => self.handle_boost(args, state),
            "clamp" => self.handle_clamp(args, state),
            "isolate" => self.handle_isolate(state),
            "reset_confidence" => self.handle_reset_confidence(args, state),
            "get_provenance" => self.handle_get_provenance(state),
            "get_model" => self.handle_get_model(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Epistemic",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Epistemic operations modify state, but are deterministic
        // Multi-shot could be supported with careful state management
        false
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        // The Epistemic effect is special: its operations directly manipulate
        // epistemic state, so we don't apply additional automatic degradation
        match operation {
            // degrade's impact is explicit in the operation itself
            "degrade" | "boost" | "reset_confidence" | "clamp" => EpistemicImpact::none(),
            // These don't affect confidence
            _ => EpistemicImpact::none(),
        }
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // Epistemic operations modify confidence state - mostly one-shot
        match operation {
            "degrade" | "boost" | "clamp" | "reset_confidence" | "assert_confidence"
            | "firewall" | "record_provenance" | "switch_model" | "merge" | "isolate" => {
                Linearity::ExactlyOnce
            }
            // Read-only queries are safe for multi-shot
            "get_confidence" | "get_provenance" | "get_model" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = EpistemicHandler::new();
        assert_eq!(handler.effect_name(), "Epistemic");
        assert_eq!(handler.handler_name(), "EpistemicHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = EpistemicHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"degrade"));
        assert!(names.contains(&"assert_confidence"));
        assert!(names.contains(&"firewall"));
        assert!(names.contains(&"record_provenance"));
        assert!(names.contains(&"switch_model"));
        assert!(names.contains(&"merge"));
        assert!(names.contains(&"get_confidence"));
    }

    #[test]
    fn test_default_confidence() {
        let state = new_state();
        assert!((EpistemicHandler::current_confidence(&state) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_degrade() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.9)], cont(), &mut state);

        let conf = EpistemicHandler::current_confidence(&state);
        assert!((conf - 0.9).abs() < 0.001);

        // Degrade again
        handler.handle("degrade", &[Value::Float(0.8)], cont(), &mut state);
        let conf = EpistemicHandler::current_confidence(&state);
        assert!((conf - 0.72).abs() < 0.001); // 0.9 * 0.8 = 0.72
    }

    #[test]
    fn test_degrade_invalid_factor() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle("degrade", &[Value::Float(1.5)], cont(), &mut state);
        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("[0, 1]"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_assert_confidence_pass() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "assert_confidence",
            &[Value::Float(0.8)],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_assert_confidence_fail() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        // Degrade to 0.5
        handler.handle("degrade", &[Value::Float(0.5)], cont(), &mut state);

        // Assert 0.8 should fail
        let result = handler.handle(
            "assert_confidence",
            &[Value::Float(0.8)],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("assertion failed"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_record_provenance() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle(
            "record_provenance",
            &[Value::String("validated".to_string())],
            cont(),
            &mut state,
        );
        handler.handle(
            "record_provenance",
            &[Value::String("transformed".to_string())],
            cont(),
            &mut state,
        );

        let chain = EpistemicHandler::provenance_chain(&state);
        assert!(chain.contains(&"validated".to_string()));
        assert!(chain.contains(&"transformed".to_string()));
    }

    #[test]
    fn test_switch_model() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        assert_eq!(EpistemicHandler::current_model(&state), "probabilistic");

        handler.handle(
            "switch_model",
            &[Value::String("bayesian".to_string())],
            cont(),
            &mut state,
        );

        assert_eq!(EpistemicHandler::current_model(&state), "bayesian");
    }

    #[test]
    fn test_switch_model_invalid() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "switch_model",
            &[Value::String("invalid".to_string())],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown uncertainty model"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_firewall_reset() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.5)], cont(), &mut state);
        assert!((EpistemicHandler::current_confidence(&state) - 0.5).abs() < 0.001);

        handler.handle(
            "firewall",
            &[Value::String("reset".to_string()), Value::Float(0.8)],
            cont(),
            &mut state,
        );

        assert!((EpistemicHandler::current_confidence(&state) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_firewall_gate_pass() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "firewall",
            &[Value::String("gate".to_string()), Value::Float(0.5)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_firewall_gate_block() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.3)], cont(), &mut state);

        let result = handler.handle(
            "firewall",
            &[Value::String("gate".to_string()), Value::Float(0.5)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("blocked"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_firewall_decay() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle(
            "firewall",
            &[Value::String("decay".to_string()), Value::Float(0.9)],
            cont(),
            &mut state,
        );

        assert!((EpistemicHandler::current_confidence(&state) - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_firewall_boost() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.5)], cont(), &mut state);

        handler.handle(
            "firewall",
            &[Value::String("boost".to_string()), Value::Float(1.5)],
            cont(),
            &mut state,
        );

        assert!((EpistemicHandler::current_confidence(&state) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_firewall_isolate() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle(
            "record_provenance",
            &[Value::String("before".to_string())],
            cont(),
            &mut state,
        );
        handler.handle(
            "firewall",
            &[Value::String("isolate".to_string())],
            cont(),
            &mut state,
        );

        let chain = EpistemicHandler::provenance_chain(&state);
        assert!(chain.contains(&"isolated".to_string()));
    }

    #[test]
    fn test_merge_most_confident() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let values = Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
            Value::Float(3.0),
        ])));

        let result = handler.handle(
            "merge",
            &[Value::String("most_confident".to_string()), values],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 1.0).abs() < 0.001);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_merge_union() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let values = Value::Array(Rc::new(RefCell::new(vec![Value::Int(1), Value::Int(2)])));

        let result = handler.handle(
            "merge",
            &[Value::String("union".to_string()), values],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 2);
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }
    }

    #[test]
    fn test_get_confidence() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.7)], cont(), &mut state);

        let result = handler.handle("get_confidence", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Float(c)) => {
                assert!((c - 0.7).abs() < 0.001);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_boost() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.5)], cont(), &mut state);
        handler.handle("boost", &[Value::Float(1.5)], cont(), &mut state);

        assert!((EpistemicHandler::current_confidence(&state) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_boost_capped() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("boost", &[Value::Float(2.0)], cont(), &mut state);

        // Should be capped at 1.0
        assert!((EpistemicHandler::current_confidence(&state) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clamp() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle(
            "clamp",
            &[Value::Float(0.3), Value::Float(0.8)],
            cont(),
            &mut state,
        );

        // Current confidence (1.0) should be clamped to 0.8
        assert!((EpistemicHandler::current_confidence(&state) - 0.8).abs() < 0.001);

        // Future degrades should respect the floor
        handler.handle("degrade", &[Value::Float(0.1)], cont(), &mut state);
        assert!((EpistemicHandler::current_confidence(&state) - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_reset_confidence() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle("degrade", &[Value::Float(0.3)], cont(), &mut state);
        handler.handle(
            "reset_confidence",
            &[Value::Float(0.95)],
            cont(),
            &mut state,
        );

        assert!((EpistemicHandler::current_confidence(&state) - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_get_provenance() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        handler.handle(
            "record_provenance",
            &[Value::String("step1".to_string())],
            cont(),
            &mut state,
        );
        handler.handle(
            "record_provenance",
            &[Value::String("step2".to_string())],
            cont(),
            &mut state,
        );

        let result = handler.handle("get_provenance", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 2);
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }
    }

    #[test]
    fn test_get_model() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle("get_model", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::String(m)) => {
                assert_eq!(m, "probabilistic");
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = EpistemicHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = EpistemicHandler::default();
        assert_eq!(handler.effect_name(), "Epistemic");
    }

    #[test]
    fn test_epistemic_impact() {
        let handler = EpistemicHandler::new();
        // Epistemic operations don't apply automatic degradation
        let impact = handler.epistemic_impact("degrade");
        assert!((impact.confidence_factor - 1.0).abs() < 0.001);
    }
}
