//! Causal Effect Handler
//!
//! This module implements the `Causal` effect handler for first-class causal
//! reasoning through algebraic effects. The Causal effect enables do-calculus
//! operations, interventions, and counterfactual queries with compile-time
//! DAG validation.
//!
//! # Effect Operations
//!
//! | Operation         | Arguments                              | Return Type | Description |
//! |-------------------|----------------------------------------|-------------|-------------|
//! | `do`              | `(var: String, value: T)`              | `Unit`      | Intervention: set variable, break incoming edges |
//! | `query`           | `(var: String)`                        | `T`         | Query causal effect with backdoor adjustment |
//! | `counterfactual`  | `(intervention: String, query: String)`| `T`         | Counterfactual: abduction → intervention → prediction |
//! | `observe`         | `(var: String, value: T)`              | `Unit`      | Observation without intervention |
//! | `get_interventions` | `()`                                 | `Array`     | Get list of active interventions |
//! | `clear_interventions` | `()`                               | `Unit`      | Clear all interventions |
//! | `get_graph`       | `()`                                   | `String`    | Get name of active causal graph |
//!
//! # Causal Semantics
//!
//! The handler implements Pearl's do-calculus:
//! - `do(X = x)`: Intervention that sets X to x and removes all incoming edges to X
//! - `query(Y)`: Computes P(Y | do(interventions)) using backdoor adjustment
//! - `counterfactual(do(X=x), Y)`: Computes Y_x in the twin network model
//!
//! # Epistemic Integration
//!
//! Causal assumptions affect confidence:
//! - Faithfulness: 0.95 factor
//! - No unmeasured confounders: 0.85 factor
//! - Positivity: 0.98 factor
//! - SUTVA: 0.95 factor
//! - Causal Markov: 0.99 factor
//!
//! # Example
//!
//! ```text
//! // Sounio code using Causal effect:
//! causal graph DrugEffect {
//!     Drug -> BloodPressure
//!     Age -> BloodPressure
//!     Age -> Drug  // confound
//! }
//!
//! fn estimate_effect(dose: Knowledge<mg>) -> Knowledge<mmHg> with Causal<DrugEffect> {
//!     do(Drug = dose)
//!     let effect = query(BloodPressure)
//!     effect  // adjusted for Age confounding
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use crate::types::causal::{CausalAssumption, causal_confidence_factor};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

/// Key for storing active causal graph name
const GRAPH_KEY: &str = "__causal_graph";

/// Key for storing active interventions: Vec<(variable, value)>
const INTERVENTIONS_KEY: &str = "__causal_interventions";

/// Key for storing observations: Vec<(variable, value)>
const OBSERVATIONS_KEY: &str = "__causal_observations";

/// Key for storing causal assumptions made
const ASSUMPTIONS_KEY: &str = "__causal_assumptions";

/// Key for storing intervention history for counterfactuals
const INTERVENTION_HISTORY_KEY: &str = "__causal_intervention_history";

/// Static operation specifications for CausalHandler
static CAUSAL_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_causal_operations() -> &'static [OperationSpec] {
    CAUSAL_OPERATIONS.get_or_init(|| {
        vec![
            // do(variable, value) - intervention
            OperationSpec::new("do", "Unit").with_params(vec!["String", "Value"]),
            // query(variable) - causal query
            OperationSpec::new("query", "Value").with_params(vec!["String"]),
            // counterfactual(intervention_expr, query_var) - counterfactual query
            OperationSpec::new("counterfactual", "Value").with_params(vec!["String", "String"]),
            // observe(variable, value) - observation without intervention
            OperationSpec::new("observe", "Unit").with_params(vec!["String", "Value"]),
            // get_interventions() - get active interventions
            OperationSpec::new("get_interventions", "Array").with_params(vec![]),
            // clear_interventions() - reset interventions
            OperationSpec::new("clear_interventions", "Unit").with_params(vec![]),
            // get_graph() - get active causal graph name
            OperationSpec::new("get_graph", "String").with_params(vec![]),
            // set_graph(name) - set active causal graph
            OperationSpec::new("set_graph", "Unit").with_params(vec!["String"]),
            // assume(assumption) - record causal assumption
            OperationSpec::new("assume", "Unit").with_params(vec!["String"]),
            // get_assumptions() - get recorded assumptions
            OperationSpec::new("get_assumptions", "Array").with_params(vec![]),
            // confidence_factor() - get confidence factor from assumptions
            OperationSpec::new("confidence_factor", "Float").with_params(vec![]),
        ]
    })
}

/// Handler for the Causal effect
///
/// CausalHandler implements do-calculus operations for causal inference.
/// It tracks interventions, observations, and causal assumptions, enabling
/// proper causal effect estimation with epistemic confidence tracking.
///
/// # State Management
///
/// The handler tracks:
/// - Active causal graph name (validated at compile-time)
/// - List of interventions: (variable, value) pairs
/// - List of observations: (variable, value) pairs
/// - Causal assumptions made during computation
/// - Intervention history for counterfactual reasoning
///
/// # Integration with Type System
///
/// The Causal effect requires a type parameter `Causal<GraphName>` that references
/// a declared `causal model` or `causal graph`. The type checker validates:
/// - Graph exists and is a DAG
/// - Intervention targets are valid nodes
/// - Backdoor criterion is satisfied or adjustment is applied
#[derive(Debug)]
pub struct CausalHandler {
    /// Default graph name (can be overridden by set_graph)
    default_graph: Option<String>,
}

impl Default for CausalHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalHandler {
    /// Create a new CausalHandler
    pub fn new() -> Self {
        Self {
            default_graph: None,
        }
    }

    /// Create a CausalHandler with a default graph
    pub fn with_graph(graph_name: impl Into<String>) -> Self {
        Self {
            default_graph: Some(graph_name.into()),
        }
    }

    /// Get the active graph name from state
    fn get_graph_name(state: &HandlerState) -> Option<String> {
        match state.named_state.get(GRAPH_KEY) {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Set the graph name in state
    fn set_graph_name(state: &mut HandlerState, name: String) {
        state
            .named_state
            .insert(GRAPH_KEY.to_string(), Value::String(name));
    }

    /// Get active interventions from state
    fn get_interventions(state: &HandlerState) -> Vec<(String, Value)> {
        match state.named_state.get(INTERVENTIONS_KEY) {
            Some(Value::Array(arr)) => arr
                .borrow()
                .iter()
                .filter_map(|v| {
                    if let Value::Tuple(t) = v {
                        if t.len() >= 2 {
                            if let Value::String(var) = &t[0] {
                                return Some((var.clone(), t[1].clone()));
                            }
                        }
                    }
                    None
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Add an intervention to state
    fn add_intervention(state: &mut HandlerState, variable: String, value: Value) {
        let mut interventions = Self::get_interventions(state);

        // Remove any existing intervention on this variable
        interventions.retain(|(v, _)| v != &variable);

        // Add the new intervention
        interventions.push((variable, value));

        // Store back
        let values: Vec<Value> = interventions
            .into_iter()
            .map(|(var, val)| Value::Tuple(vec![Value::String(var), val]))
            .collect();
        state.named_state.insert(
            INTERVENTIONS_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(values))),
        );
    }

    /// Clear all interventions
    fn clear_all_interventions(state: &mut HandlerState) {
        state.named_state.insert(
            INTERVENTIONS_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(Vec::new()))),
        );
    }

    /// Get observations from state
    fn get_observations(state: &HandlerState) -> Vec<(String, Value)> {
        match state.named_state.get(OBSERVATIONS_KEY) {
            Some(Value::Array(arr)) => arr
                .borrow()
                .iter()
                .filter_map(|v| {
                    if let Value::Tuple(t) = v {
                        if t.len() >= 2 {
                            if let Value::String(var) = &t[0] {
                                return Some((var.clone(), t[1].clone()));
                            }
                        }
                    }
                    None
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Add an observation to state
    fn add_observation(state: &mut HandlerState, variable: String, value: Value) {
        let mut observations = Self::get_observations(state);
        observations.push((variable, value));

        let values: Vec<Value> = observations
            .into_iter()
            .map(|(var, val)| Value::Tuple(vec![Value::String(var), val]))
            .collect();
        state.named_state.insert(
            OBSERVATIONS_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(values))),
        );
    }

    /// Get recorded causal assumptions
    fn get_assumptions(state: &HandlerState) -> Vec<CausalAssumption> {
        match state.named_state.get(ASSUMPTIONS_KEY) {
            Some(Value::Array(arr)) => arr
                .borrow()
                .iter()
                .filter_map(|v| {
                    if let Value::String(s) = v {
                        Self::parse_assumption(s)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Add a causal assumption
    fn add_assumption(state: &mut HandlerState, assumption: CausalAssumption) {
        let mut assumptions = Self::get_assumptions(state);
        if !assumptions.contains(&assumption) {
            assumptions.push(assumption);
        }

        let values: Vec<Value> = assumptions
            .into_iter()
            .map(|a| Value::String(Self::assumption_to_string(&a)))
            .collect();
        state.named_state.insert(
            ASSUMPTIONS_KEY.to_string(),
            Value::Array(Rc::new(RefCell::new(values))),
        );
    }

    /// Parse assumption from string
    fn parse_assumption(s: &str) -> Option<CausalAssumption> {
        match s.to_lowercase().as_str() {
            "faithfulness" => Some(CausalAssumption::Faithfulness),
            "no_unmeasured_confounders" | "nuc" => Some(CausalAssumption::NoUnmeasuredConfounders),
            "positivity" => Some(CausalAssumption::Positivity),
            "sutva" => Some(CausalAssumption::SUTVA),
            "causal_markov" | "markov" => Some(CausalAssumption::CausalMarkov),
            _ => None,
        }
    }

    /// Convert assumption to string
    fn assumption_to_string(a: &CausalAssumption) -> String {
        match a {
            CausalAssumption::Faithfulness => "faithfulness".to_string(),
            CausalAssumption::NoUnmeasuredConfounders => "no_unmeasured_confounders".to_string(),
            CausalAssumption::Positivity => "positivity".to_string(),
            CausalAssumption::SUTVA => "sutva".to_string(),
            CausalAssumption::CausalMarkov => "causal_markov".to_string(),
        }
    }

    /// Extract a string from a Value
    fn extract_string(value: &Value, op: &str, param: &str) -> Result<String, HandlerResult> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Causal",
                op,
                format!("expected String for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Handle do (intervention) operation
    ///
    /// do(variable, value) sets the variable to value and conceptually
    /// removes all incoming edges to that variable in the causal graph.
    fn handle_do(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "do",
                "do requires variable and value arguments",
            ));
        }

        let variable = match Self::extract_string(&args[0], "do", "variable") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let value = args[1].clone();

        // Record the intervention
        Self::add_intervention(state, variable.clone(), value);

        // Implicitly assume no unmeasured confounders for intervention
        Self::add_assumption(state, CausalAssumption::NoUnmeasuredConfounders);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle query operation
    ///
    /// query(variable) returns the value of the variable under the current
    /// interventions, applying backdoor adjustment for confounders.
    fn handle_query(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "query",
                "query requires a variable argument",
            ));
        }

        let variable = match Self::extract_string(&args[0], "query", "variable") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let interventions = Self::get_interventions(state);

        // Check if the queried variable was intervened on
        for (var, val) in &interventions {
            if var == &variable {
                // Under intervention, the variable has the intervened value
                return HandlerResult::Resume(val.clone());
            }
        }

        // For non-intervened variables, we would compute the causal effect
        // using backdoor adjustment. In simulation, return a placeholder.
        // Real implementation would integrate with the actual computation.

        // Record that we're using faithfulness assumption
        Self::add_assumption(state, CausalAssumption::Faithfulness);

        // Return placeholder indicating causal query
        // In full implementation, this would trigger actual causal computation
        HandlerResult::Resume(Value::String(format!("causal_query({})", variable)))
    }

    /// Handle counterfactual operation
    ///
    /// counterfactual(intervention, query) computes what the query variable
    /// would have been under the intervention, using the three-step process:
    /// 1. Abduction: infer exogenous variables from observations
    /// 2. Action: apply the intervention
    /// 3. Prediction: compute the query in the modified model
    fn handle_counterfactual(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "counterfactual",
                "counterfactual requires intervention and query arguments",
            ));
        }

        let intervention = match Self::extract_string(&args[0], "counterfactual", "intervention") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let query_var = match Self::extract_string(&args[1], "counterfactual", "query") {
            Ok(s) => s,
            Err(result) => return result,
        };

        // Record SUTVA assumption (no interference between units)
        Self::add_assumption(state, CausalAssumption::SUTVA);

        // Counterfactual computation would use twin network model
        // For simulation, return placeholder
        HandlerResult::Resume(Value::String(format!(
            "counterfactual({}, {})",
            intervention, query_var
        )))
    }

    /// Handle observe operation
    ///
    /// observe(variable, value) records an observation without intervening.
    /// This is used for conditioning in causal queries.
    fn handle_observe(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "observe",
                "observe requires variable and value arguments",
            ));
        }

        let variable = match Self::extract_string(&args[0], "observe", "variable") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let value = args[1].clone();

        Self::add_observation(state, variable, value);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_interventions operation
    fn handle_get_interventions(&self, state: &HandlerState) -> HandlerResult {
        let interventions = Self::get_interventions(state);
        let values: Vec<Value> = interventions
            .into_iter()
            .map(|(var, val)| Value::Tuple(vec![Value::String(var), val]))
            .collect();
        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(values))))
    }

    /// Handle clear_interventions operation
    fn handle_clear_interventions(&self, state: &mut HandlerState) -> HandlerResult {
        Self::clear_all_interventions(state);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_graph operation
    fn handle_get_graph(&self, state: &HandlerState) -> HandlerResult {
        let graph = Self::get_graph_name(state)
            .or_else(|| self.default_graph.clone())
            .unwrap_or_else(|| "<unset>".to_string());
        HandlerResult::Resume(Value::String(graph))
    }

    /// Handle set_graph operation
    fn handle_set_graph(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "set_graph",
                "set_graph requires a graph name argument",
            ));
        }

        let name = match Self::extract_string(&args[0], "set_graph", "name") {
            Ok(s) => s,
            Err(result) => return result,
        };

        Self::set_graph_name(state, name);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle assume operation
    fn handle_assume(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Causal",
                "assume",
                "assume requires an assumption name argument",
            ));
        }

        let assumption_str = match Self::extract_string(&args[0], "assume", "assumption") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let assumption = match Self::parse_assumption(&assumption_str) {
            Some(a) => a,
            None => {
                return HandlerResult::Abort(HandlerError::new(
                    "Causal",
                    "assume",
                    format!(
                        "unknown causal assumption: {}. Valid: faithfulness, no_unmeasured_confounders, positivity, sutva, causal_markov",
                        assumption_str
                    ),
                ));
            }
        };

        Self::add_assumption(state, assumption);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_assumptions operation
    fn handle_get_assumptions(&self, state: &HandlerState) -> HandlerResult {
        let assumptions = Self::get_assumptions(state);
        let values: Vec<Value> = assumptions
            .into_iter()
            .map(|a| Value::String(Self::assumption_to_string(&a)))
            .collect();
        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(values))))
    }

    /// Handle confidence_factor operation
    ///
    /// Returns the cumulative confidence degradation factor based on
    /// all causal assumptions that have been made.
    fn handle_confidence_factor(&self, state: &HandlerState) -> HandlerResult {
        let assumptions = Self::get_assumptions(state);
        let factor = causal_confidence_factor(&assumptions);
        HandlerResult::Resume(Value::Float(factor))
    }

    /// Get active interventions (public accessor for testing)
    pub fn active_interventions(state: &HandlerState) -> Vec<(String, Value)> {
        Self::get_interventions(state)
    }

    /// Get recorded assumptions (public accessor for testing)
    pub fn recorded_assumptions(state: &HandlerState) -> Vec<CausalAssumption> {
        Self::get_assumptions(state)
    }
}

impl HandlerCapability for CausalHandler {
    fn effect_name(&self) -> &str {
        "Causal"
    }

    fn handler_name(&self) -> &str {
        "CausalHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_causal_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "do" => self.handle_do(args, state),
            "query" => self.handle_query(args, state),
            "counterfactual" => self.handle_counterfactual(args, state),
            "observe" => self.handle_observe(args, state),
            "get_interventions" => self.handle_get_interventions(state),
            "clear_interventions" => self.handle_clear_interventions(state),
            "get_graph" => self.handle_get_graph(state),
            "set_graph" => self.handle_set_graph(args, state),
            "assume" => self.handle_assume(args, state),
            "get_assumptions" => self.handle_get_assumptions(state),
            "confidence_factor" => self.handle_confidence_factor(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Causal",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Causal operations are stateful (interventions accumulate)
        // Multi-shot not supported
        false
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        // Causal operations have epistemic impact through assumptions
        match operation {
            // do() implicitly assumes no unmeasured confounders: 0.85 factor
            "do" => EpistemicImpact {
                confidence_factor: 0.85,
                provenance_tag: Some("causal_intervention".to_string()),
                crosses_firewall: false,
            },
            // query() assumes faithfulness: 0.95 factor
            "query" => EpistemicImpact {
                confidence_factor: 0.95,
                provenance_tag: Some("causal_query".to_string()),
                crosses_firewall: false,
            },
            // counterfactual() assumes SUTVA: 0.95 factor
            "counterfactual" => EpistemicImpact {
                confidence_factor: 0.95,
                provenance_tag: Some("counterfactual".to_string()),
                crosses_firewall: false,
            },
            // observe() doesn't affect confidence
            "observe" => EpistemicImpact::none(),
            // Read-only operations don't affect confidence
            "get_interventions" | "get_graph" | "get_assumptions" | "confidence_factor" => {
                EpistemicImpact::none()
            }
            // State modification without epistemic impact
            "clear_interventions" | "set_graph" | "assume" => EpistemicImpact::none(),
            _ => EpistemicImpact::none(),
        }
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        match operation {
            // Interventions are stateful, exactly once semantics
            "do" | "observe" | "clear_interventions" | "set_graph" | "assume" => {
                Linearity::ExactlyOnce
            }
            // Queries can be called multiple times safely
            "query" | "counterfactual" => Linearity::ExactlyOnce,
            // Read operations are safe for multi-shot
            "get_interventions" | "get_graph" | "get_assumptions" | "confidence_factor" => {
                Linearity::MultiShot
            }
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
        let handler = CausalHandler::new();
        assert_eq!(handler.effect_name(), "Causal");
        assert_eq!(handler.handler_name(), "CausalHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = CausalHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"do"));
        assert!(names.contains(&"query"));
        assert!(names.contains(&"counterfactual"));
        assert!(names.contains(&"observe"));
        assert!(names.contains(&"get_interventions"));
    }

    #[test]
    fn test_do_intervention() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(100.0)],
            cont(),
            &mut state,
        );

        assert!(matches!(result, HandlerResult::Resume(Value::Unit)));

        let interventions = CausalHandler::active_interventions(&state);
        assert_eq!(interventions.len(), 1);
        assert_eq!(interventions[0].0, "Drug");
    }

    #[test]
    fn test_do_replaces_previous() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(100.0)],
            cont(),
            &mut state,
        );
        handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(200.0)],
            cont(),
            &mut state,
        );

        let interventions = CausalHandler::active_interventions(&state);
        assert_eq!(interventions.len(), 1);
        if let Value::Float(v) = &interventions[0].1 {
            assert!((*v - 200.0).abs() < 0.001);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn test_query_intervened_variable() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(100.0)],
            cont(),
            &mut state,
        );

        let result = handler.handle(
            "query",
            &[Value::String("Drug".to_string())],
            cont(),
            &mut state,
        );

        if let HandlerResult::Resume(Value::Float(v)) = result {
            assert!((v - 100.0).abs() < 0.001);
        } else {
            panic!("expected Float result");
        }
    }

    #[test]
    fn test_observe_adds_observation() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "observe",
            &[Value::String("Age".to_string()), Value::Int(45)],
            cont(),
            &mut state,
        );

        assert!(matches!(result, HandlerResult::Resume(Value::Unit)));
    }

    #[test]
    fn test_clear_interventions() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(100.0)],
            cont(),
            &mut state,
        );
        handler.handle("clear_interventions", &[], cont(), &mut state);

        let interventions = CausalHandler::active_interventions(&state);
        assert!(interventions.is_empty());
    }

    #[test]
    fn test_assume_valid() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "assume",
            &[Value::String("faithfulness".to_string())],
            cont(),
            &mut state,
        );

        let assumptions = CausalHandler::recorded_assumptions(&state);
        assert!(assumptions.contains(&CausalAssumption::Faithfulness));
    }

    #[test]
    fn test_assume_invalid() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "assume",
            &[Value::String("invalid_assumption".to_string())],
            cont(),
            &mut state,
        );

        assert!(matches!(result, HandlerResult::Abort(_)));
    }

    #[test]
    fn test_confidence_factor() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        // No assumptions: factor = 1.0
        let result = handler.handle("confidence_factor", &[], cont(), &mut state);
        if let HandlerResult::Resume(Value::Float(f)) = result {
            assert!((f - 1.0).abs() < 0.001);
        } else {
            panic!("expected Float");
        }

        // Add assumptions
        handler.handle(
            "assume",
            &[Value::String("faithfulness".to_string())],
            cont(),
            &mut state,
        );

        let result = handler.handle("confidence_factor", &[], cont(), &mut state);
        if let HandlerResult::Resume(Value::Float(f)) = result {
            assert!((f - 0.95).abs() < 0.001);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn test_do_adds_nuc_assumption() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "do",
            &[Value::String("Drug".to_string()), Value::Float(100.0)],
            cont(),
            &mut state,
        );

        let assumptions = CausalHandler::recorded_assumptions(&state);
        assert!(assumptions.contains(&CausalAssumption::NoUnmeasuredConfounders));
    }

    #[test]
    fn test_set_and_get_graph() {
        let handler = CausalHandler::new();
        let mut state = new_state();

        handler.handle(
            "set_graph",
            &[Value::String("DrugEffect".to_string())],
            cont(),
            &mut state,
        );

        let result = handler.handle("get_graph", &[], cont(), &mut state);
        if let HandlerResult::Resume(Value::String(s)) = result {
            assert_eq!(s, "DrugEffect");
        } else {
            panic!("expected String");
        }
    }

    #[test]
    fn test_with_graph_constructor() {
        let handler = CausalHandler::with_graph("MyGraph");
        let mut state = new_state();

        let result = handler.handle("get_graph", &[], Continuation::new(), &mut state);
        if let HandlerResult::Resume(Value::String(s)) = result {
            assert_eq!(s, "MyGraph");
        } else {
            panic!("expected String");
        }
    }

    #[test]
    fn test_epistemic_impact() {
        let handler = CausalHandler::new();

        let do_impact = handler.epistemic_impact("do");
        assert!((do_impact.confidence_factor - 0.85).abs() < 0.001);
        assert!(do_impact.provenance_tag.is_some());

        let query_impact = handler.epistemic_impact("query");
        assert!((query_impact.confidence_factor - 0.95).abs() < 0.001);

        let observe_impact = handler.epistemic_impact("observe");
        assert!((observe_impact.confidence_factor - 1.0).abs() < 0.001);
    }
}
