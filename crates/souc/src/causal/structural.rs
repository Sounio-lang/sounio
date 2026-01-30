//! Structural Causal Model (SCM)
//!
//! Implements full structural causal models for Level 3 reasoning:
//!
//! SCM M = ⟨U, V, F, P(U)⟩
//!
//! where:
//! - U = exogenous variables (not modeled)
//! - V = endogenous variables (modeled)
//! - F = structural equations
//! - P(U) = distribution over exogenous variables

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::graph::CausalGraph;
use super::intervention::Distribution;

/// Structural equation: V_i = f_i(PA_i, U_i)
#[derive(Clone)]
pub struct StructuralEquation {
    /// Variable this equation defines
    pub variable: String,
    /// Parent variables (PA_i)
    pub parents: Vec<String>,
    /// Associated exogenous variable
    pub exogenous: String,
    /// Structural function: f(parents, exogenous) -> value
    pub function: Arc<dyn Fn(&HashMap<String, f64>, f64) -> f64 + Send + Sync>,
}

impl StructuralEquation {
    /// Create a new structural equation
    pub fn new<F>(
        variable: impl Into<String>,
        parents: Vec<String>,
        exogenous: impl Into<String>,
        function: F,
    ) -> Self
    where
        F: Fn(&HashMap<String, f64>, f64) -> f64 + Send + Sync + 'static,
    {
        StructuralEquation {
            variable: variable.into(),
            parents,
            exogenous: exogenous.into(),
            function: Arc::new(function),
        }
    }

    /// Create a linear equation: V = Σ(β_i * PA_i) + U
    pub fn linear(
        variable: impl Into<String>,
        coefficients: Vec<(String, f64)>,
        exogenous: impl Into<String>,
        intercept: f64,
    ) -> Self {
        let parents: Vec<String> = coefficients.iter().map(|(p, _)| p.clone()).collect();
        let coeffs = coefficients.clone();

        StructuralEquation {
            variable: variable.into(),
            parents,
            exogenous: exogenous.into(),
            function: Arc::new(move |pa, u| {
                let mut sum = intercept;
                for (parent, coef) in &coeffs {
                    sum += coef * pa.get(parent).unwrap_or(&0.0);
                }
                sum + u
            }),
        }
    }

    /// Create an exogenous variable (no parents)
    pub fn exogenous(variable: impl Into<String>, exogenous: impl Into<String>) -> Self {
        StructuralEquation {
            variable: variable.into(),
            parents: vec![],
            exogenous: exogenous.into(),
            function: Arc::new(|_, u| u),
        }
    }

    /// Evaluate the equation given parent values and exogenous value
    pub fn evaluate(&self, parents: &HashMap<String, f64>, exogenous: f64) -> f64 {
        (self.function)(parents, exogenous)
    }
}

impl fmt::Debug for StructuralEquation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralEquation")
            .field("variable", &self.variable)
            .field("parents", &self.parents)
            .field("exogenous", &self.exogenous)
            .finish()
    }
}

/// Full Structural Causal Model
#[derive(Clone)]
pub struct StructuralCausalModel {
    /// Causal graph
    pub graph: CausalGraph,
    /// Structural equations for each endogenous variable
    pub equations: HashMap<String, StructuralEquation>,
    /// Distributions for exogenous variables
    pub exogenous_distributions: HashMap<String, Distribution>,
}

impl Default for StructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuralCausalModel {
    /// Create a new empty SCM
    pub fn new() -> Self {
        StructuralCausalModel {
            graph: CausalGraph::new(),
            equations: HashMap::new(),
            exogenous_distributions: HashMap::new(),
        }
    }

    /// Add a structural equation
    pub fn add_equation(&mut self, eq: StructuralEquation) {
        use super::graph::{CausalNode, EdgeType};

        // Add node for variable if not exists
        if !self.graph.contains_node(&eq.variable) {
            self.graph.add_node(CausalNode::observed(&eq.variable));
        }

        // Add edges from parents
        for parent in &eq.parents {
            if !self.graph.contains_node(parent) {
                self.graph.add_node(CausalNode::observed(parent));
            }
            let _ = self.graph.add_edge(parent, &eq.variable, EdgeType::Direct);
        }

        self.equations.insert(eq.variable.clone(), eq);
    }

    /// Add exogenous distribution
    pub fn add_exogenous(&mut self, name: impl Into<String>, distribution: Distribution) {
        self.exogenous_distributions
            .insert(name.into(), distribution);
    }

    /// Create intervened model M_x (do(X=x))
    ///
    /// Replaces structural equation for X with constant X := x
    pub fn intervene(&self, variable: &str, value: f64) -> StructuralCausalModel {
        let mut m_x = self.clone();

        // Replace equation with constant
        if m_x.equations.contains_key(variable) {
            let old_eq = m_x.equations.get(variable).unwrap();
            let new_eq = StructuralEquation {
                variable: variable.to_string(),
                parents: vec![],
                exogenous: old_eq.exogenous.clone(),
                function: Arc::new(move |_, _| value),
            };
            m_x.equations.insert(variable.to_string(), new_eq);
        }

        // Update graph (remove incoming edges to X)
        m_x.graph = m_x.graph.graph_do(variable);

        m_x
    }

    /// Sample exogenous variables from their distributions
    pub fn sample_exogenous(&self) -> HashMap<String, f64> {
        self.exogenous_distributions
            .iter()
            .map(|(name, dist)| (name.clone(), dist.sample()))
            .collect()
    }

    /// Evaluate all variables given exogenous values
    ///
    /// Uses topological order to ensure parents are computed first
    pub fn evaluate(&self, exogenous: &HashMap<String, f64>) -> HashMap<String, f64> {
        let order = self.graph.topological_order();
        let mut values = HashMap::new();

        for var in order {
            if let Some(eq) = self.equations.get(&var) {
                // Get parent values
                let parent_values: HashMap<String, f64> = eq
                    .parents
                    .iter()
                    .filter_map(|p| values.get(p).map(|v| (p.clone(), *v)))
                    .collect();

                // Get exogenous value
                let u_value = exogenous.get(&eq.exogenous).copied().unwrap_or(0.0);

                // Evaluate
                let value = eq.evaluate(&parent_values, u_value);
                values.insert(var, value);
            }
        }

        values
    }

    /// Simulate N samples from the model
    pub fn simulate(&self, n_samples: usize) -> Vec<HashMap<String, f64>> {
        (0..n_samples)
            .map(|_| {
                let u = self.sample_exogenous();
                self.evaluate(&u)
            })
            .collect()
    }

    /// Simulate interventional distribution P(V | do(X=x))
    pub fn simulate_intervention(
        &self,
        variable: &str,
        value: f64,
        n_samples: usize,
    ) -> Vec<HashMap<String, f64>> {
        let m_x = self.intervene(variable, value);
        m_x.simulate(n_samples)
    }

    /// Get all endogenous variable names
    pub fn variables(&self) -> Vec<&String> {
        self.equations.keys().collect()
    }

    /// Get all exogenous variable names
    pub fn exogenous_variables(&self) -> Vec<&String> {
        self.exogenous_distributions.keys().collect()
    }
}

impl fmt::Debug for StructuralCausalModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralCausalModel")
            .field("variables", &self.equations.keys().collect::<Vec<_>>())
            .field(
                "exogenous",
                &self.exogenous_distributions.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Builder for SCMs
pub struct SCMBuilder {
    model: StructuralCausalModel,
}

impl Default for SCMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SCMBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        SCMBuilder {
            model: StructuralCausalModel::new(),
        }
    }

    /// Add a variable with linear structural equation
    pub fn linear_variable(
        mut self,
        name: impl Into<String>,
        coefficients: Vec<(String, f64)>,
        exogenous_name: impl Into<String>,
        intercept: f64,
    ) -> Self {
        let exo_name = exogenous_name.into();
        let eq = StructuralEquation::linear(name, coefficients, &exo_name, intercept);
        self.model.add_equation(eq);

        // Add standard normal for exogenous if not specified
        if !self.model.exogenous_distributions.contains_key(&exo_name) {
            self.model.add_exogenous(
                &exo_name,
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            );
        }

        self
    }

    /// Add an exogenous variable
    pub fn exogenous_variable(
        mut self,
        name: impl Into<String>,
        exo_name: impl Into<String>,
        distribution: Distribution,
    ) -> Self {
        let exo = exo_name.into();
        let eq = StructuralEquation::exogenous(name, &exo);
        self.model.add_equation(eq);
        self.model.add_exogenous(&exo, distribution);
        self
    }

    /// Add custom structural equation
    pub fn custom_variable<F>(
        mut self,
        name: impl Into<String>,
        parents: Vec<String>,
        exogenous_name: impl Into<String>,
        function: F,
    ) -> Self
    where
        F: Fn(&HashMap<String, f64>, f64) -> f64 + Send + Sync + 'static,
    {
        let exo_name = exogenous_name.into();
        let eq = StructuralEquation::new(name, parents, &exo_name, function);
        self.model.add_equation(eq);

        // Add standard normal for exogenous if not specified
        if !self.model.exogenous_distributions.contains_key(&exo_name) {
            self.model.add_exogenous(
                &exo_name,
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            );
        }

        self
    }

    /// Set distribution for an exogenous variable
    pub fn with_distribution(mut self, exo_name: impl Into<String>, dist: Distribution) -> Self {
        self.model.add_exogenous(exo_name, dist);
        self
    }

    /// Build the SCM
    pub fn build(self) -> StructuralCausalModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_scm() -> StructuralCausalModel {
        SCMBuilder::new()
            .exogenous_variable(
                "X",
                "U_X",
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .linear_variable("Y", vec![("X".to_string(), 0.5)], "U_Y", 1.0)
            .build()
    }

    fn mediated_scm() -> StructuralCausalModel {
        SCMBuilder::new()
            .exogenous_variable(
                "X",
                "U_X",
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .linear_variable("M", vec![("X".to_string(), 0.8)], "U_M", 0.0)
            .linear_variable("Y", vec![("M".to_string(), 0.6)], "U_Y", 0.5)
            .build()
    }

    #[test]
    fn test_create_scm() {
        let scm = simple_scm();
        assert_eq!(scm.variables().len(), 2);
        assert_eq!(scm.exogenous_variables().len(), 2);
    }

    #[test]
    fn test_evaluate_scm() {
        let scm = simple_scm();

        let exo: HashMap<String, f64> = [("U_X".to_string(), 2.0), ("U_Y".to_string(), 0.0)]
            .into_iter()
            .collect();

        let values = scm.evaluate(&exo);

        // X = U_X = 2.0
        assert!((values["X"] - 2.0).abs() < 0.001);

        // Y = 0.5 * X + 1.0 + U_Y = 0.5 * 2.0 + 1.0 + 0.0 = 2.0
        assert!((values["Y"] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_intervention() {
        let scm = simple_scm();
        let m_x = scm.intervene("X", 5.0);

        let exo: HashMap<String, f64> = [("U_X".to_string(), 0.0), ("U_Y".to_string(), 0.0)]
            .into_iter()
            .collect();

        let values = m_x.evaluate(&exo);

        // X = 5.0 (intervened)
        assert!((values["X"] - 5.0).abs() < 0.001);

        // Y = 0.5 * 5.0 + 1.0 = 3.5
        assert!((values["Y"] - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_mediated_effect() {
        let scm = mediated_scm();

        let exo: HashMap<String, f64> = [
            ("U_X".to_string(), 1.0),
            ("U_M".to_string(), 0.0),
            ("U_Y".to_string(), 0.0),
        ]
        .into_iter()
        .collect();

        let values = scm.evaluate(&exo);

        // X = 1.0
        assert!((values["X"] - 1.0).abs() < 0.001);

        // M = 0.8 * X = 0.8
        assert!((values["M"] - 0.8).abs() < 0.001);

        // Y = 0.6 * M + 0.5 = 0.6 * 0.8 + 0.5 = 0.98
        assert!((values["Y"] - 0.98).abs() < 0.001);
    }

    #[test]
    fn test_simulate() {
        let scm = simple_scm();
        let samples = scm.simulate(100);

        assert_eq!(samples.len(), 100);
        assert!(samples[0].contains_key("X"));
        assert!(samples[0].contains_key("Y"));
    }

    #[test]
    fn test_custom_equation() {
        let scm = SCMBuilder::new()
            .exogenous_variable("X", "U_X", Distribution::Uniform { min: 0.0, max: 1.0 })
            .custom_variable("Y", vec!["X".to_string()], "U_Y", |pa, u| {
                let x = pa.get("X").unwrap_or(&0.0);
                x.powi(2) + u // Y = X² + U
            })
            .build();

        let exo: HashMap<String, f64> = [("U_X".to_string(), 3.0), ("U_Y".to_string(), 1.0)]
            .into_iter()
            .collect();

        let values = scm.evaluate(&exo);

        // Y = 3² + 1 = 10
        assert!((values["Y"] - 10.0).abs() < 0.001);
    }
}
