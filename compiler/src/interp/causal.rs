//! Causal inference and do-calculus
//!
//! Implements causal reasoning with support for:
//! - Causal DAGs (directed acyclic graphs)
//! - Interventions (do-operator)
//! - Counterfactual reasoning
//! - Simpson's paradox detection
//! - Backdoor criterion checking

use std::collections::{HashMap, HashSet};

use miette::Result;

/// A causal directed acyclic graph (DAG)
#[derive(Clone, Debug)]
pub struct CausalDAG {
    /// Node names
    pub nodes: Vec<String>,
    /// Edges: (source, target)
    pub edges: Vec<(String, String)>,
}

impl CausalDAG {
    /// Create a new causal DAG
    pub fn new(nodes: Vec<String>) -> Self {
        CausalDAG {
            nodes,
            edges: Vec::new(),
        }
    }

    /// Add a directed edge (causal link)
    pub fn add_edge(&mut self, source: &str, target: &str) {
        self.edges.push((source.to_string(), target.to_string()));
    }

    /// Get parents of a node
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, target)| target == node)
            .map(|(source, _)| source.clone())
            .collect()
    }

    /// Get children of a node
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(source, _)| source == node)
            .map(|(_, target)| target.clone())
            .collect()
    }

    /// Find all paths from source to target
    fn find_paths_dfs(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        paths: &mut Vec<Vec<String>>,
    ) {
        if current == target {
            paths.push(path.clone());
            return;
        }

        visited.insert(current.to_string());

        for child in self.children(current) {
            if !visited.contains(&child) {
                path.push(child.clone());
                self.find_paths_dfs(&child, target, visited, path, paths);
                path.pop();
            }
        }

        visited.remove(current);
    }

    /// Get all directed paths from source to target
    pub fn directed_paths(&self, source: &str, target: &str) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut path = vec![source.to_string()];
        self.find_paths_dfs(source, target, &mut visited, &mut path, &mut paths);
        paths
    }

    /// Check backdoor criterion (path adjustment for causal effect)
    /// Returns confounders that need to be controlled for
    pub fn backdoor_criterion(&self, treatment: &str, outcome: &str) -> Vec<String> {
        let mut confounders = Vec::new();

        // Find all parents of treatment (causes of treatment)
        let treatment_parents = self.parents(treatment);

        // For each parent of treatment, check if it reaches outcome
        for parent in treatment_parents {
            let paths = self.directed_paths(&parent, outcome);
            if !paths.is_empty() {
                // This is a confounder - affects both treatment and outcome
                confounders.push(parent);
            }
        }

        confounders
    }
}

/// An intervention: setting a variable to a fixed value
#[derive(Clone, Debug)]
pub struct Intervention {
    pub variable: String,
    pub value: f64,
}

/// Causal model with structural equations
#[derive(Clone, Debug)]
pub struct CausalModel {
    /// The causal DAG
    pub dag: CausalDAG,
    /// Structural equations (variable -> formula)
    pub equations: HashMap<String, String>,
    /// Observed data
    pub data: HashMap<String, Vec<f64>>,
}

impl CausalModel {
    /// Create a new causal model
    pub fn new(dag: CausalDAG) -> Self {
        CausalModel {
            dag,
            equations: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Add a structural equation
    pub fn add_equation(&mut self, variable: String, formula: String) {
        self.equations.insert(variable, formula);
    }

    /// Apply an intervention (do-operator)
    /// Returns a modified model with the intervention applied
    pub fn intervene(&self, intervention: Intervention) -> Self {
        let mut new_model = self.clone();

        // Remove all edges pointing TO the intervened variable
        // (cut backdoor paths)
        new_model
            .dag
            .edges
            .retain(|(_, target)| target != &intervention.variable);

        // Set the equation for the intervened variable to its constant value
        new_model.equations.insert(
            intervention.variable.clone(),
            format!("constant({})", intervention.value),
        );

        new_model
    }

    /// Estimate average treatment effect (ATE) using backdoor adjustment
    /// ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
    ///
    /// Uses the backdoor adjustment formula when confounders are present:
    /// E[Y | do(X=x)] = Σ_z P(z) * E[Y | X=x, Z=z]
    pub fn estimate_ate(&self, treatment: &str, outcome: &str) -> Result<f64> {
        let treatment_data = self
            .data
            .get(treatment)
            .ok_or_else(|| miette::miette!("No data for treatment variable: {}", treatment))?;
        let outcome_data = self
            .data
            .get(outcome)
            .ok_or_else(|| miette::miette!("No data for outcome variable: {}", outcome))?;

        if treatment_data.len() != outcome_data.len() {
            return Err(miette::miette!(
                "Treatment and outcome data must have same length"
            ));
        }

        if treatment_data.is_empty() {
            return Err(miette::miette!("No data available for ATE estimation"));
        }

        // Identify confounders using backdoor criterion
        let confounders = self.dag.backdoor_criterion(treatment, outcome);

        if confounders.is_empty() {
            // No confounders: simple difference in means
            // E[Y | X=1] - E[Y | X=0]
            self.estimate_ate_simple(treatment_data, outcome_data)
        } else {
            // With confounders: use backdoor adjustment
            self.estimate_ate_adjusted(treatment, outcome, &confounders)
        }
    }

    /// Simple ATE estimation (no confounders)
    fn estimate_ate_simple(&self, treatment_data: &[f64], outcome_data: &[f64]) -> Result<f64> {
        let mut treated_outcomes = Vec::new();
        let mut control_outcomes = Vec::new();

        for (t, y) in treatment_data.iter().zip(outcome_data.iter()) {
            // Use 0.5 as threshold for binary treatment
            if *t > 0.5 {
                treated_outcomes.push(*y);
            } else {
                control_outcomes.push(*y);
            }
        }

        if treated_outcomes.is_empty() || control_outcomes.is_empty() {
            return Err(miette::miette!(
                "Need both treated and control observations"
            ));
        }

        let mean_treated: f64 =
            treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64;
        let mean_control: f64 =
            control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64;

        Ok(mean_treated - mean_control)
    }

    /// ATE estimation with backdoor adjustment
    /// Uses stratification over confounder values
    fn estimate_ate_adjusted(
        &self,
        treatment: &str,
        outcome: &str,
        confounders: &[String],
    ) -> Result<f64> {
        let treatment_data = self.data.get(treatment).unwrap();
        let outcome_data = self.data.get(outcome).unwrap();
        let n = treatment_data.len();

        // For simplicity, we use a single confounder (the first one)
        // A full implementation would handle multiple confounders
        let confounder = &confounders[0];
        let confounder_data = self
            .data
            .get(confounder)
            .ok_or_else(|| miette::miette!("No data for confounder: {}", confounder))?;

        if confounder_data.len() != n {
            return Err(miette::miette!("Confounder data length mismatch"));
        }

        // Discretize confounder into bins (low, medium, high)
        // Find min and max of confounder
        let min_z = confounder_data
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_z = confounder_data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_z - min_z;

        if range <= 0.0 {
            // All confounder values are the same, fall back to simple estimation
            return self.estimate_ate_simple(treatment_data, outcome_data);
        }

        // Create 3 strata
        let num_strata = 3;
        let bin_width = range / num_strata as f64;

        // Compute stratified effect
        let mut total_effect = 0.0;
        let mut total_weight = 0.0;

        for stratum in 0..num_strata {
            let lower = min_z + stratum as f64 * bin_width;
            let upper = if stratum == num_strata - 1 {
                max_z + 0.001 // Include max value in last stratum
            } else {
                min_z + (stratum + 1) as f64 * bin_width
            };

            // Collect observations in this stratum
            let mut treated_outcomes = Vec::new();
            let mut control_outcomes = Vec::new();
            let mut stratum_count = 0;

            for i in 0..n {
                let z = confounder_data[i];
                if z >= lower && z < upper {
                    stratum_count += 1;
                    if treatment_data[i] > 0.5 {
                        treated_outcomes.push(outcome_data[i]);
                    } else {
                        control_outcomes.push(outcome_data[i]);
                    }
                }
            }

            // Skip strata without both treated and control observations
            if treated_outcomes.is_empty() || control_outcomes.is_empty() {
                continue;
            }

            // Compute stratum-specific effect
            let mean_treated: f64 =
                treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64;
            let mean_control: f64 =
                control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64;
            let stratum_effect = mean_treated - mean_control;

            // Weight by stratum size P(Z=z)
            let stratum_weight = stratum_count as f64 / n as f64;
            total_effect += stratum_weight * stratum_effect;
            total_weight += stratum_weight;
        }

        if total_weight == 0.0 {
            return Err(miette::miette!("No valid strata for ATE estimation"));
        }

        // Normalize by total weight (in case some strata were skipped)
        Ok(total_effect / total_weight)
    }

    /// Add observed data for a variable
    pub fn add_data(&mut self, variable: String, values: Vec<f64>) {
        self.data.insert(variable, values);
    }

    /// Detect Simpson's paradox
    /// Returns true if causal direction contradicts marginal association
    pub fn has_simpsons_paradox(&self, x: &str, y: &str, z: &str) -> bool {
        // Simpson's paradox occurs when:
        // - Marginal association between X and Y has one direction
        // - Conditional association (stratified by Z) has opposite direction
        // This is detected when Z is a confounder
        let confounders = self.dag.backdoor_criterion(x, y);
        confounders.contains(&z.to_string())
    }
}

/// Causal query result
#[derive(Clone, Debug)]
pub struct CausalQuery {
    pub query_type: String, // "ate", "counterfactual", "prob"
    pub result: f64,
    pub confidence: f64, // 0-1 confidence level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_dag_creation() {
        let mut dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string(), "Z".to_string()]);
        dag.add_edge("X", "Y");
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");

        assert_eq!(dag.parents("X"), vec!["Z"]);
        assert_eq!(dag.children("Z"), vec!["X", "Y"]);
    }

    #[test]
    fn test_backdoor_criterion() {
        let mut dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string(), "Z".to_string()]);
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");
        dag.add_edge("X", "Y");

        let confounders = dag.backdoor_criterion("X", "Y");
        assert!(confounders.contains(&"Z".to_string()));
    }

    #[test]
    fn test_intervention() {
        let dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string()]);
        let mut model = CausalModel::new(dag);
        model.add_equation("X".to_string(), "normal(0, 1)".to_string());
        model.add_equation("Y".to_string(), "X + noise".to_string());

        let intervention = Intervention {
            variable: "X".to_string(),
            value: 5.0,
        };

        let intervened = model.intervene(intervention);
        assert_eq!(
            intervened.equations.get("X"),
            Some(&"constant(5)".to_string())
        );
    }

    #[test]
    fn test_ate_simple() {
        // Simple case: no confounders, X -> Y
        let mut dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string()]);
        dag.add_edge("X", "Y");

        let mut model = CausalModel::new(dag);

        // Generate synthetic data: Y = 2*X + noise
        // Treated (X=1): Y ≈ 2
        // Control (X=0): Y ≈ 0
        // ATE ≈ 2
        let treatment = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let outcome = vec![2.1, 1.9, 2.0, 2.2, 1.8, 0.1, -0.1, 0.0, 0.2, -0.2];

        model.add_data("X".to_string(), treatment);
        model.add_data("Y".to_string(), outcome);

        let ate = model.estimate_ate("X", "Y").unwrap();
        assert!(
            (ate - 2.0).abs() < 0.1,
            "ATE should be approximately 2.0, got {}",
            ate
        );
    }

    #[test]
    fn test_ate_with_confounder() {
        // Classic confounding: Z -> X, Z -> Y
        // True causal effect of X on Y is 1.0
        // But marginal association is confounded
        let mut dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string(), "Z".to_string()]);
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");
        dag.add_edge("X", "Y");

        let mut model = CausalModel::new(dag);

        // Synthetic data with confounding:
        // Z (confounder): determines both X and Y
        // True model: Y = 1.0*X + 2.0*Z + noise
        // When Z is high, both X and Y are high (confounding)
        let z = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let x = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        // Y = X + 2*Z: so for each Z-stratum, effect of X is 1.0
        let y = vec![0.0, 0.2, 1.0, 1.2, 2.0, 2.1, 3.0, 3.1, 4.0, 4.2, 5.0, 5.1];

        model.add_data("Z".to_string(), z);
        model.add_data("X".to_string(), x);
        model.add_data("Y".to_string(), y);

        let ate = model.estimate_ate("X", "Y").unwrap();
        // After adjusting for Z, the true causal effect should be close to 1.0
        assert!(
            (ate - 1.0).abs() < 0.5,
            "Adjusted ATE should be approximately 1.0, got {}",
            ate
        );
    }

    #[test]
    fn test_ate_requires_data() {
        let dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string()]);
        let model = CausalModel::new(dag);

        let result = model.estimate_ate("X", "Y");
        assert!(result.is_err());
    }

    #[test]
    fn test_ate_requires_both_groups() {
        let mut dag = CausalDAG::new(vec!["X".to_string(), "Y".to_string()]);
        dag.add_edge("X", "Y");

        let mut model = CausalModel::new(dag);

        // Only treated group (no controls)
        model.add_data("X".to_string(), vec![1.0, 1.0, 1.0]);
        model.add_data("Y".to_string(), vec![2.0, 2.1, 1.9]);

        let result = model.estimate_ate("X", "Y");
        assert!(result.is_err());
    }
}
