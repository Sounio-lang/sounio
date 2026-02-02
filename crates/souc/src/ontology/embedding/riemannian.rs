// Sounio Compiler - Riemannian Optimization on Hyperbolic Manifolds
// Gradient descent with tangent space operations, retractions, and adaptive learning rates

use crate::ontology::loader::IRI;
use std::collections::HashMap;

/// Configuration for Riemannian gradient descent
#[derive(Clone, Debug)]
pub struct RiemannianConfig {
    /// Initial learning rate (typically 0.1)
    pub learning_rate: f64,
    /// Learning rate decay per epoch (e.g., 0.95)
    pub decay_rate: f64,
    /// Gradient norm threshold for convergence (e.g., 1e-6)
    pub convergence_threshold: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Batch size for stochastic gradient descent
    pub batch_size: usize,
    /// L2 regularization strength
    pub regularization: f64,
}

impl Default for RiemannianConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            decay_rate: 0.95,
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            batch_size: 32,
            regularization: 0.01,
        }
    }
}

/// Riemannian gradient descent on Poincaré ball manifold
pub struct RiemannianGradientDescent {
    config: RiemannianConfig,
    iteration: usize,
    best_loss: f64,
    loss_history: Vec<f64>,
}

impl RiemannianGradientDescent {
    pub fn new(config: RiemannianConfig) -> Self {
        Self {
            config,
            iteration: 0,
            best_loss: f64::INFINITY,
            loss_history: Vec::new(),
        }
    }

    /// Compute gradient in tangent space at origin
    /// For loss L(exp_0(v)), gradient w.r.t. v is just dL/dv
    pub fn tangent_space_gradient(&self, gradients: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // In tangent space at origin of Poincaré ball, gradient is simply Euclidean gradient
        // This is because exp_0 is identity at origin
        gradients.to_vec()
    }

    /// Retract from tangent space back to manifold using exponential map
    /// Moves point along geodesic: p' = exp_p(α * grad)
    pub fn retract(
        &self,
        point: &[f64],
        tangent_vec: &[f64],
        step_size: f64,
        curvature: f64,
    ) -> Vec<f64> {
        let norm_sq: f64 = tangent_vec.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        if norm < 1e-10 {
            return point.to_vec();
        }

        // Exponential map on Poincaré ball
        // exp_p(v) using Möbius addition: exp_p(v) = p ⊕ tanh(||v||/2) * v/||v||
        let alpha = (norm / 2.0).tanh();
        let scaled_tangent: Vec<f64> = tangent_vec
            .iter()
            .map(|x| step_size * alpha * x / norm)
            .collect();

        // Apply Möbius addition: p ⊕ v
        mobius_add(point, &scaled_tangent, curvature)
    }

    /// Learning rate schedule with exponential decay
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate * self.config.decay_rate.powi(self.iteration as i32)
    }

    /// Check convergence based on gradient norm
    pub fn has_converged(&self, gradient_norm: f64) -> bool {
        gradient_norm < self.config.convergence_threshold
    }

    /// Update iteration counter and return current learning rate
    pub fn step(&mut self, loss: f64) {
        self.iteration += 1;
        self.loss_history.push(loss);
        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    pub fn get_loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    pub fn get_best_loss(&self) -> f64 {
        self.best_loss
    }
}

/// Möbius addition in Poincaré ball
fn mobius_add(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let x_norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let y_norm_sq: f64 = y.iter().map(|yi| yi * yi).sum();
    let xy_dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

    let numerator_factor = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
    let denominator_factor = 1.0 + c * x_norm_sq;

    let result: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            (numerator_factor * xi + c * denominator_factor * yi)
                / (denominator_factor + c * numerator_factor)
        })
        .collect();

    result
}

/// Learn hierarchical embeddings via Riemannian optimization
pub struct HierarchyLearner {
    config: RiemannianConfig,
    optimizer: RiemannianGradientDescent,
    embeddings: HashMap<IRI, Vec<f64>>,
    curvature: f64,
}

impl HierarchyLearner {
    pub fn new(config: RiemannianConfig) -> Self {
        let config_clone = config.clone();
        Self {
            config,
            optimizer: RiemannianGradientDescent::new(config_clone),
            embeddings: HashMap::new(),
            curvature: 1.0,
        }
    }

    /// Learn hierarchy: minimize Σ d(parent, child)² + λ||embedding||²
    pub fn learn_hierarchy(
        &mut self,
        samples: Vec<(IRI, Vec<IRI>)>,
    ) -> Result<HashMap<IRI, Vec<f64>>, String> {
        if samples.is_empty() {
            return Err("No samples provided".to_string());
        }

        let dims = 64; // Default embedding dimension

        // Initialize embeddings at origin
        let mut embeddings: HashMap<IRI, Vec<f64>> = HashMap::new();
        for (parent, children) in &samples {
            embeddings
                .entry(parent.clone())
                .or_insert_with(|| vec![0.0; dims]);
            for child in children {
                embeddings
                    .entry(child.clone())
                    .or_insert_with(|| vec![0.0; dims]);
            }
        }

        // Riemannian gradient descent
        for iteration in 0..self.config.max_iterations {
            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;

            // Process samples in batches
            for batch_samples in samples.chunks(self.config.batch_size) {
                let mut gradients: HashMap<IRI, Vec<f64>> = HashMap::new();

                // Compute loss and gradients
                for (parent, children) in batch_samples {
                    let parent_emb = embeddings.get(parent).unwrap();

                    for child in children {
                        let child_emb = embeddings.get(child).unwrap();

                        // Loss: d(parent, child)²
                        let dist = hyperbolic_distance(parent_emb, child_emb, self.curvature);
                        total_loss += dist * dist;

                        // Gradient w.r.t. parent and child (simplified)
                        let grad_parent =
                            gradient_distance(parent_emb, child_emb, true, self.curvature);
                        let grad_child =
                            gradient_distance(parent_emb, child_emb, false, self.curvature);

                        gradients
                            .entry(parent.clone())
                            .or_insert_with(|| vec![0.0; dims])
                            .iter_mut()
                            .zip(grad_parent.iter())
                            .for_each(|(g, dg)| *g += dg);

                        gradients
                            .entry(child.clone())
                            .or_insert_with(|| vec![0.0; dims])
                            .iter_mut()
                            .zip(grad_child.iter())
                            .for_each(|(g, dg)| *g += dg);
                    }
                }

                // Add L2 regularization: λ||embedding||²
                total_loss += self.config.regularization
                    * embeddings
                        .values()
                        .map(|e| e.iter().map(|x| x * x).sum::<f64>())
                        .sum::<f64>();

                // Update embeddings via retraction
                let lr = self.optimizer.learning_rate();
                for (iri, grad) in &gradients {
                    let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                    total_grad_norm += grad_norm;

                    if let Some(emb) = embeddings.get_mut(iri) {
                        let updated = self.optimizer.retract(emb, grad, lr, self.curvature);
                        *emb = updated;
                    }
                }
            }

            self.optimizer.step(total_loss);
            total_loss /= samples.len() as f64;
            total_grad_norm /= samples.len() as f64;

            if self.optimizer.has_converged(total_grad_norm) {
                break;
            }
        }

        self.embeddings = embeddings.clone();
        Ok(embeddings)
    }

    pub fn get_embedding(&self, iri: &IRI) -> Option<&Vec<f64>> {
        self.embeddings.get(iri)
    }
}

/// Hyperbolic distance (for reference in loss computation)
fn hyperbolic_distance(x: &[f64], y: &[f64], c: f64) -> f64 {
    let x_norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let y_norm_sq: f64 = y.iter().map(|yi| yi * yi).sum();
    let diff_sq: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();

    let numerator = 2.0 * c * diff_sq;
    let denominator = (1.0 - c * x_norm_sq).max(1e-10) * (1.0 - c * y_norm_sq).max(1e-10);
    let arg = 1.0 + numerator / denominator;
    (1.0 / c.sqrt()) * arg.ln() // Simplification of arcosh
}

/// Approximate gradient of distance w.r.t. embedding
fn gradient_distance(x: &[f64], y: &[f64], wrt_x: bool, c: f64) -> Vec<f64> {
    let eps = 1e-5;
    let mut grad = vec![0.0; x.len()];

    let base_dist = hyperbolic_distance(x, y, c);

    for i in 0..x.len() {
        let mut x_plus = if wrt_x { x.to_vec() } else { y.to_vec() };
        x_plus[i] += eps;

        let dist_plus = if wrt_x {
            hyperbolic_distance(&x_plus, y, c)
        } else {
            hyperbolic_distance(x, &x_plus, c)
        };

        grad[i] = (dist_plus - base_dist) / eps;
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemannian_config_default() {
        let config = RiemannianConfig::default();
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_learning_rate_decay() {
        let config = RiemannianConfig::default();
        let mut optimizer = RiemannianGradientDescent::new(config);

        let lr_0 = optimizer.learning_rate();
        optimizer.step(1.0);
        let lr_1 = optimizer.learning_rate();

        assert!(lr_1 < lr_0);
        assert!((lr_1 - lr_0 * 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_retract_origin() {
        let config = RiemannianConfig::default();
        let optimizer = RiemannianGradientDescent::new(config);

        let origin = vec![0.0, 0.0];
        let tangent = vec![0.1, 0.0];

        let retracted = optimizer.retract(&origin, &tangent, 0.1, 1.0);
        assert!(retracted.len() == 2);
        // Retracted point should be inside ball
        let norm: f64 = retracted.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0);
    }

    #[test]
    fn test_hierarchy_learner_convergence() {
        let config = RiemannianConfig {
            learning_rate: 0.05,
            max_iterations: 100,
            ..Default::default()
        };
        let mut learner = HierarchyLearner::new(config);

        let samples = vec![(
            IRI::new("http://test/parent1"),
            vec![
                IRI::new("http://test/child1"),
                IRI::new("http://test/child2"),
            ],
        )];

        let result = learner.learn_hierarchy(samples);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert!(embeddings.len() >= 3);
    }
}
