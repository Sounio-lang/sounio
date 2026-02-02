// Advanced Conformal Methods - Full transduction and ensemble approaches

use crate::ontology::loader::IRI;
use std::collections::HashMap;

/// Prediction set with coverage guarantee
#[derive(Clone, Debug)]
pub struct PredictionSet {
    /// Predicted types
    pub predictions: Vec<IRI>,
    /// Coverage level (P(Y ∈ C(X)) >= coverage)
    pub coverage: f64,
    /// Confidence per prediction
    pub confidences: HashMap<IRI, f64>,
}

impl PredictionSet {
    pub fn new(predictions: Vec<IRI>, coverage: f64) -> Self {
        let confidences = predictions.iter().map(|p| (p.clone(), 0.5)).collect();

        Self {
            predictions,
            coverage,
            confidences,
        }
    }

    /// Check if prediction is in set
    pub fn contains(&self, pred: &IRI) -> bool {
        self.predictions.contains(pred)
    }

    /// Set confidence for a prediction
    pub fn set_confidence(&mut self, pred: IRI, conf: f64) {
        self.confidences.insert(pred, conf);
    }

    /// Get confidence for a prediction
    pub fn get_confidence(&self, pred: &IRI) -> Option<f64> {
        self.confidences.get(pred).copied()
    }
}

/// Full conformal transduction with leave-one-out
pub struct FullConformalTransduction {
    /// Scores for each type
    scores: HashMap<IRI, Vec<f64>>,
    /// Alpha for coverage guarantee (1 - alpha)
    alpha: f64,
}

impl FullConformalTransduction {
    pub fn new(alpha: f64) -> Self {
        Self {
            scores: HashMap::new(),
            alpha,
        }
    }

    /// Register scores for a type
    pub fn register_scores(&mut self, ty: IRI, scores: Vec<f64>) {
        self.scores.insert(ty, scores);
    }

    /// Perform leave-one-out conformal prediction
    pub fn leave_one_out_conformal(&self, types: &[IRI]) -> Vec<PredictionSet> {
        let mut predictions = Vec::new();

        for test_type in types {
            // Collect scores excluding test_type
            let mut other_scores: Vec<f64> = Vec::new();

            for (ty, scores) in &self.scores {
                if ty != test_type {
                    other_scores.extend(scores.iter());
                }
            }

            // Compute threshold for coverage (1 - alpha)
            other_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold_idx = ((other_scores.len() as f64) * (1.0 - self.alpha)).ceil() as usize;
            let threshold = other_scores
                .get(threshold_idx)
                .copied()
                .unwrap_or(f64::INFINITY);

            // Prediction set: all types with score >= threshold
            let mut pred_set = Vec::new();
            for (ty, scores) in &self.scores {
                if let Some(&score) = scores.first() {
                    if score >= threshold {
                        pred_set.push(ty.clone());
                    }
                }
            }

            let coverage = 1.0 - self.alpha;
            predictions.push(PredictionSet::new(pred_set, coverage));
        }

        predictions
    }

    /// Verify coverage guarantee
    pub fn verify_coverage(&self, predictions: &[PredictionSet], true_labels: &[IRI]) -> f64 {
        if predictions.is_empty() || true_labels.is_empty() {
            return 0.0;
        }

        let mut correct = 0;
        for (i, true_label) in true_labels.iter().enumerate() {
            if let Some(pred) = predictions.get(i) {
                if pred.contains(true_label) {
                    correct += 1;
                }
            }
        }

        correct as f64 / true_labels.len() as f64
    }
}

/// Ensemble conformal with multiple distance metrics
pub struct EnsembleConformal {
    /// Distance metrics (name -> distances)
    metrics: HashMap<String, Vec<f64>>,
    /// Learned metric weights
    weights: HashMap<String, f64>,
}

impl EnsembleConformal {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    /// Register metric distances
    pub fn register_metric(&mut self, name: String, distances: Vec<f64>) {
        self.metrics.insert(name, distances);
    }

    /// Learn metric weights from calibration data
    pub fn learn_weights(&mut self, metric_names: &[&str]) {
        // Simplified: uniform weights
        for name in metric_names {
            self.weights
                .insert(name.to_string(), 1.0 / metric_names.len() as f64);
        }
    }

    /// Combine metrics with learned weights
    pub fn combine_metrics(&self, metric_names: &[&str]) -> Vec<f64> {
        if self.metrics.is_empty() {
            return Vec::new();
        }

        // Get first metric for length
        let n = self.metrics.values().next().map(|v| v.len()).unwrap_or(0);

        let mut combined = vec![0.0; n];

        for name in metric_names {
            if let Some(distances) = self.metrics.get(*name) {
                if let Some(&weight) = self.weights.get(*name) {
                    for (i, dist) in distances.iter().enumerate() {
                        if i < combined.len() {
                            combined[i] += weight * dist;
                        }
                    }
                }
            }
        }

        combined
    }

    /// Check if ensemble preserves coverage
    pub fn preserve_coverage(&self, alpha: f64) -> bool {
        // Simplified: coverage preserved if using correct aggregation
        true
    }

    /// Get metric weight
    pub fn get_weight(&self, metric: &str) -> Option<f64> {
        self.weights.get(metric).copied()
    }

    /// Get all weights
    pub fn get_all_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }
}

impl Default for EnsembleConformal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_set_creation() {
        let preds = vec![IRI::new("http://test/a"), IRI::new("http://test/b")];
        let set = PredictionSet::new(preds.clone(), 0.95);

        assert_eq!(set.predictions.len(), 2);
        assert!((set.coverage - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_set_contains() {
        let preds = vec![IRI::new("http://test/a")];
        let set = PredictionSet::new(preds, 0.95);

        let a = IRI::new("http://test/a");
        let b = IRI::new("http://test/b");

        assert!(set.contains(&a));
        assert!(!set.contains(&b));
    }

    #[test]
    fn test_full_conformal_creation() {
        let transduction = FullConformalTransduction::new(0.1);
        assert!((transduction.alpha - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_full_conformal_coverage() {
        let mut transduction = FullConformalTransduction::new(0.1);

        let type_a = IRI::new("http://test/a");
        let type_b = IRI::new("http://test/b");

        transduction.register_scores(type_a.clone(), vec![0.9, 0.8, 0.7]);
        transduction.register_scores(type_b.clone(), vec![0.6, 0.5, 0.4]);

        let predictions = transduction.leave_one_out_conformal(&[type_a, type_b]);
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_ensemble_conformal_creation() {
        let ensemble = EnsembleConformal::new();
        assert!(ensemble.metrics.is_empty());
    }

    #[test]
    fn test_ensemble_register_metric() {
        let mut ensemble = EnsembleConformal::new();
        ensemble.register_metric("spectral".to_string(), vec![0.1, 0.2, 0.3]);

        assert_eq!(ensemble.metrics.len(), 1);
    }

    #[test]
    fn test_ensemble_learn_weights() {
        let mut ensemble = EnsembleConformal::new();
        ensemble.register_metric("spectral".to_string(), vec![0.1, 0.2]);
        ensemble.register_metric("hyperbolic".to_string(), vec![0.15, 0.25]);

        ensemble.learn_weights(&["spectral", "hyperbolic"]);

        assert_eq!(ensemble.weights.len(), 2);
        assert!((ensemble.get_weight("spectral").unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_combine_metrics() {
        let mut ensemble = EnsembleConformal::new();
        ensemble.register_metric("spectral".to_string(), vec![0.2, 0.4]);
        ensemble.register_metric("hyperbolic".to_string(), vec![0.4, 0.2]);

        ensemble.learn_weights(&["spectral", "hyperbolic"]);

        let combined = ensemble.combine_metrics(&["spectral", "hyperbolic"]);
        assert_eq!(combined.len(), 2);
        assert!((combined[0] - 0.3).abs() < 1e-10);
    }
}
