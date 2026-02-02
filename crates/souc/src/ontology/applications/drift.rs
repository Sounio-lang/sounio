// Concept Drift Detection - Monitor semantic distance changes over time

use crate::ontology::loader::IRI;
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Alert for significant concept drift
#[derive(Clone, Debug)]
pub struct DriftAlert {
    pub concept: IRI,
    pub magnitude: f64,
    pub timestamp: SystemTime,
    pub evidence: String,
}

impl DriftAlert {
    pub fn new(concept: IRI, magnitude: f64, evidence: String) -> Self {
        Self {
            concept,
            magnitude,
            timestamp: SystemTime::now(),
            evidence,
        }
    }
}

/// Concept drift detector with windowed statistics
pub struct ConceptDriftDetector {
    /// Sliding window of observations
    observations: VecDeque<(SystemTime, f64)>,
    /// Window size
    window_size: Duration,
    /// Threshold for drift detection
    drift_threshold: f64,
}

impl ConceptDriftDetector {
    pub fn new(window_size: Duration, drift_threshold: f64) -> Self {
        Self {
            observations: VecDeque::new(),
            window_size,
            drift_threshold,
        }
    }

    /// Add observation of semantic distance for a concept
    pub fn observe(&mut self, distance: f64) {
        self.observations.push_back((SystemTime::now(), distance));

        // Remove old observations outside window
        let cutoff = SystemTime::now() - self.window_size;
        while let Some(&(time, _)) = self.observations.front() {
            if time < cutoff {
                self.observations.pop_front();
            } else {
                break;
            }
        }
    }

    /// Check if significant drift is detected
    pub fn detect_significant_change(&self, concept: &IRI) -> Option<DriftAlert> {
        if self.observations.len() < 2 {
            return None;
        }

        // Compute mean distance in window
        let mean: f64 =
            self.observations.iter().map(|(_, d)| d).sum::<f64>() / self.observations.len() as f64;

        // Compute variance
        let variance = self
            .observations
            .iter()
            .map(|(_, d)| (d - mean).powi(2))
            .sum::<f64>()
            / self.observations.len() as f64;
        let stddev = variance.sqrt();

        // Detect if recent observations deviate significantly from mean
        if let Some((_, recent_dist)) = self.observations.back() {
            let z_score = (recent_dist - mean).abs() / (stddev + 1e-10);

            if z_score > self.drift_threshold {
                let magnitude = z_score;
                return Some(DriftAlert::new(
                    concept.clone(),
                    magnitude,
                    format!(
                        "Z-score: {:.2}, Mean: {:.4}, Recent: {:.4}",
                        z_score, mean, recent_dist
                    ),
                ));
            }
        }

        None
    }

    /// Get recent statistics for a concept
    pub fn get_statistics(&self) -> Option<(f64, f64)> {
        if self.observations.is_empty() {
            return None;
        }

        let mean =
            self.observations.iter().map(|(_, d)| d).sum::<f64>() / self.observations.len() as f64;
        let variance = self
            .observations
            .iter()
            .map(|(_, d)| (d - mean).powi(2))
            .sum::<f64>()
            / self.observations.len() as f64;

        Some((mean, variance.sqrt()))
    }

    /// Clear observations
    pub fn reset(&mut self) {
        self.observations.clear();
    }

    /// Get observation count
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_alert_creation() {
        let concept = IRI::new("http://test/concept");
        let alert = DriftAlert::new(concept.clone(), 3.5, "test evidence".to_string());

        assert_eq!(alert.concept, concept);
        assert!((alert.magnitude - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_drift_detector_observation() {
        let mut detector = ConceptDriftDetector::new(Duration::from_secs(60), 2.0);

        detector.observe(0.5);
        detector.observe(0.5);
        detector.observe(0.5);

        assert_eq!(detector.observation_count(), 3);
    }

    #[test]
    fn test_drift_detector_statistics() {
        let mut detector = ConceptDriftDetector::new(Duration::from_secs(60), 2.0);

        detector.observe(1.0);
        detector.observe(2.0);
        detector.observe(3.0);

        if let Some((mean, stddev)) = detector.get_statistics() {
            assert!((mean - 2.0).abs() < 0.1);
            assert!(stddev > 0.0);
        }
    }

    #[test]
    fn test_drift_detection() {
        let mut detector = ConceptDriftDetector::new(Duration::from_secs(60), 1.5);

        // Baseline observations
        for _ in 0..5 {
            detector.observe(1.0);
        }

        // Sudden shift
        detector.observe(5.0);

        let concept = IRI::new("http://test/concept");
        let alert = detector.detect_significant_change(&concept);

        // Should detect drift
        assert!(alert.is_some());
    }

    #[test]
    fn test_drift_detector_reset() {
        let mut detector = ConceptDriftDetector::new(Duration::from_secs(60), 2.0);

        detector.observe(0.5);
        detector.observe(0.5);

        detector.reset();
        assert_eq!(detector.observation_count(), 0);
    }
}
