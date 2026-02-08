//! Geometric Predicates
//!
//! First-order predicates for symbolic geometry reasoning.
//! Each predicate carries epistemic metadata (confidence, provenance).

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::merkle::{Hash256, hash};
use crate::epistemic::{Revisability, Source};

/// Kind of geometric predicate
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum PredicateKind {
    // Point relations
    /// Three points are collinear
    Collinear,
    /// Four points are concyclic (lie on a circle)
    Concyclic,
    /// Point is on a line
    OnLine,
    /// Point is on a circle
    OnCircle,
    /// Two points are equal
    EqualPoints,

    // Line relations
    /// Two lines are parallel
    Parallel,
    /// Two lines are perpendicular
    Perpendicular,

    // Length relations
    /// Two segments have equal length
    EqualLength,
    /// Ratio of two lengths equals a value
    LengthRatio,

    // Angle relations
    /// Two angles are equal
    EqualAngle,
    /// Angle is right (90 degrees)
    RightAngle,
    /// Sum of angles equals a value
    AngleSum,

    // Triangle relations
    /// Two triangles are similar
    Similar,
    /// Two triangles are congruent
    Congruent,

    // Special points
    /// Point is midpoint of segment
    Midpoint,
    /// Line is angle bisector
    AngleBisector,
    /// Line is perpendicular bisector
    PerpBisector,
    /// Point is circumcenter
    Circumcenter,
    /// Point is incenter
    Incenter,
    /// Point is centroid
    Centroid,
    /// Point is orthocenter
    Orthocenter,

    // Circle relations
    /// Line is tangent to circle at point
    Tangent,
    /// Point is center of circle
    CircleCenter,

    // Algebraic equality (for AR)
    /// Two expressions are equal
    AlgebraicEqual,
}

impl PredicateKind {
    /// Get the arity (number of arguments) for this predicate kind
    pub fn arity(&self) -> usize {
        match self {
            PredicateKind::Collinear => 3,
            PredicateKind::Concyclic => 4,
            PredicateKind::OnLine => 2,   // point, line (2 points)
            PredicateKind::OnCircle => 2, // point, circle
            PredicateKind::EqualPoints => 2,
            PredicateKind::Parallel => 4, // line1 (2 pts), line2 (2 pts)
            PredicateKind::Perpendicular => 4,
            PredicateKind::EqualLength => 4, // seg1 (2 pts), seg2 (2 pts)
            PredicateKind::LengthRatio => 5, // seg1, seg2, ratio
            PredicateKind::EqualAngle => 6,  // angle1 (3 pts), angle2 (3 pts)
            PredicateKind::RightAngle => 3,
            PredicateKind::AngleSum => 7, // angle1, angle2, sum
            PredicateKind::Similar => 6,  // tri1, tri2
            PredicateKind::Congruent => 6,
            PredicateKind::Midpoint => 3,      // mid, p1, p2
            PredicateKind::AngleBisector => 4, // line, angle
            PredicateKind::PerpBisector => 4,  // line, segment
            PredicateKind::Circumcenter => 4,  // center, p1, p2, p3
            PredicateKind::Incenter => 4,
            PredicateKind::Centroid => 4,
            PredicateKind::Orthocenter => 4,
            PredicateKind::Tangent => 4, // line, circle, point
            PredicateKind::CircleCenter => 2,
            PredicateKind::AlgebraicEqual => 2, // expr1, expr2
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            PredicateKind::Collinear => "collinear",
            PredicateKind::Concyclic => "concyclic",
            PredicateKind::OnLine => "on_line",
            PredicateKind::OnCircle => "on_circle",
            PredicateKind::EqualPoints => "equal_points",
            PredicateKind::Parallel => "parallel",
            PredicateKind::Perpendicular => "perpendicular",
            PredicateKind::EqualLength => "equal_length",
            PredicateKind::LengthRatio => "length_ratio",
            PredicateKind::EqualAngle => "equal_angle",
            PredicateKind::RightAngle => "right_angle",
            PredicateKind::AngleSum => "angle_sum",
            PredicateKind::Similar => "similar",
            PredicateKind::Congruent => "congruent",
            PredicateKind::Midpoint => "midpoint",
            PredicateKind::AngleBisector => "angle_bisector",
            PredicateKind::PerpBisector => "perp_bisector",
            PredicateKind::Circumcenter => "circumcenter",
            PredicateKind::Incenter => "incenter",
            PredicateKind::Centroid => "centroid",
            PredicateKind::Orthocenter => "orthocenter",
            PredicateKind::Tangent => "tangent",
            PredicateKind::CircleCenter => "circle_center",
            PredicateKind::AlgebraicEqual => "algebraic_equal",
        }
    }
}

/// Epistemic status of a predicate
///
/// Uses Beta distributions for full uncertainty quantification.
/// Beta(α, β) where:
/// - α = number of "successes" (evidence supporting the predicate)
/// - β = number of "failures" (evidence against)
/// - mean = α / (α + β)
/// - variance = αβ / ((α+β)²(α+β+1))
///
/// # Merkle Provenance
///
/// Each predicate carries a Merkle hash that cryptographically commits to:
/// - The predicate content (kind + arguments)
/// - Parent predicate hashes (for derived predicates)
/// - The derivation rule used
/// - Timestamp/depth information
///
/// This enables:
/// - Verifiable audit trails for regulatory compliance
/// - Efficient proof verification without re-running inference
/// - Tamper-evident provenance for scientific reproducibility
#[derive(Debug, Clone)]
pub struct PredicateEpistemic {
    /// Confidence in the predicate (Beta distribution)
    pub confidence: BetaConfidence,
    /// How this predicate was derived
    pub source: Source,
    /// Whether this can be revised
    pub revisability: Revisability,
    /// Depth in proof tree (0 = axiom)
    pub depth: usize,
    /// IDs of predicates this was derived from
    pub derived_from: Vec<PredicateId>,
    /// Merkle hash for provenance tracking
    /// H(content || parent_hashes || source || depth)
    pub merkle_hash: Option<Hash256>,
    /// Parent Merkle hashes (for verification)
    pub parent_hashes: Vec<Hash256>,
}

impl Default for PredicateEpistemic {
    fn default() -> Self {
        PredicateEpistemic {
            confidence: BetaConfidence::uniform_prior(), // Start uncertain
            source: Source::Unknown,
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".to_string()],
            },
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }
}

impl PredicateEpistemic {
    /// Create axiom epistemic status (from problem statement)
    ///
    /// Axioms have very high confidence: Beta(100, 1) ≈ 0.99 mean, low variance
    pub fn axiom() -> Self {
        PredicateEpistemic {
            confidence: BetaConfidence::new(100.0, 1.0), // Very high confidence, low variance
            source: Source::Axiom,
            revisability: Revisability::NonRevisable,
            depth: 0,
            derived_from: vec![],
            merkle_hash: None, // Will be computed when predicate content is known
            parent_hashes: vec![],
        }
    }

    /// Create axiom with Merkle hash computed from predicate content
    pub fn axiom_with_hash(predicate_key: &str) -> Self {
        let merkle_hash = Self::compute_axiom_hash(predicate_key);
        PredicateEpistemic {
            confidence: BetaConfidence::new(100.0, 1.0),
            source: Source::Axiom,
            revisability: Revisability::NonRevisable,
            depth: 0,
            derived_from: vec![],
            merkle_hash: Some(merkle_hash),
            parent_hashes: vec![],
        }
    }

    /// Create derived epistemic status with Bayesian combination and Merkle provenance
    ///
    /// For derived predicates, we combine parent Beta distributions:
    /// - Multiply α values (intersection of evidence)
    /// - Apply decay to model rule reliability
    ///
    /// Merkle hash commits to: content || parent_hashes || rule || depth
    pub fn derived(parents: &[&Predicate], rule_name: &str, decay: f64) -> Self {
        let parent_ids: Vec<PredicateId> = parents.iter().map(|p| p.id).collect();
        let max_depth = parents.iter().map(|p| p.epistemic.depth).max().unwrap_or(0);

        // Collect parent Merkle hashes for provenance chain
        let parent_hashes: Vec<Hash256> = parents
            .iter()
            .filter_map(|p| p.epistemic.merkle_hash)
            .collect();

        // Combine parent Beta distributions
        // For conjunction (all parents must be true), we use the minimum confidence approach
        // but preserve the variance information
        let combined = if parents.is_empty() {
            BetaConfidence::uniform_prior()
        } else {
            // Aggregate: product of means with geometric mean of sample sizes
            let combined_mean: f64 = parents
                .iter()
                .map(|p| p.epistemic.confidence.mean())
                .product();

            // Aggregate sample size (geometric mean preserves uncertainty)
            let combined_n: f64 = parents
                .iter()
                .map(|p| p.epistemic.confidence.sample_size())
                .fold(1.0, |acc, n| acc * n)
                .powf(1.0 / parents.len() as f64);

            // Apply decay to model rule reliability
            let decayed_mean = combined_mean * decay;

            BetaConfidence::from_confidence(decayed_mean, combined_n)
        };

        PredicateEpistemic {
            confidence: combined,
            source: Source::Derivation(rule_name.to_string()),
            revisability: Revisability::Revisable {
                conditions: vec!["parent_revision".to_string()],
            },
            depth: max_depth + 1,
            derived_from: parent_ids,
            merkle_hash: None, // Will be computed when predicate content is known
            parent_hashes,
        }
    }

    /// Create derived epistemic with full Merkle hash
    pub fn derived_with_hash(
        parents: &[&Predicate],
        rule_name: &str,
        decay: f64,
        predicate_key: &str,
    ) -> Self {
        let mut epistemic = Self::derived(parents, rule_name, decay);
        epistemic.merkle_hash = Some(Self::compute_derived_hash(
            predicate_key,
            &epistemic.parent_hashes,
            rule_name,
            epistemic.depth,
        ));
        epistemic
    }

    /// Compute Merkle hash for an axiom predicate
    fn compute_axiom_hash(predicate_key: &str) -> Hash256 {
        // H(predicate_key || "axiom" || depth=0)
        let mut data = Vec::new();
        data.extend_from_slice(predicate_key.as_bytes());
        data.extend_from_slice(b"::axiom::0");
        hash(&data)
    }

    /// Compute Merkle hash for a derived predicate
    fn compute_derived_hash(
        predicate_key: &str,
        parent_hashes: &[Hash256],
        rule_name: &str,
        depth: usize,
    ) -> Hash256 {
        // H(predicate_key || parent_hashes || rule_name || depth)
        let mut data = Vec::new();
        data.extend_from_slice(predicate_key.as_bytes());
        for parent_hash in parent_hashes {
            data.extend_from_slice(parent_hash.as_bytes());
        }
        data.extend_from_slice(b"::");
        data.extend_from_slice(rule_name.as_bytes());
        data.extend_from_slice(b"::");
        data.extend_from_slice(depth.to_string().as_bytes());
        hash(&data)
    }

    /// Compute and set the Merkle hash for this epistemic status
    pub fn compute_hash(&mut self, predicate_key: &str) {
        self.merkle_hash = Some(match &self.source {
            Source::Axiom => Self::compute_axiom_hash(predicate_key),
            Source::Derivation(rule) => {
                Self::compute_derived_hash(predicate_key, &self.parent_hashes, rule, self.depth)
            }
            Source::ModelPrediction { model, .. } => Self::compute_derived_hash(
                predicate_key,
                &[],
                &format!("neural:{}", model),
                self.depth,
            ),
            _ => Self::compute_derived_hash(
                predicate_key,
                &self.parent_hashes,
                "unknown",
                self.depth,
            ),
        });
    }

    /// Verify the Merkle hash is valid
    pub fn verify_hash(&self, predicate_key: &str) -> bool {
        match self.merkle_hash {
            Some(stored_hash) => {
                let computed = match &self.source {
                    Source::Axiom => Self::compute_axiom_hash(predicate_key),
                    Source::Derivation(rule) => Self::compute_derived_hash(
                        predicate_key,
                        &self.parent_hashes,
                        rule,
                        self.depth,
                    ),
                    Source::ModelPrediction { model, .. } => Self::compute_derived_hash(
                        predicate_key,
                        &[],
                        &format!("neural:{}", model),
                        self.depth,
                    ),
                    _ => Self::compute_derived_hash(
                        predicate_key,
                        &self.parent_hashes,
                        "unknown",
                        self.depth,
                    ),
                };
                stored_hash == computed
            }
            None => true, // No hash to verify
        }
    }

    /// Get the Merkle hash, computing it if necessary
    pub fn get_or_compute_hash(&mut self, predicate_key: &str) -> Hash256 {
        if self.merkle_hash.is_none() {
            self.compute_hash(predicate_key);
        }
        self.merkle_hash.unwrap()
    }

    /// Create from neural prediction
    ///
    /// Neural predictions have higher uncertainty than symbolic derivations.
    /// We use lower sample size to reflect this.
    pub fn from_neural(model: &str, confidence: f64) -> Self {
        // Neural predictions have sample_size ~5-10 (uncertain)
        let beta = BetaConfidence::from_confidence(confidence, 5.0);

        PredicateEpistemic {
            confidence: beta,
            source: Source::ModelPrediction {
                model: model.to_string(),
                version: None,
            },
            revisability: Revisability::Revisable {
                conditions: vec!["verification".to_string()],
            },
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }

    /// Create from neural prediction with epistemic output
    pub fn from_neural_epistemic(model: &str, beta: BetaConfidence) -> Self {
        PredicateEpistemic {
            confidence: beta,
            source: Source::ModelPrediction {
                model: model.to_string(),
                version: None,
            },
            revisability: Revisability::Revisable {
                conditions: vec!["verification".to_string()],
            },
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }

    /// Decay confidence (apply multiplicative factor)
    pub fn decay(&self, factor: f64) -> Self {
        let new_mean = self.confidence.mean() * factor;
        let sample_size = self.confidence.sample_size();
        PredicateEpistemic {
            confidence: BetaConfidence::from_confidence(new_mean, sample_size),
            ..self.clone()
        }
    }

    /// Update with new evidence (Bayesian update)
    pub fn update(&mut self, success: f64, failure: f64) {
        self.confidence.update(success, failure);
    }

    /// Get variance (epistemic uncertainty)
    pub fn variance(&self) -> f64 {
        self.confidence.variance()
    }

    /// Get 95% credible interval
    pub fn credible_interval(&self) -> (f64, f64) {
        self.confidence.credible_interval(0.95)
    }

    /// Check if this predicate is uncertain (high variance)
    pub fn is_uncertain(&self, threshold: f64) -> bool {
        self.confidence.variance() > threshold
    }

    /// Get confidence value (mean of Beta distribution)
    pub fn value(&self) -> f64 {
        self.confidence.mean()
    }
}

/// Unique identifier for predicates
pub type PredicateId = u64;

/// A geometric predicate with epistemic metadata
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Unique identifier
    pub id: PredicateId,
    /// Kind of predicate
    pub kind: PredicateKind,
    /// Arguments (point labels, in canonical order)
    pub args: Vec<String>,
    /// Epistemic metadata
    pub epistemic: PredicateEpistemic,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(kind: PredicateKind, args: Vec<String>) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Generate ID from kind + args
        let mut hasher = DefaultHasher::new();
        kind.hash(&mut hasher);
        for arg in &args {
            arg.hash(&mut hasher);
        }
        let id = hasher.finish();

        Predicate {
            id,
            kind,
            args,
            epistemic: PredicateEpistemic::default(),
        }
    }

    /// Create with epistemic status
    pub fn with_epistemic(mut self, epistemic: PredicateEpistemic) -> Self {
        self.epistemic = epistemic;
        self
    }

    /// Create collinear predicate
    pub fn collinear(p1: &str, p2: &str, p3: &str) -> Self {
        let mut args = vec![p1.to_string(), p2.to_string(), p3.to_string()];
        args.sort(); // Canonical order
        Predicate::new(PredicateKind::Collinear, args)
    }

    /// Create concyclic predicate
    pub fn concyclic(p1: &str, p2: &str, p3: &str, p4: &str) -> Self {
        let mut args = vec![
            p1.to_string(),
            p2.to_string(),
            p3.to_string(),
            p4.to_string(),
        ];
        args.sort();
        Predicate::new(PredicateKind::Concyclic, args)
    }

    /// Create parallel predicate
    pub fn parallel(l1_p1: &str, l1_p2: &str, l2_p1: &str, l2_p2: &str) -> Self {
        // Canonical: sort within each line, then sort lines
        let mut l1 = vec![l1_p1.to_string(), l1_p2.to_string()];
        let mut l2 = vec![l2_p1.to_string(), l2_p2.to_string()];
        l1.sort();
        l2.sort();

        let (first, second) = if l1 <= l2 { (l1, l2) } else { (l2, l1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::Parallel, args)
    }

    /// Create perpendicular predicate
    pub fn perpendicular(l1_p1: &str, l1_p2: &str, l2_p1: &str, l2_p2: &str) -> Self {
        let mut l1 = vec![l1_p1.to_string(), l1_p2.to_string()];
        let mut l2 = vec![l2_p1.to_string(), l2_p2.to_string()];
        l1.sort();
        l2.sort();

        let (first, second) = if l1 <= l2 { (l1, l2) } else { (l2, l1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::Perpendicular, args)
    }

    /// Create equal length predicate
    pub fn equal_length(s1_p1: &str, s1_p2: &str, s2_p1: &str, s2_p2: &str) -> Self {
        let mut s1 = vec![s1_p1.to_string(), s1_p2.to_string()];
        let mut s2 = vec![s2_p1.to_string(), s2_p2.to_string()];
        s1.sort();
        s2.sort();

        let (first, second) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::EqualLength, args)
    }

    /// Create midpoint predicate
    pub fn midpoint(mid: &str, p1: &str, p2: &str) -> Self {
        let mut endpoints = [p1.to_string(), p2.to_string()];
        endpoints.sort();
        let args = vec![mid.to_string(), endpoints[0].clone(), endpoints[1].clone()];
        Predicate::new(PredicateKind::Midpoint, args)
    }

    /// Create right angle predicate
    pub fn right_angle(p1: &str, vertex: &str, p2: &str) -> Self {
        let mut rays = [p1.to_string(), p2.to_string()];
        rays.sort();
        let args = vec![rays[0].clone(), vertex.to_string(), rays[1].clone()];
        Predicate::new(PredicateKind::RightAngle, args)
    }

    /// Create on_circle predicate
    pub fn on_circle(point: &str, center: &str, on_circle: &str) -> Self {
        Predicate::new(
            PredicateKind::OnCircle,
            vec![point.to_string(), center.to_string(), on_circle.to_string()],
        )
    }

    /// Get canonical key for deduplication
    pub fn key(&self) -> String {
        format!("{}:{}", self.kind.name(), self.args.join(","))
    }

    /// Check if this predicate is high confidence
    pub fn is_high_confidence(&self, threshold: f64) -> bool {
        self.epistemic.confidence.mean() >= threshold
    }

    /// Check if this predicate is high confidence with low uncertainty
    pub fn is_certain(&self, confidence_threshold: f64, max_variance: f64) -> bool {
        self.epistemic.confidence.mean() >= confidence_threshold
            && self.epistemic.confidence.variance() <= max_variance
    }

    /// Get the epistemic uncertainty (variance) of this predicate
    pub fn uncertainty(&self) -> f64 {
        self.epistemic.confidence.variance()
    }

    /// Get all referenced point labels
    pub fn referenced_points(&self) -> &[String] {
        &self.args
    }
}

impl PartialEq for Predicate {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.args == other.args
    }
}

impl Eq for Predicate {}

impl Hash for Predicate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.args.hash(state);
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.kind.name(), self.args.join(", "))
    }
}

/// Pattern for matching predicates in rules
#[derive(Debug, Clone)]
pub struct PredicatePattern {
    /// Kind to match
    pub kind: PredicateKind,
    /// Variable names for arguments (for binding)
    pub vars: Vec<String>,
}

impl PredicatePattern {
    pub fn new(kind: PredicateKind, vars: Vec<&str>) -> Self {
        PredicatePattern {
            kind,
            vars: vars.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Try to match a predicate, returning variable bindings if successful
    pub fn match_predicate(&self, pred: &Predicate) -> Option<HashMap<String, String>> {
        if self.kind != pred.kind || self.vars.len() != pred.args.len() {
            return None;
        }

        let mut bindings = HashMap::new();
        for (var, arg) in self.vars.iter().zip(pred.args.iter()) {
            if let Some(existing) = bindings.get(var) {
                if existing != arg {
                    return None; // Conflict
                }
            } else {
                bindings.insert(var.clone(), arg.clone());
            }
        }

        Some(bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_canonical() {
        let p1 = Predicate::collinear("A", "B", "C");
        let p2 = Predicate::collinear("C", "A", "B");
        assert_eq!(p1.key(), p2.key());
    }

    #[test]
    fn test_parallel_canonical() {
        let p1 = Predicate::parallel("A", "B", "C", "D");
        let p2 = Predicate::parallel("C", "D", "B", "A");
        assert_eq!(p1.key(), p2.key());
    }

    #[test]
    fn test_pattern_match() {
        let pattern = PredicatePattern::new(PredicateKind::Collinear, vec!["X", "Y", "Z"]);
        let pred = Predicate::collinear("A", "B", "C");

        let bindings = pattern.match_predicate(&pred).unwrap();
        assert_eq!(bindings.len(), 3);
    }

    #[test]
    fn test_epistemic_decay() {
        let epi = PredicateEpistemic::axiom();
        // Axiom has Beta(100, 1), mean ≈ 0.99
        assert!(epi.confidence.mean() > 0.98);

        let decayed = epi.decay(0.95);
        // After decay, mean should be ~0.95 * 0.99 ≈ 0.94
        assert!(decayed.confidence.mean() < epi.confidence.mean());
    }

    #[test]
    fn test_epistemic_variance() {
        // Axiom has low variance (high certainty)
        let axiom = PredicateEpistemic::axiom();
        assert!(axiom.variance() < 0.01);

        // Neural prediction has higher variance (more uncertain)
        let neural = PredicateEpistemic::from_neural("test_model", 0.8);
        assert!(neural.variance() > axiom.variance());
    }

    #[test]
    fn test_epistemic_update() {
        let mut epi = PredicateEpistemic::from_neural("test", 0.5);
        let initial_mean = epi.confidence.mean();

        // Update with positive evidence
        epi.update(5.0, 0.0);

        // Mean should increase
        assert!(epi.confidence.mean() > initial_mean);
    }

    // =========================================================================
    // Merkle Provenance Tests
    // =========================================================================

    #[test]
    fn test_axiom_merkle_hash() {
        let epi = PredicateEpistemic::axiom_with_hash("collinear:A,B,C");
        assert!(epi.merkle_hash.is_some());
        assert!(epi.parent_hashes.is_empty());
    }

    #[test]
    fn test_axiom_hash_deterministic() {
        let epi1 = PredicateEpistemic::axiom_with_hash("collinear:A,B,C");
        let epi2 = PredicateEpistemic::axiom_with_hash("collinear:A,B,C");
        assert_eq!(epi1.merkle_hash, epi2.merkle_hash);
    }

    #[test]
    fn test_different_predicates_different_hashes() {
        let epi1 = PredicateEpistemic::axiom_with_hash("collinear:A,B,C");
        let epi2 = PredicateEpistemic::axiom_with_hash("collinear:A,B,D");
        assert_ne!(epi1.merkle_hash, epi2.merkle_hash);
    }

    #[test]
    fn test_derived_merkle_hash() {
        // Create parent predicates with hashes
        let mut p1 = Predicate::collinear("A", "B", "C");
        p1.epistemic = PredicateEpistemic::axiom_with_hash(&p1.key());

        let mut p2 = Predicate::collinear("A", "B", "D");
        p2.epistemic = PredicateEpistemic::axiom_with_hash(&p2.key());

        // Create derived predicate
        let derived_key = "collinear:B,C,D";
        let epi = PredicateEpistemic::derived_with_hash(
            &[&p1, &p2],
            "collinear_trans",
            0.99,
            derived_key,
        );

        assert!(epi.merkle_hash.is_some());
        assert_eq!(epi.parent_hashes.len(), 2);
    }

    #[test]
    fn test_merkle_hash_verification() {
        let key = "midpoint:M,A,B";
        let epi = PredicateEpistemic::axiom_with_hash(key);

        // Should verify correctly
        assert!(epi.verify_hash(key));

        // Should fail with wrong key
        assert!(!epi.verify_hash("midpoint:M,A,C"));
    }

    #[test]
    fn test_compute_hash_lazy() {
        let mut epi = PredicateEpistemic::axiom();
        assert!(epi.merkle_hash.is_none());

        let key = "parallel:A,B,C,D";
        let hash = epi.get_or_compute_hash(key);

        assert!(epi.merkle_hash.is_some());
        assert_eq!(epi.merkle_hash.unwrap(), hash);
    }

    #[test]
    fn test_derived_hash_includes_parents() {
        // Create parent with hash
        let mut p1 = Predicate::collinear("A", "B", "C");
        p1.epistemic = PredicateEpistemic::axiom_with_hash(&p1.key());

        // Create derived with parent
        let derived_key = "collinear:A,B,D";
        let epi_with_parent =
            PredicateEpistemic::derived_with_hash(&[&p1], "rule1", 0.99, derived_key);

        // Create derived without parent (empty parents)
        let epi_no_parent = PredicateEpistemic::derived_with_hash(&[], "rule1", 0.99, derived_key);

        // Hashes should be different because parent hashes are included
        assert_ne!(epi_with_parent.merkle_hash, epi_no_parent.merkle_hash);
    }

    #[test]
    fn test_merkle_provenance_chain() {
        // Build a chain: axiom -> derived1 -> derived2
        let mut axiom = Predicate::collinear("A", "B", "C");
        axiom.epistemic = PredicateEpistemic::axiom_with_hash(&axiom.key());

        let derived1_key = "midpoint:M,A,B";
        let mut derived1 = Predicate::midpoint("M", "A", "B");
        derived1.epistemic =
            PredicateEpistemic::derived_with_hash(&[&axiom], "midpoint_rule", 0.99, derived1_key);

        let derived2_key = "equal_length:A,M,M,B";
        let derived2_epi = PredicateEpistemic::derived_with_hash(
            &[&derived1],
            "midpoint_equal",
            0.98,
            derived2_key,
        );

        // Verify the chain
        assert!(axiom.epistemic.parent_hashes.is_empty());
        assert_eq!(derived1.epistemic.parent_hashes.len(), 1);
        assert_eq!(derived2_epi.parent_hashes.len(), 1);

        // Parent hash of derived2 should be derived1's hash
        assert_eq!(
            derived2_epi.parent_hashes[0],
            derived1.epistemic.merkle_hash.unwrap()
        );
    }

    #[test]
    fn test_neural_prediction_merkle() {
        let mut epi = PredicateEpistemic::from_neural("alpha_geometry_v1", 0.85);
        assert!(epi.merkle_hash.is_none());

        let key = "on_circle:P,O,A";
        epi.compute_hash(key);

        assert!(epi.merkle_hash.is_some());
        assert!(epi.verify_hash(key));
    }
}
