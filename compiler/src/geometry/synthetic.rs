//! Synthetic Geometry Problem Generator
//!
//! Generates random geometry problems for self-play training with:
//! - Controllable difficulty scaling
//! - Diverse theorem types (parallel, perpendicular, equal length, cyclic, etc.)
//! - Construction-rich problems that require auxiliary points
//! - Epistemic difficulty estimation based on symbolic solver performance
//!
//! # Key Innovation
//!
//! Problems are generated with estimated difficulty based on:
//! 1. Number of premises
//! 2. Theorem type complexity
//! 3. Required auxiliary constructions (estimated)
//! 4. Historical solve rate (updated during training)

use std::collections::HashMap;

use crate::epistemic::bayesian::BetaConfidence;

use super::predicates::{Predicate, PredicateKind};
use super::proof_state::ProofState;

// =============================================================================
// Problem Templates
// =============================================================================

/// Types of synthetic problems we can generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemTemplate {
    /// Midpoint theorem variants: M midpoint of AB, N midpoint of AC => MN || BC
    MidpointTheorem,
    /// Isoceles properties: AB = AC => various conclusions
    IsocelesPerpendicular,
    /// Triangle congruence: SAS, SSS, ASA variants
    TriangleCongruence,
    /// Cyclic quadrilateral: points on circle => angle relations
    CyclicQuadrilateral,
    /// Parallel line transversal: parallel lines cut by transversal
    ParallelTransversal,
    /// Angle bisector properties
    AngleBisector,
    /// Circumcircle/incircle properties
    CircleCenter,
    /// Orthocenter properties
    Orthocenter,
    /// Centroid properties
    Centroid,
    /// Nine-point circle
    NinePointCircle,
    /// Simson line
    SimsonLine,
    /// Power of a point
    PowerOfPoint,
    /// Radical axis
    RadicalAxis,
    /// Ceva's theorem variants
    Ceva,
    /// Menelaus theorem variants
    Menelaus,
}

impl ProblemTemplate {
    /// Get all templates
    pub fn all() -> Vec<Self> {
        vec![
            Self::MidpointTheorem,
            Self::IsocelesPerpendicular,
            Self::TriangleCongruence,
            Self::CyclicQuadrilateral,
            Self::ParallelTransversal,
            Self::AngleBisector,
            Self::CircleCenter,
            Self::Orthocenter,
            Self::Centroid,
            Self::NinePointCircle,
            Self::SimsonLine,
            Self::PowerOfPoint,
            Self::RadicalAxis,
            Self::Ceva,
            Self::Menelaus,
        ]
    }

    /// Base difficulty for this template type (0-1)
    pub fn base_difficulty(&self) -> f64 {
        match self {
            Self::MidpointTheorem => 0.2,
            Self::IsocelesPerpendicular => 0.3,
            Self::TriangleCongruence => 0.4,
            Self::ParallelTransversal => 0.3,
            Self::AngleBisector => 0.5,
            Self::CyclicQuadrilateral => 0.6,
            Self::CircleCenter => 0.5,
            Self::Orthocenter => 0.6,
            Self::Centroid => 0.4,
            Self::NinePointCircle => 0.8,
            Self::SimsonLine => 0.7,
            Self::PowerOfPoint => 0.6,
            Self::RadicalAxis => 0.7,
            Self::Ceva => 0.7,
            Self::Menelaus => 0.7,
        }
    }

    /// Expected number of auxiliary constructions needed
    pub fn expected_constructions(&self) -> usize {
        match self {
            Self::MidpointTheorem => 0,
            Self::IsocelesPerpendicular => 0,
            Self::TriangleCongruence => 0,
            Self::ParallelTransversal => 1,
            Self::AngleBisector => 1,
            Self::CyclicQuadrilateral => 1,
            Self::CircleCenter => 1,
            Self::Orthocenter => 2,
            Self::Centroid => 1,
            Self::NinePointCircle => 3,
            Self::SimsonLine => 2,
            Self::PowerOfPoint => 1,
            Self::RadicalAxis => 2,
            Self::Ceva => 2,
            Self::Menelaus => 2,
        }
    }
}

// =============================================================================
// Synthetic Problem
// =============================================================================

/// A synthetically generated geometry problem
#[derive(Debug, Clone)]
pub struct SyntheticProblem {
    /// Unique problem ID
    pub id: String,
    /// Template used to generate
    pub template: ProblemTemplate,
    /// Initial proof state with premises
    pub state: ProofState,
    /// Goal predicate to prove
    pub goal: Predicate,
    /// Estimated difficulty (0-1)
    pub difficulty: f64,
    /// Generation seed for reproducibility
    pub seed: u64,
    /// Number of premises
    pub num_premises: usize,
    /// Expected constructions needed
    pub expected_constructions: usize,
    /// Historical solve rate (updated during training)
    pub historical_solve_rate: BetaConfidence,
    /// Variation parameters used
    pub variation: ProblemVariation,
}

/// Variation parameters for problem generation
#[derive(Debug, Clone, Default)]
pub struct ProblemVariation {
    /// Number of extra points added
    pub extra_points: usize,
    /// Number of extra premises (distractors)
    pub extra_premises: usize,
    /// Whether to use non-standard point labels
    pub randomize_labels: bool,
    /// Complexity multiplier
    pub complexity: f64,
}

// =============================================================================
// Problem Generator
// =============================================================================

/// Configuration for synthetic problem generation
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Minimum difficulty
    pub min_difficulty: f64,
    /// Maximum difficulty
    pub max_difficulty: f64,
    /// Target difficulty (for curriculum)
    pub target_difficulty: f64,
    /// Difficulty variance (how much to vary around target)
    pub difficulty_variance: f64,
    /// Whether to include distractor premises
    pub include_distractors: bool,
    /// Maximum extra points to add
    pub max_extra_points: usize,
    /// Whether to randomize point labels
    pub randomize_labels: bool,
    /// Weight for easy problems in curriculum
    pub easy_weight: f64,
    /// Weight for hard problems in curriculum
    pub hard_weight: f64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_difficulty: 0.1,
            max_difficulty: 0.9,
            target_difficulty: 0.5,
            difficulty_variance: 0.2,
            include_distractors: true,
            max_extra_points: 2,
            randomize_labels: false,
            easy_weight: 0.3,
            hard_weight: 0.3,
        }
    }
}

/// Generator for synthetic geometry problems
pub struct SyntheticProblemGenerator {
    /// Configuration
    pub config: GeneratorConfig,
    /// Counter for unique IDs
    counter: u64,
    /// Pseudo-random state
    rng_state: u64,
    /// Template weights (updated based on solve rates)
    template_weights: HashMap<ProblemTemplate, f64>,
    /// Historical solve rates per template
    template_solve_rates: HashMap<ProblemTemplate, BetaConfidence>,
    /// Statistics
    pub stats: GeneratorStats,
}

/// Statistics for the generator
#[derive(Debug, Clone, Default)]
pub struct GeneratorStats {
    pub problems_generated: usize,
    pub by_template: HashMap<ProblemTemplate, usize>,
    pub avg_difficulty: f64,
    pub difficulty_variance: f64,
}

impl SyntheticProblemGenerator {
    /// Create a new generator
    pub fn new(config: GeneratorConfig) -> Self {
        let mut template_weights = HashMap::new();
        let mut template_solve_rates = HashMap::new();

        for template in ProblemTemplate::all() {
            template_weights.insert(template, 1.0);
            template_solve_rates.insert(template, BetaConfidence::uniform_prior());
        }

        Self {
            config,
            counter: 0,
            rng_state: 42,
            template_weights,
            template_solve_rates,
            stats: GeneratorStats::default(),
        }
    }

    /// Create with default config
    pub fn with_default_config() -> Self {
        Self::new(GeneratorConfig::default())
    }

    /// Set the random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Generate next pseudo-random number
    fn next_random(&mut self) -> f64 {
        // Simple xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Generate a random integer in [0, max)
    fn next_int(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_random() * max as f64) as usize
    }

    /// Select a template based on weights and curriculum
    fn select_template(&mut self) -> ProblemTemplate {
        let templates = ProblemTemplate::all();
        let mut weights: Vec<f64> = templates
            .iter()
            .map(|t| {
                let base = self.template_weights.get(t).copied().unwrap_or(1.0);
                let solve_rate = self
                    .template_solve_rates
                    .get(t)
                    .map(|b| b.mean())
                    .unwrap_or(0.5);

                // Prioritize templates with low solve rate (harder to learn)
                // This implements ignorance-driven curriculum
                let ignorance_bonus = 1.0 - solve_rate;
                let variance_bonus = self
                    .template_solve_rates
                    .get(t)
                    .map(|b| b.variance().sqrt())
                    .unwrap_or(0.25);

                base * (1.0 + ignorance_bonus + variance_bonus)
            })
            .collect();

        // Normalize
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        // Sample
        let threshold = self.next_random();
        let mut cumsum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum >= threshold {
                return templates[i];
            }
        }

        templates[0]
    }

    /// Generate a problem from a specific template
    pub fn generate_from_template(&mut self, template: ProblemTemplate) -> SyntheticProblem {
        self.counter += 1;
        let id = format!(
            "synthetic_{}_{:?}_{}",
            self.counter, template, self.rng_state
        );
        let seed = self.rng_state;

        // Determine variation
        let variation = ProblemVariation {
            extra_points: self.next_int(self.config.max_extra_points + 1),
            extra_premises: if self.config.include_distractors {
                self.next_int(3)
            } else {
                0
            },
            randomize_labels: self.config.randomize_labels && self.next_random() > 0.5,
            complexity: 0.5 + self.next_random(),
        };

        // Generate based on template
        let (state, goal, num_premises) = match template {
            ProblemTemplate::MidpointTheorem => self.gen_midpoint_theorem(&variation),
            ProblemTemplate::IsocelesPerpendicular => self.gen_isoceles_perpendicular(&variation),
            ProblemTemplate::TriangleCongruence => self.gen_triangle_congruence(&variation),
            ProblemTemplate::CyclicQuadrilateral => self.gen_cyclic_quadrilateral(&variation),
            ProblemTemplate::ParallelTransversal => self.gen_parallel_transversal(&variation),
            ProblemTemplate::AngleBisector => self.gen_angle_bisector(&variation),
            ProblemTemplate::CircleCenter => self.gen_circle_center(&variation),
            ProblemTemplate::Orthocenter => self.gen_orthocenter(&variation),
            ProblemTemplate::Centroid => self.gen_centroid(&variation),
            ProblemTemplate::NinePointCircle => self.gen_nine_point_circle(&variation),
            ProblemTemplate::SimsonLine => self.gen_simson_line(&variation),
            ProblemTemplate::PowerOfPoint => self.gen_power_of_point(&variation),
            ProblemTemplate::RadicalAxis => self.gen_radical_axis(&variation),
            ProblemTemplate::Ceva => self.gen_ceva(&variation),
            ProblemTemplate::Menelaus => self.gen_menelaus(&variation),
        };

        // Calculate difficulty
        let base_diff = template.base_difficulty();
        let complexity_factor = variation.complexity;
        let premise_factor = (num_premises as f64 / 10.0).min(1.0);
        let distractor_factor = variation.extra_premises as f64 * 0.05;

        let difficulty = (base_diff * complexity_factor + premise_factor * 0.2 + distractor_factor)
            .clamp(self.config.min_difficulty, self.config.max_difficulty);

        // Update stats
        self.stats.problems_generated += 1;
        *self.stats.by_template.entry(template).or_insert(0) += 1;
        self.stats.avg_difficulty =
            (self.stats.avg_difficulty * (self.stats.problems_generated - 1) as f64 + difficulty)
                / self.stats.problems_generated as f64;

        SyntheticProblem {
            id,
            template,
            state,
            goal,
            difficulty,
            seed,
            num_premises,
            expected_constructions: template.expected_constructions(),
            historical_solve_rate: self
                .template_solve_rates
                .get(&template)
                .cloned()
                .unwrap_or_else(BetaConfidence::uniform_prior),
            variation,
        }
    }

    /// Generate a problem with automatic template selection
    pub fn generate(&mut self) -> SyntheticProblem {
        let template = self.select_template();
        self.generate_from_template(template)
    }

    /// Generate a problem targeting a specific difficulty
    pub fn generate_with_difficulty(&mut self, target_difficulty: f64) -> SyntheticProblem {
        // Select template based on target difficulty
        let templates = ProblemTemplate::all();
        let mut best_template = templates[0];
        let mut best_diff = f64::MAX;

        for template in &templates {
            let diff = (template.base_difficulty() - target_difficulty).abs();
            if diff < best_diff {
                best_diff = diff;
                best_template = *template;
            }
        }

        self.generate_from_template(best_template)
    }

    /// Update solve rate for a template (for curriculum learning)
    pub fn update_solve_rate(&mut self, template: ProblemTemplate, solved: bool) {
        let entry = self
            .template_solve_rates
            .entry(template)
            .or_insert_with(BetaConfidence::uniform_prior);

        if solved {
            *entry = BetaConfidence::new(entry.alpha + 1.0, entry.beta);
        } else {
            *entry = BetaConfidence::new(entry.alpha, entry.beta + 1.0);
        }
    }

    // =========================================================================
    // Template-specific generators
    // =========================================================================

    fn gen_midpoint_theorem(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        // Base triangle
        state.add_points(&["A", "B", "C"]);

        // Midpoints
        state.add_point("M");
        state.add_point("N");
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        // Add extra points if requested
        for i in 0..var.extra_points {
            let name = format!("P{}", i);
            state.add_point(&name);
        }

        // Add distractor premises
        for _ in 0..var.extra_premises {
            // Add some irrelevant collinearity
            if self.next_random() > 0.5 {
                state.add_axiom(Predicate::collinear("A", "B", "C"));
            }
        }

        let goal = Predicate::parallel("M", "N", "B", "C");
        let num_premises = 2 + var.extra_premises;

        (state, goal, num_premises)
    }

    fn gen_isoceles_perpendicular(
        &mut self,
        var: &ProblemVariation,
    ) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "M"]);
        state.add_axiom(Predicate::equal_length("A", "B", "A", "C"));
        state.add_axiom(Predicate::midpoint("M", "B", "C"));

        for _ in 0..var.extra_premises {
            if self.next_random() > 0.5 {
                state.add_axiom(Predicate::collinear("A", "M", "M"));
            }
        }

        let goal = Predicate::perpendicular("A", "M", "B", "C");
        (state, goal, 2 + var.extra_premises)
    }

    fn gen_triangle_congruence(
        &mut self,
        var: &ProblemVariation,
    ) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        // Two triangles
        state.add_points(&["A", "B", "C", "D", "E", "F"]);

        // SAS conditions
        state.add_axiom(Predicate::equal_length("A", "B", "D", "E"));
        state.add_axiom(Predicate::equal_length("A", "C", "D", "F"));
        state.add_axiom(Predicate::new(
            PredicateKind::EqualAngle,
            vec![
                "B".to_string(),
                "A".to_string(),
                "C".to_string(),
                "E".to_string(),
                "D".to_string(),
                "F".to_string(),
            ],
        ));

        let goal = Predicate::new(
            PredicateKind::Congruent,
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
                "F".to_string(),
            ],
        );

        (state, goal, 3 + var.extra_premises)
    }

    fn gen_cyclic_quadrilateral(
        &mut self,
        var: &ProblemVariation,
    ) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "D", "O"]);
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["A".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["B".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["C".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["D".to_string(), "O".to_string()],
        ));

        // Goal: opposite angles sum to 180 (represented as concyclic)
        let goal = Predicate::new(
            PredicateKind::Concyclic,
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
        );

        (state, goal, 4 + var.extra_premises)
    }

    fn gen_parallel_transversal(
        &mut self,
        var: &ProblemVariation,
    ) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "D", "P", "Q"]);

        // AB || CD
        state.add_axiom(Predicate::parallel("A", "B", "C", "D"));

        // Transversal through P and Q
        state.add_axiom(Predicate::collinear("A", "P", "C"));
        state.add_axiom(Predicate::collinear("B", "Q", "D"));

        // Goal: alternate interior angles equal
        let goal = Predicate::new(
            PredicateKind::EqualAngle,
            vec![
                "A".to_string(),
                "P".to_string(),
                "C".to_string(),
                "B".to_string(),
                "Q".to_string(),
                "D".to_string(),
            ],
        );

        (state, goal, 3 + var.extra_premises)
    }

    fn gen_angle_bisector(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "D"]);

        // D is on angle bisector of angle BAC
        state.add_axiom(Predicate::new(
            PredicateKind::AngleBisector,
            vec![
                "D".to_string(),
                "B".to_string(),
                "A".to_string(),
                "C".to_string(),
            ],
        ));

        // Goal: angle BAD = angle DAC
        let goal = Predicate::new(
            PredicateKind::EqualAngle,
            vec![
                "B".to_string(),
                "A".to_string(),
                "D".to_string(),
                "D".to_string(),
                "A".to_string(),
                "C".to_string(),
            ],
        );

        (state, goal, 1 + var.extra_premises)
    }

    fn gen_circle_center(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "O"]);

        // O is circumcenter
        state.add_axiom(Predicate::new(
            PredicateKind::Circumcenter,
            vec![
                "O".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ));

        // Goal: OA = OB = OC
        let goal = Predicate::equal_length("O", "A", "O", "B");

        (state, goal, 1 + var.extra_premises)
    }

    fn gen_orthocenter(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "H", "D", "E"]);

        // H is orthocenter
        state.add_axiom(Predicate::new(
            PredicateKind::Orthocenter,
            vec![
                "H".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ));

        // D is foot from A to BC
        state.add_axiom(Predicate::perpendicular("A", "D", "B", "C"));
        state.add_axiom(Predicate::collinear("B", "D", "C"));

        // Goal: A, H, D collinear
        let goal = Predicate::collinear("A", "H", "D");

        (state, goal, 3 + var.extra_premises)
    }

    fn gen_centroid(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "G", "M"]);

        // G is centroid
        state.add_axiom(Predicate::new(
            PredicateKind::Centroid,
            vec![
                "G".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ));

        // M is midpoint of BC
        state.add_axiom(Predicate::midpoint("M", "B", "C"));

        // Goal: A, G, M collinear (median)
        let goal = Predicate::collinear("A", "G", "M");

        (state, goal, 2 + var.extra_premises)
    }

    fn gen_nine_point_circle(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "H", "Ma", "Mb", "Mc", "Ha", "Hb", "Hc"]);

        // Orthocenter
        state.add_axiom(Predicate::new(
            PredicateKind::Orthocenter,
            vec![
                "H".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ));

        // Midpoints of sides
        state.add_axiom(Predicate::midpoint("Ma", "B", "C"));
        state.add_axiom(Predicate::midpoint("Mb", "C", "A"));
        state.add_axiom(Predicate::midpoint("Mc", "A", "B"));

        // Feet of altitudes
        state.add_axiom(Predicate::perpendicular("A", "Ha", "B", "C"));
        state.add_axiom(Predicate::collinear("B", "Ha", "C"));

        // Goal: These 6 points are concyclic
        let goal = Predicate::new(
            PredicateKind::Concyclic,
            vec![
                "Ma".to_string(),
                "Mb".to_string(),
                "Mc".to_string(),
                "Ha".to_string(),
            ],
        );

        (state, goal, 6 + var.extra_premises)
    }

    fn gen_simson_line(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "P", "D", "E", "F", "O"]);

        // P on circumcircle
        state.add_axiom(Predicate::new(
            PredicateKind::Circumcenter,
            vec![
                "O".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["P".to_string(), "O".to_string()],
        ));

        // Feet of perpendiculars
        state.add_axiom(Predicate::perpendicular("P", "D", "B", "C"));
        state.add_axiom(Predicate::perpendicular("P", "E", "C", "A"));
        state.add_axiom(Predicate::perpendicular("P", "F", "A", "B"));

        // Goal: D, E, F collinear
        let goal = Predicate::collinear("D", "E", "F");

        (state, goal, 5 + var.extra_premises)
    }

    fn gen_power_of_point(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["O", "P", "A", "B", "C", "D"]);

        // Circle with center O
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["A".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["B".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["C".to_string(), "O".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["D".to_string(), "O".to_string()],
        ));

        // Lines through P
        state.add_axiom(Predicate::collinear("P", "A", "B"));
        state.add_axiom(Predicate::collinear("P", "C", "D"));

        // Goal: PA * PB = PC * PD (represented as algebraic equality)
        let goal = Predicate::new(
            PredicateKind::AlgebraicEqual,
            vec![
                "P".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
        );

        (state, goal, 6 + var.extra_premises)
    }

    fn gen_radical_axis(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["O1", "O2", "P", "A", "B", "C", "D"]);

        // Two circles
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["A".to_string(), "O1".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["B".to_string(), "O1".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["C".to_string(), "O2".to_string()],
        ));
        state.add_axiom(Predicate::new(
            PredicateKind::OnCircle,
            vec!["D".to_string(), "O2".to_string()],
        ));

        // P on both
        state.add_axiom(Predicate::collinear("P", "A", "B"));
        state.add_axiom(Predicate::collinear("P", "C", "D"));

        // Goal: P is on radical axis
        let goal = Predicate::perpendicular("P", "P", "O1", "O2");

        (state, goal, 6 + var.extra_premises)
    }

    fn gen_ceva(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "D", "E", "F", "P"]);

        // Cevians through P
        state.add_axiom(Predicate::collinear("A", "P", "D"));
        state.add_axiom(Predicate::collinear("B", "P", "E"));
        state.add_axiom(Predicate::collinear("C", "P", "F"));

        // D on BC, E on CA, F on AB
        state.add_axiom(Predicate::collinear("B", "D", "C"));
        state.add_axiom(Predicate::collinear("C", "E", "A"));
        state.add_axiom(Predicate::collinear("A", "F", "B"));

        // Goal: (BD/DC) * (CE/EA) * (AF/FB) = 1 (cevians concurrent)
        let goal = Predicate::collinear("A", "P", "D");

        (state, goal, 6 + var.extra_premises)
    }

    fn gen_menelaus(&mut self, var: &ProblemVariation) -> (ProofState, Predicate, usize) {
        let mut state = ProofState::new();

        state.add_points(&["A", "B", "C", "D", "E", "F"]);

        // Points on sides (extended)
        state.add_axiom(Predicate::collinear("B", "D", "C"));
        state.add_axiom(Predicate::collinear("C", "E", "A"));
        state.add_axiom(Predicate::collinear("A", "F", "B"));

        // D, E, F collinear (transversal)
        state.add_axiom(Predicate::collinear("D", "E", "F"));

        // Goal: (BD/DC) * (CE/EA) * (AF/FB) = -1 (transversal collinearity preserved)
        let goal = Predicate::collinear("D", "E", "F");

        (state, goal, 4 + var.extra_premises)
    }
}

// =============================================================================
// Batch Generation
// =============================================================================

/// Generate a batch of synthetic problems
pub fn generate_batch(
    generator: &mut SyntheticProblemGenerator,
    count: usize,
) -> Vec<SyntheticProblem> {
    (0..count).map(|_| generator.generate()).collect()
}

/// Generate problems targeting specific difficulty range
pub fn generate_curriculum_batch(
    generator: &mut SyntheticProblemGenerator,
    count: usize,
    min_diff: f64,
    max_diff: f64,
) -> Vec<SyntheticProblem> {
    let mut problems = Vec::with_capacity(count);
    let step = (max_diff - min_diff) / count as f64;

    for i in 0..count {
        let target = min_diff + step * i as f64;
        problems.push(generator.generate_with_difficulty(target));
    }

    problems
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = SyntheticProblemGenerator::with_default_config();
        assert_eq!(generator.stats.problems_generated, 0);
    }

    #[test]
    fn test_generate_single() {
        let mut generator = SyntheticProblemGenerator::with_default_config();
        let problem = generator.generate();

        assert!(!problem.id.is_empty());
        assert!(problem.difficulty >= 0.0 && problem.difficulty <= 1.0);
        assert!(problem.state.points.len() >= 3);
    }

    #[test]
    fn test_generate_batch() {
        let mut generator = SyntheticProblemGenerator::with_default_config();
        let problems = generate_batch(&mut generator, 10);

        assert_eq!(problems.len(), 10);
        assert_eq!(generator.stats.problems_generated, 10);
    }

    #[test]
    fn test_difficulty_range() {
        let mut generator = SyntheticProblemGenerator::with_default_config();

        for _ in 0..20 {
            let problem = generator.generate();
            assert!(problem.difficulty >= 0.1);
            assert!(problem.difficulty <= 0.9);
        }
    }

    #[test]
    fn test_template_selection() {
        let mut generator = SyntheticProblemGenerator::with_default_config();

        // Generate many problems to check distribution
        for _ in 0..100 {
            generator.generate();
        }

        // Should have generated from multiple templates
        assert!(generator.stats.by_template.len() >= 5);
    }

    #[test]
    fn test_solve_rate_update() {
        let mut generator = SyntheticProblemGenerator::with_default_config();

        // Initially uniform
        let initial = generator
            .template_solve_rates
            .get(&ProblemTemplate::MidpointTheorem)
            .unwrap()
            .mean();
        assert!((initial - 0.5).abs() < 0.01);

        // Update with solves
        generator.update_solve_rate(ProblemTemplate::MidpointTheorem, true);
        generator.update_solve_rate(ProblemTemplate::MidpointTheorem, true);
        generator.update_solve_rate(ProblemTemplate::MidpointTheorem, false);

        let updated = generator
            .template_solve_rates
            .get(&ProblemTemplate::MidpointTheorem)
            .unwrap()
            .mean();
        assert!(updated > initial);
    }

    #[test]
    fn test_midpoint_theorem_generation() {
        let mut generator = SyntheticProblemGenerator::with_default_config();
        let problem = generator.generate_from_template(ProblemTemplate::MidpointTheorem);

        assert!(problem.state.points.contains_key("M"));
        assert!(problem.state.points.contains_key("N"));
    }

    #[test]
    fn test_reproducibility() {
        let mut gen1 = SyntheticProblemGenerator::with_default_config();
        gen1.set_seed(12345);

        let mut gen2 = SyntheticProblemGenerator::with_default_config();
        gen2.set_seed(12345);

        let p1 = gen1.generate();
        let p2 = gen2.generate();

        assert_eq!(p1.template, p2.template);
        assert_eq!(p1.difficulty, p2.difficulty);
    }
}
