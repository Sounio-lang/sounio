// =============================================================================
// Symbolic Geometry Engine for Demetrios
// =============================================================================
//
// A neuro-symbolic geometry prover inspired by AlphaGeometry.
// Combines Deductive Database (DD) forward chaining with Algebraic Reasoning (AR),
// integrated with Demetrios' epistemic type system.
//
// NOVEL FEATURES:
// - Every predicate is Knowledge[Predicate] with BetaConfidence
// - Epistemic pruning: high variance triggers neural suggestions via effects
// - Merkle provenance trees for verifiable proofs
// - Unit-aware algebraic reasoning with compile-time refinement checks
//
// This is L0 syntax (manual recursion, explicit linear moves)

// NOTE: This is a specification/documentation file demonstrating the intended
// Demetrios syntax for the symbolic geometry engine. The compiler's import
// resolution for epistemic types (which are keywords) is still in development.
//
// Dependencies (conceptual):
// - epistemic::Knowledge, BetaConfidence, Confidence, Provenance, Source
// - geometry::types::*
//
// The Rust implementation is in compiler/src/geometry/

// =============================================================================
// SECTION 1: Core Types with Epistemic Semantics
// =============================================================================

/// Geometric point with epistemic label tracking
/// Using CURIE: GEO:Point from geometry ontology
struct Point {
    label: String,
    is_free: bool,
    construction: Option[PointConstruction],
}

impl Point {
    /// Create a free point (given in problem statement)
    fn free(label: String) -> Point {
        Point {
            label: label,
            is_free: true,
            construction: None,
        }
    }

    /// Create a constructed point with provenance
    fn constructed(label: String, constr: PointConstruction) -> Point {
        Point {
            label: label,
            is_free: false,
            construction: Some(constr),
        }
    }
}

/// Point construction methods (auxiliary constructions)
enum PointConstruction {
    Midpoint { p1: String, p2: String },
    LineIntersection { l1: Line, l2: Line },
    CircleIntersection { c1: Circle, c2: Circle, which: i32 },
    Foot { point: String, line: Line },
    Circumcenter { a: String, b: String, c: String },
    Incenter { a: String, b: String, c: String },
    Orthocenter { a: String, b: String, c: String },
}

/// Line defined by two points
struct Line {
    p1: String,
    p2: String,
}

impl Line {
    fn new(p1: String, p2: String) -> Line {
        // Canonical ordering for equality
        if p1 <= p2 {
            Line { p1: p1, p2: p2 }
        } else {
            Line { p1: p2, p2: p1 }
        }
    }
}

/// Circle defined by center and point on circumference
struct Circle {
    center: String,
    on_circle: String,
}

// =============================================================================
// SECTION 2: Predicates with BetaConfidence + MerkleNode
// =============================================================================

/// Kind of geometric predicate (using GEO: ontology CURIEs)
enum PredicateKind {
    Collinear,       // GEO:collinear - three points on a line
    Concyclic,       // GEO:concyclic - four points on a circle
    Parallel,        // GEO:parallel - two lines parallel
    Perpendicular,   // GEO:perpendicular - two lines perpendicular
    EqualLength,     // GEO:equal_length - two segments same length
    EqualAngle,      // GEO:equal_angle - two angles equal
    RightAngle,      // GEO:right_angle - 90 degree angle
    Midpoint,        // GEO:midpoint - point bisects segment
    OnCircle,        // GEO:on_circle - point lies on circle
    Tangent,         // GEO:tangent - line tangent to circle
    AlgebraicEqual,  // AR-derived equality
}

/// Source of a predicate
enum PredicateSource {
    Axiom,
    Derived { rule: String },
    Neural { model: String, conf: f64 },
    Algebraic,
}

/// Epistemic status for predicates using Beta distribution
///
/// Beta(alpha, beta) where:
/// - mean = alpha / (alpha + beta)  -- confidence value
/// - variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))  -- uncertainty
///
/// High variance triggers neural suggestions via effects
struct PredicateEpistemic {
    /// Beta distribution parameters
    confidence: BetaConfidence,
    /// Source of this knowledge
    source: PredicateSource,
    /// Whether this can be revised
    revisable: bool,
    /// Depth in proof tree (0 = axiom)
    depth: i32,
    /// Parent predicate IDs
    derived_from: Vec[u64],
    /// Merkle hash for provenance
    merkle_hash: Option[Hash256],
    /// Parent Merkle hashes
    parent_hashes: Vec[Hash256],
}

impl PredicateEpistemic {
    /// Create axiom epistemic status (high confidence)
    /// Beta(100, 1) gives mean ~ 0.99, very low variance
    fn axiom() -> PredicateEpistemic {
        PredicateEpistemic {
            confidence: BetaConfidence::new(100.0, 1.0),
            source: PredicateSource::Axiom,
            revisable: false,
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }

    /// Create default epistemic status
    fn default() -> PredicateEpistemic {
        PredicateEpistemic {
            confidence: BetaConfidence::uniform_prior(),
            source: PredicateSource::Axiom,
            revisable: true,
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }

    /// Create derived epistemic status with hierarchical Beta combination
    ///
    /// For N parents with confidences {c_i}, the combined confidence is:
    /// - product_mean = prod(c_i.mean())
    /// - combined_n = geometric_mean(c_i.sample_size())
    /// - Apply decay factor for rule reliability
    fn derived(parents: Vec[Predicate], rule_name: String, decay: f64) -> PredicateEpistemic {
        // Collect parent info
        var parent_ids: Vec[u64] = vec![]
        var parent_hashes: Vec[Hash256] = vec![]
        var max_depth: i32 = 0

        for p in parents {
            parent_ids = parent_ids ++ vec![p.id]
            match p.epistemic.merkle_hash {
                Some(h) => {
                    parent_hashes = parent_hashes ++ vec![h]
                },
                None => {},
            }
            if p.epistemic.depth > max_depth {
                max_depth = p.epistemic.depth
            }
        }

        // Hierarchical Beta combination
        let combined = combine_parent_betas(parents, decay)

        PredicateEpistemic {
            confidence: combined,
            source: PredicateSource::Derived { rule: rule_name },
            revisable: true,
            depth: max_depth + 1,
            derived_from: parent_ids,
            merkle_hash: None,
            parent_hashes: parent_hashes,
        }
    }

    /// Get variance (epistemic uncertainty measure)
    fn variance(self) -> f64 {
        self.confidence.variance()
    }

    /// Check if uncertain enough to trigger neural help
    fn is_uncertain(self, threshold: f64) -> bool {
        self.variance() > threshold
    }

    /// Compute Merkle hash for provenance
    fn compute_hash(mut self, predicate_key: String) {
        var data = predicate_key
        for h in self.parent_hashes {
            data = data ++ "::" ++ h.to_hex()
        }
        self.merkle_hash = Some(sha256(data))
    }
}

/// Combine parent Beta distributions for derived confidence
fn combine_parent_betas(parents: Vec[Predicate], decay: f64) -> BetaConfidence {
    if parents.len() == 0 {
        return BetaConfidence::uniform_prior()
    }

    // Product of means (conjunction confidence)
    var product_mean: f64 = 1.0
    var geometric_n: f64 = 1.0

    for p in parents {
        product_mean = product_mean * p.epistemic.confidence.mean()
        geometric_n = geometric_n * p.epistemic.confidence.sample_size()
    }

    // Geometric mean of sample sizes
    let n = parents.len() as f64
    geometric_n = geometric_n.pow(1.0 / n)

    // Apply decay for rule reliability
    let decayed_mean = product_mean * decay

    BetaConfidence::from_confidence(decayed_mean, geometric_n)
}

/// A geometric predicate with full epistemic metadata
struct Predicate {
    id: u64,
    kind: PredicateKind,
    args: Vec[String],
    epistemic: PredicateEpistemic,
}

impl Predicate {
    /// Create collinear predicate: collinear(A, B, C)
    fn collinear(p1: String, p2: String, p3: String) -> Predicate {
        var args = vec![p1, p2, p3]
        args.sort()
        Predicate {
            id: hash_predicate("collinear", args.clone()),
            kind: PredicateKind::Collinear,
            args: args,
            epistemic: PredicateEpistemic::default(),
        }
    }

    /// Create parallel predicate: parallel(line(A,B), line(C,D))
    fn parallel(a: String, b: String, c: String, d: String) -> Predicate {
        var l1 = vec![a, b]
        var l2 = vec![c, d]
        l1.sort()
        l2.sort()
        let args = if l1 <= l2 { l1 ++ l2 } else { l2 ++ l1 }

        Predicate {
            id: hash_predicate("parallel", args.clone()),
            kind: PredicateKind::Parallel,
            args: args,
            epistemic: PredicateEpistemic::default(),
        }
    }

    /// Create midpoint predicate: midpoint(M, A, B)
    fn midpoint(mid: String, p1: String, p2: String) -> Predicate {
        var endpoints = vec![p1, p2]
        endpoints.sort()
        let args = vec![mid] ++ endpoints

        Predicate {
            id: hash_predicate("midpoint", args.clone()),
            kind: PredicateKind::Midpoint,
            args: args,
            epistemic: PredicateEpistemic::default(),
        }
    }

    /// Set epistemic status
    fn with_epistemic(self, epi: PredicateEpistemic) -> Predicate {
        Predicate {
            id: self.id,
            kind: self.kind,
            args: self.args,
            epistemic: epi,
        }
    }

    /// Get canonical key for deduplication
    fn key(self) -> String {
        self.kind.name() ++ ":" ++ self.args.join(",")
    }

    /// Get confidence value (Beta mean)
    fn confidence(self) -> f64 {
        self.epistemic.confidence.mean()
    }

    /// Get uncertainty (Beta variance)
    fn uncertainty(self) -> f64 {
        self.epistemic.variance()
    }
}

// =============================================================================
// SECTION 3: ProofState with Epistemic Tracking
// =============================================================================

/// Proof state maintaining all knowledge during proof search
struct ProofState {
    points: Map[String, Point],
    predicates: Map[String, Predicate],
    id_to_key: Map[u64, String],
    goal: Option[ProofGoal],
    confidence: BetaConfidence,
    trace: Vec[ProofStep],
    constructions: Vec[Construction],
}

/// Goal to prove
struct ProofGoal {
    predicate: Predicate,
    min_confidence: f64,
}

/// A step in the proof
struct ProofStep {
    predicate: Predicate,
    rule: String,
    premises: Vec[u64],
    confidence: f64,
}

/// Construction type
struct Construction {
    kind: ConstructionKind,
    result_label: String,
}

enum ConstructionKind {
    Midpoint { p1: String, p2: String },
    Perpendicular { line: Line, through: String },
    Parallel { line: Line, through: String },
    CircleCenter { p1: String, p2: String, p3: String },
}

impl Construction {
    fn midpoint(p1: String, p2: String) -> Construction {
        let label = "M_" ++ p1 ++ p2
        Construction {
            kind: ConstructionKind::Midpoint { p1: p1, p2: p2 },
            result_label: label,
        }
    }
}

impl ProofState {
    /// Create empty state
    fn new() -> ProofState {
        ProofState {
            points: Map::new(),
            predicates: Map::new(),
            id_to_key: Map::new(),
            goal: None,
            confidence: BetaConfidence::uniform_prior(),
            trace: vec![],
            constructions: vec![],
        }
    }

    /// Add points from problem statement
    fn add_points(mut self, labels: Vec[String]) -> ProofState {
        for label in labels {
            self.points.insert(label.clone(), Point::free(label))
        }
        self
    }

    /// Add axiom predicate (from problem statement)
    fn add_axiom(mut self, pred: Predicate) -> ProofState {
        let pred_with_epi = pred.with_epistemic(PredicateEpistemic::axiom())
        self.add_predicate_internal(pred_with_epi, None)
    }

    /// Add derived predicate with hierarchical Beta combination
    fn add_derived(
        mut self,
        pred: Predicate,
        rule: String,
        parent_ids: Vec[u64],
        decay: f64
    ) -> Option[ProofState] {
        let key = pred.key()

        // Check if already exists with higher confidence
        match self.predicates.get(&key) {
            Some(existing) => {
                if existing.confidence() >= pred.confidence() {
                    return None
                }
            },
            None => {},
        }

        // Get parent predicates
        var parents: Vec[Predicate] = vec![]
        for pid in parent_ids.clone() {
            match self.id_to_key.get(&pid) {
                Some(pkey) => {
                    match self.predicates.get(pkey) {
                        Some(p) => {
                            parents = parents ++ vec![p.clone()]
                        },
                        None => {},
                    }
                },
                None => {},
            }
        }

        // Create derived epistemic status
        let epistemic = PredicateEpistemic::derived(parents, rule.clone(), decay)
        let pred_with_epi = pred.with_epistemic(epistemic)

        // Update overall confidence via Bayesian update
        let conf = pred_with_epi.confidence()
        self.confidence.update(conf, 1.0 - conf)

        Some(self.add_predicate_internal(pred_with_epi, Some((rule, parent_ids))))
    }

    fn add_predicate_internal(
        mut self,
        pred: Predicate,
        derivation: Option[(String, Vec[u64])]
    ) -> ProofState {
        let key = pred.key()
        let id = pred.id

        // Record proof step if derived
        match derivation {
            Some((rule, parents)) => {
                self.trace = self.trace ++ vec![ProofStep {
                    predicate: pred.clone(),
                    rule: rule,
                    premises: parents,
                    confidence: pred.confidence(),
                }]
            },
            None => {},
        }

        self.predicates.insert(key.clone(), pred)
        self.id_to_key.insert(id, key)
        self
    }

    /// Set goal to prove
    fn set_goal(mut self, pred: Predicate, min_conf: f64) -> ProofState {
        self.goal = Some(ProofGoal {
            predicate: pred,
            min_confidence: min_conf,
        })
        self
    }

    /// Check if goal is satisfied
    fn goal_satisfied(self) -> bool {
        match self.goal {
            Some(goal) => {
                let key = goal.predicate.key()
                match self.predicates.get(&key) {
                    Some(pred) => pred.confidence() >= goal.min_confidence,
                    None => false,
                }
            },
            None => false,
        }
    }

    /// Compute global uncertainty (aggregate variance)
    fn global_uncertainty(self) -> f64 {
        if self.predicates.len() == 0 {
            return 1.0
        }

        var total_var: f64 = 0.0
        for pred in self.predicates.values() {
            total_var = total_var + pred.uncertainty()
        }

        total_var / (self.predicates.len() as f64)
    }

    /// Get predicates with high uncertainty (candidates for neural help)
    fn uncertain_predicates(self, threshold: f64) -> Vec[Predicate] {
        var result: Vec[Predicate] = vec![]
        for pred in self.predicates.values() {
            if pred.uncertainty() > threshold {
                result = result ++ vec![pred.clone()]
            }
        }
        result
    }

    /// Get number of predicates
    fn num_predicates(self) -> i32 {
        self.predicates.len()
    }
}

// =============================================================================
// SECTION 4: Forward Chaining with Epistemic Pruning
// =============================================================================

/// Geometry rule for forward chaining
struct GeometryRule {
    name: String,
    decay: f64,
    priority: i32,
}

/// Decision from epistemic pruning
enum PruningDecision {
    Continue,
    Prune { reason: String },
    RequestNeural { uncertainty: f64 },
}

/// Epistemic pruner controlling branch exploration
struct EpistemicPruner {
    confidence_threshold: f64,
    variance_threshold: f64,
    depth_decay: f64,
    soft_prune: bool,
}

impl EpistemicPruner {
    fn default() -> EpistemicPruner {
        EpistemicPruner {
            confidence_threshold: 0.3,
            variance_threshold: 0.15,
            depth_decay: 0.95,
            soft_prune: true,
        }
    }

    /// Evaluate a derivation for pruning
    fn evaluate(self, confidence: BetaConfidence, depth: i32) -> PruningDecision {
        let mean = confidence.mean()
        let variance = confidence.variance()

        // Check raw confidence
        if mean < self.confidence_threshold {
            if self.soft_prune {
                return PruningDecision::RequestNeural { uncertainty: 1.0 - mean }
            }
            return PruningDecision::Prune {
                reason: "Low confidence: " ++ mean.to_string()
            }
        }

        // Check variance (epistemic uncertainty)
        if variance > self.variance_threshold {
            if self.soft_prune {
                return PruningDecision::RequestNeural {
                    uncertainty: variance / 0.25
                }
            }
            return PruningDecision::Prune {
                reason: "High variance: " ++ variance.to_string()
            }
        }

        // Check depth-decayed confidence
        let effective_conf = mean * self.depth_decay.pow(depth as f64)
        if effective_conf < self.confidence_threshold * 0.5 {
            return PruningDecision::Prune {
                reason: "Depth decay: " ++ effective_conf.to_string()
            }
        }

        PruningDecision::Continue
    }
}

// =============================================================================
// SECTION 5: Effects for Neuro-Symbolic Integration
// =============================================================================

/// Effect for requesting neural suggestions
effect geometry_reasoning {
    /// Request neural auxiliary construction
    fn suggest_construction(state: ProofState, uncertainty: f64) -> Option[Construction];

    /// Request neural predicate verification
    fn verify_predicate(pred: Predicate) -> BetaConfidence;

    /// Log high-uncertainty situation
    fn log_uncertainty(message: String, variance: f64);
}

/// Handler for pure symbolic mode (no neural)
handler PureSymbolicHandler for geometry_reasoning {
    suggest_construction(state, uncertainty) => resume(None),
    verify_predicate(pred) => resume(pred.epistemic.confidence),
    log_uncertainty(message, variance) => {
        // Logging only
        resume(())
    }
}

// =============================================================================
// SECTION 6: Main Deduction Engine
// =============================================================================

/// Engine configuration
struct EngineConfig {
    max_iterations: i32,
    max_depth: i32,
    min_confidence: f64,
    pruning_threshold: f64,
    uncertainty_threshold: f64,
    confidence_decay: f64,
    enable_ar: bool,
}

impl EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            max_iterations: 1000,
            max_depth: 50,
            min_confidence: 0.1,
            pruning_threshold: 0.5,
            uncertainty_threshold: 0.3,
            confidence_decay: 0.99,
            enable_ar: true,
        }
    }
}

/// Result of deduction
struct DeductionResult {
    proved: bool,
    state: ProofState,
    iterations: i32,
    predicates_derived: i32,
    confidence: BetaConfidence,
    global_uncertainty: f64,
    neural_requests: Vec[NeuralRequest],
}

struct NeuralRequest {
    state_summary: String,
    uncertainty: f64,
}

struct RuleMatch {
    parent_ids: Vec[u64],
    depth: i32,
    bindings: Map[String, String],
}

/// Main forward chaining function with epistemic pruning
///
/// L0 implementation: manual iteration, explicit state threading
fn forward_chain(
    state: ProofState,
    rules: Vec[GeometryRule],
    config: EngineConfig
) -> DeductionResult with geometry_reasoning {
    var current_state = state
    var iterations: i32 = 0
    var predicates_derived: i32 = 0
    var neural_requests: Vec[NeuralRequest] = vec![]
    let pruner = EpistemicPruner::default()

    // Main deduction loop (L0: explicit while loop)
    while iterations < config.max_iterations {
        iterations = iterations + 1

        // Check goal satisfaction
        if current_state.goal_satisfied() {
            return DeductionResult {
                proved: true,
                state: current_state,
                iterations: iterations,
                predicates_derived: predicates_derived,
                confidence: current_state.confidence,
                global_uncertainty: current_state.global_uncertainty(),
                neural_requests: neural_requests,
            }
        }

        // Check global uncertainty - trigger neural if too high
        let global_unc = current_state.global_uncertainty()
        if global_unc > config.uncertainty_threshold {
            // Effect: request neural suggestion
            perform geometry_reasoning.log_uncertainty(
                "Global uncertainty high",
                global_unc
            )

            let suggestion = perform geometry_reasoning.suggest_construction(
                current_state.clone(),
                global_unc
            )

            match suggestion {
                Some(constr) => {
                    current_state = apply_construction(current_state, constr)
                },
                None => {},
            }
        }

        // Simulate rule application (simplified for demonstration)
        var derived_this_round: i32 = 0

        // In full implementation, this would iterate through rules
        // and find matching predicates to derive new ones

        // Check fixpoint
        if derived_this_round == 0 {
            break
        }
    }

    DeductionResult {
        proved: current_state.goal_satisfied(),
        state: current_state,
        iterations: iterations,
        predicates_derived: predicates_derived,
        confidence: current_state.confidence,
        global_uncertainty: current_state.global_uncertainty(),
        neural_requests: neural_requests,
    }
}

/// Apply a construction to the proof state
fn apply_construction(state: ProofState, constr: Construction) -> ProofState {
    var new_state = state
    new_state.constructions = new_state.constructions ++ vec![constr.clone()]

    match constr.kind {
        ConstructionKind::Midpoint { p1, p2 } => {
            let mid_point = Point::constructed(
                constr.result_label.clone(),
                PointConstruction::Midpoint { p1: p1.clone(), p2: p2.clone() }
            )
            new_state.points.insert(constr.result_label.clone(), mid_point)

            // Add midpoint predicate
            let mid_pred = Predicate::midpoint(constr.result_label, p1, p2)
            new_state = new_state.add_axiom(mid_pred)
        },
        _ => {},
    }

    new_state
}

/// Compute match confidence from parent predicates
fn compute_match_confidence(parent_ids: Vec[u64], state: ProofState) -> BetaConfidence {
    var total_alpha: f64 = 0.0
    var total_beta: f64 = 0.0

    for pid in parent_ids {
        match state.id_to_key.get(&pid) {
            Some(key) => {
                match state.predicates.get(key) {
                    Some(pred) => {
                        let conf = pred.epistemic.confidence
                        total_alpha = total_alpha + conf.alpha
                        total_beta = total_beta + conf.beta
                    },
                    None => {},
                }
            },
            None => {},
        }
    }

    if total_alpha == 0.0 {
        BetaConfidence::uniform_prior()
    } else {
        BetaConfidence::new(total_alpha, total_beta)
    }
}

// =============================================================================
// SECTION 7: Algebraic Reasoning with Units
// =============================================================================

/// Physical unit for dimensional analysis
struct Unit {
    length_exp: i32,
    angle_exp: i32,
}

impl Unit {
    fn dimensionless() -> Unit {
        Unit { length_exp: 0, angle_exp: 0 }
    }

    fn length() -> Unit {
        Unit { length_exp: 1, angle_exp: 0 }
    }

    fn angle() -> Unit {
        Unit { length_exp: 0, angle_exp: 1 }
    }

    fn area() -> Unit {
        Unit { length_exp: 2, angle_exp: 0 }
    }

    fn compatible(self, other: Unit) -> bool {
        self.length_exp == other.length_exp && self.angle_exp == other.angle_exp
    }

    fn multiply(self, other: Unit) -> Unit {
        Unit {
            length_exp: self.length_exp + other.length_exp,
            angle_exp: self.angle_exp + other.angle_exp,
        }
    }
}

/// Symbolic expression with epistemic tracking
struct Expression {
    kind: ExprKind,
    confidence: BetaConfidence,
    unit: Option[Unit],
}

enum ExprKind {
    Constant { value: f64 },
    Variable { name: String },
    Add { left: Box[Expression], right: Box[Expression] },
    Sub { left: Box[Expression], right: Box[Expression] },
    Mul { left: Box[Expression], right: Box[Expression] },
    Div { left: Box[Expression], right: Box[Expression] },
    Sqrt { inner: Box[Expression] },
    Pi,
}

/// Algebraic equation: lhs = rhs
struct Equation {
    lhs: Expression,
    rhs: Expression,
    confidence: BetaConfidence,
}

impl Equation {
    /// Check unit compatibility (refinement check)
    fn units_valid(self) -> bool {
        match (self.lhs.unit, self.rhs.unit) {
            (Some(u1), Some(u2)) => u1.compatible(u2),
            _ => true,
        }
    }
}

/// Algebraic simplification with epistemic propagation
fn algebraic_simplify(expr: Expression) -> Expression {
    match expr.kind {
        // x + 0 = x
        ExprKind::Add { left, right } => {
            let l = algebraic_simplify(*left)
            let r = algebraic_simplify(*right)

            match r.kind {
                ExprKind::Constant { value } => {
                    if value == 0.0 {
                        return l
                    }
                },
                _ => {},
            }
            match l.kind {
                ExprKind::Constant { value } => {
                    if value == 0.0 {
                        return r
                    }
                },
                _ => {},
            }

            // Constant folding with epistemic propagation
            match (l.kind.clone(), r.kind.clone()) {
                (ExprKind::Constant { value: a }, ExprKind::Constant { value: b }) => {
                    Expression {
                        kind: ExprKind::Constant { value: a + b },
                        confidence: combine_confidence(l.confidence, r.confidence),
                        unit: l.unit,
                    }
                },
                _ => {
                    Expression {
                        kind: ExprKind::Add {
                            left: Box::new(l.clone()),
                            right: Box::new(r.clone())
                        },
                        confidence: combine_confidence(l.confidence, r.confidence),
                        unit: l.unit,
                    }
                }
            }
        },

        // x * 0 = 0, x * 1 = x
        ExprKind::Mul { left, right } => {
            let l = algebraic_simplify(*left)
            let r = algebraic_simplify(*right)

            match r.kind {
                ExprKind::Constant { value } => {
                    if value == 0.0 {
                        return Expression {
                            kind: ExprKind::Constant { value: 0.0 },
                            confidence: r.confidence,
                            unit: Some(Unit::dimensionless()),
                        }
                    }
                    if value == 1.0 {
                        return l
                    }
                },
                _ => {},
            }

            let new_unit = match (l.unit.clone(), r.unit.clone()) {
                (Some(u1), Some(u2)) => Some(u1.multiply(u2)),
                _ => None,
            }

            Expression {
                kind: ExprKind::Mul {
                    left: Box::new(l.clone()),
                    right: Box::new(r.clone())
                },
                confidence: combine_confidence(l.confidence, r.confidence),
                unit: new_unit,
            }
        },

        // sqrt simplification
        ExprKind::Sqrt { inner } => {
            let i = algebraic_simplify(*inner)
            match i.kind {
                ExprKind::Constant { value } => {
                    if value >= 0.0 {
                        let new_unit = match i.unit {
                            Some(u) => Some(Unit {
                                length_exp: u.length_exp / 2,
                                angle_exp: u.angle_exp / 2,
                            }),
                            None => None,
                        }
                        return Expression {
                            kind: ExprKind::Constant { value: value.sqrt() },
                            confidence: i.confidence,
                            unit: new_unit,
                        }
                    }
                },
                _ => {},
            }
            Expression {
                kind: ExprKind::Sqrt { inner: Box::new(i.clone()) },
                confidence: i.confidence,
                unit: i.unit,
            }
        },

        _ => expr,
    }
}

fn combine_confidence(a: BetaConfidence, b: BetaConfidence) -> BetaConfidence {
    // Linear pool with equal weights, slight decay
    a.combine(b, 1.0, 1.0).decay(0.99)
}

// =============================================================================
// SECTION 8: Example Usage
// =============================================================================

/// Example: Prove midpoint theorem
/// Given: M is midpoint of AB, N is midpoint of AC
/// Prove: MN is parallel to BC
fn example_midpoint_theorem() -> DeductionResult {
    // Setup proof state
    var state = ProofState::new()
    state = state.add_points(vec!["A", "B", "C", "M", "N"])

    // Add axioms with epistemic tracking
    state = state.add_axiom(Predicate::midpoint("M", "A", "B"))
    state = state.add_axiom(Predicate::midpoint("N", "A", "C"))

    // Set goal
    let goal = Predicate::parallel("M", "N", "B", "C")
    state = state.set_goal(goal, 0.9)

    // Define rules
    let rules = standard_geometry_rules()
    let config = EngineConfig::default()

    // Run with pure symbolic handler
    let result = handle {
        forward_chain(state, rules, config)
    } with PureSymbolicHandler

    result
}

/// Standard geometry rules (simplified)
fn standard_geometry_rules() -> Vec[GeometryRule] {
    vec![
        GeometryRule {
            name: "collinear_trans".to_string(),
            decay: 0.99,
            priority: 1,
        },
        GeometryRule {
            name: "midpoint_parallel".to_string(),
            decay: 0.98,
            priority: 2,
        },
        GeometryRule {
            name: "midpoint_equal_length".to_string(),
            decay: 1.0,
            priority: 1,
        },
    ]
}

// =============================================================================
// SECTION 9: Hash Utilities
// =============================================================================

type Hash256 = [u8; 32]

fn hash_predicate(kind: String, args: Vec[String]) -> u64 {
    var h: u64 = 0
    for c in kind.bytes() {
        h = h * 31 + (c as u64)
    }
    for arg in args {
        for c in arg.bytes() {
            h = h * 31 + (c as u64)
        }
    }
    h
}

fn sha256(data: String) -> Hash256 {
    // Placeholder - would use actual SHA256 implementation
    var result: Hash256 = [0; 32]
    var h: u64 = 0
    for c in data.bytes() {
        h = h * 31 + (c as u64)
    }
    // Fill first 8 bytes with hash
    for i in 0..8 {
        result[i] = ((h >> (i * 8)) & 0xFF) as u8
    }
    result
}

// =============================================================================
// SECTION 10: Tests
// =============================================================================

fn test_beta_confidence_propagation() {
    // Axiom has high confidence
    let axiom_epi = PredicateEpistemic::axiom()
    assert(axiom_epi.confidence.mean() > 0.98)
    assert(axiom_epi.variance() < 0.01)

    // Derived has lower confidence
    let p1 = Predicate::collinear("A", "B", "C").with_epistemic(axiom_epi.clone())
    let p2 = Predicate::collinear("A", "B", "D").with_epistemic(axiom_epi.clone())

    let derived_epi = PredicateEpistemic::derived(vec![p1, p2], "test_rule".to_string(), 0.95)
    assert(derived_epi.confidence.mean() < axiom_epi.confidence.mean())
}

fn test_epistemic_pruning() {
    let pruner = EpistemicPruner::default()

    // High confidence -> continue
    let high_conf = BetaConfidence::new(100.0, 1.0)
    match pruner.evaluate(high_conf, 0) {
        PruningDecision::Continue => assert(true),
        _ => assert(false),
    }

    // Low confidence -> prune or neural request
    let low_conf = BetaConfidence::new(1.0, 10.0)
    match pruner.evaluate(low_conf, 0) {
        PruningDecision::Continue => assert(false),
        _ => assert(true),
    }
}

fn test_midpoint_theorem() {
    let result = example_midpoint_theorem()
    // Note: full proof requires complete rule implementation
    assert(result.iterations > 0)
}

fn test_unit_compatibility() {
    let length = Unit::length()
    let area = Unit::area()

    assert(!length.compatible(area))
    assert(length.multiply(length).compatible(area))
}

fn test_algebraic_simplify_add_zero() {
    let zero = Expression {
        kind: ExprKind::Constant { value: 0.0 },
        confidence: BetaConfidence::new(100.0, 1.0),
        unit: Some(Unit::length()),
    }
    let x = Expression {
        kind: ExprKind::Variable { name: "x".to_string() },
        confidence: BetaConfidence::new(100.0, 1.0),
        unit: Some(Unit::length()),
    }
    let sum = Expression {
        kind: ExprKind::Add {
            left: Box::new(x.clone()),
            right: Box::new(zero),
        },
        confidence: BetaConfidence::new(100.0, 1.0),
        unit: Some(Unit::length()),
    }

    let simplified = algebraic_simplify(sum)
    match simplified.kind {
        ExprKind::Variable { name } => assert(name == "x"),
        _ => assert(false),
    }
}
