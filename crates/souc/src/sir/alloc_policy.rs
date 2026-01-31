//! A/B Policy Comparison Infrastructure for Register Allocation
//!
//! This module provides instrumentation for comparing register allocation
//! policies: Classic (structural heuristics) vs Attention (epistemic-aware).
//!
//! Usage: souc build --alloc-policy=attention --emit-metrics=metrics.json

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

// =============================================================================
// Policy Enum
// =============================================================================

/// Register allocation policy selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AllocPolicy {
    /// Classic linear scan: spill by furthest next use.
    Classic,

    /// Attention-based: spill by lowest epistemic score.
    #[default]
    Attention,

    /// Attention with BO-calibrated weights.
    AttentionCalibrated,
}

impl std::str::FromStr for AllocPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "classic" | "c" => Ok(AllocPolicy::Classic),
            "attention" | "a" => Ok(AllocPolicy::Attention),
            "calibrated" | "bo" => Ok(AllocPolicy::AttentionCalibrated),
            _ => Err(format!(
                "Unknown policy: '{}'. Use: classic, attention, calibrated",
                s
            )),
        }
    }
}

impl std::fmt::Display for AllocPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocPolicy::Classic => write!(f, "classic"),
            AllocPolicy::Attention => write!(f, "attention"),
            AllocPolicy::AttentionCalibrated => write!(f, "calibrated"),
        }
    }
}

// =============================================================================
// Attention Configuration
// =============================================================================

/// Weights for attention-based allocation scoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionConfig {
    // Classical components
    pub w_use_density: f64,
    pub w_crosses_call: f64,
    pub w_next_use_distance: f64,

    // Epistemic components
    pub w_confidence: f64,
    pub w_uncertainty: f64,
    pub w_provenance: f64,

    // Classification thresholds
    pub high_confidence_threshold: f64,
    pub low_confidence_threshold: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            // Classical (baseline)
            w_use_density: 1.0,
            w_crosses_call: 0.5,
            w_next_use_distance: 0.3,

            // Epistemic (initial, pre-BO)
            w_confidence: 0.5,
            w_uncertainty: 0.3,
            w_provenance: 0.2,

            // Thresholds
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
        }
    }
}

impl AttentionConfig {
    /// BO-calibrated configuration (optimized 2025-12-28).
    ///
    /// These weights were found via Bayesian Optimization on simulated
    /// epistemic workloads. Key insight: high w_confidence (0.94) confirms
    /// the strategy of prioritizing high-confidence values for register retention.
    pub fn calibrated() -> Self {
        Self {
            w_use_density: 1.7088,
            w_crosses_call: 0.4527,
            w_next_use_distance: 0.8802,
            w_confidence: 0.9374,
            w_uncertainty: 0.2411,
            w_provenance: 0.2013,
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
        }
    }

    /// Scientific configuration prioritizing epistemic properties.
    ///
    /// For scientific computing where data provenance and confidence
    /// are critical for reproducibility and correctness.
    pub fn scientific() -> Self {
        Self {
            w_use_density: 0.8,
            w_crosses_call: 0.4,
            w_next_use_distance: 0.5,
            w_confidence: 1.2, // High priority for confidence
            w_uncertainty: 0.5,
            w_provenance: 0.6, // Strong provenance preservation
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
        }
    }

    /// Performance configuration prioritizing classical heuristics.
    ///
    /// For performance-critical code where raw speed matters more
    /// than epistemic properties.
    pub fn performance() -> Self {
        Self {
            w_use_density: 2.0, // Strong preference for use density
            w_crosses_call: 0.8,
            w_next_use_distance: 1.0,
            w_confidence: 0.3, // Lower epistemic priority
            w_uncertainty: 0.2,
            w_provenance: 0.1,
            high_confidence_threshold: 0.7,
            low_confidence_threshold: 0.3,
        }
    }

    /// Load from JSON file.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: AttentionConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save to JSON file.
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

// =============================================================================
// Confidence Classification
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceClass {
    High,
    Medium,
    Low,
}

// =============================================================================
// Epistemic Metadata for Live Intervals
// =============================================================================

/// Epistemic metadata attached to live intervals.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpistemicMetadata {
    /// Confidence/trust level τ ∈ [0, 1]
    pub confidence: Option<f64>,

    /// Uncertainty level σ ∈ [0, 1]
    pub uncertainty: Option<f64>,

    /// Has provenance tracking
    pub has_provenance: bool,

    /// Provenance chain depth
    pub provenance_depth: u32,

    /// Source operation (for debugging)
    pub source_op: Option<String>,
}

impl EpistemicMetadata {
    /// Classify confidence level.
    pub fn confidence_class(&self, config: &AttentionConfig) -> ConfidenceClass {
        let tau = self.confidence.unwrap_or(0.5);
        if tau >= config.high_confidence_threshold {
            ConfidenceClass::High
        } else if tau <= config.low_confidence_threshold {
            ConfidenceClass::Low
        } else {
            ConfidenceClass::Medium
        }
    }
}

// =============================================================================
// Attention Score Computation
// =============================================================================

/// Compute attention score for a live interval.
pub fn compute_attention_score(
    use_density: f64,
    crosses_call: bool,
    next_use_distance: f64,
    epistemic: &EpistemicMetadata,
    config: &AttentionConfig,
) -> f64 {
    // Classical components
    let crosses_call_val = if crosses_call { 1.0 } else { 0.0 };
    let next_use = 1.0 / (1.0 + next_use_distance * 0.01); // Normalize

    let classic = config.w_use_density * use_density
        + config.w_crosses_call * crosses_call_val
        + config.w_next_use_distance * next_use;

    // Epistemic components
    let tau = epistemic.confidence.unwrap_or(0.5);
    let sigma = epistemic.uncertainty.unwrap_or(0.5);
    let rho = if epistemic.has_provenance { 1.0 } else { 0.0 };

    let epistemic_score =
        config.w_confidence * tau - config.w_uncertainty * sigma + config.w_provenance * rho;

    classic + epistemic_score
}

// =============================================================================
// Allocation Metrics
// =============================================================================

/// Detailed metrics for allocation quality analysis.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AllocationMetrics {
    // === Spill statistics ===
    pub total_spills: usize,
    pub total_reloads: usize,

    // === Spills by confidence class ===
    pub high_confidence_spills: usize,
    pub medium_confidence_spills: usize,
    pub low_confidence_spills: usize,

    // === Provenance tracking ===
    pub provenance_values_total: usize,
    pub provenance_values_spilled: usize,
    pub provenance_values_preserved: usize,

    // === Register utilization ===
    pub max_active_intervals: usize,
    pub avg_active_intervals: f64,
    pub callee_saved_used: usize,

    // === Attention scores ===
    pub min_attention_score: f64,
    pub max_attention_score: f64,
    pub avg_attention_score: f64,
    pub spilled_avg_score: f64,
    pub kept_avg_score: f64,

    // === Timing ===
    #[serde(with = "duration_serde")]
    pub allocation_time: Duration,

    // === Policy used ===
    pub policy: String,
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

impl AllocationMetrics {
    /// Epistemic quality score: ratio of "good" spills to "bad" spills.
    /// Higher is better (we want to spill low-confidence values).
    pub fn epistemic_quality(&self) -> f64 {
        if self.total_spills == 0 {
            return 1.0;
        }
        let good = self.low_confidence_spills as f64;
        let bad = self.high_confidence_spills as f64;
        (good + 1.0) / (bad + 1.0)
    }

    /// Provenance preservation rate.
    pub fn provenance_preservation_rate(&self) -> f64 {
        if self.provenance_values_total == 0 {
            return 1.0;
        }
        self.provenance_values_preserved as f64 / self.provenance_values_total as f64
    }

    /// Score separation: difference between kept and spilled average scores.
    /// Higher means the policy is discriminating well.
    pub fn score_separation(&self) -> f64 {
        self.kept_avg_score - self.spilled_avg_score
    }

    /// Export to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(self.to_json().as_bytes())?;
        Ok(())
    }

    /// Record a spill with given confidence and provenance.
    /// Convenience method for testing.
    pub fn record_spill(&mut self, confidence: f64, has_provenance: bool) {
        self.total_spills += 1;

        // Classify by confidence
        if confidence >= 0.7 {
            self.high_confidence_spills += 1;
        } else if confidence <= 0.3 {
            self.low_confidence_spills += 1;
        } else {
            self.medium_confidence_spills += 1;
        }

        // Track provenance
        if has_provenance {
            self.provenance_values_spilled += 1;
        }
    }
}

// =============================================================================
// Spill Event
// =============================================================================

/// Spill event for detailed logging.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpillEvent {
    pub value: u32,
    pub position: u32,
    pub attention_score: f64,
    pub confidence: Option<f64>,
    pub uncertainty: Option<f64>,
    pub has_provenance: bool,
    pub confidence_class: ConfidenceClass,
    pub reason: SpillReason,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SpillReason {
    RegisterPressure,
    CallerSavedEviction,
    PolicyDecision,
}

// =============================================================================
// Metrics Collector
// =============================================================================

/// Collects allocation metrics during register allocation.
#[derive(Default, Debug)]
pub struct MetricsCollector {
    pub enabled: bool,
    pub metrics: AllocationMetrics,
    pub spill_log: Vec<SpillEvent>,
    pub config: AttentionConfig,

    // Internal accumulators
    score_samples: Vec<f64>,
    kept_scores: Vec<f64>,
    spilled_scores: Vec<f64>,
    start_time: Option<Instant>,
}

impl MetricsCollector {
    pub fn new(policy: AllocPolicy, enabled: bool) -> Self {
        let config = match policy {
            AllocPolicy::Classic => AttentionConfig::default(),
            AllocPolicy::Attention => AttentionConfig::default(),
            AllocPolicy::AttentionCalibrated => AttentionConfig::calibrated(),
        };

        Self {
            enabled,
            metrics: AllocationMetrics {
                policy: policy.to_string(),
                ..Default::default()
            },
            spill_log: Vec::new(),
            config,
            score_samples: Vec::new(),
            kept_scores: Vec::new(),
            spilled_scores: Vec::new(),
            start_time: None,
        }
    }

    pub fn start(&mut self) {
        if self.enabled {
            self.start_time = Some(Instant::now());
        }
    }

    pub fn record_score(&mut self, score: f64) {
        if self.enabled {
            self.score_samples.push(score);
        }
    }

    pub fn record_kept(&mut self, score: f64) {
        if self.enabled {
            self.kept_scores.push(score);
        }
    }

    pub fn record_spill(&mut self, event: SpillEvent) {
        if self.enabled {
            self.spilled_scores.push(event.attention_score);
            self.metrics.total_spills += 1;

            match event.confidence_class {
                ConfidenceClass::High => self.metrics.high_confidence_spills += 1,
                ConfidenceClass::Medium => self.metrics.medium_confidence_spills += 1,
                ConfidenceClass::Low => self.metrics.low_confidence_spills += 1,
            }

            if event.has_provenance {
                self.metrics.provenance_values_spilled += 1;
            }

            self.spill_log.push(event);
        }
    }

    pub fn set_provenance_total(&mut self, count: usize) {
        if self.enabled {
            self.metrics.provenance_values_total = count;
        }
    }

    pub fn update_max_active(&mut self, count: usize) {
        if self.enabled && count > self.metrics.max_active_intervals {
            self.metrics.max_active_intervals = count;
        }
    }

    pub fn finalize(&mut self) {
        if !self.enabled {
            return;
        }

        if let Some(start) = self.start_time {
            self.metrics.allocation_time = start.elapsed();
        }

        // Attention score statistics
        if !self.score_samples.is_empty() {
            self.metrics.min_attention_score = self
                .score_samples
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            self.metrics.max_attention_score = self
                .score_samples
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            self.metrics.avg_attention_score =
                self.score_samples.iter().sum::<f64>() / self.score_samples.len() as f64;
        }

        if !self.spilled_scores.is_empty() {
            self.metrics.spilled_avg_score =
                self.spilled_scores.iter().sum::<f64>() / self.spilled_scores.len() as f64;
        }

        if !self.kept_scores.is_empty() {
            self.metrics.kept_avg_score =
                self.kept_scores.iter().sum::<f64>() / self.kept_scores.len() as f64;
        }

        // Provenance preservation
        self.metrics.provenance_values_preserved = self
            .metrics
            .provenance_values_total
            .saturating_sub(self.metrics.provenance_values_spilled);
    }

    /// Reset the collector for a new function/allocation pass.
    pub fn reset(&mut self) {
        self.metrics = AllocationMetrics {
            policy: self.metrics.policy.clone(),
            ..Default::default()
        };
        self.spill_log.clear();
        self.score_samples.clear();
        self.kept_scores.clear();
        self.spilled_scores.clear();
        self.start_time = None;
    }
}

// =============================================================================
// A/B Comparison Result
// =============================================================================

/// Results from A/B comparison.
#[derive(Debug, Serialize, Deserialize)]
pub struct ABComparisonResult {
    pub classic_metrics: AllocationMetrics,
    pub attention_metrics: AllocationMetrics,
    pub calibrated_metrics: Option<AllocationMetrics>,

    // Derived comparisons
    pub spill_reduction: f64,
    pub epistemic_quality_gain: f64,
    pub provenance_preservation_gain: f64,
    pub score_separation_gain: f64,
}

impl ABComparisonResult {
    pub fn compute(
        classic: AllocationMetrics,
        attention: AllocationMetrics,
        calibrated: Option<AllocationMetrics>,
    ) -> Self {
        let classic_spills = classic.total_spills as f64;
        let attention_spills = attention.total_spills as f64;
        let spill_reduction = if classic_spills > 0.0 {
            (classic_spills - attention_spills) / classic_spills
        } else {
            0.0
        };

        let classic_eq = classic.epistemic_quality();
        let attention_eq = attention.epistemic_quality();
        let epistemic_quality_gain = if classic_eq > 0.0 {
            (attention_eq - classic_eq) / classic_eq
        } else {
            0.0
        };

        let classic_pp = classic.provenance_preservation_rate();
        let attention_pp = attention.provenance_preservation_rate();
        let provenance_preservation_gain = attention_pp - classic_pp;

        let score_separation_gain = attention.score_separation() - classic.score_separation();

        Self {
            classic_metrics: classic,
            attention_metrics: attention,
            calibrated_metrics: calibrated,
            spill_reduction,
            epistemic_quality_gain,
            provenance_preservation_gain,
            score_separation_gain,
        }
    }

    pub fn summary(&self) -> String {
        let timing_overhead = {
            let c = self.classic_metrics.allocation_time.as_secs_f64();
            let a = self.attention_metrics.allocation_time.as_secs_f64();
            if c == 0.0 {
                0.0
            } else {
                (a - c) / c
            }
        };

        let verdict = {
            let quality_improved = self.epistemic_quality_gain > 0.0;
            let provenance_improved = self.provenance_preservation_gain >= 0.0;
            let overhead_acceptable = timing_overhead < 0.10;

            if quality_improved && provenance_improved && overhead_acceptable {
                "ATTENTION POLICY SUPERIOR - Recommend adoption"
            } else if quality_improved && overhead_acceptable {
                "ATTENTION POLICY MARGINAL - Consider for epistemic workloads"
            } else if !overhead_acceptable {
                "ATTENTION POLICY OVERHEAD TOO HIGH - Optimize implementation"
            } else {
                "CLASSIC POLICY PREFERRED - No improvement observed"
            }
        };

        format!(
            r#"
A/B POLICY COMPARISON RESULTS
=============================

SPILL STATISTICS
                        Classic     Attention   Delta
Total Spills            {:>10}  {:>10}  {:>+10.1}%
High-Confidence Spills  {:>10}  {:>10}
Low-Confidence Spills   {:>10}  {:>10}

QUALITY METRICS
                        Classic     Attention   Delta
Epistemic Quality       {:>10.3}  {:>10.3}  {:>+10.1}%
Provenance Preservation {:>9.1}%  {:>9.1}%  {:>+10.1}%
Score Separation        {:>10.3}  {:>10.3}

TIMING
Allocation Time         {:>7.2}us  {:>7.2}us  {:>+7.1}% overhead

VERDICT: {}
"#,
            self.classic_metrics.total_spills,
            self.attention_metrics.total_spills,
            self.spill_reduction * 100.0,
            self.classic_metrics.high_confidence_spills,
            self.attention_metrics.high_confidence_spills,
            self.classic_metrics.low_confidence_spills,
            self.attention_metrics.low_confidence_spills,
            self.classic_metrics.epistemic_quality(),
            self.attention_metrics.epistemic_quality(),
            self.epistemic_quality_gain * 100.0,
            self.classic_metrics.provenance_preservation_rate() * 100.0,
            self.attention_metrics.provenance_preservation_rate() * 100.0,
            self.provenance_preservation_gain * 100.0,
            self.classic_metrics.score_separation(),
            self.attention_metrics.score_separation(),
            self.classic_metrics.allocation_time.as_secs_f64() * 1_000_000.0,
            self.attention_metrics.allocation_time.as_secs_f64() * 1_000_000.0,
            timing_overhead * 100.0,
            verdict
        )
    }
}

// =============================================================================
// Spill Candidate Selection
// =============================================================================

/// A candidate for spilling during register allocation.
///
/// This struct captures all the information needed to make a spill decision
/// for a live interval, combining classical heuristics with epistemic metadata.
#[derive(Debug, Clone)]
pub struct SpillCandidate {
    /// Index into the active interval list
    pub index: usize,

    /// Computed attention score (higher = more valuable, less likely to spill)
    pub score: f64,

    /// Epistemic metadata for this interval
    pub epistemic: EpistemicMetadata,

    /// Confidence classification
    pub confidence_class: ConfidenceClass,

    /// Use density (uses per instruction)
    pub use_density: f64,

    /// Whether this interval crosses a call
    pub crosses_call: bool,

    /// Distance to next use (in instructions)
    pub next_use_distance: f64,
}

impl SpillCandidate {
    /// Create a new spill candidate.
    pub fn new(
        index: usize,
        use_density: f64,
        crosses_call: bool,
        next_use_distance: f64,
        epistemic: EpistemicMetadata,
        config: &AttentionConfig,
    ) -> Self {
        let confidence_class = epistemic.confidence_class(config);
        let score = compute_attention_score(
            use_density,
            crosses_call,
            next_use_distance,
            &epistemic,
            config,
        );

        Self {
            index,
            score,
            epistemic,
            confidence_class,
            use_density,
            crosses_call,
            next_use_distance,
        }
    }
}

/// Select the best spill victim from a set of candidates.
///
/// This function implements the core spill selection logic for attention-based
/// register allocation. It selects the candidate with the lowest attention score,
/// meaning the least valuable interval to keep in a register.
///
/// # Arguments
/// * `candidates` - Slice of spill candidates to choose from
/// * `policy` - The allocation policy to use for selection
///
/// # Returns
/// The index of the selected spill victim, or None if no candidates exist
///
/// # Selection Strategy
///
/// **Classic policy**: Uses only classical heuristics (use density, call crossing,
/// next use distance) to select the spill victim.
///
/// **Attention policy**: Combines classical heuristics with epistemic metadata
/// (confidence, uncertainty, provenance) using default weights.
///
/// **Calibrated policy**: Uses BO-optimized weights that have been empirically
/// tuned for epistemic workloads.
pub fn select_spill_victim(candidates: &[SpillCandidate], policy: AllocPolicy) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }

    match policy {
        AllocPolicy::Classic => {
            // Classic: prefer spilling intervals with:
            // 1. Low use density (fewer uses = less spill cost)
            // 2. Not crossing calls (easier to reload)
            // 3. Furthest next use (more time before reload needed)
            candidates
                .iter()
                .min_by(|a, b| {
                    // Lower use density is better for spilling
                    let density_cmp = a
                        .use_density
                        .partial_cmp(&b.use_density)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if density_cmp != std::cmp::Ordering::Equal {
                        return density_cmp;
                    }

                    // Prefer not crossing calls
                    let call_cmp = a.crosses_call.cmp(&b.crosses_call);
                    if call_cmp != std::cmp::Ordering::Equal {
                        return call_cmp;
                    }

                    // Furthest next use is better (negate for min)
                    b.next_use_distance
                        .partial_cmp(&a.next_use_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|c| c.index)
        }

        AllocPolicy::Attention | AllocPolicy::AttentionCalibrated => {
            // Attention-based: select lowest attention score (least valuable)
            candidates
                .iter()
                .min_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|c| c.index)
        }
    }
}

/// Select multiple spill victims to achieve a target register count.
///
/// This is useful when register pressure requires spilling multiple intervals
/// at once. The function greedily selects the lowest-scoring candidates.
///
/// # Arguments
/// * `candidates` - Mutable slice of spill candidates (will be sorted)
/// * `policy` - The allocation policy to use
/// * `count` - Number of victims to select
///
/// # Returns
/// Vector of indices of selected spill victims
pub fn select_spill_victims(
    candidates: &mut [SpillCandidate],
    policy: AllocPolicy,
    count: usize,
) -> Vec<usize> {
    if candidates.is_empty() || count == 0 {
        return Vec::new();
    }

    match policy {
        AllocPolicy::Classic => {
            // Sort by classic criteria (ascending = worse for keeping)
            candidates.sort_by(|a, b| {
                let density_cmp = a
                    .use_density
                    .partial_cmp(&b.use_density)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if density_cmp != std::cmp::Ordering::Equal {
                    return density_cmp;
                }
                a.crosses_call.cmp(&b.crosses_call)
            });
        }

        AllocPolicy::Attention | AllocPolicy::AttentionCalibrated => {
            // Sort by attention score (ascending = lower score = better to spill)
            candidates.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    candidates.iter().take(count).map(|c| c.index).collect()
}

/// Classify a confidence value into High/Medium/Low categories.
///
/// This is a standalone function version of `EpistemicMetadata::confidence_class()`.
pub fn classify_confidence(confidence: f64, config: &AttentionConfig) -> ConfidenceClass {
    if confidence >= config.high_confidence_threshold {
        ConfidenceClass::High
    } else if confidence <= config.low_confidence_threshold {
        ConfidenceClass::Low
    } else {
        ConfidenceClass::Medium
    }
}

/// Compute the attention score for a given set of interval characteristics.
///
/// This is the core scoring function that combines classical register allocation
/// heuristics with epistemic metadata. Higher scores indicate more valuable
/// intervals that should be kept in registers.
///
/// This is an alias for `compute_attention_score` with a more intuitive name.
pub fn attention_score(
    use_density: f64,
    crosses_call: bool,
    next_use_distance: f64,
    epistemic: &EpistemicMetadata,
    config: &AttentionConfig,
) -> f64 {
    compute_attention_score(
        use_density,
        crosses_call,
        next_use_distance,
        epistemic,
        config,
    )
}

// =============================================================================
// CLI Options
// =============================================================================

/// Compiler options for allocation policy.
#[derive(Debug, Default)]
pub struct AllocatorOptions {
    pub policy: AllocPolicy,
    pub emit_metrics: Option<String>,
    pub attention_config_path: Option<String>,
    pub run_ab_comparison: bool,
}

impl AllocatorOptions {
    /// Parse from command line args.
    pub fn from_args(args: &[String]) -> Self {
        let mut opts = AllocatorOptions::default();

        for arg in args {
            if let Some(policy_str) = arg.strip_prefix("--alloc-policy=") {
                opts.policy = policy_str.parse().unwrap_or_default();
            }
            if let Some(path) = arg.strip_prefix("--emit-metrics=") {
                opts.emit_metrics = Some(path.to_string());
            }
            if let Some(path) = arg.strip_prefix("--attention-config=") {
                opts.attention_config_path = Some(path.to_string());
            }
            if arg == "--ab-compare" {
                opts.run_ab_comparison = true;
            }
        }

        opts
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_score_ordering() {
        let config = AttentionConfig::default();

        // High-value: high confidence, low uncertainty, has provenance
        let critical = EpistemicMetadata {
            confidence: Some(0.95),
            uncertainty: Some(0.1),
            has_provenance: true,
            provenance_depth: 1,
            source_op: None,
        };

        // Low-value: low confidence, high uncertainty, no provenance
        let dispensable = EpistemicMetadata {
            confidence: Some(0.2),
            uncertainty: Some(0.9),
            has_provenance: false,
            provenance_depth: 0,
            source_op: None,
        };

        let score_critical = compute_attention_score(0.5, false, 10.0, &critical, &config);
        let score_dispensable = compute_attention_score(0.2, false, 50.0, &dispensable, &config);

        assert!(
            score_critical > score_dispensable,
            "Critical value must have higher score. critical={:.3}, dispensable={:.3}",
            score_critical,
            score_dispensable
        );
    }

    #[test]
    fn test_confidence_classification() {
        let config = AttentionConfig::default();

        let high = EpistemicMetadata {
            confidence: Some(0.85),
            ..Default::default()
        };
        let medium = EpistemicMetadata {
            confidence: Some(0.5),
            ..Default::default()
        };
        let low = EpistemicMetadata {
            confidence: Some(0.2),
            ..Default::default()
        };

        assert_eq!(high.confidence_class(&config), ConfidenceClass::High);
        assert_eq!(medium.confidence_class(&config), ConfidenceClass::Medium);
        assert_eq!(low.confidence_class(&config), ConfidenceClass::Low);
    }

    #[test]
    fn test_epistemic_quality_formula() {
        let mut metrics = AllocationMetrics::default();

        // Good case: mostly low-confidence spills
        metrics.total_spills = 10;
        metrics.high_confidence_spills = 1;
        metrics.low_confidence_spills = 8;
        let good_quality = metrics.epistemic_quality();

        // Bad case: mostly high-confidence spills
        metrics.high_confidence_spills = 8;
        metrics.low_confidence_spills = 1;
        let bad_quality = metrics.epistemic_quality();

        assert!(
            good_quality > bad_quality,
            "Spilling low-confidence should yield higher quality. Good: {:.3}, Bad: {:.3}",
            good_quality,
            bad_quality
        );
    }

    #[test]
    fn test_metrics_serialization() {
        let mut metrics = AllocationMetrics::default();
        metrics.total_spills = 5;
        metrics.high_confidence_spills = 1;
        metrics.low_confidence_spills = 3;
        metrics.provenance_values_total = 10;
        metrics.provenance_values_preserved = 8;

        let json = metrics.to_json();
        assert!(json.contains("total_spills"));
        assert!(json.contains("5"));
    }

    #[test]
    fn test_policy_parsing() {
        assert_eq!(
            "classic".parse::<AllocPolicy>().unwrap(),
            AllocPolicy::Classic
        );
        assert_eq!(
            "attention".parse::<AllocPolicy>().unwrap(),
            AllocPolicy::Attention
        );
        assert_eq!(
            "calibrated".parse::<AllocPolicy>().unwrap(),
            AllocPolicy::AttentionCalibrated
        );
        assert_eq!(
            "bo".parse::<AllocPolicy>().unwrap(),
            AllocPolicy::AttentionCalibrated
        );
    }

    #[test]
    fn test_spill_candidate_creation() {
        let config = AttentionConfig::default();
        let epistemic = EpistemicMetadata {
            confidence: Some(0.8),
            uncertainty: Some(0.2),
            has_provenance: true,
            provenance_depth: 2,
            source_op: Some("mul".to_string()),
        };

        let candidate = SpillCandidate::new(
            0, 0.5,   // use_density
            false, // crosses_call
            10.0,  // next_use_distance
            epistemic, &config,
        );

        assert_eq!(candidate.index, 0);
        assert!(candidate.score > 0.0);
        assert_eq!(candidate.confidence_class, ConfidenceClass::High);
        assert_eq!(candidate.use_density, 0.5);
        assert!(!candidate.crosses_call);
        assert_eq!(candidate.next_use_distance, 10.0);
    }

    #[test]
    fn test_select_spill_victim_attention() {
        let config = AttentionConfig::calibrated();

        // High-value candidate (should NOT be spilled)
        let high_value = SpillCandidate::new(
            0,
            0.8,  // high use density
            true, // crosses call
            5.0,  // close next use
            EpistemicMetadata {
                confidence: Some(0.95),
                uncertainty: Some(0.1),
                has_provenance: true,
                ..Default::default()
            },
            &config,
        );

        // Low-value candidate (should be spilled)
        let low_value = SpillCandidate::new(
            1,
            0.1,   // low use density
            false, // no call crossing
            100.0, // far next use
            EpistemicMetadata {
                confidence: Some(0.2),
                uncertainty: Some(0.8),
                has_provenance: false,
                ..Default::default()
            },
            &config,
        );

        let candidates = vec![high_value, low_value];

        // Attention policy should select low-value (index 1)
        let victim = select_spill_victim(&candidates, AllocPolicy::Attention);
        assert_eq!(
            victim,
            Some(1),
            "Should select low-value candidate for spilling"
        );

        // Calibrated should also select low-value
        let victim_cal = select_spill_victim(&candidates, AllocPolicy::AttentionCalibrated);
        assert_eq!(
            victim_cal,
            Some(1),
            "Calibrated should also select low-value candidate"
        );
    }

    #[test]
    fn test_select_spill_victim_classic() {
        let config = AttentionConfig::default();

        // Low density, no call, far use (best for spilling in classic)
        let best_spill = SpillCandidate::new(
            0,
            0.1,   // low density
            false, // no call
            100.0, // far use
            EpistemicMetadata::default(),
            &config,
        );

        // High density, crosses call, close use (worst for spilling)
        let worst_spill = SpillCandidate::new(
            1,
            0.9,  // high density
            true, // crosses call
            5.0,  // close use
            EpistemicMetadata::default(),
            &config,
        );

        let candidates = vec![best_spill, worst_spill];

        // Classic should select best_spill (index 0) based on density
        let victim = select_spill_victim(&candidates, AllocPolicy::Classic);
        assert_eq!(
            victim,
            Some(0),
            "Classic should select based on use density"
        );
    }

    #[test]
    fn test_select_spill_victims_multiple() {
        let config = AttentionConfig::calibrated();

        let mut candidates: Vec<SpillCandidate> = (0..5)
            .map(|i| {
                SpillCandidate::new(
                    i,
                    0.1 * (i as f64 + 1.0), // increasing density
                    false,
                    50.0,
                    EpistemicMetadata {
                        confidence: Some(0.2 * (i as f64 + 1.0).min(1.0)),
                        ..Default::default()
                    },
                    &config,
                )
            })
            .collect();

        // Select 2 victims
        let victims = select_spill_victims(&mut candidates, AllocPolicy::Attention, 2);
        assert_eq!(victims.len(), 2);

        // Should select the two lowest-scoring candidates
        // With increasing confidence, candidates 0 and 1 should be lowest
        assert!(victims.contains(&0) || victims.contains(&1));
    }

    #[test]
    fn test_classify_confidence_standalone() {
        let config = AttentionConfig::default();

        assert_eq!(classify_confidence(0.9, &config), ConfidenceClass::High);
        assert_eq!(classify_confidence(0.5, &config), ConfidenceClass::Medium);
        assert_eq!(classify_confidence(0.1, &config), ConfidenceClass::Low);
    }

    #[test]
    fn test_attention_score_alias() {
        let config = AttentionConfig::default();
        let epistemic = EpistemicMetadata {
            confidence: Some(0.7),
            uncertainty: Some(0.3),
            has_provenance: true,
            ..Default::default()
        };

        let score1 = attention_score(0.5, false, 20.0, &epistemic, &config);
        let score2 = compute_attention_score(0.5, false, 20.0, &epistemic, &config);

        assert!(
            (score1 - score2).abs() < f64::EPSILON,
            "attention_score and compute_attention_score should be identical"
        );
    }

    #[test]
    fn test_calibrated_weights_values() {
        let calibrated = AttentionConfig::calibrated();

        // Verify BO-optimized weights from 2025-12-28
        assert!((calibrated.w_use_density - 1.7088).abs() < 0.0001);
        assert!((calibrated.w_crosses_call - 0.4527).abs() < 0.0001);
        assert!((calibrated.w_next_use_distance - 0.8802).abs() < 0.0001);
        assert!((calibrated.w_confidence - 0.9374).abs() < 0.0001);
        assert!((calibrated.w_uncertainty - 0.2411).abs() < 0.0001);
        assert!((calibrated.w_provenance - 0.2013).abs() < 0.0001);
    }

    #[test]
    fn test_select_spill_empty_candidates() {
        let empty: Vec<SpillCandidate> = vec![];

        assert_eq!(select_spill_victim(&empty, AllocPolicy::Classic), None);
        assert_eq!(select_spill_victim(&empty, AllocPolicy::Attention), None);
        assert_eq!(
            select_spill_victim(&empty, AllocPolicy::AttentionCalibrated),
            None
        );
    }

    #[test]
    fn test_calibrated_prefers_high_confidence() {
        let config = AttentionConfig::calibrated();

        // Two candidates with same classical metrics but different confidence
        let high_conf = SpillCandidate::new(
            0,
            0.5,
            false,
            50.0,
            EpistemicMetadata {
                confidence: Some(0.95),
                uncertainty: Some(0.05),
                has_provenance: true,
                ..Default::default()
            },
            &config,
        );

        let low_conf = SpillCandidate::new(
            1,
            0.5,
            false,
            50.0,
            EpistemicMetadata {
                confidence: Some(0.2),
                uncertainty: Some(0.8),
                has_provenance: false,
                ..Default::default()
            },
            &config,
        );

        // High confidence should have higher score
        assert!(
            high_conf.score > low_conf.score,
            "High confidence should have higher attention score. high={:.3}, low={:.3}",
            high_conf.score,
            low_conf.score
        );

        // Therefore low_conf should be selected for spilling
        let candidates = vec![high_conf, low_conf];
        let victim = select_spill_victim(&candidates, AllocPolicy::AttentionCalibrated);
        assert_eq!(victim, Some(1), "Should spill low-confidence value");
    }
}
