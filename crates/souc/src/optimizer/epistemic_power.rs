//! Epistemic Power utility and runtime normalization helpers.
//!
//! Canonical form:
//! EpistemicPower = exp(sum(w_i * ln(clamp(m_i))) - penalties)

pub const EPISTEMIC_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicMetrics {
    pub runtime: f64,
    pub gum_fidelity: f64,
    pub provenance_depth: f64,
    pub formal_coverage: f64,
    pub conformance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicWeights {
    pub runtime: f64,
    pub gum_fidelity: f64,
    pub provenance_depth: f64,
    pub formal_coverage: f64,
    pub conformance: f64,
}

impl Default for EpistemicWeights {
    fn default() -> Self {
        Self {
            runtime: 0.35,
            gum_fidelity: 0.25,
            provenance_depth: 0.15,
            formal_coverage: 0.15,
            conformance: 0.10,
        }
    }
}

impl EpistemicWeights {
    pub fn sum(self) -> f64 {
        self.runtime
            + self.gum_fidelity
            + self.provenance_depth
            + self.formal_coverage
            + self.conformance
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeNormalizationBand {
    pub floor_speedup: f64,
    pub cap_speedup: f64,
}

impl RuntimeNormalizationBand {
    pub const fn new(floor_speedup: f64, cap_speedup: f64) -> Self {
        Self {
            floor_speedup,
            cap_speedup,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadFamily {
    DenseLinear,
    Attention,
    EpistemicElementwise,
    MonteCarlo,
    QuantumParallel,
}

impl WorkloadFamily {
    pub fn default_band(self) -> RuntimeNormalizationBand {
        match self {
            WorkloadFamily::DenseLinear => RuntimeNormalizationBand::new(0.95, 3.0),
            WorkloadFamily::Attention => RuntimeNormalizationBand::new(0.95, 3.0),
            WorkloadFamily::EpistemicElementwise => RuntimeNormalizationBand::new(0.95, 2.5),
            WorkloadFamily::MonteCarlo => RuntimeNormalizationBand::new(0.95, 2.5),
            WorkloadFamily::QuantumParallel => RuntimeNormalizationBand::new(0.95, 2.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicPenalties {
    pub budget_penalty: f64,
    pub bit_exact_penalty: f64,
    pub compile_overhead_penalty: f64,
}

impl EpistemicPenalties {
    pub fn total(self) -> f64 {
        (self.budget_penalty + self.bit_exact_penalty + self.compile_overhead_penalty).max(0.0)
    }
}

pub fn clamp_metric(x: f64) -> f64 {
    x.clamp(EPISTEMIC_EPSILON, 1.0)
}

pub fn normalize_runtime_speedup(speedup: f64, band: RuntimeNormalizationBand) -> f64 {
    if band.cap_speedup <= band.floor_speedup {
        return EPISTEMIC_EPSILON;
    }
    let normalized = (speedup - band.floor_speedup) / (band.cap_speedup - band.floor_speedup);
    clamp_metric(normalized)
}

pub fn epistemic_power_log(
    metrics: EpistemicMetrics,
    weights: EpistemicWeights,
    penalties: EpistemicPenalties,
) -> f64 {
    let weighted = weights.runtime * clamp_metric(metrics.runtime).ln()
        + weights.gum_fidelity * clamp_metric(metrics.gum_fidelity).ln()
        + weights.provenance_depth * clamp_metric(metrics.provenance_depth).ln()
        + weights.formal_coverage * clamp_metric(metrics.formal_coverage).ln()
        + weights.conformance * clamp_metric(metrics.conformance).ln();

    weighted - penalties.total()
}

pub fn epistemic_power(
    metrics: EpistemicMetrics,
    weights: EpistemicWeights,
    penalties: EpistemicPenalties,
) -> f64 {
    epistemic_power_log(metrics, weights, penalties).exp()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicIntrinsicKind {
    KnowledgeFma,
    KnowledgeAdd,
    KnowledgeMul,
    KnowledgeDiv,
    KnowledgePropVar,
    KnowledgeMergeProv,
    Other,
}

impl EpistemicIntrinsicKind {
    pub fn from_name(name: &str) -> Self {
        match name {
            "knowledge.fma" => Self::KnowledgeFma,
            "knowledge.add" => Self::KnowledgeAdd,
            "knowledge.mul" => Self::KnowledgeMul,
            "knowledge.div" => Self::KnowledgeDiv,
            "knowledge.prop_var" => Self::KnowledgePropVar,
            "knowledge.merge_prov" => Self::KnowledgeMergeProv,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpistemicIntrinsicCounters {
    pub knowledge_fma: u64,
    pub knowledge_add: u64,
    pub knowledge_mul: u64,
    pub knowledge_div: u64,
    pub knowledge_prop_var: u64,
    pub knowledge_merge_prov: u64,
    pub other: u64,
    pub fallback: u64,
}

impl EpistemicIntrinsicCounters {
    pub fn total_intrinsics(self) -> u64 {
        self.knowledge_fma
            + self.knowledge_add
            + self.knowledge_mul
            + self.knowledge_div
            + self.knowledge_prop_var
            + self.knowledge_merge_prov
            + self.other
    }
}

/// Runtime tracker for intrinsic usage that can be folded into EpistemicPower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicPowerTracker {
    counters: EpistemicIntrinsicCounters,
    gum_fidelity_bonus: f64,
    provenance_depth_bonus: f64,
    formal_coverage_bonus: f64,
}

impl Default for EpistemicPowerTracker {
    fn default() -> Self {
        Self {
            counters: EpistemicIntrinsicCounters::default(),
            gum_fidelity_bonus: 0.0,
            provenance_depth_bonus: 0.0,
            formal_coverage_bonus: 0.0,
        }
    }
}

impl EpistemicPowerTracker {
    pub fn record_intrinsic(&mut self, name: &str, contribution: f64) {
        let c = contribution.clamp(0.0, 1.0);
        match EpistemicIntrinsicKind::from_name(name) {
            EpistemicIntrinsicKind::KnowledgeFma => {
                self.counters.knowledge_fma += 1;
                self.gum_fidelity_bonus += 0.004 * c;
            }
            EpistemicIntrinsicKind::KnowledgeAdd => {
                self.counters.knowledge_add += 1;
                self.gum_fidelity_bonus += 0.002 * c;
            }
            EpistemicIntrinsicKind::KnowledgeMul => {
                self.counters.knowledge_mul += 1;
                self.gum_fidelity_bonus += 0.003 * c;
            }
            EpistemicIntrinsicKind::KnowledgeDiv => {
                self.counters.knowledge_div += 1;
                self.gum_fidelity_bonus += 0.003 * c;
            }
            EpistemicIntrinsicKind::KnowledgePropVar => {
                self.counters.knowledge_prop_var += 1;
                self.gum_fidelity_bonus += 0.0025 * c;
            }
            EpistemicIntrinsicKind::KnowledgeMergeProv => {
                self.counters.knowledge_merge_prov += 1;
                self.provenance_depth_bonus += 0.003 * c;
            }
            EpistemicIntrinsicKind::Other => {
                self.counters.other += 1;
            }
        }
        self.formal_coverage_bonus += 0.001 * c;
    }

    pub fn record_fallback(&mut self) {
        self.counters.fallback += 1;
        // Fallback should not erase accumulated credit, only dampen it.
        self.gum_fidelity_bonus *= 0.995;
        self.provenance_depth_bonus *= 0.995;
        self.formal_coverage_bonus *= 0.995;
    }

    pub fn counters(self) -> EpistemicIntrinsicCounters {
        self.counters
    }

    pub fn intrinsic_count(self) -> u64 {
        self.counters.total_intrinsics()
    }

    pub fn fallback_count(self) -> u64 {
        self.counters.fallback
    }

    /// Apply intrinsic bonuses with fallback damping and clamp to [ε, 1].
    pub fn adjust_metrics(self, metrics: EpistemicMetrics) -> EpistemicMetrics {
        let total = self.intrinsic_count() as f64;
        let fallback = self.fallback_count() as f64;
        let fallback_ratio = if total > 0.0 { fallback / total } else { 0.0 };
        let damping = (1.0 - 0.5 * fallback_ratio).clamp(0.5, 1.0);

        let gum = clamp_metric(metrics.gum_fidelity + self.gum_fidelity_bonus * damping);
        let prov = clamp_metric(metrics.provenance_depth + self.provenance_depth_bonus * damping);
        let formal = clamp_metric(metrics.formal_coverage + self.formal_coverage_bonus * damping);

        EpistemicMetrics {
            runtime: clamp_metric(metrics.runtime),
            gum_fidelity: gum,
            provenance_depth: prov,
            formal_coverage: formal,
            conformance: clamp_metric(metrics.conformance),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_default_to_one() {
        let sum = EpistemicWeights::default().sum();
        assert!((sum - 1.0).abs() < 1e-9, "weights sum={sum}");
    }

    #[test]
    fn runtime_normalization_is_clamped() {
        let band = WorkloadFamily::Attention.default_band();
        assert_eq!(normalize_runtime_speedup(0.10, band), EPISTEMIC_EPSILON);
        assert!((normalize_runtime_speedup(3.0, band) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn penalty_monotonicity_holds() {
        let metrics = EpistemicMetrics {
            runtime: 0.8,
            gum_fidelity: 0.9,
            provenance_depth: 0.7,
            formal_coverage: 0.6,
            conformance: 1.0,
        };
        let weights = EpistemicWeights::default();
        let lo = EpistemicPenalties {
            budget_penalty: 0.0,
            bit_exact_penalty: 0.0,
            compile_overhead_penalty: 0.0,
        };
        let hi = EpistemicPenalties {
            budget_penalty: 0.2,
            bit_exact_penalty: 0.0,
            compile_overhead_penalty: 0.0,
        };
        assert!(epistemic_power(metrics, weights, lo) > epistemic_power(metrics, weights, hi));
    }

    #[test]
    fn better_metrics_increase_score() {
        let weights = EpistemicWeights::default();
        let penalties = EpistemicPenalties {
            budget_penalty: 0.0,
            bit_exact_penalty: 0.0,
            compile_overhead_penalty: 0.0,
        };
        let m1 = EpistemicMetrics {
            runtime: 0.8,
            gum_fidelity: 0.8,
            provenance_depth: 0.8,
            formal_coverage: 0.8,
            conformance: 0.8,
        };
        let m2 = EpistemicMetrics {
            runtime: 0.85,
            gum_fidelity: 0.85,
            provenance_depth: 0.85,
            formal_coverage: 0.85,
            conformance: 0.85,
        };
        assert!(epistemic_power(m2, weights, penalties) > epistemic_power(m1, weights, penalties));
    }

    #[test]
    fn tracker_counts_intrinsics_and_fallback() {
        let mut tracker = EpistemicPowerTracker::default();
        tracker.record_intrinsic("knowledge.fma", 1.0);
        tracker.record_intrinsic("knowledge.add", 0.5);
        tracker.record_intrinsic("knowledge.merge_prov", 1.0);
        tracker.record_fallback();

        let counts = tracker.counters();
        assert_eq!(counts.knowledge_fma, 1);
        assert_eq!(counts.knowledge_add, 1);
        assert_eq!(counts.knowledge_merge_prov, 1);
        assert_eq!(counts.fallback, 1);
        assert_eq!(tracker.intrinsic_count(), 3);
    }

    #[test]
    fn tracker_adjusts_metrics_monotonically_without_fallback() {
        let mut tracker = EpistemicPowerTracker::default();
        tracker.record_intrinsic("knowledge.fma", 1.0);
        tracker.record_intrinsic("knowledge.prop_var", 1.0);
        tracker.record_intrinsic("knowledge.merge_prov", 1.0);

        let base = EpistemicMetrics {
            runtime: 0.8,
            gum_fidelity: 0.8,
            provenance_depth: 0.7,
            formal_coverage: 0.6,
            conformance: 1.0,
        };
        let adjusted = tracker.adjust_metrics(base);
        assert!(adjusted.gum_fidelity >= base.gum_fidelity);
        assert!(adjusted.provenance_depth >= base.provenance_depth);
        assert!(adjusted.formal_coverage >= base.formal_coverage);
    }
}
