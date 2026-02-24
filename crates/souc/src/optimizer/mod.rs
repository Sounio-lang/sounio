//! Optimization Passes for Sounio HIR
//!
//! This module contains optimization passes that run on the HIR before
//! lowering to HLIR. The key optimizer is the EpistemicOptimizer which
//! handles Knowledge type optimization.

pub mod epistemic;
pub mod epistemic_power;

pub use epistemic::EpistemicOptimizer;
pub use epistemic_power::{
    EPISTEMIC_EPSILON, EpistemicIntrinsicCounters, EpistemicIntrinsicKind, EpistemicMetrics,
    EpistemicPenalties, EpistemicPowerTracker, EpistemicWeights, RuntimeNormalizationBand,
    WorkloadFamily, clamp_metric, epistemic_power, epistemic_power_log, normalize_runtime_speedup,
};
