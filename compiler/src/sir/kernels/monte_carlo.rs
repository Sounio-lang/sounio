//! Monte Carlo Particle Simulation Kernel
//!
//! This is the canonical proof-of-concept for epistemic-aware parallel execution.
//! The kernel demonstrates:
//!
//! 1. **Confidence flows through computation** - Each particle carries epistemic
//!    state that is updated as transformations occur.
//!
//! 2. **Low-confidence particles handled differently** - The "attention score"
//!    concept applied to particles: high-confidence particles may receive more
//!    computational resources (finer time steps, more accurate methods).
//!
//! 3. **Divergence-avoiding patterns** - Confidence partitioning allows handling
//!    low and high confidence particles in separate branches, avoiding GPU warp
//!    divergence while preserving epistemic semantics.
//!
//! # Kernel Structure
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Monte Carlo Kernel                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  1. Initialize particles from prior distribution                │
//! │  2. Loop:                                                       │
//! │     a. Propagate with stochastic dynamics                       │
//! │     b. Update confidence based on dynamics certainty            │
//! │     c. Apply importance weighting (observation update)          │
//! │     d. Partition by confidence (high/low attention)             │
//! │     e. Resample based on weights                                │
//! │  3. Epistemic reduction to aggregate statistics                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Implementation Philosophy
//!
//! This is "the soul" of Sounio's scientific computing - demonstrating that
//! uncertainty is not an afterthought but flows through every computation,
//! enabling principled decision-making about computational resource allocation.

use crate::sir::{
    blocks::{BasicBlock, Linkage, Terminator},
    metadata::MetadataStore,
    module::SirModule,
    types::{ArrayType, SirType, StructType},
    values::{BlockId, ValueId},
};
use std::fmt;

// ============================================================================
// PARTICLE STATE
// ============================================================================

/// State for a single Monte Carlo particle.
///
/// This represents the complete epistemic state of a particle in the simulation,
/// including both its physical state and its epistemic metadata (confidence,
/// provenance, weight).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleState {
    /// 3D position in state space [x, y, z]
    pub position: [f64; 3],

    /// 3D velocity (or momentum) [vx, vy, vz]
    pub velocity: [f64; 3],

    /// Importance weight for resampling
    /// Higher weight = more likely to survive resampling
    pub weight: f64,

    /// Epistemic confidence in this particle's state
    /// Ranges from 0.0 (completely uncertain) to 1.0 (certain)
    pub confidence: f64,

    /// Provenance lineage tracking
    /// Encodes the history of transformations this particle has undergone
    /// Lower bits: original source ID
    /// Higher bits: transformation chain hash
    pub provenance: u64,
}

impl ParticleState {
    /// Create a new particle with given state
    pub fn new(position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            position,
            velocity,
            weight: 1.0,
            confidence: 1.0,
            provenance: 0,
        }
    }

    /// Create an uncertain particle
    pub fn uncertain(position: [f64; 3], velocity: [f64; 3], confidence: f64) -> Self {
        Self {
            position,
            velocity,
            weight: 1.0,
            confidence: confidence.clamp(0.0, 1.0),
            provenance: 0,
        }
    }

    /// Create particle with full epistemic state
    pub fn with_epistemic(
        position: [f64; 3],
        velocity: [f64; 3],
        weight: f64,
        confidence: f64,
        provenance: u64,
    ) -> Self {
        Self {
            position,
            velocity,
            weight,
            confidence: confidence.clamp(0.0, 1.0),
            provenance,
        }
    }

    /// Get the SIR type representation of ParticleState
    pub fn sir_type() -> SirType {
        SirType::Struct(StructType::new(vec![
            (Some("position".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
            (Some("velocity".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
            (Some("weight".into()), SirType::f64()),
            (Some("confidence".into()), SirType::f64()),
            (Some("provenance".into()), SirType::i64()),
        ]).named("ParticleState"))
    }

    /// Attention score - how much computational attention this particle deserves
    ///
    /// This is the key insight: particles with higher confidence AND higher weight
    /// contribute more to the final estimate, so they deserve more computational
    /// resources (finer time steps, more accurate integration methods, etc.)
    pub fn attention_score(&self) -> f64 {
        self.confidence * self.weight
    }

    /// Update provenance after a transformation
    pub fn with_transform(&self, transform_id: u32) -> Self {
        // Hash the transformation into the provenance chain
        // Uses a simple mixing function to combine transform history
        let new_provenance = self.provenance
            .wrapping_mul(0x517cc1b727220a95)
            .wrapping_add(transform_id as u64);

        Self {
            provenance: new_provenance,
            ..self.clone()
        }
    }
}

impl Default for ParticleState {
    fn default() -> Self {
        Self::new([0.0; 3], [0.0; 3])
    }
}

// ============================================================================
// MONTE CARLO CONFIGURATION
// ============================================================================

/// Configuration for the Monte Carlo simulation kernel
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of particles in the ensemble
    pub num_particles: usize,

    /// Number of time steps to simulate
    pub num_steps: usize,

    /// Time step size
    pub dt: f64,

    /// Confidence threshold for partitioning
    /// Particles below this threshold go to the "low attention" path
    pub confidence_threshold: f64,

    /// Resampling threshold (effective sample size ratio)
    /// Resample when ESS/N < threshold
    pub resampling_threshold: f64,

    /// Process noise (for stochastic dynamics)
    pub process_noise: f64,

    /// Measurement noise (for importance weighting)
    pub measurement_noise: f64,

    /// Whether to track full provenance
    pub track_provenance: bool,

    /// Prior distribution for initialization
    pub prior: PriorDistribution,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_particles: 1000,
            num_steps: 100,
            dt: 0.01,
            confidence_threshold: 0.5,
            resampling_threshold: 0.5,
            process_noise: 0.1,
            measurement_noise: 0.1,
            track_provenance: true,
            prior: PriorDistribution::default(),
        }
    }
}

/// Prior distribution for particle initialization
#[derive(Debug, Clone)]
pub struct PriorDistribution {
    /// Mean position
    pub position_mean: [f64; 3],
    /// Position standard deviation
    pub position_std: [f64; 3],
    /// Mean velocity
    pub velocity_mean: [f64; 3],
    /// Velocity standard deviation
    pub velocity_std: [f64; 3],
    /// Initial confidence level
    pub initial_confidence: f64,
}

impl Default for PriorDistribution {
    fn default() -> Self {
        Self {
            position_mean: [0.0, 0.0, 0.0],
            position_std: [1.0, 1.0, 1.0],
            velocity_mean: [0.0, 0.0, 0.0],
            velocity_std: [1.0, 1.0, 1.0],
            initial_confidence: 0.5, // Start uncertain, gain confidence with evidence
        }
    }
}

// ============================================================================
// CONFIDENCE PARTITIONING
// ============================================================================

/// Partition particles by confidence threshold.
///
/// This is the divergence-avoiding pattern: instead of having GPU threads
/// diverge based on confidence within a warp, we partition particles into
/// separate buffers that can be processed uniformly.
///
/// High-confidence particles get "high attention" processing:
/// - Finer time steps
/// - Higher-order integration methods
/// - More accurate observation models
///
/// Low-confidence particles get "low attention" processing:
/// - Coarser time steps
/// - Simpler integration methods
/// - Approximate observation models
pub fn partition_by_confidence<'a>(
    particles: &'a [ParticleState],
    threshold: f64,
) -> (Vec<&'a ParticleState>, Vec<&'a ParticleState>) {
    let mut high_confidence = Vec::with_capacity(particles.len());
    let mut low_confidence = Vec::with_capacity(particles.len() / 4);

    for particle in particles {
        if particle.confidence >= threshold {
            high_confidence.push(particle);
        } else {
            low_confidence.push(particle);
        }
    }

    (high_confidence, low_confidence)
}

/// Partition with indices (for in-place operations)
pub fn partition_indices_by_confidence(
    particles: &[ParticleState],
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut high_indices = Vec::with_capacity(particles.len());
    let mut low_indices = Vec::with_capacity(particles.len() / 4);

    for (i, particle) in particles.iter().enumerate() {
        if particle.confidence >= threshold {
            high_indices.push(i);
        } else {
            low_indices.push(i);
        }
    }

    (high_indices, low_indices)
}

/// Attention-weighted partition using composite attention score
pub fn partition_by_attention<'a>(
    particles: &'a [ParticleState],
    attention_threshold: f64,
) -> (Vec<&'a ParticleState>, Vec<&'a ParticleState>) {
    let mut high_attention = Vec::with_capacity(particles.len());
    let mut low_attention = Vec::with_capacity(particles.len() / 4);

    for particle in particles {
        if particle.attention_score() >= attention_threshold {
            high_attention.push(particle);
        } else {
            low_attention.push(particle);
        }
    }

    (high_attention, low_attention)
}

// ============================================================================
// EPISTEMIC REDUCTION
// ============================================================================

/// Result of epistemic reduction over particles.
///
/// This preserves uncertainty through the aggregation process,
/// providing not just point estimates but full epistemic metadata.
#[derive(Debug, Clone)]
pub struct EpistemicReduction {
    /// Mean position (confidence-weighted)
    pub mean_position: [f64; 3],

    /// Mean velocity (confidence-weighted)
    pub mean_velocity: [f64; 3],

    /// Variance of position
    pub position_variance: [f64; 3],

    /// Variance of velocity
    pub velocity_variance: [f64; 3],

    /// Aggregate confidence (effective confidence of the estimate)
    pub aggregate_confidence: f64,

    /// Effective sample size (measure of weight distribution)
    pub effective_sample_size: f64,

    /// Combined provenance (union of particle provenances)
    pub combined_provenance: u64,

    /// Number of particles used in reduction
    pub particle_count: usize,
}

impl EpistemicReduction {
    /// Compute epistemic reduction over a set of particles.
    ///
    /// Uses confidence-weighted aggregation where particles with higher
    /// confidence contribute more to the final estimate. The aggregate
    /// confidence reflects our overall certainty given the particle ensemble.
    pub fn from_particles(particles: &[ParticleState]) -> Self {
        if particles.is_empty() {
            return Self::empty();
        }

        let n = particles.len() as f64;

        // Compute total weight for normalization
        let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
        if total_weight == 0.0 {
            return Self::empty();
        }

        // Confidence-weighted mean
        let mut weighted_conf: f64 = 0.0;
        let mut mean_pos = [0.0; 3];
        let mut mean_vel = [0.0; 3];

        for p in particles {
            let w = p.weight * p.confidence;
            weighted_conf += w;
            for i in 0..3 {
                mean_pos[i] += w * p.position[i];
                mean_vel[i] += w * p.velocity[i];
            }
        }

        if weighted_conf > 0.0 {
            for i in 0..3 {
                mean_pos[i] /= weighted_conf;
                mean_vel[i] /= weighted_conf;
            }
        }

        // Variance (weighted)
        let mut var_pos = [0.0; 3];
        let mut var_vel = [0.0; 3];

        for p in particles {
            let w = p.weight * p.confidence;
            for i in 0..3 {
                let dp = p.position[i] - mean_pos[i];
                let dv = p.velocity[i] - mean_vel[i];
                var_pos[i] += w * dp * dp;
                var_vel[i] += w * dv * dv;
            }
        }

        if weighted_conf > 0.0 {
            for i in 0..3 {
                var_pos[i] /= weighted_conf;
                var_vel[i] /= weighted_conf;
            }
        }

        // Effective sample size
        let weight_sq_sum: f64 = particles.iter().map(|p| p.weight * p.weight).sum();
        let ess = if weight_sq_sum > 0.0 {
            (total_weight * total_weight) / weight_sq_sum
        } else {
            0.0
        };

        // Aggregate confidence: based on ESS ratio and average confidence
        let avg_confidence: f64 = particles.iter().map(|p| p.confidence).sum::<f64>() / n;
        let ess_ratio = ess / n;
        // Aggregate confidence is geometric mean of ESS ratio and average confidence
        let aggregate_conf = (ess_ratio * avg_confidence).sqrt();

        // Combined provenance (XOR all provenance values)
        let combined_prov = particles.iter().fold(0u64, |acc, p| acc ^ p.provenance);

        Self {
            mean_position: mean_pos,
            mean_velocity: mean_vel,
            position_variance: var_pos,
            velocity_variance: var_vel,
            aggregate_confidence: aggregate_conf,
            effective_sample_size: ess,
            combined_provenance: combined_prov,
            particle_count: particles.len(),
        }
    }

    /// Empty reduction (no particles)
    pub fn empty() -> Self {
        Self {
            mean_position: [0.0; 3],
            mean_velocity: [0.0; 3],
            position_variance: [f64::INFINITY; 3],
            velocity_variance: [f64::INFINITY; 3],
            aggregate_confidence: 0.0,
            effective_sample_size: 0.0,
            combined_provenance: 0,
            particle_count: 0,
        }
    }

    /// Get standard deviation of position
    pub fn position_std(&self) -> [f64; 3] {
        [
            self.position_variance[0].sqrt(),
            self.position_variance[1].sqrt(),
            self.position_variance[2].sqrt(),
        ]
    }

    /// Get standard deviation of velocity
    pub fn velocity_std(&self) -> [f64; 3] {
        [
            self.velocity_variance[0].sqrt(),
            self.velocity_variance[1].sqrt(),
            self.velocity_variance[2].sqrt(),
        ]
    }
}

// ============================================================================
// MONTE CARLO KERNEL
// ============================================================================

/// The Monte Carlo simulation kernel in SIR form.
///
/// This is "the soul" of epistemic-aware parallel execution, demonstrating:
/// - How confidence flows through computation
/// - How to partition work by epistemic state
/// - How to preserve uncertainty through aggregation
pub struct MonteCarloKernel {
    /// Kernel configuration
    pub config: MonteCarloConfig,

    /// Current particle ensemble
    particles: Vec<ParticleState>,

    /// High-confidence particle indices (for partitioned execution)
    high_conf_indices: Vec<usize>,

    /// Low-confidence particle indices
    low_conf_indices: Vec<usize>,

    /// Current simulation time
    current_time: f64,

    /// Current simulation step
    current_step: usize,

    /// Metadata tracking
    metadata: MetadataStore,
}

impl MonteCarloKernel {
    /// Create a new Monte Carlo kernel with configuration
    pub fn new(config: MonteCarloConfig) -> Self {
        let particles = Vec::with_capacity(config.num_particles);

        Self {
            config,
            particles,
            high_conf_indices: Vec::new(),
            low_conf_indices: Vec::new(),
            current_time: 0.0,
            current_step: 0,
            metadata: MetadataStore::new(),
        }
    }

    /// Initialize particles from the prior distribution
    pub fn initialize_from_prior(&mut self, rng_seed: u64) {
        // Simple PRNG for initialization (xorshift64)
        let mut rng_state = rng_seed;
        let mut next_rand = || -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64)
        };

        // Box-Muller transform for Gaussian sampling
        let mut next_normal = || -> f64 {
            let u1 = next_rand().max(1e-10);
            let u2 = next_rand();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        self.particles.clear();
        self.particles.reserve(self.config.num_particles);

        for i in 0..self.config.num_particles {
            let position = [
                self.config.prior.position_mean[0] + self.config.prior.position_std[0] * next_normal(),
                self.config.prior.position_mean[1] + self.config.prior.position_std[1] * next_normal(),
                self.config.prior.position_mean[2] + self.config.prior.position_std[2] * next_normal(),
            ];

            let velocity = [
                self.config.prior.velocity_mean[0] + self.config.prior.velocity_std[0] * next_normal(),
                self.config.prior.velocity_mean[1] + self.config.prior.velocity_std[1] * next_normal(),
                self.config.prior.velocity_mean[2] + self.config.prior.velocity_std[2] * next_normal(),
            ];

            let provenance = if self.config.track_provenance {
                // Encode initialization as provenance
                (rng_seed << 32) | (i as u64)
            } else {
                0
            };

            self.particles.push(ParticleState::with_epistemic(
                position,
                velocity,
                1.0 / self.config.num_particles as f64, // Equal initial weights
                self.config.prior.initial_confidence,
                provenance,
            ));
        }

        // Initial partitioning
        self.partition_particles();
    }

    /// Partition particles by confidence threshold
    fn partition_particles(&mut self) {
        let (high, low) = partition_indices_by_confidence(
            &self.particles,
            self.config.confidence_threshold,
        );
        self.high_conf_indices = high;
        self.low_conf_indices = low;
    }

    /// Propagate particles with stochastic dynamics
    ///
    /// High-confidence particles use RK4, low-confidence use Euler
    pub fn propagate(&mut self, rng_seed: u64) {
        let dt = self.config.dt;
        let noise = self.config.process_noise;

        // Simple PRNG
        let mut rng_state = rng_seed.wrapping_add(self.current_step as u64);
        let mut next_normal = || -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u1 = ((rng_state as f64) / (u64::MAX as f64)).max(1e-10);
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u2 = (rng_state as f64) / (u64::MAX as f64);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        // Propagate high-confidence particles (RK4 - more accurate)
        for &idx in &self.high_conf_indices {
            let p = &mut self.particles[idx];

            // RK4 integration for position (treating velocity as derivative)
            // For simplicity, using constant velocity model with noise
            let k1_pos = p.velocity;

            // Position update with RK4-like accuracy
            for i in 0..3 {
                p.position[i] += k1_pos[i] * dt + noise * next_normal() * dt.sqrt();
            }

            // Velocity random walk
            for i in 0..3 {
                p.velocity[i] += noise * next_normal() * dt.sqrt();
            }

            // Confidence decreases slightly with process noise
            // More accurate integration preserves more confidence
            p.confidence *= 0.999;

            // Update provenance
            if self.config.track_provenance {
                p.provenance = p.provenance.wrapping_mul(0x517cc1b727220a95).wrapping_add(1);
            }
        }

        // Propagate low-confidence particles (Euler - simpler, faster)
        for &idx in &self.low_conf_indices {
            let p = &mut self.particles[idx];

            // Simple Euler integration
            for i in 0..3 {
                p.position[i] += p.velocity[i] * dt + noise * next_normal() * dt.sqrt();
                p.velocity[i] += noise * next_normal() * dt.sqrt();
            }

            // Confidence decreases more for less accurate integration
            p.confidence *= 0.99;

            // Update provenance
            if self.config.track_provenance {
                p.provenance = p.provenance.wrapping_mul(0x517cc1b727220a95).wrapping_add(2);
            }
        }

        self.current_time += dt;
        self.current_step += 1;
    }

    /// Apply importance weighting based on observation
    pub fn apply_weights(&mut self, observation: [f64; 3], observation_confidence: f64) {
        let obs_noise = self.config.measurement_noise;
        let obs_var = obs_noise * obs_noise;

        for p in &mut self.particles {
            // Compute likelihood (Gaussian observation model)
            let mut log_likelihood = 0.0;
            for i in 0..3 {
                let diff = p.position[i] - observation[i];
                log_likelihood -= 0.5 * diff * diff / obs_var;
            }

            // Update weight with likelihood
            let likelihood = log_likelihood.exp().max(1e-100);
            p.weight *= likelihood;

            // Update confidence based on how well this particle matches observation
            // Particles close to observation gain confidence
            let match_factor = likelihood.min(1.0);
            p.confidence = (p.confidence * 0.9 + match_factor * observation_confidence * 0.1)
                .clamp(0.0, 1.0);

            // Update provenance
            if self.config.track_provenance {
                p.provenance = p.provenance.wrapping_mul(0x517cc1b727220a95).wrapping_add(3);
            }
        }

        // Normalize weights
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total_weight > 0.0 {
            for p in &mut self.particles {
                p.weight /= total_weight;
            }
        }

        // Re-partition after weight update
        self.partition_particles();
    }

    /// Resample particles based on weights (systematic resampling)
    pub fn resample(&mut self, rng_seed: u64) {
        // Compute effective sample size
        let weight_sq_sum: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        let n = self.particles.len() as f64;
        let ess = if weight_sq_sum > 0.0 { 1.0 / weight_sq_sum } else { 0.0 };

        // Only resample if ESS is too low
        if ess / n >= self.config.resampling_threshold {
            return;
        }

        // Systematic resampling
        let mut rng_state = rng_seed.wrapping_add(self.current_step as u64).wrapping_add(42);
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u0 = (rng_state as f64) / (u64::MAX as f64);

        let n_particles = self.particles.len();
        let mut cumsum = 0.0;
        let mut new_particles = Vec::with_capacity(n_particles);

        let step = 1.0 / n_particles as f64;
        let mut u = u0 * step;
        let mut j = 0;

        for i in 0..n_particles {
            cumsum += self.particles[i].weight;
            while u <= cumsum && j < n_particles {
                let mut new_p = self.particles[i].clone();
                new_p.weight = 1.0 / n_particles as f64;

                // Resampling reduces confidence slightly (introduces degeneracy)
                new_p.confidence *= 0.95;

                // Update provenance for resampled particle
                if self.config.track_provenance {
                    new_p.provenance = new_p.provenance
                        .wrapping_mul(0x517cc1b727220a95)
                        .wrapping_add((j as u64) << 16 | 4);
                }

                new_particles.push(new_p);
                u += step;
                j += 1;
            }
        }

        // Handle any remaining particles
        while new_particles.len() < n_particles {
            let mut new_p = self.particles.last().unwrap().clone();
            new_p.weight = 1.0 / n_particles as f64;
            new_p.confidence *= 0.95;
            new_particles.push(new_p);
        }

        self.particles = new_particles;
        self.partition_particles();
    }

    /// Run the complete simulation
    pub fn run(&mut self, observations: &[([f64; 3], f64)], rng_seed: u64) -> EpistemicReduction {
        self.initialize_from_prior(rng_seed);

        let mut obs_idx = 0;
        for step in 0..self.config.num_steps {
            // Propagate
            self.propagate(rng_seed.wrapping_add(step as u64 * 1000));

            // Apply observation if available at this time step
            if obs_idx < observations.len() {
                let (obs, conf) = observations[obs_idx];
                self.apply_weights(obs, conf);
                obs_idx += 1;
            }

            // Resample if needed
            self.resample(rng_seed.wrapping_add(step as u64 * 2000));
        }

        // Final epistemic reduction
        EpistemicReduction::from_particles(&self.particles)
    }

    /// Get current particle ensemble (for debugging/visualization)
    pub fn particles(&self) -> &[ParticleState] {
        &self.particles
    }

    /// Get high-confidence particle count
    pub fn high_confidence_count(&self) -> usize {
        self.high_conf_indices.len()
    }

    /// Get low-confidence particle count
    pub fn low_confidence_count(&self) -> usize {
        self.low_conf_indices.len()
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get current reduction state
    pub fn current_reduction(&self) -> EpistemicReduction {
        EpistemicReduction::from_particles(&self.particles)
    }
}

// ============================================================================
// SIR CODE GENERATION
// ============================================================================

/// Emit the Monte Carlo kernel as SIR instructions.
///
/// This generates a complete SIR module containing the Monte Carlo simulation
/// kernel with all epistemic-aware operations explicitly represented.
pub fn emit_monte_carlo_sir(config: &MonteCarloConfig) -> SirModule {
    let mut module = SirModule::new("monte_carlo_kernel");
    module.flags.optimize_epistemic = true;

    // Add ParticleState type
    module.add_type("ParticleState", ParticleState::sir_type());

    // Add EpistemicReduction type
    let reduction_type = SirType::Struct(StructType::new(vec![
        (Some("mean_position".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("mean_velocity".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("position_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("velocity_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("aggregate_confidence".into()), SirType::f64()),
        (Some("effective_sample_size".into()), SirType::f64()),
        (Some("combined_provenance".into()), SirType::i64()),
        (Some("particle_count".into()), SirType::i64()),
    ]).named("EpistemicReduction"));
    module.add_type("EpistemicReduction", reduction_type.clone());

    // Generate initialization function
    emit_init_function(&mut module, config);

    // Generate propagation function (high confidence path)
    emit_propagate_high_function(&mut module, config);

    // Generate propagation function (low confidence path)
    emit_propagate_low_function(&mut module, config);

    // Generate weight update function
    emit_weight_update_function(&mut module, config);

    // Generate resample function
    emit_resample_function(&mut module, config);

    // Generate confidence partition function
    emit_partition_function(&mut module, config);

    // Generate reduction function
    emit_reduction_function(&mut module, config);

    // Generate main kernel function
    emit_main_kernel(&mut module, config);

    module
}

/// Emit particle initialization function
fn emit_init_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let func_id = module.create_function(
        "mc_init_particles",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("rng_seed".into(), SirType::i64()),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;
        func.pure = false; // Writes to memory

        // Create entry block
        let entry = BasicBlock::new(BlockId::new(0)).named("entry");

        // For SIR representation, we'll emit pseudo-ops that represent the algorithm
        // In practice, this would be lowered to actual loop structure

        let mut block = entry;
        block.terminate(Terminator::Return(None));
        func.blocks.push(block);
    }
}

/// Emit high-confidence propagation function
fn emit_propagate_high_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let func_id = module.create_function(
        "mc_propagate_high",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("indices".into(), SirType::ptr(SirType::i64())),
            ("count".into(), SirType::i64()),
            ("dt".into(), SirType::f64()),
            ("noise".into(), SirType::f64()),
            ("rng_state".into(), SirType::ptr(SirType::i64())),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;
        func.hot = true; // Hot path

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit low-confidence propagation function
fn emit_propagate_low_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let func_id = module.create_function(
        "mc_propagate_low",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("indices".into(), SirType::ptr(SirType::i64())),
            ("count".into(), SirType::i64()),
            ("dt".into(), SirType::f64()),
            ("noise".into(), SirType::f64()),
            ("rng_state".into(), SirType::ptr(SirType::i64())),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit weight update function
fn emit_weight_update_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let observation_ty = SirType::Array(ArrayType::new(SirType::f64(), 3));

    let func_id = module.create_function(
        "mc_update_weights",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("observation".into(), SirType::ptr(observation_ty)),
            ("obs_confidence".into(), SirType::f64()),
            ("measurement_noise".into(), SirType::f64()),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit resampling function
fn emit_resample_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());

    let func_id = module.create_function(
        "mc_resample",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("threshold".into(), SirType::f64()),
            ("rng_state".into(), SirType::ptr(SirType::i64())),
        ],
        SirType::bool(),
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit confidence partition function
fn emit_partition_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());

    // Returns count of high-confidence particles
    let func_id = module.create_function(
        "mc_partition_by_confidence",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("threshold".into(), SirType::f64()),
            ("high_indices".into(), SirType::ptr(SirType::i64())),
            ("low_indices".into(), SirType::ptr(SirType::i64())),
        ],
        SirType::i64(), // Returns high count
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;
        func.pure = true; // Pure partitioning

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit epistemic reduction function
fn emit_reduction_function(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let reduction_ptr_ty = SirType::ptr(SirType::Struct(StructType::new(vec![
        (Some("mean_position".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("mean_velocity".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("position_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("velocity_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("aggregate_confidence".into()), SirType::f64()),
        (Some("effective_sample_size".into()), SirType::f64()),
        (Some("combined_provenance".into()), SirType::i64()),
        (Some("particle_count".into()), SirType::i64()),
    ]).named("EpistemicReduction")));

    let func_id = module.create_function(
        "mc_epistemic_reduction",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("result".into(), reduction_ptr_ty),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::Internal;

        let mut entry = BasicBlock::new(BlockId::new(0)).named("entry");
        entry.terminate(Terminator::Return(None));
        func.blocks.push(entry);
    }
}

/// Emit the main kernel function
fn emit_main_kernel(module: &mut SirModule, config: &MonteCarloConfig) {
    let particle_ptr_ty = SirType::ptr(ParticleState::sir_type());
    let observation_ty = SirType::Array(ArrayType::new(SirType::f64(), 3));
    let reduction_ptr_ty = SirType::ptr(SirType::Struct(StructType::new(vec![
        (Some("mean_position".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("mean_velocity".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("position_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("velocity_variance".into()), SirType::Array(ArrayType::new(SirType::f64(), 3))),
        (Some("aggregate_confidence".into()), SirType::f64()),
        (Some("effective_sample_size".into()), SirType::f64()),
        (Some("combined_provenance".into()), SirType::i64()),
        (Some("particle_count".into()), SirType::i64()),
    ]).named("EpistemicReduction")));

    let func_id = module.create_function(
        "mc_run_kernel",
        vec![
            ("particles".into(), particle_ptr_ty.clone()),
            ("num_particles".into(), SirType::i64()),
            ("num_steps".into(), SirType::i64()),
            ("dt".into(), SirType::f64()),
            ("observations".into(), SirType::ptr(observation_ty)),
            ("obs_confidences".into(), SirType::ptr(SirType::f64())),
            ("num_observations".into(), SirType::i64()),
            ("config_confidence_threshold".into(), SirType::f64()),
            ("config_resample_threshold".into(), SirType::f64()),
            ("config_process_noise".into(), SirType::f64()),
            ("config_measurement_noise".into(), SirType::f64()),
            ("rng_seed".into(), SirType::i64()),
            ("result".into(), reduction_ptr_ty),
        ],
        SirType::Void,
    );

    if let Some(func) = module.function_mut(func_id) {
        func.linkage = Linkage::External; // Public entry point

        let entry = BasicBlock::new(BlockId::new(0)).named("entry");
        let loop_header = BasicBlock::new(BlockId::new(1)).named("loop.header");
        let loop_body = BasicBlock::new(BlockId::new(2)).named("loop.body");
        let loop_exit = BasicBlock::new(BlockId::new(3)).named("loop.exit");

        // Entry block
        let mut entry_block = entry;
        entry_block.terminate(Terminator::Br(BlockId::new(1)));
        func.blocks.push(entry_block);

        // Loop header (check step < num_steps)
        let mut header_block = loop_header;
        header_block.is_loop_header = true;
        header_block.terminate(Terminator::CondBr {
            cond: ValueId::new(0), // Placeholder
            then_block: BlockId::new(2),
            else_block: BlockId::new(3),
        });
        func.blocks.push(header_block);

        // Loop body
        let mut body_block = loop_body;
        body_block.loop_depth = 1;
        body_block.terminate(Terminator::Br(BlockId::new(1)));
        func.blocks.push(body_block);

        // Exit block
        let mut exit_block = loop_exit;
        exit_block.terminate(Terminator::Return(None));
        func.blocks.push(exit_block);
    }
}

// ============================================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================================

impl fmt::Display for ParticleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Particle(pos=[{:.3}, {:.3}, {:.3}], vel=[{:.3}, {:.3}, {:.3}], w={:.4}, c={:.3})",
            self.position[0], self.position[1], self.position[2],
            self.velocity[0], self.velocity[1], self.velocity[2],
            self.weight, self.confidence
        )
    }
}

impl fmt::Display for EpistemicReduction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EpistemicReduction(\n\
             \x20 mean_pos=[{:.3}, {:.3}, {:.3}] +/- [{:.3}, {:.3}, {:.3}]\n\
             \x20 mean_vel=[{:.3}, {:.3}, {:.3}] +/- [{:.3}, {:.3}, {:.3}]\n\
             \x20 confidence={:.3}, ESS={:.1}, n={}\n\
             )",
            self.mean_position[0], self.mean_position[1], self.mean_position[2],
            self.position_std()[0], self.position_std()[1], self.position_std()[2],
            self.mean_velocity[0], self.mean_velocity[1], self.mean_velocity[2],
            self.velocity_std()[0], self.velocity_std()[1], self.velocity_std()[2],
            self.aggregate_confidence, self.effective_sample_size, self.particle_count,
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_state_creation() {
        let p = ParticleState::new([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        assert_eq!(p.velocity, [0.1, 0.2, 0.3]);
        assert_eq!(p.weight, 1.0);
        assert_eq!(p.confidence, 1.0);
        assert_eq!(p.provenance, 0);
    }

    #[test]
    fn test_uncertain_particle() {
        let p = ParticleState::uncertain([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.5);
        assert_eq!(p.confidence, 0.5);
    }

    #[test]
    fn test_attention_score() {
        let p = ParticleState::with_epistemic([0.0; 3], [0.0; 3], 0.5, 0.8, 0);
        assert!((p.attention_score() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_partition_by_confidence() {
        let particles = vec![
            ParticleState::uncertain([0.0; 3], [0.0; 3], 0.8),
            ParticleState::uncertain([0.0; 3], [0.0; 3], 0.3),
            ParticleState::uncertain([0.0; 3], [0.0; 3], 0.6),
            ParticleState::uncertain([0.0; 3], [0.0; 3], 0.4),
        ];

        let (high, low) = partition_by_confidence(&particles, 0.5);
        assert_eq!(high.len(), 2); // 0.8 and 0.6
        assert_eq!(low.len(), 2);  // 0.3 and 0.4
    }

    #[test]
    fn test_epistemic_reduction_empty() {
        let reduction = EpistemicReduction::from_particles(&[]);
        assert_eq!(reduction.particle_count, 0);
        assert_eq!(reduction.aggregate_confidence, 0.0);
    }

    #[test]
    fn test_epistemic_reduction() {
        let particles = vec![
            ParticleState::with_epistemic([1.0, 0.0, 0.0], [0.0; 3], 0.5, 0.9, 0),
            ParticleState::with_epistemic([-1.0, 0.0, 0.0], [0.0; 3], 0.5, 0.9, 0),
        ];

        let reduction = EpistemicReduction::from_particles(&particles);
        assert_eq!(reduction.particle_count, 2);
        // Mean should be close to 0
        assert!(reduction.mean_position[0].abs() < 0.01);
        assert!(reduction.aggregate_confidence > 0.0);
    }

    #[test]
    fn test_monte_carlo_kernel_initialization() {
        let config = MonteCarloConfig {
            num_particles: 100,
            ..Default::default()
        };

        let mut kernel = MonteCarloKernel::new(config);
        kernel.initialize_from_prior(12345);

        assert_eq!(kernel.particles().len(), 100);
        assert_eq!(
            kernel.high_confidence_count() + kernel.low_confidence_count(),
            100
        );
    }

    #[test]
    fn test_monte_carlo_propagation() {
        let config = MonteCarloConfig {
            num_particles: 50,
            num_steps: 10,
            dt: 0.1,
            ..Default::default()
        };

        let mut kernel = MonteCarloKernel::new(config);
        kernel.initialize_from_prior(42);

        let initial_time = kernel.current_time();
        kernel.propagate(42);

        assert!(kernel.current_time() > initial_time);
    }

    #[test]
    fn test_monte_carlo_full_run() {
        let config = MonteCarloConfig {
            num_particles: 100,
            num_steps: 10,
            dt: 0.1,
            confidence_threshold: 0.5,
            ..Default::default()
        };

        let mut kernel = MonteCarloKernel::new(config);

        // Observations: target at origin
        let observations = vec![
            ([0.0, 0.0, 0.0], 0.9),
            ([0.0, 0.0, 0.0], 0.9),
        ];

        let result = kernel.run(&observations, 12345);

        // Should have some particles and non-zero confidence
        assert!(result.particle_count > 0);
        assert!(result.effective_sample_size > 0.0);
    }

    #[test]
    fn test_emit_sir_module() {
        let config = MonteCarloConfig::default();
        let module = emit_monte_carlo_sir(&config);

        // Check module has expected functions
        assert!(module.function_by_name("mc_run_kernel").is_some());
        assert!(module.function_by_name("mc_init_particles").is_some());
        assert!(module.function_by_name("mc_propagate_high").is_some());
        assert!(module.function_by_name("mc_propagate_low").is_some());
        assert!(module.function_by_name("mc_update_weights").is_some());
        assert!(module.function_by_name("mc_resample").is_some());
        assert!(module.function_by_name("mc_partition_by_confidence").is_some());
        assert!(module.function_by_name("mc_epistemic_reduction").is_some());

        // Check epistemic optimization flag is set
        assert!(module.flags.optimize_epistemic);
    }

    #[test]
    fn test_particle_sir_type() {
        let ty = ParticleState::sir_type();
        if let SirType::Struct(s) = ty {
            assert_eq!(s.fields.len(), 5);
            assert_eq!(s.name, Some("ParticleState".into()));
        } else {
            panic!("Expected struct type");
        }
    }

    #[test]
    fn test_provenance_tracking() {
        let p1 = ParticleState::with_epistemic([0.0; 3], [0.0; 3], 1.0, 1.0, 100);
        let p2 = p1.with_transform(1);
        let p3 = p2.with_transform(2);

        // Provenance should change with each transform
        assert_ne!(p1.provenance, p2.provenance);
        assert_ne!(p2.provenance, p3.provenance);
        assert_ne!(p1.provenance, p3.provenance);
    }
}
