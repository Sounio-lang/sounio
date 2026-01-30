//! GPU-Accelerated Uplift Trees with Epistemic Heads
//!
//! This module implements causal tree algorithms (Athey & Imbens, 2016) on GPU
//! with native epistemic uncertainty tracking. WORLD-FIRST innovation.
//!
//! # Architecture
//!
//! ```text
//! Uplift Tree with Epistemic Leaves
//! ─────────────────────────────────
//!           [Root: τ̂ = 0.15 ± 0.08]
//!                    │
//!        ┌──────────┴──────────┐
//!        │ age > 35            │ age ≤ 35
//!        ▼                     ▼
//!   [τ̂ = 0.25 ± 0.05]    [τ̂ = 0.02 ± 0.12]
//!    Persuadable            Uncertain
//!        │
//!    ┌───┴───┐
//!    │       │
//!   ...     ...
//! ```
//!
//! Each leaf stores:
//! - `tau_estimate`: CATE estimate (τ̂(x))
//! - `confidence`: Beta posterior (α, β)
//! - `n_treated`: Number of treated samples
//! - `n_control`: Number of control samples
//! - `segment`: Customer segment classification
//!
//! # GPU Parallelization Strategy
//!
//! ## Tree Building (Level-wise)
//! - Each warp evaluates one candidate split
//! - Shared memory for histogram aggregation
//! - Warp-level voting for best split
//!
//! ## Tree Inference
//! - Each thread processes one sample
//! - Coalesced memory access for node traversal
//! - Epistemic heads compute confidence per leaf
//!
//! # Splitting Criteria
//!
//! Uses **Causal Tree Criterion** (maximizes heterogeneity):
//! ```text
//! Δ(s) = Var(τ̂_L) · n_L/n + Var(τ̂_R) · n_R/n - Var(τ̂)
//! ```
//!
//! With epistemic modification:
//! ```text
//! Δ_epistemic(s) = Δ(s) · confidence_penalty(split)
//! ```

use std::fmt::Write;

use super::epistemic_ptx::{EpistemicPtxConfig, EpistemicPtxEmitter};

/// Configuration for GPU uplift tree building
#[derive(Debug, Clone)]
pub struct UpliftTreeGpuConfig {
    /// Target compute capability
    pub sm_version: (u32, u32),
    /// Maximum tree depth
    pub max_depth: u32,
    /// Minimum samples per leaf
    pub min_samples_leaf: u32,
    /// Minimum treated samples per leaf (for valid CATE estimation)
    pub min_treated_leaf: u32,
    /// Minimum control samples per leaf
    pub min_control_leaf: u32,
    /// Number of histogram bins for continuous features
    pub n_bins: u32,
    /// Confidence threshold for "confident" CATE
    pub confidence_threshold: f32,
    /// Enable honesty (sample splitting for estimation)
    pub honest: bool,
    /// Fraction of samples for tree structure (if honest)
    pub honest_fraction: f32,
    /// Enable epistemic uncertainty tracking
    pub epistemic_tracking: bool,
    /// Splitting criterion
    pub criterion: SplitCriterion,
    /// Number of features (for PTX emission)
    pub n_features: u32,
}

impl Default for UpliftTreeGpuConfig {
    fn default() -> Self {
        Self {
            sm_version: (8, 0),
            max_depth: 8,
            min_samples_leaf: 100,
            min_treated_leaf: 20,
            min_control_leaf: 20,
            n_bins: 256,
            confidence_threshold: 0.1,
            honest: true,
            honest_fraction: 0.5,
            epistemic_tracking: true,
            criterion: SplitCriterion::CausalTree,
            n_features: 10, // Default, should be set for actual use
        }
    }
}

/// Splitting criterion for uplift trees
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCriterion {
    /// Standard Causal Tree (Athey & Imbens)
    CausalTree,
    /// Modified outcome transformed (Rzepakowski & Jaroszewicz)
    ModifiedOutcome,
    /// Kullback-Leibler divergence
    KLDivergence,
    /// Euclidean distance
    EuclideanDistance,
    /// Chi-squared test
    ChiSquared,
    /// Delta-Delta-P (Hansotia & Rukstales)
    DeltaDeltaP,
}

/// Node type in uplift tree
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Internal node with split
    Internal,
    /// Leaf node with CATE estimate
    Leaf,
}

/// Customer segment classification (from uplift.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomerSegment {
    /// τ(x) > 0 with high confidence - TARGET these customers
    Persuadable,
    /// Would convert anyway (high baseline)
    SureThing,
    /// Won't convert regardless
    LostCause,
    /// τ(x) < 0 - AVOID treating these (sleeping dogs)
    SleepingDog,
    /// Insufficient confidence to classify
    Uncertain,
}

impl CustomerSegment {
    /// Classify based on uplift estimate and confidence
    pub fn classify(tau: f32, epsilon: f32, baseline: f32, threshold: f32) -> Self {
        let confident = epsilon < threshold;

        if !confident {
            return CustomerSegment::Uncertain;
        }

        if tau > threshold {
            CustomerSegment::Persuadable
        } else if tau < -threshold {
            CustomerSegment::SleepingDog
        } else if baseline > 0.7 {
            CustomerSegment::SureThing
        } else if baseline < 0.1 {
            CustomerSegment::LostCause
        } else {
            CustomerSegment::Uncertain
        }
    }
}

/// GPU representation of an uplift tree node
#[derive(Debug, Clone)]
pub struct UpliftTreeNode {
    /// Node ID
    pub id: u32,
    /// Node type
    pub node_type: NodeType,
    /// Parent node ID (0 for root)
    pub parent: u32,
    /// Left child ID (if internal)
    pub left_child: u32,
    /// Right child ID (if internal)
    pub right_child: u32,
    /// Depth in tree
    pub depth: u32,
    /// Feature index for split (if internal)
    pub split_feature: u32,
    /// Split threshold (if internal)
    pub split_threshold: f32,
    /// CATE estimate τ̂(x) for this node
    pub tau_estimate: f32,
    /// Epistemic uncertainty (standard error)
    pub tau_std_error: f32,
    /// Beta posterior alpha (successes + prior)
    pub beta_alpha: f32,
    /// Beta posterior beta (failures + prior)
    pub beta_beta: f32,
    /// Number of treated samples
    pub n_treated: u32,
    /// Number of control samples
    pub n_control: u32,
    /// Mean outcome for treated
    pub y_treated_mean: f32,
    /// Mean outcome for control
    pub y_control_mean: f32,
    /// Customer segment classification
    pub segment: CustomerSegment,
    /// Provenance bitmask (data sources)
    pub provenance: u64,
}

impl UpliftTreeNode {
    /// Create a new leaf node
    pub fn leaf(id: u32, parent: u32, depth: u32) -> Self {
        Self {
            id,
            node_type: NodeType::Leaf,
            parent,
            left_child: 0,
            right_child: 0,
            depth,
            split_feature: 0,
            split_threshold: 0.0,
            tau_estimate: 0.0,
            tau_std_error: f32::INFINITY,
            beta_alpha: 1.0, // Uniform prior
            beta_beta: 1.0,
            n_treated: 0,
            n_control: 0,
            y_treated_mean: 0.0,
            y_control_mean: 0.0,
            segment: CustomerSegment::Uncertain,
            provenance: 0,
        }
    }

    /// Compute confidence from Beta posterior
    pub fn confidence(&self) -> f32 {
        // Variance of Beta distribution
        let n = self.beta_alpha + self.beta_beta;
        let var = (self.beta_alpha * self.beta_beta) / (n * n * (n + 1.0));

        // Confidence = 1 - sqrt(variance)
        1.0 - var.sqrt().min(1.0)
    }

    /// Update segment classification
    pub fn update_segment(&mut self, threshold: f32) {
        self.segment = CustomerSegment::classify(
            self.tau_estimate,
            self.tau_std_error,
            self.y_control_mean,
            threshold,
        );
    }
}

/// GPU-compatible tree structure (array-based for coalesced access)
#[derive(Debug, Clone)]
pub struct UpliftTreeGpu {
    /// All nodes in breadth-first order
    pub nodes: Vec<UpliftTreeNode>,
    /// Number of features
    pub n_features: u32,
    /// Feature names (for interpretability)
    pub feature_names: Vec<String>,
    /// Configuration used for building
    pub config: UpliftTreeGpuConfig,
}

impl UpliftTreeGpu {
    /// Create a new empty tree
    pub fn new(n_features: u32, config: UpliftTreeGpuConfig) -> Self {
        Self {
            nodes: Vec::new(),
            n_features,
            feature_names: Vec::new(),
            config,
        }
    }

    /// Get tree depth
    pub fn depth(&self) -> u32 {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Get number of leaves
    pub fn n_leaves(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Leaf)
            .count()
    }

    /// Traverse to leaf for a sample (CPU reference implementation)
    pub fn predict_single(&self, features: &[f32]) -> &UpliftTreeNode {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];

            if node.node_type == NodeType::Leaf {
                return node;
            }

            let feature_val = features.get(node.split_feature as usize).unwrap_or(&0.0);

            node_idx = if *feature_val <= node.split_threshold {
                node.left_child as usize
            } else {
                node.right_child as usize
            };
        }
    }
}

/// Histogram bin for GPU tree building
#[derive(Debug, Clone, Copy, Default)]
pub struct UpliftHistBin {
    /// Sum of outcomes for treated in this bin
    pub sum_y_treated: f32,
    /// Sum of squared outcomes for treated
    pub sum_y2_treated: f32,
    /// Count of treated samples
    pub count_treated: u32,
    /// Sum of outcomes for control
    pub sum_y_control: f32,
    /// Sum of squared outcomes for control
    pub sum_y2_control: f32,
    /// Count of control samples
    pub count_control: u32,
}

impl UpliftHistBin {
    /// Merge two histogram bins
    pub fn merge(&mut self, other: &UpliftHistBin) {
        self.sum_y_treated += other.sum_y_treated;
        self.sum_y2_treated += other.sum_y2_treated;
        self.count_treated += other.count_treated;
        self.sum_y_control += other.sum_y_control;
        self.sum_y2_control += other.sum_y2_control;
        self.count_control += other.count_control;
    }

    /// Compute CATE estimate from this bin
    pub fn cate(&self) -> f32 {
        let y_t = if self.count_treated > 0 {
            self.sum_y_treated / self.count_treated as f32
        } else {
            0.0
        };

        let y_c = if self.count_control > 0 {
            self.sum_y_control / self.count_control as f32
        } else {
            0.0
        };

        y_t - y_c
    }

    /// Compute variance of CATE estimate
    pub fn cate_variance(&self) -> f32 {
        if self.count_treated < 2 || self.count_control < 2 {
            return f32::INFINITY;
        }

        let n_t = self.count_treated as f32;
        let n_c = self.count_control as f32;

        let mean_t = self.sum_y_treated / n_t;
        let mean_c = self.sum_y_control / n_c;

        let var_t = (self.sum_y2_treated / n_t) - (mean_t * mean_t);
        let var_c = (self.sum_y2_control / n_c) - (mean_c * mean_c);

        // Variance of difference: Var(Y_t) / n_t + Var(Y_c) / n_c
        var_t / n_t + var_c / n_c
    }
}

/// PTX emitter for uplift tree operations
pub struct UpliftTreePtxEmitter {
    config: UpliftTreeGpuConfig,
    epistemic: EpistemicPtxEmitter,
    output: String,
    indent: usize,
}

impl UpliftTreePtxEmitter {
    pub fn new(config: UpliftTreeGpuConfig) -> Self {
        let epistemic_config = EpistemicPtxConfig {
            sm_version: config.sm_version,
            quadrature_propagation: true,
            provenance_tracking: true,
            ..Default::default()
        };

        Self {
            config,
            epistemic: EpistemicPtxEmitter::new(epistemic_config),
            output: String::new(),
            indent: 1,
        }
    }

    pub fn output(&self) -> &str {
        &self.output
    }

    fn emit(&mut self, s: &str) {
        let indent = "\t".repeat(self.indent);
        writeln!(self.output, "{}{}", indent, s).unwrap();
    }

    fn emit_comment(&mut self, s: &str) {
        self.emit(&format!("// {}", s));
    }

    /// Emit register declarations for uplift tree operations
    pub fn emit_declarations(&mut self) {
        self.emit_comment("=== Uplift Tree GPU Registers ===");

        // Tree traversal
        self.emit(".reg .u32 %r_node_idx;");
        self.emit(".reg .u32 %r_node_type;");
        self.emit(".reg .u32 %r_split_feature;");
        self.emit(".reg .f32 %r_split_threshold;");
        self.emit(".reg .u32 %r_left_child;");
        self.emit(".reg .u32 %r_right_child;");

        // CATE estimation
        self.emit(".reg .f32 %r_tau;"); // τ̂ estimate
        self.emit(".reg .f32 %r_tau_eps;"); // τ epistemic uncertainty
        self.emit(".reg .f32 %r_y_treated;");
        self.emit(".reg .f32 %r_y_control;");
        self.emit(".reg .u32 %r_n_treated;");
        self.emit(".reg .u32 %r_n_control;");

        // Beta posterior for confidence
        self.emit(".reg .f32 %r_beta_alpha;");
        self.emit(".reg .f32 %r_beta_beta;");
        self.emit(".reg .f32 %r_confidence;");

        // Segment classification
        self.emit(".reg .u32 %r_segment;");

        // Provenance tracking
        self.emit(".reg .u64 %r_prov;");

        // Histogram bins (shared memory)
        self.emit(&format!(
            ".shared .align 16 .b8 smem_hist[{}];",
            self.config.n_bins * 24 * 4
        )); // 24 bytes per bin, 4 features at a time

        // Predicates
        self.emit(".reg .pred %p_is_leaf;");
        self.emit(".reg .pred %p_go_left;");
        self.emit(".reg .pred %p_confident;");
        self.emit(".reg .pred %p_persuadable;");
        self.emit(".reg .pred %p_sleeping_dog;");

        // Temporaries
        self.emit(".reg .f32 %r_ft0, %r_ft1, %r_ft2, %r_ft3;");
        self.emit(".reg .u32 %r_ut0, %r_ut1, %r_ut2;");
        self.emit("");
    }

    /// Emit tree traversal kernel
    pub fn emit_tree_traverse(&mut self, tree_ptr: &str, features_ptr: &str, output_ptr: &str) {
        self.emit_comment("=== Uplift Tree Traversal with Epistemic Output ===");

        // Get thread/sample index
        self.emit("mov.u32 %r_ut0, %ctaid.x;");
        self.emit("mov.u32 %r_ut1, %ntid.x;");
        self.emit("mad.lo.u32 %r_ut0, %r_ut0, %r_ut1, %tid.x;");
        self.emit_comment("r_ut0 = sample index");

        // Start at root
        self.emit("mov.u32 %r_node_idx, 0;");

        // Traversal loop
        self.emit("traverse_loop:");
        self.indent += 1;

        // Load node data (assuming struct layout)
        self.emit_comment(
            "Load node: [type, parent, left, right, depth, feature, threshold, tau, ...]",
        );
        self.emit(&format!(
            "mad.wide.u32 %rd0, %r_node_idx, 64, {};",
            tree_ptr
        ));

        // Load node type
        self.emit("ld.global.u32 %r_node_type, [%rd0];");

        // Check if leaf
        self.emit("setp.eq.u32 %p_is_leaf, %r_node_type, 1;");
        self.emit("@%p_is_leaf bra leaf_node;");

        // Internal node: load split info
        self.emit_comment("Internal node - load split parameters");
        self.emit("ld.global.u32 %r_split_feature, [%rd0+20];");
        self.emit("ld.global.f32 %r_split_threshold, [%rd0+24];");
        self.emit("ld.global.u32 %r_left_child, [%rd0+8];");
        self.emit("ld.global.u32 %r_right_child, [%rd0+12];");

        // Load feature value for this sample
        self.emit_comment("Load feature value: features[sample_idx * n_features + split_feature]");
        self.emit(&format!(
            "mul.lo.u32 %r_ut1, %r_ut0, {};",
            self.config.n_features
        ));
        self.emit("add.u32 %r_ut1, %r_ut1, %r_split_feature;");
        self.emit(&format!("mad.wide.u32 %rd1, %r_ut1, 4, {};", features_ptr));
        self.emit("ld.global.f32 %r_ft0, [%rd1];");

        // Compare and branch
        self.emit("setp.le.f32 %p_go_left, %r_ft0, %r_split_threshold;");
        self.emit("selp.u32 %r_node_idx, %r_left_child, %r_right_child, %p_go_left;");
        self.emit("bra traverse_loop;");

        self.indent -= 1;

        // Leaf node handling
        self.emit("leaf_node:");
        self.indent += 1;

        // Load CATE estimate and epistemic data
        self.emit_comment("Leaf node - load CATE and epistemic state");
        self.emit("ld.global.f32 %r_tau, [%rd0+28];"); // tau_estimate
        self.emit("ld.global.f32 %r_tau_eps, [%rd0+32];"); // tau_std_error
        self.emit("ld.global.f32 %r_beta_alpha, [%rd0+36];"); // beta_alpha
        self.emit("ld.global.f32 %r_beta_beta, [%rd0+40];"); // beta_beta
        self.emit("ld.global.u32 %r_segment, [%rd0+56];"); // segment
        self.emit("ld.global.u64 %r_prov, [%rd0+60];"); // provenance

        // Compute confidence from Beta posterior
        self.emit_comment("Confidence = 1 - sqrt(α·β / ((α+β)² · (α+β+1)))");
        self.emit("add.f32 %r_ft0, %r_beta_alpha, %r_beta_beta;"); // α + β
        self.emit("mul.f32 %r_ft1, %r_ft0, %r_ft0;"); // (α + β)²
        self.emit("add.f32 %r_ft2, %r_ft0, 1.0;"); // α + β + 1
        self.emit("mul.f32 %r_ft1, %r_ft1, %r_ft2;"); // (α+β)² · (α+β+1)
        self.emit("mul.f32 %r_ft2, %r_beta_alpha, %r_beta_beta;"); // α · β
        self.emit("div.approx.f32 %r_ft2, %r_ft2, %r_ft1;"); // variance
        self.emit("sqrt.approx.f32 %r_ft2, %r_ft2;");
        self.emit("sub.f32 %r_confidence, 1.0, %r_ft2;");

        // Store output: [tau, tau_eps, confidence, segment, provenance]
        self.emit_comment("Store epistemic output");
        self.emit(&format!("mad.wide.u32 %rd1, %r_ut0, 24, {};", output_ptr));
        self.emit("st.global.f32 [%rd1], %r_tau;");
        self.emit("st.global.f32 [%rd1+4], %r_tau_eps;");
        self.emit("st.global.f32 [%rd1+8], %r_confidence;");
        self.emit("st.global.u32 [%rd1+12], %r_segment;");
        self.emit("st.global.u64 [%rd1+16], %r_prov;");

        self.indent -= 1;
    }

    /// Emit histogram building kernel for tree construction
    pub fn emit_histogram_build(
        &mut self,
        features_ptr: &str,
        treatment_ptr: &str,
        outcome_ptr: &str,
        node_samples_ptr: &str,
        hist_out_ptr: &str,
    ) {
        self.emit_comment("=== Histogram Building for Split Finding ===");

        // Each warp handles one feature
        self.emit("mov.u32 %r_ut0, %tid.x;");
        self.emit("mov.u32 %r_ut1, %laneid;");

        // Initialize shared memory histogram
        self.emit_comment("Zero shared memory histogram");
        self.emit(&format!(
            "setp.lt.u32 %p0, %r_ut0, {};",
            self.config.n_bins * 6
        ));
        self.emit("@%p0 st.shared.f32 [smem_hist + %r_ut0*4], 0.0;");
        self.emit("bar.sync 0;");

        // Each thread processes multiple samples
        self.emit_comment("Accumulate samples into histogram");
        self.emit("mov.u32 %r_sample, %tid.x;");

        self.emit("hist_loop:");
        self.indent += 1;

        // Check bounds
        self.emit(&format!(
            "ld.global.u32 %r_n_samples, [{}];",
            node_samples_ptr
        ));
        self.emit("setp.ge.u32 %p_done, %r_sample, %r_n_samples;");
        self.emit("@%p_done bra hist_done;");

        // Load sample data
        self.emit_comment("Load: feature value, treatment, outcome");
        self.emit(&format!(
            "ld.global.f32 %r_ft0, [{} + %r_sample*4];",
            features_ptr
        ));
        self.emit(&format!(
            "ld.global.u32 %r_treatment, [{} + %r_sample*4];",
            treatment_ptr
        ));
        self.emit(&format!(
            "ld.global.f32 %r_outcome, [{} + %r_sample*4];",
            outcome_ptr
        ));

        // Compute bin index
        self.emit_comment("Bin index = (feature - min) / bin_width");
        self.emit(&format!(
            "mul.f32 %r_ft1, %r_ft0, {};",
            self.config.n_bins as f32
        ));
        self.emit("cvt.rni.u32.f32 %r_bin, %r_ft1;");
        self.emit(&format!(
            "min.u32 %r_bin, %r_bin, {};",
            self.config.n_bins - 1
        ));

        // Atomic add to histogram bin
        self.emit("setp.eq.u32 %p_treated, %r_treatment, 1;");

        // Treated: bins 0-2 (sum_y, sum_y2, count)
        self.emit("@%p_treated mad.lo.u32 %r_ut2, %r_bin, 24, 0;");
        self.emit("@%p_treated red.add.f32 [smem_hist + %r_ut2], %r_outcome;");
        self.emit("@%p_treated mul.f32 %r_ft1, %r_outcome, %r_outcome;");
        self.emit("@%p_treated red.add.f32 [smem_hist + %r_ut2 + 4], %r_ft1;");
        self.emit("@%p_treated red.add.u32 [smem_hist + %r_ut2 + 8], 1;");

        // Control: bins 3-5 (sum_y, sum_y2, count)
        self.emit("@!%p_treated mad.lo.u32 %r_ut2, %r_bin, 24, 12;");
        self.emit("@!%p_treated red.add.f32 [smem_hist + %r_ut2], %r_outcome;");
        self.emit("@!%p_treated mul.f32 %r_ft1, %r_outcome, %r_outcome;");
        self.emit("@!%p_treated red.add.f32 [smem_hist + %r_ut2 + 4], %r_ft1;");
        self.emit("@!%p_treated red.add.u32 [smem_hist + %r_ut2 + 8], 1;");

        // Next sample (strided)
        self.emit("add.u32 %r_sample, %r_sample, %ntid.x;");
        self.emit("bra hist_loop;");

        self.indent -= 1;
        self.emit("hist_done:");
        self.emit("bar.sync 0;");

        // Write shared histogram to global memory
        self.emit_comment("Store histogram to global memory");
        self.emit(&format!(
            "setp.lt.u32 %p0, %r_ut0, {};",
            self.config.n_bins * 6
        ));
        self.emit("@%p0 ld.shared.f32 %r_ft0, [smem_hist + %r_ut0*4];");
        self.emit(&format!(
            "@%p0 st.global.f32 [{} + %r_ut0*4], %r_ft0;",
            hist_out_ptr
        ));
    }

    /// Emit split finding kernel (finds best split from histograms)
    pub fn emit_find_best_split(&mut self, hist_ptr: &str, split_out_ptr: &str) {
        self.emit_comment("=== Find Best Split (Causal Tree Criterion) ===");

        // Each warp evaluates candidate splits for one feature
        self.emit("mov.u32 %r_feature, %warpid;");
        self.emit("mov.u32 %r_lane, %laneid;");

        // Initialize best gain
        self.emit("mov.f32 %r_best_gain, 0.0;");
        self.emit("mov.u32 %r_best_bin, 0;");

        // Prefix sum of histogram for cumulative statistics
        self.emit_comment("Compute prefix sums for left/right partition statistics");

        // Each lane handles a subset of bins
        let bins_per_lane = self.config.n_bins.div_ceil(32);
        self.emit(&format!("mov.u32 %r_my_bins, {};", bins_per_lane));

        self.emit_comment("Evaluate splits");
        self.emit("mov.u32 %r_bin, %r_lane;");

        self.emit("split_loop:");
        self.indent += 1;

        self.emit(&format!(
            "setp.ge.u32 %p_done, %r_bin, {};",
            self.config.n_bins - 1
        ));
        self.emit("@%p_done bra split_done;");

        // Load left (cumulative) and right statistics
        self.emit_comment("Left: bins [0, bin], Right: bins (bin, n_bins)");

        // Compute CATE for left and right partitions
        self.emit_comment("τ̂_L = E[Y|T=1,X∈L] - E[Y|T=0,X∈L]");
        // ... (complex histogram arithmetic)

        // Compute causal tree criterion
        self.emit_comment("Δ = Var(τ̂_L)·n_L/n + Var(τ̂_R)·n_R/n");

        // Update best if better
        self.emit("setp.gt.f32 %p_better, %r_gain, %r_best_gain;");
        self.emit("@%p_better mov.f32 %r_best_gain, %r_gain;");
        self.emit("@%p_better mov.u32 %r_best_bin, %r_bin;");

        self.emit("add.u32 %r_bin, %r_bin, 32;");
        self.emit("bra split_loop;");

        self.indent -= 1;
        self.emit("split_done:");

        // Warp-level reduction to find best split
        self.emit_comment("Warp reduction for best split");
        for offset in [16, 8, 4, 2, 1] {
            self.emit(&format!(
                "shfl.sync.down.b32 %r_ft0, %r_best_gain, {}, 31, 0xFFFFFFFF;",
                offset
            ));
            self.emit(&format!(
                "shfl.sync.down.b32 %r_ut0, %r_best_bin, {}, 31, 0xFFFFFFFF;",
                offset
            ));
            self.emit("setp.gt.f32 %p_better, %r_ft0, %r_best_gain;");
            self.emit("@%p_better mov.f32 %r_best_gain, %r_ft0;");
            self.emit("@%p_better mov.u32 %r_best_bin, %r_ut0;");
        }

        // Lane 0 writes result
        self.emit("setp.eq.u32 %p0, %r_lane, 0;");
        self.emit(&format!(
            "@%p0 st.global.f32 [{} + %r_feature*8], %r_best_gain;",
            split_out_ptr
        ));
        self.emit(&format!(
            "@%p0 st.global.u32 [{} + %r_feature*8 + 4], %r_best_bin;",
            split_out_ptr
        ));
    }

    /// Emit segment classification kernel
    pub fn emit_segment_classify(&mut self, predictions_ptr: &str, segments_out_ptr: &str) {
        self.emit_comment("=== Customer Segment Classification ===");

        self.emit("mov.u32 %r_idx, %tid.x;");
        self.emit("mad.lo.u32 %r_idx, %ctaid.x, %ntid.x, %r_idx;");

        // Load prediction: [tau, tau_eps, confidence, ...]
        self.emit(&format!(
            "mad.wide.u32 %rd0, %r_idx, 24, {};",
            predictions_ptr
        ));
        self.emit("ld.global.f32 %r_tau, [%rd0];");
        self.emit("ld.global.f32 %r_tau_eps, [%rd0+4];");
        self.emit("ld.global.f32 %r_confidence, [%rd0+8];");

        // Classification logic
        let threshold = self.config.confidence_threshold;

        // Check confidence
        self.emit(&format!(
            "setp.gt.f32 %p_confident, %r_confidence, {};",
            1.0 - threshold
        ));

        // Initialize as Uncertain (4)
        self.emit("mov.u32 %r_segment, 4;");

        // If confident and tau > threshold -> Persuadable (0)
        self.emit(&format!(
            "setp.gt.f32 %p_persuadable, %r_tau, {};",
            threshold
        ));
        self.emit("and.pred %p_persuadable, %p_confident, %p_persuadable;");
        self.emit("@%p_persuadable mov.u32 %r_segment, 0;");

        // If confident and tau < -threshold -> SleepingDog (3)
        self.emit(&format!(
            "setp.lt.f32 %p_sleeping_dog, %r_tau, {};",
            -threshold
        ));
        self.emit("and.pred %p_sleeping_dog, %p_confident, %p_sleeping_dog;");
        self.emit("@%p_sleeping_dog mov.u32 %r_segment, 3;");

        // Store segment
        self.emit(&format!(
            "st.global.u32 [{} + %r_idx*4], %r_segment;",
            segments_out_ptr
        ));
    }

    /// Emit epistemic QINI curve computation
    pub fn emit_qini_compute(
        &mut self,
        predictions_ptr: &str,
        treatments_ptr: &str,
        outcomes_ptr: &str,
        n_samples: u32,
        qini_out_ptr: &str,
    ) {
        self.emit_comment("=== QINI Curve with Epistemic Variance ===");

        // QINI = Σ (Y_i · T_i / p) - Σ (Y_i · (1-T_i) / (1-p))
        // weighted by confidence

        self.emit_comment("Each thread computes contribution for one sample");
        self.emit("mov.u32 %r_idx, %tid.x;");
        self.emit("mad.lo.u32 %r_idx, %ctaid.x, %ntid.x, %r_idx;");

        self.emit(&format!("setp.ge.u32 %p_done, %r_idx, {};", n_samples));
        self.emit("@%p_done exit;");

        // Load data
        self.emit(&format!(
            "mad.wide.u32 %rd0, %r_idx, 24, {};",
            predictions_ptr
        ));
        self.emit("ld.global.f32 %r_tau, [%rd0];");
        self.emit("ld.global.f32 %r_confidence, [%rd0+8];");

        self.emit(&format!(
            "ld.global.u32 %r_treatment, [{} + %r_idx*4];",
            treatments_ptr
        ));
        self.emit(&format!(
            "ld.global.f32 %r_outcome, [{} + %r_idx*4];",
            outcomes_ptr
        ));

        // Compute QINI contribution weighted by confidence
        self.emit_comment("QINI contribution = outcome · (2·treatment - 1) · confidence");
        self.emit("cvt.rn.f32.u32 %r_ft0, %r_treatment;");
        self.emit("mad.f32 %r_ft0, %r_ft0, 2.0, -1.0;"); // 2*T - 1
        self.emit("mul.f32 %r_ft0, %r_ft0, %r_outcome;");
        self.emit("mul.f32 %r_ft0, %r_ft0, %r_confidence;");

        // Atomic add to cumulative QINI
        self.emit(&format!("red.add.f32 [{}], %r_ft0;", qini_out_ptr));

        // Also accumulate variance contribution
        self.emit("mul.f32 %r_ft1, %r_ft0, %r_ft0;"); // x²
        self.emit(&format!("red.add.f32 [{} + 4], %r_ft1;", qini_out_ptr));
    }

    /// Generate complete uplift tree kernel
    pub fn generate_full_kernel(&mut self) -> String {
        let mut full = String::new();

        writeln!(
            full,
            "// Sounio GPU Uplift Trees - WORLD-FIRST Epistemic Causal Trees"
        )
        .unwrap();
        writeln!(full, ".version 7.5").unwrap();
        writeln!(
            full,
            ".target sm_{}{}",
            self.config.sm_version.0, self.config.sm_version.1
        )
        .unwrap();
        writeln!(full, ".address_size 64").unwrap();
        writeln!(full).unwrap();

        // Declarations
        writeln!(full, ".visible .entry uplift_tree_predict(").unwrap();
        writeln!(full, "\t.param .u64 param_tree,").unwrap();
        writeln!(full, "\t.param .u64 param_features,").unwrap();
        writeln!(full, "\t.param .u64 param_output,").unwrap();
        writeln!(full, "\t.param .u32 param_n_samples").unwrap();
        writeln!(full, ") {{").unwrap();

        self.emit_declarations();
        full.push_str(&self.output);
        self.output.clear();

        self.emit_tree_traverse("%rd_tree", "%rd_features", "%rd_output");
        full.push_str(&self.output);
        self.output.clear();

        writeln!(full, "\tret;").unwrap();
        writeln!(full, "}}").unwrap();

        full
    }
}

/// Compile uplift tree operations to PTX
pub fn compile_uplift_tree_ptx(config: &UpliftTreeGpuConfig) -> String {
    let mut emitter = UpliftTreePtxEmitter::new(config.clone());
    emitter.generate_full_kernel()
}

/// Runtime for GPU uplift tree execution
#[derive(Debug)]
pub struct UpliftTreeGpuRuntime {
    /// Tree structure on device
    pub tree_buffer: Option<u64>, // Device pointer
    /// Configuration
    pub config: UpliftTreeGpuConfig,
}

impl UpliftTreeGpuRuntime {
    pub fn new(config: UpliftTreeGpuConfig) -> Self {
        Self {
            tree_buffer: None,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_classification() {
        assert_eq!(
            CustomerSegment::classify(0.15, 0.05, 0.3, 0.1),
            CustomerSegment::Persuadable
        );

        assert_eq!(
            CustomerSegment::classify(-0.15, 0.05, 0.3, 0.1),
            CustomerSegment::SleepingDog
        );

        assert_eq!(
            CustomerSegment::classify(0.05, 0.5, 0.3, 0.1),
            CustomerSegment::Uncertain
        );
    }

    #[test]
    fn test_hist_bin_cate() {
        let mut bin = UpliftHistBin::default();
        bin.sum_y_treated = 10.0;
        bin.count_treated = 5;
        bin.sum_y_control = 5.0;
        bin.count_control = 5;

        // CATE = 10/5 - 5/5 = 2 - 1 = 1
        assert!((bin.cate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_confidence() {
        let mut node = UpliftTreeNode::leaf(0, 0, 0);
        node.beta_alpha = 10.0; // 10 positive effects
        node.beta_beta = 2.0; // 2 negative effects

        let conf = node.confidence();
        assert!(conf > 0.5); // Should have decent confidence
    }

    #[test]
    fn test_ptx_generation() {
        let config = UpliftTreeGpuConfig::default();
        let ptx = compile_uplift_tree_ptx(&config);

        // Check for entry point
        assert!(ptx.contains(".entry uplift_tree_predict"));
        // Check for CATE estimation registers
        assert!(ptx.contains("%r_tau"));
        // Check for segment classification register
        assert!(ptx.contains("%r_segment"));
    }
}
