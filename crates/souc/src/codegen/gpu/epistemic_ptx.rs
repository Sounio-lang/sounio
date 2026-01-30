//! Epistemic PTX Code Generation - Shadow Registers for Uncertainty Tracking
//!
//! This module extends PTX code generation to track epistemic state through GPU
//! computation using shadow registers. This is a WORLD-FIRST innovation.
//!
//! # Architecture
//!
//! Every epistemic value has three shadow components tracked in registers:
//!
//! ```text
//! Value: %r_val     (f32/f64)    - The actual computed value
//! ├── Epsilon: %r_eps   (f32)    - Uncertainty bound
//! ├── Validity: %p_valid (pred)  - Validity predicate
//! └── Provenance: %r_prov (u64)  - Data lineage bitmask
//! ```
//!
//! # Epistemic Operations
//!
//! ## Additive Operations (add, sub)
//! ```ptx
//! // Value: c = a + b
//! add.f32 %r_c, %r_a, %r_b;
//! // Epsilon: ε_c = sqrt(ε_a² + ε_b²)  [Quadrature]
//! mul.f32 %r_t1, %r_eps_a, %r_eps_a;
//! mul.f32 %r_t2, %r_eps_b, %r_eps_b;
//! add.f32 %r_t3, %r_t1, %r_t2;
//! sqrt.approx.f32 %r_eps_c, %r_t3;
//! // Validity: v_c = v_a ∧ v_b
//! and.pred %p_valid_c, %p_valid_a, %p_valid_b;
//! // Provenance: prov_c = prov_a ⊕ prov_b
//! xor.b64 %r_prov_c, %r_prov_a, %r_prov_b;
//! ```
//!
//! ## Multiplicative Operations (mul, div)
//! ```ptx
//! // Value: c = a * b
//! mul.f32 %r_c, %r_a, %r_b;
//! // Epsilon: ε_c ≈ |a|·ε_b + |b|·ε_a (first-order approximation)
//! abs.f32 %r_abs_a, %r_a;
//! abs.f32 %r_abs_b, %r_b;
//! mul.f32 %r_t1, %r_abs_a, %r_eps_b;
//! mul.f32 %r_t2, %r_abs_b, %r_eps_a;
//! add.f32 %r_eps_c, %r_t1, %r_t2;
//! ```
//!
//! ## Confidence-Gated Execution
//! ```ptx
//! // Only execute if confidence > threshold
//! setp.lt.f32 %p_confident, %r_eps, 0.05;
//! @%p_confident expensive_operation;
//! @!%p_confident fallback_operation;
//! ```
//!
//! ## Warp-Level Epistemic Aggregation
//! ```ptx
//! // Aggregate confidence across warp
//! vote.sync.ballot.b32 %r_ballot, %p_valid, 0xFFFFFFFF;
//! popc.b32 %r_valid_count, %r_ballot;
//! // Warp-level confidence = valid_count / 32
//! ```

use std::fmt::Write;

/// Epistemic PTX code generator configuration
#[derive(Debug, Clone)]
pub struct EpistemicPtxConfig {
    /// Target compute capability
    pub sm_version: (u32, u32),
    /// Default uncertainty bound for constants
    pub default_epsilon: f32,
    /// Confidence threshold for gated execution
    pub confidence_threshold: f32,
    /// Enable quadrature propagation (slower but more accurate)
    pub quadrature_propagation: bool,
    /// Enable warp-level epistemic aggregation
    pub warp_aggregation: bool,
    /// Enable provenance tracking
    pub provenance_tracking: bool,
    /// Provenance bits per source (for packing)
    pub provenance_bits: u32,
}

impl Default for EpistemicPtxConfig {
    fn default() -> Self {
        Self {
            sm_version: (8, 0),
            default_epsilon: 0.0,
            confidence_threshold: 0.05,
            quadrature_propagation: true,
            warp_aggregation: true,
            provenance_tracking: true,
            provenance_bits: 8,
        }
    }
}

/// Shadow register set for an epistemic value
#[derive(Debug, Clone)]
pub struct EpistemicShadowRegs {
    /// Value register name
    pub value: String,
    /// Epsilon (uncertainty) register name
    pub epsilon: String,
    /// Validity predicate register name
    pub validity: String,
    /// Provenance register name
    pub provenance: String,
}

impl EpistemicShadowRegs {
    pub fn new(base_name: &str) -> Self {
        Self {
            value: format!("%r_{}", base_name),
            epsilon: format!("%r_{}_eps", base_name),
            validity: format!("%p_{}_valid", base_name),
            provenance: format!("%r_{}_prov", base_name),
        }
    }

    pub fn from_value_reg(value_reg: &str) -> Self {
        let base = value_reg
            .trim_start_matches("%r_")
            .trim_start_matches("%f_");
        Self {
            value: value_reg.to_string(),
            epsilon: format!("%r_{}_eps", base),
            validity: format!("%p_{}_valid", base),
            provenance: format!("%r_{}_prov", base),
        }
    }
}

/// Epistemic PTX emitter
pub struct EpistemicPtxEmitter {
    /// Configuration
    config: EpistemicPtxConfig,
    /// Output buffer
    output: String,
    /// Current indentation
    indent: usize,
    /// Shadow register counter
    shadow_counter: u32,
    /// Register to shadow mapping
    shadows: std::collections::HashMap<String, EpistemicShadowRegs>,
}

impl EpistemicPtxEmitter {
    pub fn new(config: EpistemicPtxConfig) -> Self {
        Self {
            config,
            output: String::new(),
            indent: 1,
            shadow_counter: 0,
            shadows: std::collections::HashMap::new(),
        }
    }

    /// Get the generated output
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Clear the output buffer
    pub fn clear(&mut self) {
        self.output.clear();
    }

    fn emit(&mut self, s: &str) {
        let indent = "\t".repeat(self.indent);
        writeln!(self.output, "{}{}", indent, s).unwrap();
    }

    fn emit_comment(&mut self, s: &str) {
        self.emit(&format!("// {}", s));
    }

    /// Allocate a new shadow register set
    pub fn alloc_shadow(&mut self, base_name: &str) -> EpistemicShadowRegs {
        self.shadow_counter += 1;
        let name = format!("{}_{}", base_name, self.shadow_counter);
        let shadow = EpistemicShadowRegs::new(&name);
        self.shadows.insert(shadow.value.clone(), shadow.clone());
        shadow
    }

    /// Emit shadow register declarations for a kernel
    pub fn emit_shadow_declarations(&mut self, count: u32) {
        self.emit_comment("Epistemic shadow registers");

        // Epsilon registers (f32)
        self.emit(&format!(".reg .f32 %r_eps<{}>;", count));

        // Validity predicates
        self.emit(&format!(".reg .pred %p_valid<{}>;", count));

        // Provenance registers (u64)
        if self.config.provenance_tracking {
            self.emit(&format!(".reg .u64 %r_prov<{}>;", count));
        }

        // Temporary registers for epsilon propagation
        self.emit(".reg .f32 %r_eps_t0, %r_eps_t1, %r_eps_t2, %r_eps_t3;");

        self.emit("");
    }

    /// Emit initialization for a constant (zero uncertainty)
    pub fn emit_constant_epistemic(&mut self, shadow: &EpistemicShadowRegs) {
        self.emit_comment("Initialize constant epistemic state (zero uncertainty)");

        // Zero epsilon
        self.emit(&format!("mov.f32 {}, 0.0;", shadow.epsilon));

        // Valid
        self.emit(&format!("setp.eq.u32 {}, 1, 1;", shadow.validity));

        // Zero provenance (constants have no external source)
        if self.config.provenance_tracking {
            self.emit(&format!("mov.u64 {}, 0;", shadow.provenance));
        }
    }

    /// Emit initialization from parameter with epsilon bound
    pub fn emit_param_epistemic(
        &mut self,
        shadow: &EpistemicShadowRegs,
        epsilon_bound: f32,
        provenance_id: u64,
    ) {
        self.emit_comment(&format!(
            "Initialize parameter epistemic state (ε ≤ {})",
            epsilon_bound
        ));

        // Set epsilon bound
        self.emit(&format!(
            "mov.f32 {}, 0F{:08X};",
            shadow.epsilon,
            epsilon_bound.to_bits()
        ));

        // Assume valid
        self.emit(&format!("setp.eq.u32 {}, 1, 1;", shadow.validity));

        // Set provenance ID
        if self.config.provenance_tracking {
            self.emit(&format!(
                "mov.u64 {}, 0x{:016X};",
                shadow.provenance, provenance_id
            ));
        }
    }

    /// Emit additive operation with epistemic tracking (add, sub)
    pub fn emit_epistemic_add(
        &mut self,
        result: &EpistemicShadowRegs,
        left: &EpistemicShadowRegs,
        right: &EpistemicShadowRegs,
        is_sub: bool,
    ) {
        let op = if is_sub { "sub" } else { "add" };
        self.emit_comment(&format!("Epistemic {}", op));

        // Value computation
        self.emit(&format!(
            "{}.f32 {}, {}, {};",
            op, result.value, left.value, right.value
        ));

        // Epsilon propagation via quadrature: sqrt(ε_a² + ε_b²)
        if self.config.quadrature_propagation {
            self.emit_comment("Quadrature: ε_c = sqrt(ε_a² + ε_b²)");
            self.emit(&format!(
                "mul.f32 %r_eps_t0, {}, {};",
                left.epsilon, left.epsilon
            ));
            self.emit(&format!(
                "mul.f32 %r_eps_t1, {}, {};",
                right.epsilon, right.epsilon
            ));
            self.emit("add.f32 %r_eps_t2, %r_eps_t0, %r_eps_t1;");
            self.emit(&format!("sqrt.approx.f32 {}, %r_eps_t2;", result.epsilon));
        } else {
            // Simple additive (conservative upper bound)
            self.emit(&format!(
                "add.f32 {}, {}, {};",
                result.epsilon, left.epsilon, right.epsilon
            ));
        }

        // Validity: AND
        self.emit(&format!(
            "and.pred {}, {}, {};",
            result.validity, left.validity, right.validity
        ));

        // Provenance: XOR merge
        if self.config.provenance_tracking {
            self.emit(&format!(
                "xor.b64 {}, {}, {};",
                result.provenance, left.provenance, right.provenance
            ));
        }
    }

    /// Emit multiplicative operation with epistemic tracking (mul)
    pub fn emit_epistemic_mul(
        &mut self,
        result: &EpistemicShadowRegs,
        left: &EpistemicShadowRegs,
        right: &EpistemicShadowRegs,
    ) {
        self.emit_comment("Epistemic mul");

        // Value computation
        self.emit(&format!(
            "mul.f32 {}, {}, {};",
            result.value, left.value, right.value
        ));

        // Epsilon propagation: |a|·ε_b + |b|·ε_a
        self.emit_comment("Multiplicative: ε_c = |a|·ε_b + |b|·ε_a");
        self.emit(&format!("abs.f32 %r_eps_t0, {};", left.value));
        self.emit(&format!("abs.f32 %r_eps_t1, {};", right.value));
        self.emit(&format!("mul.f32 %r_eps_t2, %r_eps_t0, {};", right.epsilon));
        self.emit(&format!("mul.f32 %r_eps_t3, %r_eps_t1, {};", left.epsilon));
        self.emit(&format!(
            "add.f32 {}, %r_eps_t2, %r_eps_t3;",
            result.epsilon
        ));

        // Validity: AND
        self.emit(&format!(
            "and.pred {}, {}, {};",
            result.validity, left.validity, right.validity
        ));

        // Provenance: XOR merge
        if self.config.provenance_tracking {
            self.emit(&format!(
                "xor.b64 {}, {}, {};",
                result.provenance, left.provenance, right.provenance
            ));
        }
    }

    /// Emit division with epistemic tracking
    pub fn emit_epistemic_div(
        &mut self,
        result: &EpistemicShadowRegs,
        left: &EpistemicShadowRegs,
        right: &EpistemicShadowRegs,
    ) {
        self.emit_comment("Epistemic div");

        // Value computation
        self.emit(&format!(
            "div.approx.f32 {}, {}, {};",
            result.value, left.value, right.value
        ));

        // Epsilon propagation: (|a|·ε_b + |b|·ε_a) / b²
        // Note: This widens near zero - may need special handling
        self.emit_comment("Division: ε_c = (|a|·ε_b + |b|·ε_a) / b²");
        self.emit(&format!("abs.f32 %r_eps_t0, {};", left.value));
        self.emit(&format!("abs.f32 %r_eps_t1, {};", right.value));
        self.emit(&format!("mul.f32 %r_eps_t2, %r_eps_t0, {};", right.epsilon));
        self.emit(&format!("mul.f32 %r_eps_t3, %r_eps_t1, {};", left.epsilon));
        self.emit("add.f32 %r_eps_t0, %r_eps_t2, %r_eps_t3;");
        self.emit(&format!(
            "mul.f32 %r_eps_t1, {}, {};",
            right.value, right.value
        ));
        self.emit(&format!(
            "div.approx.f32 {}, %r_eps_t0, %r_eps_t1;",
            result.epsilon
        ));

        // Check for near-zero divisor - widen uncertainty
        self.emit_comment("Widen ε if divisor near zero");
        self.emit(&format!("abs.f32 %r_eps_t2, {};", right.value));
        self.emit("setp.lt.f32 %p_near_zero, %r_eps_t2, 0.0001;");
        self.emit(&format!("@%p_near_zero mov.f32 {}, 1.0;", result.epsilon));

        // Validity: AND
        self.emit(&format!(
            "and.pred {}, {}, {};",
            result.validity, left.validity, right.validity
        ));

        // Provenance
        if self.config.provenance_tracking {
            self.emit(&format!(
                "xor.b64 {}, {}, {};",
                result.provenance, left.provenance, right.provenance
            ));
        }
    }

    /// Emit FMA (fused multiply-add) with epistemic tracking
    pub fn emit_epistemic_fma(
        &mut self,
        result: &EpistemicShadowRegs,
        a: &EpistemicShadowRegs,
        b: &EpistemicShadowRegs,
        c: &EpistemicShadowRegs,
    ) {
        self.emit_comment("Epistemic FMA: result = a * b + c");

        // Value computation
        self.emit(&format!(
            "fma.rn.f32 {}, {}, {}, {};",
            result.value, a.value, b.value, c.value
        ));

        // Epsilon: combine multiplicative and additive
        // ε_result = sqrt((|a|·ε_b + |b|·ε_a)² + ε_c²)
        self.emit_comment("FMA epsilon: sqrt((|a|·ε_b + |b|·ε_a)² + ε_c²)");
        self.emit(&format!("abs.f32 %r_eps_t0, {};", a.value));
        self.emit(&format!("abs.f32 %r_eps_t1, {};", b.value));
        self.emit(&format!("mul.f32 %r_eps_t2, %r_eps_t0, {};", b.epsilon));
        self.emit(&format!("mul.f32 %r_eps_t3, %r_eps_t1, {};", a.epsilon));
        self.emit("add.f32 %r_eps_t0, %r_eps_t2, %r_eps_t3;"); // mul part
        self.emit("mul.f32 %r_eps_t0, %r_eps_t0, %r_eps_t0;"); // square
        self.emit(&format!("mul.f32 %r_eps_t1, {}, {};", c.epsilon, c.epsilon)); // c²
        self.emit("add.f32 %r_eps_t2, %r_eps_t0, %r_eps_t1;");
        self.emit(&format!("sqrt.approx.f32 {}, %r_eps_t2;", result.epsilon));

        // Validity: AND all three
        self.emit(&format!(
            "and.pred %p_temp, {}, {};",
            a.validity, b.validity
        ));
        self.emit(&format!(
            "and.pred {}, %p_temp, {};",
            result.validity, c.validity
        ));

        // Provenance
        if self.config.provenance_tracking {
            self.emit(&format!(
                "xor.b64 %r_eps_t0, {}, {};",
                a.provenance, b.provenance
            ));
            self.emit(&format!(
                "xor.b64 {}, %r_eps_t0, {};",
                result.provenance, c.provenance
            ));
        }
    }

    /// Emit confidence-gated execution
    pub fn emit_confidence_gate(
        &mut self,
        shadow: &EpistemicShadowRegs,
        threshold: f32,
        high_conf_label: &str,
        low_conf_label: &str,
    ) {
        self.emit_comment(&format!("Confidence gate (threshold = {})", threshold));

        // Check if epsilon < threshold (high confidence)
        self.emit(&format!(
            "setp.lt.f32 %p_confident, {}, 0F{:08X};",
            shadow.epsilon,
            threshold.to_bits()
        ));

        // Also check validity
        self.emit(&format!(
            "and.pred %p_confident, %p_confident, {};",
            shadow.validity
        ));

        // Branch based on confidence
        self.emit(&format!("@%p_confident bra {};", high_conf_label));
        self.emit(&format!("@!%p_confident bra {};", low_conf_label));
    }

    /// Emit confidence-predicated instruction
    pub fn emit_predicated_on_confidence(
        &mut self,
        shadow: &EpistemicShadowRegs,
        instruction: &str,
    ) {
        self.emit_comment("Predicated on confidence");
        self.emit(&format!("@{} {};", shadow.validity, instruction));
    }

    /// Emit warp-level confidence voting
    pub fn emit_warp_confidence_vote(
        &mut self,
        shadow: &EpistemicShadowRegs,
        result_count: &str,
        result_mask: &str,
    ) {
        self.emit_comment("Warp-level confidence voting");

        // Get ballot of valid lanes
        self.emit(&format!(
            "vote.sync.ballot.b32 {}, {}, 0xFFFFFFFF;",
            result_mask, shadow.validity
        ));

        // Count valid lanes
        self.emit(&format!("popc.b32 {}, {};", result_count, result_mask));
    }

    /// Emit warp-level epsilon reduction (min/max/avg)
    pub fn emit_warp_epsilon_reduce(
        &mut self,
        shadow: &EpistemicShadowRegs,
        result: &str,
        op: WarpEpsilonOp,
    ) {
        self.emit_comment(&format!("Warp-level epsilon {:?}", op));

        // Use warp shuffle for reduction
        // This is a tree reduction: 16 -> 8 -> 4 -> 2 -> 1
        self.emit(&format!("mov.f32 {}, {};", result, shadow.epsilon));

        for offset in [16, 8, 4, 2, 1] {
            self.emit(&format!(
                "shfl.sync.down.b32 %r_eps_t0, {}, {}, 31, 0xFFFFFFFF;",
                result, offset
            ));
            match op {
                WarpEpsilonOp::Min => {
                    self.emit(&format!("min.f32 {}, {}, %r_eps_t0;", result, result));
                }
                WarpEpsilonOp::Max => {
                    self.emit(&format!("max.f32 {}, {}, %r_eps_t0;", result, result));
                }
                WarpEpsilonOp::Sum => {
                    self.emit(&format!("add.f32 {}, {}, %r_eps_t0;", result, result));
                }
            }
        }

        if matches!(op, WarpEpsilonOp::Sum) {
            // Divide by warp size to get average
            self.emit(&format!("mul.f32 {}, {}, 0.03125;", result, result)); // 1/32
        }
    }

    /// Emit warp-level provenance merge
    pub fn emit_warp_provenance_merge(&mut self, shadow: &EpistemicShadowRegs, result: &str) {
        if !self.config.provenance_tracking {
            return;
        }

        self.emit_comment("Warp-level provenance merge (OR reduction)");
        self.emit(&format!("mov.u64 {}, {};", result, shadow.provenance));

        for offset in [16, 8, 4, 2, 1] {
            self.emit(&format!(
                "shfl.sync.down.b32 %r_prov_lo, {}, {}, 31, 0xFFFFFFFF;",
                result, offset
            ));
            self.emit(&format!("or.b64 {}, {}, %r_prov_lo;", result, result));
        }
    }

    /// Emit WMMA (Tensor Core) with epistemic tracking
    pub fn emit_epistemic_wmma(
        &mut self,
        frag_d: &str,
        frag_a: &str,
        frag_b: &str,
        frag_c: &str,
        frag_eps_d: &str,
        frag_eps_a: &str,
        frag_eps_b: &str,
        frag_eps_c: &str,
    ) {
        self.emit_comment("Epistemic WMMA: D = A * B + C with uncertainty");

        // Standard WMMA for values
        self.emit(&format!(
            "wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {}, {}, {}, {};",
            frag_d, frag_a, frag_b, frag_c
        ));

        // Epsilon WMMA: ε_D² = ||B||² · ||ε_A||² + ||A||² · ||ε_B||² + ||ε_C||²
        // This requires computing norms - simplified here
        self.emit_comment("Epsilon propagation for WMMA (simplified)");
        self.emit(&format!(
            "wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {}, {}, {}, {};",
            frag_eps_d, frag_eps_a, frag_eps_b, frag_eps_c
        ));
    }

    /// Emit epistemic comparison (returns confidence in comparison)
    pub fn emit_epistemic_compare(
        &mut self,
        result_pred: &str,
        result_confidence: &str,
        left: &EpistemicShadowRegs,
        right: &EpistemicShadowRegs,
        cmp: &str,
    ) {
        self.emit_comment(&format!(
            "Epistemic comparison: {} {} {}",
            left.value, cmp, right.value
        ));

        // Standard comparison
        self.emit(&format!(
            "setp.{}.f32 {}, {}, {};",
            cmp, result_pred, left.value, right.value
        ));

        // Compute distance to decision boundary
        self.emit(&format!(
            "sub.f32 %r_eps_t0, {}, {};",
            left.value, right.value
        ));
        self.emit("abs.f32 %r_eps_t0, %r_eps_t0;");

        // Combined uncertainty
        if self.config.quadrature_propagation {
            self.emit(&format!(
                "mul.f32 %r_eps_t1, {}, {};",
                left.epsilon, left.epsilon
            ));
            self.emit(&format!(
                "mul.f32 %r_eps_t2, {}, {};",
                right.epsilon, right.epsilon
            ));
            self.emit("add.f32 %r_eps_t3, %r_eps_t1, %r_eps_t2;");
            self.emit("sqrt.approx.f32 %r_eps_t1, %r_eps_t3;");
        } else {
            self.emit(&format!(
                "add.f32 %r_eps_t1, {}, {};",
                left.epsilon, right.epsilon
            ));
        }

        // Confidence = distance / uncertainty (higher is better)
        self.emit_comment("Confidence = distance / combined_uncertainty");
        self.emit(&format!(
            "div.approx.f32 {}, %r_eps_t0, %r_eps_t1;",
            result_confidence
        ));

        // Clamp confidence to [0, 1]
        self.emit(&format!(
            "min.f32 {}, {}, 1.0;",
            result_confidence, result_confidence
        ));
        self.emit(&format!(
            "max.f32 {}, {}, 0.0;",
            result_confidence, result_confidence
        ));
    }

    /// Emit epistemic select (ternary with confidence)
    pub fn emit_epistemic_select(
        &mut self,
        result: &EpistemicShadowRegs,
        cond: &str,
        cond_confidence: &str,
        if_true: &EpistemicShadowRegs,
        if_false: &EpistemicShadowRegs,
    ) {
        self.emit_comment("Epistemic select");

        // Value select
        self.emit(&format!(
            "selp.f32 {}, {}, {}, {};",
            result.value, if_true.value, if_false.value, cond
        ));

        // Epsilon: max of both branches weighted by uncertainty in condition
        self.emit_comment("Epsilon: max(ε_true, ε_false) weighted by condition confidence");
        self.emit(&format!(
            "max.f32 %r_eps_t0, {}, {};",
            if_true.epsilon, if_false.epsilon
        ));
        // If condition is uncertain, increase result uncertainty
        self.emit(&format!("sub.f32 %r_eps_t1, 1.0, {};", cond_confidence)); // 1 - confidence
        self.emit(&format!(
            "fma.rn.f32 {}, %r_eps_t0, %r_eps_t1, %r_eps_t0;",
            result.epsilon
        ));

        // Validity: AND of selected branch validity
        self.emit(&format!(
            "selp.pred {}, {}, {}, {};",
            result.validity, if_true.validity, if_false.validity, cond
        ));

        // Provenance: OR of both branches
        if self.config.provenance_tracking {
            self.emit(&format!(
                "or.b64 {}, {}, {};",
                result.provenance, if_true.provenance, if_false.provenance
            ));
        }
    }

    /// Emit epistemic reduction (sum/mean with confidence weighting)
    pub fn emit_epistemic_reduce_sum(
        &mut self,
        values: &[EpistemicShadowRegs],
        result: &EpistemicShadowRegs,
    ) {
        if values.is_empty() {
            return;
        }

        self.emit_comment("Epistemic reduction (sum with quadrature)");

        // Initialize with first value
        self.emit(&format!("mov.f32 {}, {};", result.value, values[0].value));
        self.emit(&format!(
            "mul.f32 {}, {}, {};",
            result.epsilon, values[0].epsilon, values[0].epsilon
        ));
        self.emit(&format!(
            "mov.pred {}, {};",
            result.validity, values[0].validity
        ));
        if self.config.provenance_tracking {
            self.emit(&format!(
                "mov.u64 {}, {};",
                result.provenance, values[0].provenance
            ));
        }

        // Accumulate rest
        for v in &values[1..] {
            self.emit(&format!(
                "add.f32 {}, {}, {};",
                result.value, result.value, v.value
            ));
            self.emit(&format!("mul.f32 %r_eps_t0, {}, {};", v.epsilon, v.epsilon));
            self.emit(&format!(
                "add.f32 {}, {}, %r_eps_t0;",
                result.epsilon, result.epsilon
            ));
            self.emit(&format!(
                "and.pred {}, {}, {};",
                result.validity, result.validity, v.validity
            ));
            if self.config.provenance_tracking {
                self.emit(&format!(
                    "or.b64 {}, {}, {};",
                    result.provenance, result.provenance, v.provenance
                ));
            }
        }

        // Final sqrt for epsilon
        self.emit(&format!(
            "sqrt.approx.f32 {}, {};",
            result.epsilon, result.epsilon
        ));
    }

    /// Emit confidence-weighted mean reduction
    pub fn emit_epistemic_weighted_mean(
        &mut self,
        values: &[EpistemicShadowRegs],
        result: &EpistemicShadowRegs,
    ) {
        if values.is_empty() {
            return;
        }

        self.emit_comment("Confidence-weighted mean: high confidence values contribute more");

        // Weight = 1 / (1 + ε)
        // weighted_sum = Σ (value * weight)
        // weight_sum = Σ weight
        // result = weighted_sum / weight_sum

        self.emit("mov.f32 %r_eps_t0, 0.0;"); // weighted_sum
        self.emit("mov.f32 %r_eps_t1, 0.0;"); // weight_sum
        self.emit("mov.f32 %r_eps_t2, 0.0;"); // weighted_eps_sum

        for v in values {
            // weight = 1 / (1 + ε)
            self.emit(&format!("add.f32 %r_eps_t3, 1.0, {};", v.epsilon));
            self.emit("rcp.approx.f32 %r_eps_t3, %r_eps_t3;"); // 1/x

            // weighted_sum += value * weight
            self.emit(&format!(
                "fma.rn.f32 %r_eps_t0, {}, %r_eps_t3, %r_eps_t0;",
                v.value
            ));

            // weight_sum += weight
            self.emit("add.f32 %r_eps_t1, %r_eps_t1, %r_eps_t3;");

            // weighted_eps_sum += ε * weight
            self.emit(&format!(
                "fma.rn.f32 %r_eps_t2, {}, %r_eps_t3, %r_eps_t2;",
                v.epsilon
            ));
        }

        // result.value = weighted_sum / weight_sum
        self.emit(&format!(
            "div.approx.f32 {}, %r_eps_t0, %r_eps_t1;",
            result.value
        ));

        // result.epsilon = weighted_eps_sum / weight_sum
        self.emit(&format!(
            "div.approx.f32 {}, %r_eps_t2, %r_eps_t1;",
            result.epsilon
        ));

        // Validity: AND all
        self.emit(&format!("setp.eq.u32 {}, 1, 1;", result.validity));
        for v in values {
            self.emit(&format!(
                "and.pred {}, {}, {};",
                result.validity, result.validity, v.validity
            ));
        }

        // Provenance: OR all
        if self.config.provenance_tracking {
            self.emit(&format!("mov.u64 {}, 0;", result.provenance));
            for v in values {
                self.emit(&format!(
                    "or.b64 {}, {}, {};",
                    result.provenance, result.provenance, v.provenance
                ));
            }
        }
    }
}

/// Warp-level epsilon reduction operation
#[derive(Debug, Clone, Copy)]
pub enum WarpEpsilonOp {
    Min,
    Max,
    Sum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_add_emission() {
        let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());

        let left = EpistemicShadowRegs::new("a");
        let right = EpistemicShadowRegs::new("b");
        let result = EpistemicShadowRegs::new("c");

        emitter.emit_epistemic_add(&result, &left, &right, false);

        let output = emitter.output();
        assert!(output.contains("add.f32"));
        assert!(output.contains("sqrt.approx.f32")); // Quadrature
        assert!(output.contains("and.pred")); // Validity
        assert!(output.contains("xor.b64")); // Provenance
    }

    #[test]
    fn test_confidence_gate_emission() {
        let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());

        let shadow = EpistemicShadowRegs::new("x");
        emitter.emit_confidence_gate(&shadow, 0.05, "high_conf", "low_conf");

        let output = emitter.output();
        assert!(output.contains("setp.lt.f32"));
        assert!(output.contains("bra high_conf"));
        assert!(output.contains("bra low_conf"));
    }

    #[test]
    fn test_warp_reduction_emission() {
        let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());

        let shadow = EpistemicShadowRegs::new("x");
        emitter.emit_warp_epsilon_reduce(&shadow, "%r_result", WarpEpsilonOp::Max);

        let output = emitter.output();
        assert!(output.contains("shfl.sync.down.b32"));
        assert!(output.contains("max.f32"));
    }

    #[test]
    fn test_epistemic_compare_emission() {
        let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());

        let left = EpistemicShadowRegs::new("a");
        let right = EpistemicShadowRegs::new("b");

        emitter.emit_epistemic_compare("%p_result", "%r_confidence", &left, &right, "gt");

        let output = emitter.output();
        assert!(output.contains("setp.gt.f32"));
        assert!(output.contains("Confidence"));
    }
}
