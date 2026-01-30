//! Counterfactual Execution Model for Apple Metal (MSL)
//!
//! This module implements Pearl's do-calculus as Metal GPU primitives, enabling
//! causal inference at Apple Silicon speed. WORLD-FIRST for Metal GPUs.
//!
//! # Metal-Specific Architecture
//!
//! Apple Metal uses SIMD groups (32 threads) instead of CUDA warps, but the
//! fundamental parallelism model is similar. Key differences:
//!
//! - `simdgroup` instead of warp (both 32-wide on Apple Silicon)
//! - `simd_shuffle_xor` instead of `__shfl_xor_sync`
//! - `threadgroup` barriers instead of `__syncthreads()`
//! - `thread_index_in_simdgroup` instead of `laneid`
//!
//! # The Ladder of Causation on Metal
//!
//! ```text
//! Level 1: ASSOCIATION (Seeing)    - Standard Metal execution
//! Level 2: INTERVENTION (Doing)    - do(X=x) operator
//! Level 3: COUNTERFACTUAL (Imagining) - What if X had been x'?
//! ```
//!
//! # Parallel World Execution via SIMD Groups
//!
//! Different threads within a simdgroup explore different causal worlds:
//!
//! ```metal
//! // Intervention: do(X = x_cf)
//! uint lane = thread_index_in_simdgroup;
//! bool is_counterfactual = (lane & 1) != 0;  // Odd lanes = counterfactual
//!
//! float x = is_counterfactual ? x_cf : x_factual;
//!
//! // Execute model in both worlds
//! float outcome = structural_equation(x, ...);
//!
//! // Compute treatment effect (divergence between worlds)
//! float other_outcome = simd_shuffle_xor(outcome, 1);
//! float ite = is_counterfactual ? (outcome - other_outcome) : (other_outcome - outcome);
//! ```
//!
//! # Use Cases
//!
//! - **Apple Neural Engine**: Counterfactual ML explanations
//! - **macOS/iOS ML**: Causal inference on-device
//! - **Vision Pro**: Real-time causal reasoning for XR
//! - **Computational Biology**: Drug response modeling on Apple Silicon

use std::collections::HashMap;
use std::fmt::Write;

use super::counterfactual::{CounterfactualContext, WorldId};
use super::ir::MetalGpuFamily;

/// Configuration for counterfactual Metal emission
#[derive(Debug, Clone)]
pub struct CounterfactualMetalConfig {
    /// Target Metal GPU family
    pub gpu_family: MetalGpuFamily,
    /// Number of parallel worlds per simdgroup (power of 2)
    pub worlds_per_simdgroup: u32,
    /// Enable world divergence tracking
    pub track_divergence: bool,
    /// Enable causal depth tracking
    pub track_depth: bool,
    /// Maximum causal depth
    pub max_depth: u32,
    /// Use fast math optimizations
    pub fast_math: bool,
}

impl Default for CounterfactualMetalConfig {
    fn default() -> Self {
        Self {
            gpu_family: MetalGpuFamily::Apple8,
            worlds_per_simdgroup: 2, // Half simdgroup factual, half counterfactual
            track_divergence: true,
            track_depth: true,
            max_depth: 8,
            fast_math: true,
        }
    }
}

/// Counterfactual Metal Shading Language code emitter
pub struct CounterfactualMetalEmitter {
    config: CounterfactualMetalConfig,
    output: String,
    indent: usize,
}

impl CounterfactualMetalEmitter {
    pub fn new(config: CounterfactualMetalConfig) -> Self {
        Self {
            config,
            output: String::new(),
            indent: 1,
        }
    }

    pub fn output(&self) -> &str {
        &self.output
    }

    pub fn clear(&mut self) {
        self.output.clear();
    }

    fn emit(&mut self, s: &str) {
        let indent = "    ".repeat(self.indent);
        writeln!(self.output, "{}{}", indent, s).unwrap();
    }

    fn emit_raw(&mut self, s: &str) {
        writeln!(self.output, "{}", s).unwrap();
    }

    fn emit_comment(&mut self, s: &str) {
        self.emit(&format!("// {}", s));
    }

    /// Generate MSL header with required includes and using declarations
    pub fn emit_header(&mut self) {
        self.emit_raw("#include <metal_stdlib>");
        self.emit_raw("#include <simdgroup_functions.h>");
        self.emit_raw("");
        self.emit_raw("using namespace metal;");
        self.emit_raw("");
    }

    /// Emit counterfactual world context structure
    pub fn emit_cf_context_struct(&mut self) {
        self.emit_raw("// Counterfactual execution context");
        self.emit_raw("struct CounterfactualContext {");
        self.emit("uint64_t world_id;          // Current world identifier");
        self.emit("uint32_t causal_depth;      // Number of interventions from factual");
        self.emit("bool is_factual;            // True if in factual world");
        self.emit("bool is_counterfactual;     // True if in counterfactual world");
        self.emit("float divergence;           // Accumulated world divergence");
        self.emit_raw("};");
        self.emit_raw("");
    }

    /// Emit world ID constants and helpers
    pub fn emit_world_id_helpers(&mut self) {
        self.emit_raw("// World ID constants");
        self.emit_raw("constant uint64_t WORLD_FACTUAL = 0;");
        self.emit_raw("constant uint64_t WORLD_CF_MARKER = 0xCAFEBABE00000000ULL;");
        self.emit_raw("");

        self.emit_raw("// Check if world is factual");
        self.emit_raw("inline bool world_is_factual(uint64_t world_id) {");
        self.emit("return world_id == WORLD_FACTUAL;");
        self.emit_raw("}");
        self.emit_raw("");

        self.emit_raw("// Check if world is counterfactual");
        self.emit_raw("inline bool world_is_counterfactual(uint64_t world_id) {");
        self.emit("return (world_id & WORLD_CF_MARKER) == WORLD_CF_MARKER;");
        self.emit_raw("}");
        self.emit_raw("");

        self.emit_raw("// Get intervention ID from counterfactual world");
        self.emit_raw("inline uint32_t world_intervention_id(uint64_t world_id) {");
        self.emit("return uint32_t(world_id & 0xFFFFFFFF);");
        self.emit_raw("}");
        self.emit_raw("");

        self.emit_raw("// Create counterfactual world ID");
        self.emit_raw("inline uint64_t create_cf_world(uint32_t intervention_id) {");
        self.emit("return WORLD_CF_MARKER | uint64_t(intervention_id);");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit SIMD group helpers for counterfactual operations
    pub fn emit_simd_helpers(&mut self) {
        self.emit_raw("// SIMD group helpers for counterfactual execution");
        self.emit_raw("");

        // Initialize counterfactual context
        self.emit_raw("// Initialize counterfactual context for a thread");
        self.emit_raw("inline CounterfactualContext cf_init(uint simd_lane_id) {");
        self.emit("CounterfactualContext ctx;");
        self.emit("ctx.world_id = WORLD_FACTUAL;");
        self.emit("ctx.causal_depth = 0;");
        self.emit("ctx.is_factual = true;");
        self.emit("ctx.is_counterfactual = false;");
        self.emit("ctx.divergence = 0.0f;");
        self.emit("return ctx;");
        self.emit_raw("}");
        self.emit_raw("");

        // World assignment based on lane
        self.emit_raw(&format!(
            "// Assign thread to world based on lane ID ({} worlds per simdgroup)",
            self.config.worlds_per_simdgroup
        ));
        self.emit_raw("inline void cf_assign_world(thread CounterfactualContext& ctx, uint simd_lane_id, uint32_t intervention_id) {");
        self.emit(&format!(
            "uint world_index = simd_lane_id & {};",
            self.config.worlds_per_simdgroup - 1
        ));
        self.emit("ctx.is_factual = (world_index == 0);");
        self.emit("ctx.is_counterfactual = (world_index != 0);");
        self.emit("if (ctx.is_counterfactual) {");
        self.indent += 1;
        self.emit("ctx.world_id = create_cf_world(intervention_id);");
        self.emit("ctx.causal_depth += 1;");
        self.indent -= 1;
        self.emit("}");
        self.emit_raw("}");
        self.emit_raw("");

        // Intervention application
        self.emit_raw("// Apply do-operator: do(X = cf_value)");
        self.emit_raw("inline float cf_intervene(");
        self.emit("    float factual_value,");
        self.emit("    float cf_value,");
        self.emit("    thread CounterfactualContext& ctx,");
        self.emit("    uint simd_lane_id,");
        self.emit("    uint32_t intervention_id");
        self.emit_raw(") {");
        self.emit("cf_assign_world(ctx, simd_lane_id, intervention_id);");
        self.emit("return ctx.is_counterfactual ? cf_value : factual_value;");
        self.emit_raw("}");
        self.emit_raw("");

        // Individual Treatment Effect
        self.emit_raw("// Compute Individual Treatment Effect (ITE)");
        self.emit_raw("// ITE = Y(1) - Y(0) where Y(x) is outcome under treatment x");
        self.emit_raw("inline float cf_compute_ite(float outcome, bool is_counterfactual) {");
        self.emit("// Exchange outcome with paired thread in other world");
        self.emit("float other_outcome = simd_shuffle_xor(outcome, 1);");
        self.emit("");
        self.emit("// ITE from counterfactual perspective");
        self.emit("if (is_counterfactual) {");
        self.indent += 1;
        self.emit("return outcome - other_outcome;  // Y(1) - Y(0)");
        self.indent -= 1;
        self.emit("} else {");
        self.indent += 1;
        self.emit("return other_outcome - outcome;  // Y(1) - Y(0)");
        self.indent -= 1;
        self.emit("}");
        self.emit_raw("}");
        self.emit_raw("");

        // Average Treatment Effect
        self.emit_raw("// Compute Average Treatment Effect (ATE) across simdgroup");
        self.emit_raw("inline float cf_compute_ate(float ite) {");
        self.emit("float sum = simd_sum(ite);");
        let pairs = self.config.gpu_family.simd_width() / self.config.worlds_per_simdgroup;
        self.emit(&format!(
            "return sum / {}.0f;  // Average over {} pairs",
            pairs, pairs
        ));
        self.emit_raw("}");
        self.emit_raw("");

        // World divergence
        self.emit_raw("// Compute divergence between factual and counterfactual outcomes");
        self.emit_raw(
            "inline float cf_compute_divergence(float outcome, bool is_counterfactual) {",
        );
        self.emit("float other = simd_shuffle_xor(outcome, 1);");
        self.emit("return abs(outcome - other);");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit structural equation functions
    pub fn emit_structural_equations(&mut self) {
        self.emit_raw("// Structural Equation Models (SEMs) for causal inference");
        self.emit_raw("");

        // Linear structural equation: Y = β₀ + Σ βᵢXᵢ
        self.emit_raw("// Linear structural equation: Y = β₀ + β₁X₁ + β₂X₂ + ...");
        self.emit_raw("template<int N>");
        self.emit_raw("inline float sem_linear(");
        self.emit("    thread const float* x,     // Input variables");
        self.emit("    constant const float* beta // Coefficients [β₀, β₁, ...]");
        self.emit_raw(") {");
        self.emit("float y = beta[0];  // Intercept");
        self.emit("for (int i = 0; i < N; i++) {");
        self.indent += 1;
        self.emit("y = fma(beta[i + 1], x[i], y);");
        self.indent -= 1;
        self.emit("}");
        self.emit("return y;");
        self.emit_raw("}");
        self.emit_raw("");

        // Logistic structural equation: Y = σ(β₀ + Σ βᵢXᵢ)
        self.emit_raw("// Logistic structural equation: Y = sigmoid(β₀ + β₁X₁ + ...)");
        self.emit_raw("template<int N>");
        self.emit_raw("inline float sem_logistic(");
        self.emit("    thread const float* x,");
        self.emit("    constant const float* beta");
        self.emit_raw(") {");
        self.emit("float z = beta[0];");
        self.emit("for (int i = 0; i < N; i++) {");
        self.indent += 1;
        self.emit("z = fma(beta[i + 1], x[i], z);");
        self.indent -= 1;
        self.emit("}");
        self.emit("return 1.0f / (1.0f + exp(-z));  // Sigmoid");
        self.emit_raw("}");
        self.emit_raw("");

        // Multiplicative structural equation: Y = ∏ Xᵢ^βᵢ
        self.emit_raw("// Multiplicative structural equation: Y = ∏ Xᵢ^βᵢ");
        self.emit_raw("template<int N>");
        self.emit_raw("inline float sem_multiplicative(");
        self.emit("    thread const float* x,");
        self.emit("    constant const float* beta");
        self.emit_raw(") {");
        self.emit("float y = 1.0f;");
        self.emit("for (int i = 0; i < N; i++) {");
        self.indent += 1;
        self.emit("y *= pow(x[i], beta[i]);");
        self.indent -= 1;
        self.emit("}");
        self.emit("return y;");
        self.emit_raw("}");
        self.emit_raw("");

        // Threshold structural equation: Y = 1{X > θ}
        self.emit_raw("// Threshold structural equation: Y = 1 if X > θ, else 0");
        self.emit_raw("inline float sem_threshold(float x, float theta) {");
        self.emit("return x > theta ? 1.0f : 0.0f;");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit probability of causation functions
    pub fn emit_probability_causation(&mut self) {
        self.emit_raw("// Probability of Causation (PoC)");
        self.emit_raw("// P(Y_x=1 | X=0, Y=0) - probability that X=1 would have caused Y=1");
        self.emit_raw("inline float cf_probability_causation(");
        self.emit("    float x_actual,      // Actual X value");
        self.emit("    float y_actual,      // Actual Y value");
        self.emit("    float y_cf,          // Counterfactual Y under do(X=1)");
        self.emit("    float threshold      // Decision threshold (typically 0.5)");
        self.emit_raw(") {");
        self.emit("// Precondition: X=0 and Y=0");
        self.emit("bool precondition = (x_actual < threshold) && (y_actual < threshold);");
        self.emit("");
        self.emit("// Would Y have been 1 under counterfactual?");
        self.emit("bool y_cf_positive = y_cf >= threshold;");
        self.emit("");
        self.emit("// P(causation) = 1 if precond ∧ y_cf_positive, else 0");
        self.emit("return (precondition && y_cf_positive) ? 1.0f : 0.0f;");
        self.emit_raw("}");
        self.emit_raw("");

        // Aggregate PoC across simdgroup
        self.emit_raw("// Aggregate probability of causation across simdgroup");
        self.emit_raw("inline float cf_aggregate_poc(float poc) {");
        self.emit("float sum = simd_sum(poc);");
        self.emit(&format!(
            "return sum / {}.0f;",
            self.config.gpu_family.simd_width()
        ));
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit nested intervention support
    pub fn emit_nested_interventions(&mut self) {
        self.emit_raw("// Nested interventions: do(X=x) then do(Z=z)");
        self.emit_raw("// Uses different bits of lane ID for nested levels");
        self.emit_raw("inline float2 cf_nested_intervene(");
        self.emit("    float x_factual,");
        self.emit("    float x_cf,");
        self.emit("    float z_factual,");
        self.emit("    float z_cf,");
        self.emit("    thread CounterfactualContext& ctx,");
        self.emit("    uint simd_lane_id,");
        self.emit("    uint32_t intervention_id_x,");
        self.emit("    uint32_t intervention_id_z");
        self.emit_raw(") {");
        self.emit(&format!("// Check max depth ({})", self.config.max_depth));
        self.emit(&format!(
            "if (ctx.causal_depth >= {}) {{",
            self.config.max_depth
        ));
        self.indent += 1;
        self.emit("return float2(x_factual, z_factual);");
        self.indent -= 1;
        self.emit("}");
        self.emit("");
        self.emit("// First level: bit 0 of lane ID");
        self.emit("bool cf_level1 = (simd_lane_id & 1) != 0;");
        self.emit("float x = cf_level1 ? x_cf : x_factual;");
        self.emit("");
        self.emit("// Second level: bit 1 of lane ID");
        self.emit("bool cf_level2 = (simd_lane_id & 2) != 0;");
        self.emit("float z = cf_level2 ? z_cf : z_factual;");
        self.emit("");
        self.emit("// Update context");
        self.emit("ctx.is_counterfactual = cf_level1 || cf_level2;");
        self.emit("ctx.is_factual = !ctx.is_counterfactual;");
        self.emit("ctx.causal_depth += (cf_level1 ? 1 : 0) + (cf_level2 ? 1 : 0);");
        self.emit("");
        self.emit("return float2(x, z);");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit world merge operations
    pub fn emit_world_merge(&mut self) {
        self.emit_raw("// Merge results from parallel worlds");
        self.emit_raw("// Returns (factual_result, counterfactual_result)");
        self.emit_raw("inline float2 cf_world_merge(float result, bool is_counterfactual) {");
        self.emit("float other = simd_shuffle_xor(result, 1);");
        self.emit("if (is_counterfactual) {");
        self.indent += 1;
        self.emit("return float2(other, result);  // (factual, counterfactual)");
        self.indent -= 1;
        self.emit("} else {");
        self.indent += 1;
        self.emit("return float2(result, other);  // (factual, counterfactual)");
        self.indent -= 1;
        self.emit("}");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Emit causal effect estimators
    pub fn emit_causal_estimators(&mut self) {
        self.emit_raw("// Causal Effect Estimators");
        self.emit_raw("");

        // Conditional Average Treatment Effect
        self.emit_raw("// CATE: Conditional Average Treatment Effect");
        self.emit_raw("// E[Y(1) - Y(0) | X = x]");
        self.emit_raw("inline float cf_cate(");
        self.emit("    float ite,              // Individual Treatment Effect");
        self.emit("    float condition_value,  // Value to condition on");
        self.emit("    float x,                // Conditioning variable");
        self.emit("    float epsilon           // Tolerance for condition match");
        self.emit_raw(") {");
        self.emit("// Only include in average if condition matches");
        self.emit("bool matches = abs(x - condition_value) < epsilon;");
        self.emit("float masked_ite = matches ? ite : 0.0f;");
        self.emit("float count = simd_sum(matches ? 1.0f : 0.0f);");
        self.emit("float sum = simd_sum(masked_ite);");
        self.emit("return count > 0 ? sum / count : 0.0f;");
        self.emit_raw("}");
        self.emit_raw("");

        // Attributable Fraction
        self.emit_raw("// AF: Attributable Fraction");
        self.emit_raw("// Proportion of outcome attributable to exposure");
        self.emit_raw("inline float cf_attributable_fraction(");
        self.emit("    float y_exposed,      // Outcome in exposed group");
        self.emit("    float y_unexposed     // Outcome in unexposed group");
        self.emit_raw(") {");
        self.emit("if (y_exposed <= 0.0f) return 0.0f;");
        self.emit("return (y_exposed - y_unexposed) / y_exposed;");
        self.emit_raw("}");
        self.emit_raw("");

        // Number Needed to Treat
        self.emit_raw("// NNT: Number Needed to Treat");
        self.emit_raw("// 1 / ATE - how many to treat to prevent one bad outcome");
        self.emit_raw("inline float cf_nnt(float ate) {");
        self.emit("if (abs(ate) < 1e-10f) return INFINITY;");
        self.emit("return 1.0f / abs(ate);");
        self.emit_raw("}");
        self.emit_raw("");
    }

    /// Generate complete counterfactual MSL library
    pub fn generate_library(&mut self) -> String {
        self.clear();

        self.emit_raw("//");
        self.emit_raw("// Sounio Counterfactual Execution Library for Metal");
        self.emit_raw("// Pearl's do-calculus implemented as GPU primitives");
        self.emit_raw("//");
        self.emit_raw(&format!(
            "// Target: {} (MSL {})",
            match self.config.gpu_family {
                MetalGpuFamily::Apple7 => "Apple7 (M1/A14)",
                MetalGpuFamily::Apple8 => "Apple8 (M2/A15/A16)",
                MetalGpuFamily::Apple9 => "Apple9 (M3/A17)",
                MetalGpuFamily::Apple10 => "Apple10 (M4/A18)",
                MetalGpuFamily::Mac2 => "Mac2 (Intel)",
                MetalGpuFamily::Common => "Common (Portable)",
            },
            self.config.gpu_family.msl_version()
        ));
        self.emit_raw("//");
        self.emit_raw("");

        self.emit_header();
        self.emit_cf_context_struct();
        self.emit_world_id_helpers();
        self.emit_simd_helpers();
        self.emit_structural_equations();
        self.emit_probability_causation();
        self.emit_nested_interventions();
        self.emit_world_merge();
        self.emit_causal_estimators();

        self.output.clone()
    }

    /// Generate a counterfactual kernel from context
    pub fn generate_kernel(&mut self, ctx: &CounterfactualContext, kernel_name: &str) -> String {
        self.clear();

        self.emit_raw(&format!("// Counterfactual kernel: {}", kernel_name));
        self.emit_raw(&format!("// {} intervention(s)", ctx.interventions.len()));
        self.emit_raw("");

        // Kernel signature
        self.emit_raw(&format!("kernel void {}(", kernel_name));
        self.emit("    device const float* inputs [[buffer(0)]],");
        self.emit("    device float* outputs [[buffer(1)]],");
        self.emit("    device float* ite_outputs [[buffer(2)]],");
        self.emit("    uint tid [[thread_position_in_grid]],");
        self.emit("    uint simd_lane_id [[thread_index_in_simdgroup]]");
        self.emit_raw(") {");
        self.indent += 1;

        // Initialize context
        self.emit("// Initialize counterfactual context");
        self.emit("CounterfactualContext ctx = cf_init(simd_lane_id);");
        self.emit("");

        // Load factual values
        self.emit("// Load factual values");
        for (i, (var, _value)) in ctx
            .snapshots
            .get(&WorldId::FACTUAL)
            .map(|s| &s.values)
            .unwrap_or(&HashMap::new())
            .iter()
            .enumerate()
        {
            self.emit(&format!(
                "float {} = inputs[tid * {} + {}];",
                var,
                ctx.exogenous.len().max(1),
                i
            ));
        }
        self.emit("");

        // Apply interventions
        for intervention in &ctx.interventions {
            if let Some(cf_val) = intervention.value.as_f32() {
                self.emit(&format!(
                    "// Intervention: do({} = {})",
                    intervention.variable, cf_val
                ));
                self.emit(&format!(
                    "{0} = cf_intervene({0}, {1}f, ctx, simd_lane_id, {2});",
                    intervention.variable, cf_val, intervention.id
                ));
                self.emit("");
            }
        }

        // Placeholder for structural equation evaluation
        self.emit("// Evaluate structural equations (user-defined)");
        self.emit("float outcome = 0.0f;  // Replace with actual computation");
        self.emit("");

        // Compute ITE
        self.emit("// Compute Individual Treatment Effect");
        self.emit("float ite = cf_compute_ite(outcome, ctx.is_counterfactual);");
        self.emit("");

        // Store outputs
        self.emit("// Store results");
        self.emit("outputs[tid] = outcome;");
        self.emit("ite_outputs[tid] = ite;");

        self.indent -= 1;
        self.emit_raw("}");

        self.output.clone()
    }
}

/// Generate counterfactual MSL for a given context
pub fn compile_counterfactual_metal(
    ctx: &CounterfactualContext,
    gpu_family: MetalGpuFamily,
) -> String {
    let config = CounterfactualMetalConfig {
        gpu_family,
        ..Default::default()
    };

    let mut emitter = CounterfactualMetalEmitter::new(config);

    let mut output = emitter.generate_library();
    output.push_str("\n\n");
    output.push_str(&emitter.generate_kernel(ctx, "counterfactual_main"));

    output
}

/// Generate just the counterfactual library (no kernel)
pub fn generate_counterfactual_metal_library(gpu_family: MetalGpuFamily) -> String {
    let config = CounterfactualMetalConfig {
        gpu_family,
        ..Default::default()
    };

    let mut emitter = CounterfactualMetalEmitter::new(config);
    emitter.generate_library()
}

#[cfg(test)]
mod tests {
    use super::super::counterfactual::CounterfactualValue;
    use super::*;

    #[test]
    fn test_counterfactual_metal_library() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Check key components exist
        assert!(library.contains("CounterfactualContext"));
        assert!(library.contains("cf_intervene"));
        assert!(library.contains("cf_compute_ite"));
        assert!(library.contains("cf_compute_ate"));
        assert!(library.contains("simd_shuffle_xor"));
        assert!(library.contains("sem_linear"));
        assert!(library.contains("sem_logistic"));
    }

    #[test]
    fn test_counterfactual_kernel_generation() {
        let mut ctx = CounterfactualContext::new();
        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.5));
        ctx.intervene("treatment", CounterfactualValue::F32(1.0));

        let msl = compile_counterfactual_metal(&ctx, MetalGpuFamily::Apple9);

        assert!(msl.contains("kernel void counterfactual_main"));
        assert!(msl.contains("cf_intervene"));
        assert!(msl.contains("do(treatment = 1)"));
    }

    #[test]
    fn test_config_defaults() {
        let config = CounterfactualMetalConfig::default();
        assert_eq!(config.worlds_per_simdgroup, 2);
        assert_eq!(config.max_depth, 8);
        assert!(config.track_divergence);
        assert!(config.track_depth);
    }

    #[test]
    fn test_nested_interventions() {
        let config = CounterfactualMetalConfig {
            gpu_family: MetalGpuFamily::Apple8,
            max_depth: 4,
            ..Default::default()
        };

        let mut emitter = CounterfactualMetalEmitter::new(config);
        emitter.emit_header();
        emitter.emit_nested_interventions();

        let output = emitter.output();
        assert!(output.contains("cf_nested_intervene"));
        assert!(output.contains("bit 0"));
        assert!(output.contains("bit 1"));
    }
}
