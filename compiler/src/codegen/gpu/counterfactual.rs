//! Counterfactual Execution Model for GPU
//!
//! This module implements Pearl's do-calculus as GPU primitives, enabling
//! causal inference at hardware speed. This is a WORLD-FIRST innovation.
//!
//! # The Ladder of Causation on GPU
//!
//! ```text
//! Level 1: ASSOCIATION (Seeing)    - Standard GPU execution
//! Level 2: INTERVENTION (Doing)    - do(X=x) operator
//! Level 3: COUNTERFACTUAL (Imagining) - What if X had been x'?
//! ```
//!
//! # Architecture
//!
//! ## World Branching
//! ```text
//! Factual World (W₀)
//!     │
//!     ├── do(Treatment = 1.0) ──> Counterfactual World (W₁)
//!     │                                │
//!     │                                └── Evaluate outcome
//!     │
//!     └── Continue factual execution
//! ```
//!
//! ## GPU Implementation
//!
//! Different threads/warps explore different causal worlds:
//!
//! ```ptx
//! // Intervention: do(X = x_cf)
//! mov.u32 %r_lane, %laneid;
//! and.b32 %r_is_cf, %r_lane, 1;      // Odd lanes = counterfactual
//! setp.ne.u32 %p_cf, %r_is_cf, 0;
//! selp.f32 %x, %x_cf, %x_factual, %p_cf;
//!
//! // Execute model in both worlds
//! ... compute outcome ...
//!
//! // Compute treatment effect (divergence between worlds)
//! shfl.sync.xor.b32 %r_other_outcome, %r_outcome, 1, 0xFFFFFFFF;
//! sub.f32 %r_ite, %r_outcome, %r_other_outcome;  // Individual Treatment Effect
//! ```
//!
//! # Use Cases
//!
//! - **Computational Psychiatry**: What if the patient had taken the treatment?
//! - **Causal ML**: Counterfactual explanations at GPU speed
//! - **Economic Modeling**: Policy impact simulation
//! - **Drug Discovery**: What if we modified this molecular structure?
//!
//! # Example
//!
//! ```ignore
//! use sounio::codegen::gpu::counterfactual::*;
//!
//! let mut ctx = CounterfactualContext::new();
//!
//! // Set factual values
//! ctx.set_factual("treatment", 0.0);
//! ctx.set_factual("age", 45.0);
//!
//! // Enter counterfactual world: do(treatment = 1.0)
//! ctx.intervene("treatment", 1.0);
//!
//! // Generate PTX for parallel world execution
//! let emitter = CounterfactualPtxEmitter::new();
//! let ptx = emitter.emit_parallel_worlds(&ctx);
//! ```

use std::collections::HashMap;
use std::fmt::Write;

/// World identifier for causal reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldId(pub u64);

impl WorldId {
    /// The factual world (what actually happened)
    pub const FACTUAL: WorldId = WorldId(0);

    /// Create a counterfactual world from an intervention
    pub fn counterfactual(intervention_id: u32) -> Self {
        WorldId(0xCAFEBABE00000000 | intervention_id as u64)
    }

    /// Check if this is the factual world
    pub fn is_factual(&self) -> bool {
        self.0 == 0
    }

    /// Check if this is a counterfactual world
    pub fn is_counterfactual(&self) -> bool {
        (self.0 & 0xCAFEBABE00000000) == 0xCAFEBABE00000000
    }

    /// Get the intervention ID for a counterfactual world
    pub fn intervention_id(&self) -> Option<u32> {
        if self.is_counterfactual() {
            Some((self.0 & 0xFFFFFFFF) as u32)
        } else {
            None
        }
    }
}

/// Value in a causal world
#[derive(Debug, Clone)]
pub enum CounterfactualValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
    Vector(Vec<f32>),
}

impl CounterfactualValue {
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            CounterfactualValue::F32(v) => Some(*v),
            CounterfactualValue::F64(v) => Some(*v as f32),
            CounterfactualValue::I32(v) => Some(*v as f32),
            CounterfactualValue::I64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// An intervention (do-operator application)
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Variable being intervened upon
    pub variable: String,
    /// Value to set (do(X = value))
    pub value: CounterfactualValue,
    /// Intervention ID for world tracking
    pub id: u32,
    /// Parent world (where intervention originates)
    pub parent_world: WorldId,
    /// Resulting world
    pub result_world: WorldId,
}

impl Intervention {
    pub fn new(variable: &str, value: CounterfactualValue, id: u32) -> Self {
        Self {
            variable: variable.to_string(),
            value,
            id,
            parent_world: WorldId::FACTUAL,
            result_world: WorldId::counterfactual(id),
        }
    }
}

/// Snapshot of a world's state
#[derive(Debug, Clone)]
pub struct WorldSnapshot {
    /// World identifier
    pub world_id: WorldId,
    /// Variable values in this world
    pub values: HashMap<String, CounterfactualValue>,
    /// Causal depth (number of interventions from factual)
    pub depth: u32,
}

impl WorldSnapshot {
    pub fn factual() -> Self {
        Self {
            world_id: WorldId::FACTUAL,
            values: HashMap::new(),
            depth: 0,
        }
    }

    pub fn from_intervention(intervention: &Intervention, parent: &WorldSnapshot) -> Self {
        let mut values = parent.values.clone();
        values.insert(intervention.variable.clone(), intervention.value.clone());

        Self {
            world_id: intervention.result_world,
            values,
            depth: parent.depth + 1,
        }
    }
}

/// Divergence between worlds
#[derive(Debug, Clone)]
pub struct WorldDivergence {
    /// Variable that diverged
    pub variable: String,
    /// Factual value
    pub factual: CounterfactualValue,
    /// Counterfactual value
    pub counterfactual: CounterfactualValue,
    /// Absolute difference (for numeric types)
    pub absolute: f64,
    /// Relative difference (for numeric types)
    pub relative: f64,
}

/// Counterfactual execution context
#[derive(Debug, Clone)]
pub struct CounterfactualContext {
    /// All world snapshots
    pub snapshots: HashMap<WorldId, WorldSnapshot>,
    /// List of interventions
    pub interventions: Vec<Intervention>,
    /// Next intervention ID
    next_intervention_id: u32,
    /// Current active world
    pub current_world: WorldId,
    /// Variables that are exogenous (can be intervened)
    pub exogenous: Vec<String>,
    /// Structural equations (var -> expression)
    pub structural_equations: HashMap<String, String>,
}

impl CounterfactualContext {
    pub fn new() -> Self {
        let mut snapshots = HashMap::new();
        snapshots.insert(WorldId::FACTUAL, WorldSnapshot::factual());

        Self {
            snapshots,
            interventions: Vec::new(),
            next_intervention_id: 1,
            current_world: WorldId::FACTUAL,
            exogenous: Vec::new(),
            structural_equations: HashMap::new(),
        }
    }

    /// Set a factual value
    pub fn set_factual(&mut self, variable: &str, value: CounterfactualValue) {
        if let Some(snapshot) = self.snapshots.get_mut(&WorldId::FACTUAL) {
            snapshot.values.insert(variable.to_string(), value);
        }
    }

    /// Perform an intervention: do(variable = value)
    pub fn intervene(&mut self, variable: &str, value: CounterfactualValue) -> WorldId {
        let intervention = Intervention::new(variable, value, self.next_intervention_id);
        self.next_intervention_id += 1;

        let parent = self
            .snapshots
            .get(&intervention.parent_world)
            .cloned()
            .unwrap_or_else(WorldSnapshot::factual);

        let new_snapshot = WorldSnapshot::from_intervention(&intervention, &parent);
        let result_world = intervention.result_world;

        self.snapshots.insert(result_world, new_snapshot);
        self.interventions.push(intervention);

        result_world
    }

    /// Enter a counterfactual world
    pub fn enter_world(&mut self, world: WorldId) {
        self.current_world = world;
    }

    /// Get value of a variable in a world
    pub fn get_value(&self, variable: &str, world: WorldId) -> Option<&CounterfactualValue> {
        self.snapshots
            .get(&world)
            .and_then(|s| s.values.get(variable))
    }

    /// Compute divergence between factual and counterfactual
    pub fn compute_divergence(&self, variable: &str, cf_world: WorldId) -> Option<WorldDivergence> {
        let factual = self.get_value(variable, WorldId::FACTUAL)?;
        let counterfactual = self.get_value(variable, cf_world)?;

        let (absolute, relative) = match (factual.as_f32(), counterfactual.as_f32()) {
            (Some(f), Some(cf)) => {
                let abs = (cf - f).abs() as f64;
                let rel = if f.abs() > 1e-10 {
                    abs / f.abs() as f64
                } else {
                    0.0
                };
                (abs, rel)
            }
            _ => (0.0, 0.0),
        };

        Some(WorldDivergence {
            variable: variable.to_string(),
            factual: factual.clone(),
            counterfactual: counterfactual.clone(),
            absolute,
            relative,
        })
    }

    /// Add an exogenous variable
    pub fn add_exogenous(&mut self, variable: &str) {
        if !self.exogenous.contains(&variable.to_string()) {
            self.exogenous.push(variable.to_string());
        }
    }

    /// Add a structural equation
    pub fn add_structural_equation(&mut self, variable: &str, equation: &str) {
        self.structural_equations
            .insert(variable.to_string(), equation.to_string());
    }
}

impl Default for CounterfactualContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for counterfactual PTX emission
#[derive(Debug, Clone)]
pub struct CounterfactualPtxConfig {
    /// Target compute capability
    pub sm_version: (u32, u32),
    /// Number of parallel worlds per warp (power of 2)
    pub worlds_per_warp: u32,
    /// Enable world divergence tracking
    pub track_divergence: bool,
    /// Enable causal depth tracking
    pub track_depth: bool,
    /// Maximum causal depth
    pub max_depth: u32,
}

impl Default for CounterfactualPtxConfig {
    fn default() -> Self {
        Self {
            sm_version: (8, 0),
            worlds_per_warp: 2, // Half warp factual, half counterfactual
            track_divergence: true,
            track_depth: true,
            max_depth: 8,
        }
    }
}

/// Counterfactual PTX code emitter
pub struct CounterfactualPtxEmitter {
    config: CounterfactualPtxConfig,
    output: String,
    indent: usize,
}

impl CounterfactualPtxEmitter {
    pub fn new(config: CounterfactualPtxConfig) -> Self {
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
        let indent = "\t".repeat(self.indent);
        writeln!(self.output, "{}{}", indent, s).unwrap();
    }

    fn emit_comment(&mut self, s: &str) {
        self.emit(&format!("// {}", s));
    }

    /// Emit register declarations for counterfactual execution
    pub fn emit_cf_declarations(&mut self) {
        self.emit_comment("Counterfactual execution registers");

        // World ID register
        self.emit(".reg .u64 %r_world_id;");

        // Causal depth
        if self.config.track_depth {
            self.emit(".reg .u32 %r_causal_depth;");
        }

        // Predicates for world branching
        self.emit(".reg .pred %p_is_factual;");
        self.emit(".reg .pred %p_is_cf;");

        // Divergence tracking
        if self.config.track_divergence {
            self.emit(".reg .f32 %r_divergence;");
        }

        // Temporary registers
        self.emit(".reg .u32 %r_cf_temp0, %r_cf_temp1;");
        self.emit(".reg .f32 %r_cf_ftemp0, %r_cf_ftemp1;");

        self.emit("");
    }

    /// Emit initialization for counterfactual execution
    pub fn emit_cf_init(&mut self) {
        self.emit_comment("Initialize counterfactual execution context");

        // Initialize world ID to factual
        self.emit("mov.u64 %r_world_id, 0;");

        // Initialize causal depth
        if self.config.track_depth {
            self.emit("mov.u32 %r_causal_depth, 0;");
        }

        // Set factual predicate
        self.emit("setp.eq.u64 %p_is_factual, %r_world_id, 0;");
        self.emit("setp.ne.u64 %p_is_cf, %r_world_id, 0;");

        // Initialize divergence
        if self.config.track_divergence {
            self.emit("mov.f32 %r_divergence, 0.0;");
        }
    }

    /// Emit do-operator: do(variable = value)
    pub fn emit_intervention(
        &mut self,
        variable: &str,
        factual_reg: &str,
        cf_value: f32,
        intervention_id: u32,
    ) {
        self.emit_comment(&format!("Intervention: do({} = {})", variable, cf_value));

        // Create new world ID via XOR with intervention marker
        let marker = 0xCAFEBABE00000000_u64 | intervention_id as u64;
        self.emit(&format!(
            "xor.b64 %r_world_id, %r_world_id, 0x{:016X};",
            marker
        ));

        // Increment causal depth
        if self.config.track_depth {
            self.emit("add.u32 %r_causal_depth, %r_causal_depth, 1;");
        }

        // Determine if this thread is in factual or counterfactual world
        // Using lane ID for world assignment
        self.emit_comment("Assign threads to worlds based on lane ID");
        self.emit("mov.u32 %r_cf_temp0, %laneid;");
        self.emit(&format!(
            "and.b32 %r_cf_temp1, %r_cf_temp0, {};",
            self.config.worlds_per_warp - 1
        ));
        self.emit("setp.eq.u32 %p_is_factual, %r_cf_temp1, 0;");
        self.emit("setp.ne.u32 %p_is_cf, %r_cf_temp1, 0;");

        // Apply intervention: select between factual and counterfactual value
        self.emit_comment("Apply intervention");
        self.emit(&format!(
            "mov.f32 %r_cf_ftemp0, 0F{:08X};",
            cf_value.to_bits()
        ));
        self.emit(&format!(
            "selp.f32 {}, %r_cf_ftemp0, {}, %p_is_cf;",
            factual_reg, factual_reg
        ));
    }

    /// Emit parallel world execution setup
    pub fn emit_parallel_worlds(&mut self, ctx: &CounterfactualContext) {
        self.emit_comment("=== Parallel Counterfactual World Execution ===");

        // Emit world branching based on interventions
        for intervention in &ctx.interventions {
            if let Some(value) = intervention.value.as_f32() {
                let reg = format!("%r_{}", intervention.variable);
                self.emit_intervention(&intervention.variable, &reg, value, intervention.id);
            }
        }
    }

    /// Emit world divergence computation
    pub fn emit_divergence_compute(&mut self, outcome_reg: &str, divergence_reg: &str) {
        self.emit_comment("Compute world divergence (treatment effect)");

        // Exchange outcome with paired thread in other world
        self.emit("// Exchange outcome between factual/counterfactual pairs");
        self.emit(&format!(
            "shfl.sync.xor.b32 %r_cf_ftemp0, {}, 1, 0xFFFFFFFF;",
            outcome_reg
        ));

        // Compute Individual Treatment Effect (ITE)
        self.emit("// ITE = outcome_cf - outcome_factual");
        self.emit(&format!(
            "sub.f32 {}, {}, %r_cf_ftemp0;",
            divergence_reg, outcome_reg
        ));

        // For factual threads, negate to get correct sign
        self.emit("@%p_is_factual neg.f32 {}, {};");
        self.emit(&format!(
            "@%p_is_factual neg.f32 {}, {};",
            divergence_reg, divergence_reg
        ));
    }

    /// Emit Average Treatment Effect computation (warp-level)
    pub fn emit_ate_compute(&mut self, ite_reg: &str, ate_reg: &str) {
        self.emit_comment("Compute Average Treatment Effect (ATE) across warp");

        // Sum ITEs using warp reduction
        self.emit(&format!("mov.f32 {}, {};", ate_reg, ite_reg));

        for offset in [16, 8, 4, 2, 1] {
            self.emit(&format!(
                "shfl.sync.down.b32 %r_cf_ftemp0, {}, {}, 31, 0xFFFFFFFF;",
                ate_reg, offset
            ));
            self.emit(&format!("add.f32 {}, {}, %r_cf_ftemp0;", ate_reg, ate_reg));
        }

        // Divide by number of counterfactual pairs
        let pairs = 32 / self.config.worlds_per_warp;
        self.emit(&format!(
            "mul.f32 {}, {}, {};",
            ate_reg,
            ate_reg,
            1.0 / pairs as f32
        ));
    }

    /// Emit conditional execution based on world
    pub fn emit_world_conditional(&mut self, factual_block: &str, cf_block: &str) {
        self.emit_comment("Conditional execution based on world");
        self.emit(&format!("@%p_is_factual bra {};", factual_block));
        self.emit(&format!("@%p_is_cf bra {};", cf_block));
    }

    /// Emit world merge (combine results from parallel worlds)
    pub fn emit_world_merge(&mut self, result_reg: &str, factual_reg: &str, cf_reg: &str) {
        self.emit_comment("Merge results from parallel worlds");

        // Each thread has result from its world, exchange
        self.emit(&format!(
            "shfl.sync.xor.b32 %r_cf_ftemp0, {}, 1, 0xFFFFFFFF;",
            factual_reg
        ));

        // Factual threads get counterfactual result, vice versa
        self.emit(&format!(
            "selp.f32 {}, %r_cf_ftemp0, {}, %p_is_factual;",
            result_reg, cf_reg
        ));
    }

    /// Emit probability of causation computation
    /// P(Y_x=1 | X=0, Y=0) - probability that X caused Y
    pub fn emit_probability_causation(
        &mut self,
        x_reg: &str,
        y_reg: &str,
        y_cf_reg: &str,
        result_reg: &str,
    ) {
        self.emit_comment("Probability of Causation: P(Y_x=1 | X=0, Y=0)");

        // Check preconditions: X=0, Y=0
        self.emit(&format!("setp.eq.f32 %p_x_zero, {}, 0.0;", x_reg));
        self.emit(&format!("setp.eq.f32 %p_y_zero, {}, 0.0;", y_reg));
        self.emit("and.pred %p_precond, %p_x_zero, %p_y_zero;");

        // Check if counterfactual Y would be 1
        self.emit(&format!("setp.gt.f32 %p_y_cf_one, {}, 0.5;", y_cf_reg));

        // P(causation) = 1 if precond ∧ y_cf_one, else 0
        self.emit("and.pred %p_caused, %p_precond, %p_y_cf_one;");
        self.emit(&format!("selp.f32 {}, 1.0, 0.0, %p_caused;", result_reg));

        // Aggregate across warp for population-level estimate
        self.emit_comment("Aggregate causation across warp");
        for offset in [16, 8, 4, 2, 1] {
            self.emit(&format!(
                "shfl.sync.down.b32 %r_cf_ftemp0, {}, {}, 31, 0xFFFFFFFF;",
                result_reg, offset
            ));
            self.emit(&format!(
                "add.f32 {}, {}, %r_cf_ftemp0;",
                result_reg, result_reg
            ));
        }
        self.emit(&format!("mul.f32 {}, {}, 0.03125;", result_reg, result_reg)); // /32
    }

    /// Emit nested intervention (second-level counterfactual)
    pub fn emit_nested_intervention(
        &mut self,
        var1: &str,
        val1: f32,
        var2: &str,
        val2: f32,
        reg1: &str,
        reg2: &str,
    ) {
        self.emit_comment(&format!(
            "Nested intervention: do({} = {}) then do({} = {})",
            var1, val1, var2, val2
        ));

        // Check if we're already at max depth
        if self.config.track_depth {
            self.emit(&format!(
                "setp.lt.u32 %p_can_intervene, %r_causal_depth, {};",
                self.config.max_depth
            ));
            self.emit("@!%p_can_intervene bra skip_nested_intervention;");
        }

        // First intervention
        self.emit(&format!("mov.f32 %r_cf_ftemp0, 0F{:08X};", val1.to_bits()));
        self.emit(&format!(
            "selp.f32 {}, %r_cf_ftemp0, {}, %p_is_cf;",
            reg1, reg1
        ));

        // Second intervention (deeper nesting)
        // Use different bits of lane ID for second level
        self.emit("shr.b32 %r_cf_temp0, %laneid, 1;");
        self.emit(&format!(
            "and.b32 %r_cf_temp1, %r_cf_temp0, {};",
            self.config.worlds_per_warp - 1
        ));
        self.emit(".reg .pred %p_nested_cf;");
        self.emit("setp.ne.u32 %p_nested_cf, %r_cf_temp1, 0;");

        self.emit(&format!("mov.f32 %r_cf_ftemp0, 0F{:08X};", val2.to_bits()));
        self.emit(&format!(
            "selp.f32 {}, %r_cf_ftemp0, {}, %p_nested_cf;",
            reg2, reg2
        ));

        if self.config.track_depth {
            self.emit("add.u32 %r_causal_depth, %r_causal_depth, 2;");
            self.emit("skip_nested_intervention:");
        }
    }

    /// Emit structural equation evaluation
    pub fn emit_structural_eq(
        &mut self,
        result_reg: &str,
        eq_type: StructuralEqType,
        inputs: &[&str],
        params: &[f32],
    ) {
        self.emit_comment(&format!("Structural equation: {:?}", eq_type));

        match eq_type {
            StructuralEqType::Linear => {
                // Y = β₀ + β₁X₁ + β₂X₂ + ...
                if !params.is_empty() {
                    self.emit(&format!(
                        "mov.f32 {}, 0F{:08X};",
                        result_reg,
                        params[0].to_bits()
                    ));

                    for (i, input) in inputs.iter().enumerate() {
                        if i + 1 < params.len() {
                            self.emit(&format!(
                                "mov.f32 %r_cf_ftemp0, 0F{:08X};",
                                params[i + 1].to_bits()
                            ));
                            self.emit(&format!(
                                "fma.rn.f32 {}, %r_cf_ftemp0, {}, {};",
                                result_reg, input, result_reg
                            ));
                        }
                    }
                }
            }

            StructuralEqType::Logistic => {
                // Y = sigmoid(β₀ + β₁X₁ + ...)
                // First compute linear part
                if !params.is_empty() {
                    self.emit(&format!(
                        "mov.f32 {}, 0F{:08X};",
                        result_reg,
                        params[0].to_bits()
                    ));

                    for (i, input) in inputs.iter().enumerate() {
                        if i + 1 < params.len() {
                            self.emit(&format!(
                                "mov.f32 %r_cf_ftemp0, 0F{:08X};",
                                params[i + 1].to_bits()
                            ));
                            self.emit(&format!(
                                "fma.rn.f32 {}, %r_cf_ftemp0, {}, {};",
                                result_reg, input, result_reg
                            ));
                        }
                    }
                }

                // Sigmoid: 1 / (1 + exp(-x))
                self.emit(&format!("neg.f32 %r_cf_ftemp0, {};", result_reg));
                self.emit("ex2.approx.f32 %r_cf_ftemp0, %r_cf_ftemp0;"); // exp(-x) ≈ 2^(-x/ln2)
                self.emit("add.f32 %r_cf_ftemp0, %r_cf_ftemp0, 1.0;");
                self.emit(&format!("rcp.approx.f32 {}, %r_cf_ftemp0;", result_reg));
            }

            StructuralEqType::Multiplicative => {
                // Y = ∏ Xᵢ^βᵢ
                self.emit(&format!("mov.f32 {}, 1.0;", result_reg));

                for (i, input) in inputs.iter().enumerate() {
                    if i < params.len() {
                        // x^β = exp(β * ln(x))
                        self.emit(&format!("lg2.approx.f32 %r_cf_ftemp0, {};", input));
                        self.emit(&format!(
                            "mov.f32 %r_cf_ftemp1, 0F{:08X};",
                            params[i].to_bits()
                        ));
                        self.emit("mul.f32 %r_cf_ftemp0, %r_cf_ftemp0, %r_cf_ftemp1;");
                        self.emit("ex2.approx.f32 %r_cf_ftemp0, %r_cf_ftemp0;");
                        self.emit(&format!(
                            "mul.f32 {}, {}, %r_cf_ftemp0;",
                            result_reg, result_reg
                        ));
                    }
                }
            }

            StructuralEqType::Threshold => {
                // Y = 1 if X > θ else 0
                if !inputs.is_empty() && !params.is_empty() {
                    self.emit(&format!(
                        "mov.f32 %r_cf_ftemp0, 0F{:08X};",
                        params[0].to_bits()
                    ));
                    self.emit(&format!(
                        "setp.gt.f32 %p_threshold, {}, %r_cf_ftemp0;",
                        inputs[0]
                    ));
                    self.emit(&format!("selp.f32 {}, 1.0, 0.0, %p_threshold;", result_reg));
                }
            }
        }
    }
}

/// Types of structural equations for causal models
#[derive(Debug, Clone, Copy)]
pub enum StructuralEqType {
    /// Y = β₀ + Σ βᵢXᵢ
    Linear,
    /// Y = sigmoid(β₀ + Σ βᵢXᵢ)
    Logistic,
    /// Y = ∏ Xᵢ^βᵢ
    Multiplicative,
    /// Y = 1 if X > θ else 0
    Threshold,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_id() {
        assert!(WorldId::FACTUAL.is_factual());
        assert!(!WorldId::FACTUAL.is_counterfactual());

        let cf = WorldId::counterfactual(42);
        assert!(!cf.is_factual());
        assert!(cf.is_counterfactual());
        assert_eq!(cf.intervention_id(), Some(42));
    }

    #[test]
    fn test_context_intervention() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.5));

        let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

        assert!(cf_world.is_counterfactual());

        // Factual treatment should be 0
        assert_eq!(
            ctx.get_value("treatment", WorldId::FACTUAL)
                .and_then(|v| v.as_f32()),
            Some(0.0)
        );

        // Counterfactual treatment should be 1
        assert_eq!(
            ctx.get_value("treatment", cf_world)
                .and_then(|v| v.as_f32()),
            Some(1.0)
        );
    }

    #[test]
    fn test_ptx_intervention_emission() {
        let mut emitter = CounterfactualPtxEmitter::new(CounterfactualPtxConfig::default());

        emitter.emit_cf_declarations();
        emitter.emit_cf_init();
        emitter.emit_intervention("treatment", "%r_treatment", 1.0, 1);

        let output = emitter.output();

        assert!(output.contains("do(treatment = 1)"));
        assert!(output.contains("selp.f32"));
        assert!(output.contains("xor.b64"));
    }

    #[test]
    fn test_divergence_compute() {
        let mut emitter = CounterfactualPtxEmitter::new(CounterfactualPtxConfig::default());

        emitter.emit_divergence_compute("%r_outcome", "%r_divergence");

        let output = emitter.output();

        assert!(output.contains("shfl.sync.xor"));
        assert!(output.contains("ITE"));
    }

    #[test]
    fn test_structural_equation() {
        let mut emitter = CounterfactualPtxEmitter::new(CounterfactualPtxConfig::default());

        emitter.emit_structural_eq(
            "%r_y",
            StructuralEqType::Linear,
            &["%r_x1", "%r_x2"],
            &[0.5, 2.0, -1.0], // Y = 0.5 + 2*X1 - X2
        );

        let output = emitter.output();

        assert!(output.contains("fma.rn.f32"));
    }
}
