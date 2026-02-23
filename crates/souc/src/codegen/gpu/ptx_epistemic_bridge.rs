//! Bridge between PTX codegen and epistemic PTX emission
//!
//! This module connects the standard PTX code generator with the epistemic
//! shadow register tracking system, enabling automatic epistemic state
//! propagation in generated PTX code.

use super::epistemic_ptx::{EpistemicPtxConfig, EpistemicPtxEmitter, EpistemicShadowRegs};
use super::ir::{GpuModule, GpuOp, ValueId};
use super::ptx::PtxCodegen;
use crate::optimizer::{EpistemicIntrinsicCounters, EpistemicPowerTracker};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fmt::{Display, Formatter};

/// PTX register wrapper used by intrinsic emitters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PtxReg(String);

impl PtxReg {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl Display for PtxReg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn emit_knowledge_fma(out: &mut String, dst: &PtxReg, a: &PtxReg, b: &PtxReg, c: &PtxReg) {
    writeln!(
        out,
        "// .knowledge.fma.rn.f64 {}, {}, {}, {};",
        dst, a, b, c
    )
    .unwrap();
    writeln!(out, "// .knowledge.prop_var.f64 %rd_var_out, {};", a).unwrap();
    writeln!(out, "// .knowledge.merge_prov %rd_prov_out, {}, {};", a, b).unwrap();
    // PTX-safe fallback expansion.
    writeln!(out, "fma.rn.f64 {}, {}, {}, {};", dst, a, b, c).unwrap();
}

pub fn emit_knowledge_add(out: &mut String, dst: &PtxReg, a: &PtxReg, b: &PtxReg) {
    writeln!(out, "// .knowledge.add.rn.f64 {}, {}, {};", dst, a, b).unwrap();
    writeln!(out, "// variance = u_a^2 + u_b^2 (GUM, independent inputs)").unwrap();
    writeln!(out, "add.rn.f64 {}, {}, {};", dst, a, b).unwrap();
}

pub fn emit_knowledge_mul(out: &mut String, dst: &PtxReg, a: &PtxReg, b: &PtxReg) {
    writeln!(out, "// .knowledge.mul.rn.f64 {}, {}, {};", dst, a, b).unwrap();
    writeln!(out, "// variance = b^2*u_a^2 + a^2*u_b^2 (delta method)").unwrap();
    writeln!(out, "mul.rn.f64 {}, {}, {};", dst, a, b).unwrap();
}

pub fn emit_knowledge_div(out: &mut String, dst: &PtxReg, a: &PtxReg, b: &PtxReg) {
    writeln!(out, "// .knowledge.div.rn.f64 {}, {}, {};", dst, a, b).unwrap();
    writeln!(
        out,
        "// variance = (u_a^2 / b^2) + (a^2*u_b^2 / b^4) with near-zero guard in runtime stub"
    )
    .unwrap();
    writeln!(out, "div.rn.f64 {}, {}, {};", dst, a, b).unwrap();
}

pub fn emit_knowledge_prop_var(out: &mut String, dst: &PtxReg, src: &PtxReg) {
    writeln!(out, "// .knowledge.prop_var.f64 {}, {};", dst, src).unwrap();
    writeln!(out, "// epistemic degradation is applied by runtime policy").unwrap();
    writeln!(out, "mov.f64 {}, {};", dst, src).unwrap();
}

pub fn emit_knowledge_merge_prov(out: &mut String, dst: &PtxReg, a: &PtxReg, b: &PtxReg) {
    writeln!(out, "// .knowledge.merge_prov {}, {}, {};", dst, a, b).unwrap();
    writeln!(
        out,
        "// fast-path provenance merge; runtime may harden with hash mix"
    )
    .unwrap();
    writeln!(out, "xor.b64 {}, {}, {};", dst, a, b).unwrap();
}

/// Extended PTX codegen with epistemic support
pub struct EpistemicPtxCodegen {
    /// Base PTX code generator
    pub base: PtxCodegen,
    /// Epistemic shadow emitter
    pub epistemic: EpistemicPtxEmitter,
    /// Mapping from value IDs to their shadow register sets
    shadow_map: HashMap<ValueId, EpistemicShadowRegs>,
    /// Whether epistemic tracking is enabled
    enabled: bool,
    /// Intrinsic usage tracker for EpistemicPower integration
    power_tracker: EpistemicPowerTracker,
}

impl EpistemicPtxCodegen {
    /// Create a new epistemic PTX codegen
    pub fn new(sm_version: (u32, u32), epistemic_enabled: bool) -> Self {
        let config = EpistemicPtxConfig {
            sm_version,
            ..Default::default()
        };

        Self {
            base: PtxCodegen::new(sm_version),
            epistemic: EpistemicPtxEmitter::new(config),
            shadow_map: HashMap::new(),
            enabled: epistemic_enabled,
            power_tracker: EpistemicPowerTracker::default(),
        }
    }

    pub fn intrinsic_counters(&self) -> EpistemicIntrinsicCounters {
        self.power_tracker.counters()
    }

    pub fn power_tracker(&self) -> EpistemicPowerTracker {
        self.power_tracker
    }

    /// Generate PTX with optional epistemic tracking
    pub fn generate(&mut self, module: &GpuModule) -> String {
        // First generate standard PTX
        let base_ptx = self.base.generate(module);

        if !self.enabled {
            return base_ptx;
        }

        self.epistemic.clear();
        self.shadow_map.clear();
        self.power_tracker = EpistemicPowerTracker::default();

        let candidate_ops = module
            .kernels
            .values()
            .flat_map(|kernel| kernel.blocks.iter())
            .flat_map(|block| block.instructions.iter())
            .filter(|(_, op)| Self::is_epistemic_candidate(op))
            .count();

        if candidate_ops > 0 {
            self.epistemic
                .emit_shadow_declarations((candidate_ops as u32).saturating_mul(3).max(16));
        }

        let mut supported_ops = 0usize;
        let mut fallback_ops = 0usize;

        // Deterministic order keeps generated output stable across runs.
        let mut kernel_names: Vec<_> = module.kernels.keys().cloned().collect();
        kernel_names.sort();

        for kernel_name in kernel_names {
            if let Some(kernel) = module.kernels.get(&kernel_name) {
                for block in &kernel.blocks {
                    for (result, op) in &block.instructions {
                        self.seed_operand_shadows(op);
                        if Self::is_epistemic_candidate(op) {
                            if self.emit_epistemic_op(op, *result) {
                                supported_ops += 1;
                            } else {
                                fallback_ops += 1;
                                self.power_tracker.record_fallback();
                            }
                        }
                    }
                }
            }
        }

        let counters = self.intrinsic_counters();
        let mut out = base_ptx;
        out.push_str("\n// === Sounio Epistemic PTX Bridge ===\n");
        out.push_str(&format!(
            "// candidate_ops={} supported_ops={} fallback_ops={}\n",
            candidate_ops, supported_ops, fallback_ops
        ));
        out.push_str(
            "// fallback policy: keep base PTX semantics when epistemic lowering is unavailable\n",
        );
        out.push_str(&format!(
            "// intrinsic_usage: fma={} add={} mul={} div={} prop_var={} merge_prov={} other={} fallback={}\n",
            counters.knowledge_fma,
            counters.knowledge_add,
            counters.knowledge_mul,
            counters.knowledge_div,
            counters.knowledge_prop_var,
            counters.knowledge_merge_prov,
            counters.other,
            counters.fallback
        ));
        out.push_str("// intrinsic_templates (commented, PTX-safe):\n");
        out.push_str(&self.emit_intrinsic_manifest());
        out
    }

    fn emit_intrinsic_manifest(&self) -> String {
        let mut raw = String::new();
        let dst = PtxReg::new("%fd_dst");
        let a = PtxReg::new("%fd_a");
        let b = PtxReg::new("%fd_b");
        let c = PtxReg::new("%fd_c");
        let var_out = PtxReg::new("%fd_var_out");
        let prov_out = PtxReg::new("%rd_prov_out");

        emit_knowledge_fma(&mut raw, &dst, &a, &b, &c);
        emit_knowledge_add(&mut raw, &dst, &a, &b);
        emit_knowledge_mul(&mut raw, &dst, &a, &b);
        emit_knowledge_div(&mut raw, &dst, &a, &b);
        emit_knowledge_prop_var(&mut raw, &var_out, &a);
        emit_knowledge_merge_prov(&mut raw, &prov_out, &a, &b);

        let mut commented = String::new();
        for line in raw.lines() {
            commented.push_str("// ");
            commented.push_str(line);
            commented.push('\n');
        }
        commented
    }

    fn is_epistemic_candidate(op: &GpuOp) -> bool {
        matches!(
            op,
            GpuOp::FAdd(_, _)
                | GpuOp::Add(_, _)
                | GpuOp::FSub(_, _)
                | GpuOp::Sub(_, _)
                | GpuOp::FMul(_, _)
                | GpuOp::Mul(_, _)
                | GpuOp::FDiv(_, _)
                | GpuOp::Div(_, _)
                | GpuOp::FMulAdd(_, _, _)
        )
    }

    fn ensure_shadow_for(&mut self, value: ValueId) {
        if self.shadow_map.contains_key(&value) {
            return;
        }
        let _ = self.alloc_shadow(value, &format!("v{}", value.0));
    }

    fn seed_operand_shadows(&mut self, op: &GpuOp) {
        match op {
            GpuOp::FAdd(left, right)
            | GpuOp::Add(left, right)
            | GpuOp::FSub(left, right)
            | GpuOp::Sub(left, right)
            | GpuOp::FMul(left, right)
            | GpuOp::Mul(left, right)
            | GpuOp::FDiv(left, right)
            | GpuOp::Div(left, right) => {
                self.ensure_shadow_for(*left);
                self.ensure_shadow_for(*right);
            }
            GpuOp::FMulAdd(a, b, c) => {
                self.ensure_shadow_for(*a);
                self.ensure_shadow_for(*b);
                self.ensure_shadow_for(*c);
            }
            _ => {}
        }
    }

    /// Allocate shadow registers for a value
    pub fn alloc_shadow(&mut self, value: ValueId, base_reg: &str) -> EpistemicShadowRegs {
        let shadow = self.epistemic.alloc_shadow(base_reg);
        self.shadow_map.insert(value, shadow.clone());
        shadow
    }

    /// Get shadow registers for a value
    pub fn get_shadow(&self, value: ValueId) -> Option<&EpistemicShadowRegs> {
        self.shadow_map.get(&value)
    }

    /// Emit epistemic operation
    pub fn emit_epistemic_op(&mut self, op: &GpuOp, result: ValueId) -> bool {
        if !self.enabled {
            return false;
        }

        match op {
            GpuOp::FAdd(left, right) | GpuOp::Add(left, right) => {
                // Clone shadow data BEFORE calling alloc_shadow to avoid borrow conflict
                let shadow_data = match (self.shadow_map.get(left), self.shadow_map.get(right)) {
                    (Some(l), Some(r)) => Some((l.clone(), r.clone())),
                    _ => None,
                };

                if let Some((l_shadow, r_shadow)) = shadow_data {
                    let result_shadow = self.alloc_shadow(result, &format!("v{}", result.0));
                    self.epistemic
                        .emit_epistemic_add(&result_shadow, &l_shadow, &r_shadow, false);
                    self.power_tracker.record_intrinsic("knowledge.add", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.merge_prov", 1.0);
                    return true;
                }
            }

            GpuOp::FSub(left, right) | GpuOp::Sub(left, right) => {
                // Clone shadow data BEFORE calling alloc_shadow to avoid borrow conflict
                let shadow_data = match (self.shadow_map.get(left), self.shadow_map.get(right)) {
                    (Some(l), Some(r)) => Some((l.clone(), r.clone())),
                    _ => None,
                };

                if let Some((l_shadow, r_shadow)) = shadow_data {
                    let result_shadow = self.alloc_shadow(result, &format!("v{}", result.0));
                    self.epistemic
                        .emit_epistemic_add(&result_shadow, &l_shadow, &r_shadow, true);
                    self.power_tracker.record_intrinsic("knowledge.add", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.merge_prov", 1.0);
                    return true;
                }
            }

            GpuOp::FMul(left, right) | GpuOp::Mul(left, right) => {
                // Clone shadow data BEFORE calling alloc_shadow to avoid borrow conflict
                let shadow_data = match (self.shadow_map.get(left), self.shadow_map.get(right)) {
                    (Some(l), Some(r)) => Some((l.clone(), r.clone())),
                    _ => None,
                };

                if let Some((l_shadow, r_shadow)) = shadow_data {
                    let result_shadow = self.alloc_shadow(result, &format!("v{}", result.0));
                    self.epistemic
                        .emit_epistemic_mul(&result_shadow, &l_shadow, &r_shadow);
                    self.power_tracker.record_intrinsic("knowledge.mul", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.prop_var", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.merge_prov", 1.0);
                    return true;
                }
            }

            GpuOp::FDiv(left, right) | GpuOp::Div(left, right) => {
                // Clone shadow data BEFORE calling alloc_shadow to avoid borrow conflict
                let shadow_data = match (self.shadow_map.get(left), self.shadow_map.get(right)) {
                    (Some(l), Some(r)) => Some((l.clone(), r.clone())),
                    _ => None,
                };

                if let Some((l_shadow, r_shadow)) = shadow_data {
                    let result_shadow = self.alloc_shadow(result, &format!("v{}", result.0));
                    self.epistemic
                        .emit_epistemic_div(&result_shadow, &l_shadow, &r_shadow);
                    self.power_tracker.record_intrinsic("knowledge.div", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.prop_var", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.merge_prov", 1.0);
                    return true;
                }
            }

            GpuOp::FMulAdd(a, b, c) => {
                let shadow_data = match (
                    self.shadow_map.get(a),
                    self.shadow_map.get(b),
                    self.shadow_map.get(c),
                ) {
                    (Some(a_shadow), Some(b_shadow), Some(c_shadow)) => {
                        Some((a_shadow.clone(), b_shadow.clone(), c_shadow.clone()))
                    }
                    _ => None,
                };

                if let Some((a_shadow, b_shadow, c_shadow)) = shadow_data {
                    let result_shadow = self.alloc_shadow(result, &format!("v{}", result.0));
                    self.epistemic.emit_epistemic_fma(
                        &result_shadow,
                        &a_shadow,
                        &b_shadow,
                        &c_shadow,
                    );
                    self.power_tracker.record_intrinsic("knowledge.fma", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.prop_var", 1.0);
                    self.power_tracker
                        .record_intrinsic("knowledge.merge_prov", 1.0);
                    return true;
                }
            }

            _ => {
                // Other operations don't have special epistemic handling yet
            }
        }
        false
    }

    /// Get the epistemic PTX output (shadow register operations)
    pub fn epistemic_output(&self) -> &str {
        self.epistemic.output()
    }
}

#[cfg(test)]
mod tests {
    use super::super::ir::{BlockId, GpuBlock, GpuKernel, GpuTarget, GpuTerminator};
    use super::*;

    #[test]
    fn test_epistemic_codegen_creation() {
        let codegen = EpistemicPtxCodegen::new((8, 0), true);
        assert!(codegen.enabled);
    }

    #[test]
    fn test_shadow_allocation() {
        let mut codegen = EpistemicPtxCodegen::new((8, 0), true);
        let value = ValueId(42);
        let shadow = codegen.alloc_shadow(value, "test_reg");

        // alloc_shadow adds counter suffix and PTX register prefix
        assert_eq!(shadow.value, "%r_test_reg_1");
        assert!(codegen.get_shadow(value).is_some());
    }

    #[test]
    fn test_disabled_epistemic() {
        let codegen = EpistemicPtxCodegen::new((8, 0), false);
        assert!(!codegen.enabled);
        assert_eq!(codegen.epistemic_output(), "");
    }

    #[test]
    fn test_generate_injects_epistemic_bridge_section() {
        let mut module = GpuModule::new(
            "epistemic_test",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        let mut kernel = GpuKernel::new("k0");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(
            ValueId(0),
            GpuOp::ConstFloat(1.0, super::super::ir::GpuType::F32),
        );
        block.add_instruction(
            ValueId(1),
            GpuOp::ConstFloat(2.0, super::super::ir::GpuType::F32),
        );
        block.add_instruction(ValueId(2), GpuOp::FAdd(ValueId(0), ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = EpistemicPtxCodegen::new((8, 0), true);
        let ptx = codegen.generate(&module);
        assert!(ptx.contains("Sounio Epistemic PTX Bridge"));
        assert!(ptx.contains("supported_ops="));
        assert!(ptx.contains("intrinsic_usage:"));
        assert!(ptx.contains(".knowledge.add.rn.f64"));
    }

    #[test]
    fn test_emit_fmuladd_epistemic_support() {
        let mut codegen = EpistemicPtxCodegen::new((8, 0), true);
        codegen.alloc_shadow(ValueId(0), "a");
        codegen.alloc_shadow(ValueId(1), "b");
        codegen.alloc_shadow(ValueId(2), "c");
        let ok = codegen.emit_epistemic_op(
            &GpuOp::FMulAdd(ValueId(0), ValueId(1), ValueId(2)),
            ValueId(3),
        );
        assert!(ok);
        assert!(codegen.epistemic_output().contains("Epistemic FMA"));
        let counters = codegen.intrinsic_counters();
        assert_eq!(counters.knowledge_fma, 1);
    }

    #[test]
    fn test_emit_knowledge_intrinsic_templates() {
        let mut out = String::new();
        let dst = PtxReg::new("%fd0");
        let a = PtxReg::new("%fd1");
        let b = PtxReg::new("%fd2");
        let c = PtxReg::new("%fd3");
        let prov = PtxReg::new("%rd0");

        emit_knowledge_fma(&mut out, &dst, &a, &b, &c);
        emit_knowledge_add(&mut out, &dst, &a, &b);
        emit_knowledge_mul(&mut out, &dst, &a, &b);
        emit_knowledge_div(&mut out, &dst, &a, &b);
        emit_knowledge_prop_var(&mut out, &dst, &a);
        emit_knowledge_merge_prov(&mut out, &prov, &a, &b);

        assert!(out.contains(".knowledge.fma.rn.f64"));
        assert!(out.contains(".knowledge.add.rn.f64"));
        assert!(out.contains(".knowledge.mul.rn.f64"));
        assert!(out.contains(".knowledge.div.rn.f64"));
        assert!(out.contains(".knowledge.prop_var.f64"));
        assert!(out.contains(".knowledge.merge_prov"));
    }
}
