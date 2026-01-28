//! Bridge between PTX codegen and epistemic PTX emission
//!
//! This module connects the standard PTX code generator with the epistemic
//! shadow register tracking system, enabling automatic epistemic state
//! propagation in generated PTX code.

use super::epistemic_ptx::{EpistemicPtxConfig, EpistemicPtxEmitter, EpistemicShadowRegs};
use super::ir::{GpuModule, GpuOp, ValueId};
use super::ptx::PtxCodegen;
use std::collections::HashMap;

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
        }
    }

    /// Generate PTX with optional epistemic tracking
    pub fn generate(&mut self, module: &GpuModule) -> String {
        // First generate standard PTX
        let base_ptx = self.base.generate(module);

        if !self.enabled {
            return base_ptx;
        }

        // For now, we return the base PTX since shadow tracking
        // is already embedded in the GPU IR lowering phase.
        // Future enhancement: inject epistemic PTX fragments here.
        base_ptx
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
    pub fn emit_epistemic_op(&mut self, op: &GpuOp, result: ValueId) {
        if !self.enabled {
            return;
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
                }
            }

            _ => {
                // Other operations don't have special epistemic handling yet
            }
        }
    }

    /// Get the epistemic PTX output (shadow register operations)
    pub fn epistemic_output(&self) -> &str {
        self.epistemic.output()
    }
}

#[cfg(test)]
mod tests {
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
}
