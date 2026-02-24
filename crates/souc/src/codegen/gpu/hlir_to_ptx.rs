//! HLIR -> PTX lowering entrypoints wired to the epistemic PTX bridge.
//!
//! This module keeps a thin boundary between high-level lowering
//! (`hlir_to_gpu`) and final PTX emission (`ptx` + `ptx_epistemic_bridge`).

use super::hlir_to_gpu::{lower, lower_with_config, LoweringConfig};
use super::ptx::PtxCodegen;
use super::ptx_epistemic_bridge::EpistemicPtxCodegen;
use super::GpuTarget;
use crate::hlir::HlirModule;
use crate::optimizer::EpistemicIntrinsicCounters;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicPtxOutput {
    pub ptx: String,
    pub intrinsic_counters: EpistemicIntrinsicCounters,
}

/// Compile HLIR to standard PTX without epistemic bridge annotations.
pub fn compile_to_ptx(hlir: &HlirModule, sm_version: (u32, u32)) -> String {
    let target = GpuTarget::Cuda {
        compute_capability: sm_version,
    };
    let gpu_module = lower(hlir, target);
    let mut codegen = PtxCodegen::new(sm_version);
    codegen.generate(&gpu_module)
}

/// Compile HLIR to PTX with epistemic bridge enabled/disabled.
pub fn compile_to_ptx_epistemic_with_stats(
    hlir: &HlirModule,
    sm_version: (u32, u32),
    epistemic_enabled: bool,
) -> EpistemicPtxOutput {
    let config = LoweringConfig {
        target: GpuTarget::Cuda {
            compute_capability: sm_version,
        },
        epistemic_enabled,
        ..Default::default()
    };

    let gpu_module = lower_with_config(hlir, &config);

    if epistemic_enabled {
        let mut codegen = EpistemicPtxCodegen::new(sm_version, true);
        let ptx = codegen.generate(&gpu_module);
        EpistemicPtxOutput {
            ptx,
            intrinsic_counters: codegen.intrinsic_counters(),
        }
    } else {
        let mut codegen = PtxCodegen::new(sm_version);
        let ptx = codegen.generate(&gpu_module);
        EpistemicPtxOutput {
            ptx,
            intrinsic_counters: EpistemicIntrinsicCounters::default(),
        }
    }
}

/// Backward-compatible API that returns only PTX text.
pub fn compile_to_ptx_epistemic(
    hlir: &HlirModule,
    sm_version: (u32, u32),
    epistemic_enabled: bool,
) -> String {
    compile_to_ptx_epistemic_with_stats(hlir, sm_version, epistemic_enabled).ptx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlir::{
        BlockId as HlirBlockId, HlirBlock, HlirFunction, HlirModule, HlirParam, HlirTerminator,
        HlirType, Op, ValueId as HlirValueId,
    };
    use std::collections::HashMap;

    fn tiny_hlir() -> HlirModule {
        let mut module = HlirModule::new("ptx_bridge_test");
        module.functions.push(HlirFunction {
            id: crate::hlir::FunctionId(0),
            name: "k0".to_string(),
            link_name: None,
            params: vec![HlirParam {
                value: HlirValueId(0),
                name: "x".to_string(),
                ty: HlirType::F32,
            }],
            return_type: HlirType::Void,
            effects: vec![],
            blocks: vec![HlirBlock {
                id: HlirBlockId(0),
                label: "entry".to_string(),
                params: vec![],
                instructions: vec![crate::hlir::HlirInstr {
                    result: Some(HlirValueId(1)),
                    op: Op::Const(crate::hlir::HlirConstant::Float(1.0, HlirType::F32)),
                    ty: HlirType::F32,
                }],
                terminator: HlirTerminator::Return(None),
            }],
            is_kernel: true,
            locals: HashMap::new(),
            is_variadic: false,
            abi: crate::ast::Abi::Rust,
            is_exported: false,
        });
        module
    }

    #[test]
    fn emits_epistemic_bridge_metadata_when_enabled() {
        let out = compile_to_ptx_epistemic_with_stats(&tiny_hlir(), (8, 0), true);
        assert!(out.ptx.contains("Sounio Epistemic PTX Bridge"));
    }

    #[test]
    fn emits_plain_ptx_when_disabled() {
        let out = compile_to_ptx_epistemic_with_stats(&tiny_hlir(), (8, 0), false);
        assert!(!out.ptx.contains("Sounio Epistemic PTX Bridge"));
    }
}
