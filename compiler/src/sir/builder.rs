//! SIR Builder
//!
//! High-level API for constructing SIR programs.
//! Manages value numbering, block creation, and instruction insertion.

use super::blocks::{BasicBlock, Instruction, SirFunction, Terminator};
use super::metadata::{EpistemicMetadata, Metadata, MetadataStore};
use super::module::SirModule;
use super::ops::*;
use super::types::{DistributionKind, ScalarType, SirType};
use super::values::{BlockId, Constant, FuncId, Value, ValueId, ValueMetadata};

/// Builder for constructing SIR functions
pub struct SirBuilder<'m> {
    /// Module being built
    module: &'m mut SirModule,
    /// Current function
    current_func: Option<FuncId>,
    /// Current block
    current_block: Option<BlockId>,
    /// Next value ID
    next_value_id: u32,
    /// Next block ID
    next_block_id: u32,
    /// Metadata store
    metadata: MetadataStore,
    /// Value definitions
    values: Vec<Value>,
}

impl<'m> SirBuilder<'m> {
    pub fn new(module: &'m mut SirModule) -> Self {
        Self {
            module,
            current_func: None,
            current_block: None,
            next_value_id: 0,
            next_block_id: 0,
            metadata: MetadataStore::new(),
            values: vec![],
        }
    }

    /// Create a new function and set it as current
    pub fn create_function(
        &mut self,
        name: impl Into<String>,
        params: Vec<(String, SirType)>,
        ret_ty: SirType,
    ) -> FuncId {
        let func_id = self.module.create_function(name, params.clone(), ret_ty);
        self.current_func = Some(func_id);

        // Create entry block
        let entry = self.create_block_in_func("entry");
        self.current_block = Some(entry);

        // Create parameter values
        for (name, ty) in params {
            let val_id = self.alloc_value(ty.clone());
            let val = Value::new(val_id, ty).named(name);
            self.values.push(val);
        }

        func_id
    }

    /// Create a new basic block
    pub fn create_block(&mut self) -> BlockId {
        self.create_block_in_func("")
    }

    /// Create a named basic block
    pub fn create_block_named(&mut self, name: impl Into<String>) -> BlockId {
        self.create_block_in_func(name)
    }

    fn create_block_in_func(&mut self, name: impl Into<String>) -> BlockId {
        let id = BlockId::new(self.next_block_id);
        self.next_block_id += 1;

        let mut block = BasicBlock::new(id);
        let name = name.into();
        if !name.is_empty() {
            block = block.named(name);
        }

        if let Some(func_id) = self.current_func {
            if let Some(func) = self.module.function_mut(func_id) {
                func.blocks.push(block);
            }
        }

        id
    }

    /// Set current insertion point
    pub fn position_at(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Allocate a new value ID
    fn alloc_value(&mut self, ty: SirType) -> ValueId {
        let id = ValueId::new(self.next_value_id);
        self.next_value_id += 1;
        self.values.push(Value::new(id, ty));
        id
    }

    /// Get the current function
    fn current_function(&mut self) -> &mut SirFunction {
        let func_id = self.current_func.expect("No current function");
        self.module
            .function_mut(func_id)
            .expect("Function not found")
    }

    /// Get the current block
    fn current_block_mut(&mut self) -> &mut BasicBlock {
        let block_id = self.current_block.expect("No current block");
        let func = self.current_function();
        func.block_mut(block_id).expect("Block not found")
    }

    /// Insert an instruction at current position
    fn insert(&mut self, inst: Instruction) -> Option<ValueId> {
        let result = inst.result;
        self.current_block_mut().push(inst);
        result
    }

    // =========================================================================
    // CONSTANT BUILDERS
    // =========================================================================

    pub fn const_i32(&mut self, val: i32) -> ValueId {
        let id = self.alloc_value(SirType::i32());
        self.insert(Instruction::with_result(
            id,
            SirInst::Const(Constant::I32(val)),
        ));
        id
    }

    pub fn const_i64(&mut self, val: i64) -> ValueId {
        let id = self.alloc_value(SirType::i64());
        self.insert(Instruction::with_result(
            id,
            SirInst::Const(Constant::I64(val)),
        ));
        id
    }

    pub fn const_f32(&mut self, val: f32) -> ValueId {
        let id = self.alloc_value(SirType::f32());
        self.insert(Instruction::with_result(
            id,
            SirInst::Const(Constant::F32(val)),
        ));
        id
    }

    pub fn const_f64(&mut self, val: f64) -> ValueId {
        let id = self.alloc_value(SirType::f64());
        self.insert(Instruction::with_result(
            id,
            SirInst::Const(Constant::F64(val)),
        ));
        id
    }

    pub fn const_bool(&mut self, val: bool) -> ValueId {
        let id = self.alloc_value(SirType::bool());
        self.insert(Instruction::with_result(
            id,
            SirInst::Const(Constant::Bool(val)),
        ));
        id
    }

    // =========================================================================
    // ARITHMETIC BUILDERS
    // =========================================================================

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.values[lhs.index()].ty.clone();
        let id = self.alloc_value(ty);
        let op = if self.values[lhs.index()].ty.is_float() {
            ArithOp::FAdd
        } else {
            ArithOp::Add
        };
        self.insert(Instruction::with_result(
            id,
            SirInst::BinOp { op, lhs, rhs },
        ));
        id
    }

    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.values[lhs.index()].ty.clone();
        let id = self.alloc_value(ty);
        let op = if self.values[lhs.index()].ty.is_float() {
            ArithOp::FSub
        } else {
            ArithOp::Sub
        };
        self.insert(Instruction::with_result(
            id,
            SirInst::BinOp { op, lhs, rhs },
        ));
        id
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.values[lhs.index()].ty.clone();
        let id = self.alloc_value(ty);
        let op = if self.values[lhs.index()].ty.is_float() {
            ArithOp::FMul
        } else {
            ArithOp::Mul
        };
        self.insert(Instruction::with_result(
            id,
            SirInst::BinOp { op, lhs, rhs },
        ));
        id
    }

    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.values[lhs.index()].ty.clone();
        let id = self.alloc_value(ty);
        let op = if self.values[lhs.index()].ty.is_float() {
            ArithOp::FDiv
        } else {
            ArithOp::SDiv
        };
        self.insert(Instruction::with_result(
            id,
            SirInst::BinOp { op, lhs, rhs },
        ));
        id
    }

    // =========================================================================
    // COMPARISON BUILDERS
    // =========================================================================

    pub fn icmp_eq(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::bool());
        self.insert(Instruction::with_result(
            id,
            SirInst::Cmp {
                op: CmpOp::Eq,
                lhs,
                rhs,
            },
        ));
        id
    }

    pub fn icmp_ne(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::bool());
        self.insert(Instruction::with_result(
            id,
            SirInst::Cmp {
                op: CmpOp::Ne,
                lhs,
                rhs,
            },
        ));
        id
    }

    pub fn icmp_slt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::bool());
        self.insert(Instruction::with_result(
            id,
            SirInst::Cmp {
                op: CmpOp::SLt,
                lhs,
                rhs,
            },
        ));
        id
    }

    pub fn fcmp_olt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::bool());
        self.insert(Instruction::with_result(
            id,
            SirInst::Cmp {
                op: CmpOp::FOLt,
                lhs,
                rhs,
            },
        ));
        id
    }

    // =========================================================================
    // MEMORY BUILDERS
    // =========================================================================

    pub fn alloca(&mut self, ty: SirType) -> ValueId {
        let ptr_ty = SirType::ptr(ty.clone());
        let id = self.alloc_value(ptr_ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::Memory(MemoryOp::Alloca {
                ty,
                count: None,
                align: None,
            }),
        ));
        id
    }

    pub fn load(&mut self, ptr: ValueId, ty: SirType) -> ValueId {
        let id = self.alloc_value(ty.clone());
        self.insert(Instruction::with_result(
            id,
            SirInst::Memory(MemoryOp::Load {
                ptr,
                ty,
                volatile: false,
                align: None,
            }),
        ));
        id
    }

    pub fn store(&mut self, ptr: ValueId, val: ValueId) {
        self.insert(Instruction::void(SirInst::Memory(MemoryOp::Store {
            ptr,
            val,
            volatile: false,
            align: None,
        })));
    }

    // =========================================================================
    // CONTROL FLOW BUILDERS
    // =========================================================================

    pub fn ret(&mut self, val: ValueId) {
        self.current_block_mut()
            .terminate(Terminator::Return(Some(val)));
    }

    pub fn ret_void(&mut self) {
        self.current_block_mut().terminate(Terminator::Return(None));
    }

    pub fn br(&mut self, target: BlockId) {
        self.current_block_mut().terminate(Terminator::Br(target));
    }

    pub fn cond_br(&mut self, cond: ValueId, then_block: BlockId, else_block: BlockId) {
        self.current_block_mut().terminate(Terminator::CondBr {
            cond,
            then_block,
            else_block,
        });
    }

    // =========================================================================
    // EPISTEMIC BUILDERS (Domain-Specific)
    // =========================================================================

    /// Create an epistemic value from a value and confidence
    pub fn epistemic_create(&mut self, value: ValueId, confidence: ValueId) -> ValueId {
        let val_ty = self.values[value.index()].ty.clone();
        let id = self.alloc_value(SirType::Epistemic(Box::new(val_ty)));
        self.insert(Instruction::with_result(
            id,
            SirInst::Epistemic(EpistemicOp::Create { value, confidence }),
        ));
        id
    }

    /// Extract the raw value from an epistemic value
    pub fn epistemic_value(&mut self, epistemic: ValueId) -> ValueId {
        let ep_ty = &self.values[epistemic.index()].ty;
        let inner_ty = if let SirType::Epistemic(inner) = ep_ty {
            (**inner).clone()
        } else {
            SirType::f64() // Default
        };
        let id = self.alloc_value(inner_ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::Epistemic(EpistemicOp::ExtractValue(epistemic)),
        ));
        id
    }

    /// Extract confidence from an epistemic value
    pub fn epistemic_confidence(&mut self, epistemic: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::f64());
        self.insert(Instruction::with_result(
            id,
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(epistemic)),
        ));
        id
    }

    /// Fused epistemic multiplication (computes both value and confidence)
    pub fn epistemic_fused_mul(
        &mut self,
        val_a: ValueId,
        conf_a: ValueId,
        val_b: ValueId,
        conf_b: ValueId,
    ) -> ValueId {
        let val_ty = self.values[val_a.index()].ty.clone();
        let id = self.alloc_value(SirType::Epistemic(Box::new(val_ty)));
        self.insert(Instruction::with_result(
            id,
            SirInst::Epistemic(EpistemicOp::FusedMul {
                val_a,
                conf_a,
                val_b,
                conf_b,
            }),
        ));
        id
    }

    /// Mark a value as certain (confidence = 1.0)
    pub fn mark_certain(&mut self, value: ValueId) {
        self.metadata
            .attach(value, Metadata::Epistemic(EpistemicMetadata::certain()));
    }

    // =========================================================================
    // PROBABILITY BUILDERS (Domain-Specific)
    // =========================================================================

    /// Create a Normal distribution
    pub fn normal_dist(&mut self, mean: ValueId, stddev: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::Distribution(DistributionKind::Normal));
        self.insert(Instruction::with_result(
            id,
            SirInst::Prob(ProbOp::CreateDist {
                kind: DistributionKind::Normal,
                params: vec![mean, stddev],
            }),
        ));
        id
    }

    /// Sample from a distribution
    pub fn sample(&mut self, dist: ValueId, rng: ValueId) -> ValueId {
        let id = self.alloc_value(SirType::f64());
        self.insert(Instruction::with_result(
            id,
            SirInst::Prob(ProbOp::Sample { dist, rng }),
        ));
        id
    }

    /// Vectorized sampling
    pub fn sample_n(&mut self, dist: ValueId, count: ValueId, rng: ValueId) -> ValueId {
        // Returns pointer to array of samples
        let id = self.alloc_value(SirType::ptr(SirType::f64()));
        self.insert(Instruction::with_result(
            id,
            SirInst::Prob(ProbOp::SampleN { dist, count, rng }),
        ));
        id
    }

    // =========================================================================
    // SCIENTIFIC BUILDERS (Domain-Specific)
    // =========================================================================

    /// Single ODE solver step
    pub fn ode_step(
        &mut self,
        method: OdeMethod,
        state: ValueId,
        derivatives: FuncId,
        t: ValueId,
        dt: ValueId,
    ) -> ValueId {
        // Returns new state
        let state_ty = self.values[state.index()].ty.clone();
        let id = self.alloc_value(state_ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::Scientific(ScientificOp::OdeStep {
                method,
                state,
                derivatives,
                t,
                dt,
            }),
        ));
        id
    }

    /// Dot product
    pub fn dot_product(&mut self, a: ValueId, b: ValueId, len: u32) -> ValueId {
        let id = self.alloc_value(SirType::f64());
        self.insert(Instruction::with_result(
            id,
            SirInst::Scientific(ScientificOp::DotProduct {
                a,
                b,
                len,
                epistemic: false,
            }),
        ));
        id
    }

    /// Linear interpolation
    pub fn lerp(&mut self, a: ValueId, b: ValueId, t: ValueId) -> ValueId {
        let ty = self.values[a.index()].ty.clone();
        let id = self.alloc_value(ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::Scientific(ScientificOp::Lerp { a, b, t }),
        ));
        id
    }

    // =========================================================================
    // INTRINSICS
    // =========================================================================

    pub fn sqrt(&mut self, val: ValueId) -> ValueId {
        let ty = self.values[val.index()].ty.clone();
        let id = self.alloc_value(ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::UnaryFloat {
                op: UnaryFloatOp::Sqrt,
                val,
            },
        ));
        id
    }

    pub fn exp(&mut self, val: ValueId) -> ValueId {
        let ty = self.values[val.index()].ty.clone();
        let id = self.alloc_value(ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::UnaryFloat {
                op: UnaryFloatOp::Exp,
                val,
            },
        ));
        id
    }

    pub fn log(&mut self, val: ValueId) -> ValueId {
        let ty = self.values[val.index()].ty.clone();
        let id = self.alloc_value(ty);
        self.insert(Instruction::with_result(
            id,
            SirInst::UnaryFloat {
                op: UnaryFloatOp::Log,
                val,
            },
        ));
        id
    }

    /// Get the built metadata store
    pub fn into_metadata(self) -> MetadataStore {
        self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let mut module = SirModule::new("test");

        {
            let mut builder = SirBuilder::new(&mut module);

            // Create: fn add(a: i32, b: i32) -> i32 { a + b }
            builder.create_function(
                "add",
                vec![("a".into(), SirType::i32()), ("b".into(), SirType::i32())],
                SirType::i32(),
            );

            // Get parameter values (they are created during function creation)
            let a = ValueId::new(0);
            let b = ValueId::new(1);

            let sum = builder.add(a, b);
            builder.ret(sum);
        }

        // Validate
        assert!(module.validate().is_ok());
        println!("{}", module);
    }

    #[test]
    fn test_epistemic_function() {
        let mut module = SirModule::new("test");

        {
            let mut builder = SirBuilder::new(&mut module);

            // fn uncertain_calc(x: f64, conf: f64) -> epistemic<f64>
            builder.create_function(
                "uncertain_calc",
                vec![
                    ("x".into(), SirType::f64()),
                    ("conf".into(), SirType::f64()),
                ],
                SirType::Epistemic(Box::new(SirType::f64())),
            );

            let x = ValueId::new(0);
            let conf = ValueId::new(1);

            // Create epistemic value
            let ep = builder.epistemic_create(x, conf);

            // Extract and double the value
            let val = builder.epistemic_value(ep);
            let two = builder.const_f64(2.0);
            let doubled = builder.mul(val, two);

            // Create result with propagated confidence
            let one = builder.const_f64(1.0);
            let result = builder.epistemic_fused_mul(doubled, conf, one, one);

            builder.ret(result);
        }

        assert!(module.validate().is_ok());
        println!("{}", module);
    }
}
