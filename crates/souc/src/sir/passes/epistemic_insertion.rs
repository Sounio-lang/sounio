//! Epistemic Insertion Pass
//!
//! This pass transforms operations on `Knowledge<T>` values to automatically
//! propagate uncertainty according to GUM (Guide to Uncertainty in Measurement)
//! rules.
//!
//! # Strategy
//!
//! 1. **Identify Epistemic Values**: Detect values with `SirType::Epistemic` or
//!    `confidence_bounds` metadata
//! 2. **Transform Arithmetic**: For each arithmetic operation on epistemic values:
//!    - Extract value and confidence components
//!    - Perform the original operation on values
//!    - Insert confidence propagation (GUM rules)
//!    - Create result with combined uncertainty
//! 3. **Propagation Rules** (GUM-compliant):
//!    - Add/Sub: σ² = σ_a² + σ_b² (uncertainties add in quadrature)
//!    - Mul/Div: σ_rel² = σ_rel_a² + σ_rel_b² (relative uncertainties)
//!
//! # Example Transformation
//!
//! Before:
//! ```text
//! %3 = FAdd %1, %2  ; where %1, %2 are epistemic
//! ```
//!
//! After:
//! ```text
//! %1_val = epistemic.extract_value %1
//! %1_conf = epistemic.extract_confidence %1
//! %2_val = epistemic.extract_value %2
//! %2_conf = epistemic.extract_confidence %2
//! %3_val = FAdd %1_val, %2_val
//! %3_conf = epistemic.propagate_add %1_conf, %2_conf
//! %3 = epistemic.create %3_val, %3_conf
//! ```

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::sir::{
    blocks::{BasicBlock, Instruction, SirFunction},
    module::SirModule,
    ops::{ArithOp, EpistemicOp, SirInst},
    types::SirType,
    values::ValueId,
};
use std::collections::{HashMap, HashSet};

/// Epistemic insertion pass for automatic uncertainty propagation
pub struct EpistemicInsertion {
    /// Counter for generating new value IDs
    next_value_id: u32,
    /// Track which values are epistemic
    epistemic_values: HashSet<ValueId>,
    /// Map from original value to (extracted_value, extracted_confidence)
    extracted_components: HashMap<ValueId, (ValueId, ValueId)>,
}

impl EpistemicInsertion {
    pub fn new() -> Self {
        Self {
            next_value_id: 0x1000_0000, // Start high to avoid conflicts
            epistemic_values: HashSet::new(),
            extracted_components: HashMap::new(),
        }
    }

    /// Run the pass on a module
    pub fn run(&mut self, module: &mut SirModule) -> EpistemicInsertionResult {
        let mut result = EpistemicInsertionResult::default();

        for func in &mut module.functions {
            let func_result = self.run_on_function(func);
            result.operations_transformed += func_result.operations_transformed;
            result.propagations_inserted += func_result.propagations_inserted;
            result.modified |= func_result.modified;
        }

        result
    }

    /// Run on a single function
    fn run_on_function(&mut self, func: &mut SirFunction) -> EpistemicInsertionResult {
        let mut result = EpistemicInsertionResult::default();

        // Reset per-function state
        self.epistemic_values.clear();
        self.extracted_components.clear();

        // First pass: identify all epistemic values
        self.collect_epistemic_values(func);

        if self.epistemic_values.is_empty() {
            return result;
        }

        // Second pass: transform operations
        for block in &mut func.blocks {
            let block_result = self.transform_block(block);
            result.operations_transformed += block_result.operations_transformed;
            result.propagations_inserted += block_result.propagations_inserted;
            result.modified |= block_result.modified;
        }

        result
    }

    /// Collect all values that are epistemic (by type or metadata)
    fn collect_epistemic_values(&mut self, func: &SirFunction) {
        // Check function parameters
        for (i, (_name, ty)) in func.params.iter().enumerate() {
            if self.is_epistemic_type(ty) {
                // Function params are typically assigned sequential IDs
                self.epistemic_values.insert(ValueId::new(i as u32));
            }
        }

        // Check block parameters and instruction results
        for block in &func.blocks {
            for param in &block.params {
                if self.is_epistemic_type(&param.ty) {
                    self.epistemic_values.insert(param.id);
                }
                // Also check metadata
                if param.metadata.confidence_bounds.is_some() {
                    self.epistemic_values.insert(param.id);
                }
            }

            for inst in &block.instructions {
                if let Some(result_id) = inst.result {
                    // Check if instruction produces epistemic value
                    match &inst.inst {
                        SirInst::Epistemic(EpistemicOp::Create { .. }) => {
                            self.epistemic_values.insert(result_id);
                        }
                        SirInst::Const(crate::sir::values::Constant::Epistemic { .. }) => {
                            self.epistemic_values.insert(result_id);
                        }
                        // BinOp on epistemic operands produces epistemic result
                        SirInst::BinOp { lhs, rhs, .. } => {
                            if self.epistemic_values.contains(lhs)
                                || self.epistemic_values.contains(rhs)
                            {
                                self.epistemic_values.insert(result_id);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Check if a type is epistemic
    fn is_epistemic_type(&self, ty: &SirType) -> bool {
        match ty {
            SirType::Epistemic(_) => true,
            SirType::Struct(st) => {
                // Check if it's a Knowledge struct by name
                st.name
                    .as_ref()
                    .map_or(false, |n| n.starts_with("Knowledge"))
            }
            _ => false,
        }
    }

    /// Transform a basic block, inserting epistemic operations
    fn transform_block(&mut self, block: &mut BasicBlock) -> EpistemicInsertionResult {
        let mut result = EpistemicInsertionResult::default();
        let mut new_instructions = Vec::new();

        for inst in &block.instructions {
            match &inst.inst {
                SirInst::BinOp { op, lhs, rhs } => {
                    let lhs_epistemic = self.epistemic_values.contains(lhs);
                    let rhs_epistemic = self.epistemic_values.contains(rhs);

                    if lhs_epistemic || rhs_epistemic {
                        // Transform this operation
                        let transformed = self.transform_binop(
                            inst.result,
                            *op,
                            *lhs,
                            lhs_epistemic,
                            *rhs,
                            rhs_epistemic,
                        );
                        new_instructions.extend(transformed);
                        result.operations_transformed += 1;
                        result.propagations_inserted += 1;
                        result.modified = true;
                    } else {
                        // Keep original instruction
                        new_instructions.push(inst.clone());
                    }
                }
                _ => {
                    // Keep non-binop instructions unchanged
                    new_instructions.push(inst.clone());
                }
            }
        }

        if result.modified {
            block.instructions = new_instructions;
        }

        result
    }

    /// Transform a binary operation on epistemic values
    fn transform_binop(
        &mut self,
        result_id: Option<ValueId>,
        op: ArithOp,
        lhs: ValueId,
        lhs_epistemic: bool,
        rhs: ValueId,
        rhs_epistemic: bool,
    ) -> Vec<Instruction> {
        let mut instructions = Vec::new();

        // Extract components for epistemic operands
        let (lhs_val, lhs_conf) = if lhs_epistemic {
            self.get_or_extract_components(&mut instructions, lhs)
        } else {
            // Non-epistemic value: use directly with confidence 1.0
            let conf_id = self.new_value_id();
            instructions.push(Instruction::with_result(
                conf_id,
                SirInst::Const(crate::sir::values::Constant::F64(1.0)),
            ));
            (lhs, conf_id)
        };

        let (rhs_val, rhs_conf) = if rhs_epistemic {
            self.get_or_extract_components(&mut instructions, rhs)
        } else {
            // Non-epistemic value: use directly with confidence 1.0
            let conf_id = self.new_value_id();
            instructions.push(Instruction::with_result(
                conf_id,
                SirInst::Const(crate::sir::values::Constant::F64(1.0)),
            ));
            (rhs, conf_id)
        };

        // Perform the original operation on extracted values
        let result_val = self.new_value_id();
        instructions.push(Instruction::with_result(
            result_val,
            SirInst::BinOp {
                op,
                lhs: lhs_val,
                rhs: rhs_val,
            },
        ));

        // Insert confidence propagation based on operation type
        let result_conf = self.new_value_id();
        let propagate_op = match op {
            ArithOp::Add | ArithOp::FAdd => EpistemicOp::PropagateAdd {
                conf_a: lhs_conf,
                conf_b: rhs_conf,
            },
            ArithOp::Sub | ArithOp::FSub => EpistemicOp::PropagateSub {
                conf_a: lhs_conf,
                conf_b: rhs_conf,
            },
            ArithOp::Mul | ArithOp::FMul => EpistemicOp::PropagateMul {
                val_a: lhs_val,
                conf_a: lhs_conf,
                val_b: rhs_val,
                conf_b: rhs_conf,
            },
            ArithOp::SDiv | ArithOp::UDiv | ArithOp::FDiv => EpistemicOp::PropagateDiv {
                val_a: lhs_val,
                conf_a: lhs_conf,
                val_b: rhs_val,
                conf_b: rhs_conf,
            },
            // For other operations (bitwise, shifts), use meet (minimum confidence)
            _ => EpistemicOp::Meet {
                conf_a: lhs_conf,
                conf_b: rhs_conf,
            },
        };

        instructions.push(Instruction::with_result(
            result_conf,
            SirInst::Epistemic(propagate_op),
        ));

        // Create the result epistemic value
        let final_result = result_id.unwrap_or_else(|| self.new_value_id());
        instructions.push(Instruction::with_result(
            final_result,
            SirInst::Epistemic(EpistemicOp::Create {
                value: result_val,
                confidence: result_conf,
            }),
        ));

        // Track that the result is epistemic
        self.epistemic_values.insert(final_result);

        instructions
    }

    /// Get or extract value and confidence components from an epistemic value
    fn get_or_extract_components(
        &mut self,
        instructions: &mut Vec<Instruction>,
        value: ValueId,
    ) -> (ValueId, ValueId) {
        if let Some(&(val, conf)) = self.extracted_components.get(&value) {
            return (val, conf);
        }

        // Extract value component
        let val_id = self.new_value_id();
        instructions.push(Instruction::with_result(
            val_id,
            SirInst::Epistemic(EpistemicOp::ExtractValue(value)),
        ));

        // Extract confidence component
        let conf_id = self.new_value_id();
        instructions.push(Instruction::with_result(
            conf_id,
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(value)),
        ));

        // Cache for reuse
        self.extracted_components.insert(value, (val_id, conf_id));

        (val_id, conf_id)
    }

    /// Generate a new unique value ID
    fn new_value_id(&mut self) -> ValueId {
        let id = ValueId::new(self.next_value_id);
        self.next_value_id += 1;
        id
    }
}

impl Default for EpistemicInsertion {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of running the epistemic insertion pass
#[derive(Debug, Clone, Default)]
pub struct EpistemicInsertionResult {
    /// Was the module modified?
    pub modified: bool,
    /// Number of operations transformed
    pub operations_transformed: usize,
    /// Number of confidence propagation instructions inserted
    pub propagations_inserted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sir::{
        blocks::{BasicBlock, Terminator},
        module::SirModule,
        types::SirType,
        values::{BlockId, Constant, FuncId, Value, ValueId},
    };

    fn create_test_function() -> SirFunction {
        let mut func = SirFunction::new(
            FuncId(0),
            "test_epistemic_add",
            vec![
                (
                    "a".to_string(),
                    SirType::Epistemic(Box::new(SirType::f64())),
                ),
                (
                    "b".to_string(),
                    SirType::Epistemic(Box::new(SirType::f64())),
                ),
            ],
            SirType::Epistemic(Box::new(SirType::f64())),
        );

        let mut entry_block = BasicBlock::new(BlockId(0));
        entry_block.name = Some("entry".to_string());

        // Parameters as block params
        entry_block.params = vec![
            Value::new(
                ValueId::new(0),
                SirType::Epistemic(Box::new(SirType::f64())),
            ),
            Value::new(
                ValueId::new(1),
                SirType::Epistemic(Box::new(SirType::f64())),
            ),
        ];

        // %2 = fadd %0, %1
        entry_block.instructions.push(Instruction::with_result(
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FAdd,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        ));

        entry_block.terminator = Some(Terminator::Return(Some(ValueId::new(2))));

        func.blocks.push(entry_block);
        func
    }

    #[test]
    fn test_epistemic_detection() {
        let func = create_test_function();
        let mut pass = EpistemicInsertion::new();

        pass.collect_epistemic_values(&func);

        // Both parameters should be detected as epistemic
        assert!(pass.epistemic_values.contains(&ValueId::new(0)));
        assert!(pass.epistemic_values.contains(&ValueId::new(1)));
    }

    #[test]
    fn test_epistemic_transformation() {
        let func = create_test_function();
        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = EpistemicInsertion::new();
        let result = pass.run(&mut module);

        assert!(result.modified);
        assert_eq!(result.operations_transformed, 1);
        assert_eq!(result.propagations_inserted, 1);

        // Check that the function was transformed
        let transformed_func = &module.functions[0];
        let entry_block = &transformed_func.blocks[0];

        // Should have more instructions now:
        // - Extract value/conf from %0 (2 instructions)
        // - Extract value/conf from %1 (2 instructions)
        // - FAdd on values (1 instruction)
        // - PropagateAdd on confidences (1 instruction)
        // - Create epistemic result (1 instruction)
        assert!(
            entry_block.instructions.len() >= 7,
            "Expected at least 7 instructions, got {}",
            entry_block.instructions.len()
        );

        // Verify PropagateAdd was inserted
        let has_propagate = entry_block.instructions.iter().any(|inst| {
            matches!(
                &inst.inst,
                SirInst::Epistemic(EpistemicOp::PropagateAdd { .. })
            )
        });
        assert!(has_propagate, "Expected PropagateAdd instruction");
    }

    #[test]
    fn test_non_epistemic_unchanged() {
        let mut func = SirFunction::new(
            FuncId(0),
            "test_normal_add",
            vec![
                ("a".to_string(), SirType::f64()),
                ("b".to_string(), SirType::f64()),
            ],
            SirType::f64(),
        );

        let mut entry_block = BasicBlock::new(BlockId(0));
        entry_block.params = vec![
            Value::new(ValueId::new(0), SirType::f64()),
            Value::new(ValueId::new(1), SirType::f64()),
        ];

        entry_block.instructions.push(Instruction::with_result(
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FAdd,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        ));

        entry_block.terminator = Some(Terminator::Return(Some(ValueId::new(2))));
        func.blocks.push(entry_block);

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = EpistemicInsertion::new();
        let result = pass.run(&mut module);

        // Should not be modified since no epistemic values
        assert!(!result.modified);
        assert_eq!(result.operations_transformed, 0);
    }

    #[test]
    fn test_mixed_epistemic_and_scalar() {
        let mut func = SirFunction::new(
            FuncId(0),
            "test_mixed",
            vec![
                (
                    "a".to_string(),
                    SirType::Epistemic(Box::new(SirType::f64())),
                ),
                ("b".to_string(), SirType::f64()), // Non-epistemic
            ],
            SirType::Epistemic(Box::new(SirType::f64())),
        );

        let mut entry_block = BasicBlock::new(BlockId(0));
        entry_block.params = vec![
            Value::new(
                ValueId::new(0),
                SirType::Epistemic(Box::new(SirType::f64())),
            ),
            Value::new(ValueId::new(1), SirType::f64()),
        ];

        // Add epistemic + scalar
        entry_block.instructions.push(Instruction::with_result(
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FAdd,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        ));

        entry_block.terminator = Some(Terminator::Return(Some(ValueId::new(2))));
        func.blocks.push(entry_block);

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = EpistemicInsertion::new();
        let result = pass.run(&mut module);

        // Should transform because one operand is epistemic
        assert!(result.modified);
        assert_eq!(result.operations_transformed, 1);

        // Verify the scalar got confidence 1.0
        let transformed_func = &module.functions[0];
        let entry_block = &transformed_func.blocks[0];

        let has_const_1 = entry_block.instructions.iter().any(|inst| {
            matches!(&inst.inst, SirInst::Const(Constant::F64(conf)) if (*conf - 1.0).abs() < 1e-10)
        });
        assert!(has_const_1, "Expected confidence 1.0 constant for scalar");
    }

    #[test]
    fn test_multiplication_uses_relative_uncertainty() {
        let mut func = SirFunction::new(
            FuncId(0),
            "test_mul",
            vec![
                (
                    "a".to_string(),
                    SirType::Epistemic(Box::new(SirType::f64())),
                ),
                (
                    "b".to_string(),
                    SirType::Epistemic(Box::new(SirType::f64())),
                ),
            ],
            SirType::Epistemic(Box::new(SirType::f64())),
        );

        let mut entry_block = BasicBlock::new(BlockId(0));
        entry_block.params = vec![
            Value::new(
                ValueId::new(0),
                SirType::Epistemic(Box::new(SirType::f64())),
            ),
            Value::new(
                ValueId::new(1),
                SirType::Epistemic(Box::new(SirType::f64())),
            ),
        ];

        // Multiply epistemic values
        entry_block.instructions.push(Instruction::with_result(
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FMul,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        ));

        entry_block.terminator = Some(Terminator::Return(Some(ValueId::new(2))));
        func.blocks.push(entry_block);

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = EpistemicInsertion::new();
        let result = pass.run(&mut module);

        assert!(result.modified);

        // Verify PropagateMul was used (includes values for relative uncertainty)
        let transformed_func = &module.functions[0];
        let entry_block = &transformed_func.blocks[0];

        let has_propagate_mul = entry_block.instructions.iter().any(|inst| {
            matches!(
                &inst.inst,
                SirInst::Epistemic(EpistemicOp::PropagateMul { .. })
            )
        });
        assert!(has_propagate_mul, "Expected PropagateMul instruction");
    }
}
