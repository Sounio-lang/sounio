//! Unit Check Insertion Pass
//!
//! This pass analyzes operations on Quantity values and inserts runtime checks
//! or automatic conversions where needed for unit compatibility.
//!
//! # Strategy
//!
//! 1. **Analyze Operations**: For each arithmetic operation, check if operands
//!    have unit metadata attached
//! 2. **Check Compatibility**: Determine if units are compatible
//!    - Add/Sub: Units must be identical
//!    - Mul/Div: Units combine dimensionally
//! 3. **Insert Checks/Conversions**: Based on compatibility analysis:
//!    - Compatible with same scale: No-op
//!    - Compatible with different scale: Insert conversion
//!    - Incompatible: Insert assertion with descriptive error

use crate::sir::{
    blocks::{Instruction, SirFunction},
    module::SirModule,
    ops::{ArithOp, FailureMode, SirInst},
    values::{PhysicalUnit, ValueId},
};
use std::collections::HashMap;

/// Unit check insertion pass
pub struct UnitCheckInsertion {
    /// Enable automatic unit conversion (e.g., mg → g)
    pub auto_convert: bool,
    /// Be strict about dimensionality (no silent scaling)
    pub strict: bool,
}

impl UnitCheckInsertion {
    pub fn new() -> Self {
        Self {
            auto_convert: true,
            strict: false,
        }
    }

    /// Run the pass on a module
    pub fn run(&mut self, module: &mut SirModule) -> UnitCheckResult {
        let mut result = UnitCheckResult::default();

        for func in &mut module.functions {
            let func_result = self.run_on_function(func);
            result.checks_inserted += func_result.checks_inserted;
            result.conversions_inserted += func_result.conversions_inserted;
            result.modified |= func_result.modified;
        }

        result
    }

    /// Run on a single function
    fn run_on_function(&mut self, func: &mut SirFunction) -> UnitCheckResult {
        let mut result = UnitCheckResult::default();

        // Collect value to unit mapping
        let mut value_units: HashMap<ValueId, PhysicalUnit> = HashMap::new();

        // First pass: collect all values with unit metadata
        for block in &func.blocks {
            for param in &block.params {
                if let Some(unit) = &param.metadata.unit {
                    value_units.insert(param.id, unit.clone());
                }
            }
        }

        // Second pass: analyze operations and insert checks
        for block in &mut func.blocks {
            let mut new_instructions = Vec::new();

            for inst in &block.instructions {
                // Check if this is an arithmetic operation on values with units
                match &inst.inst {
                    SirInst::BinOp { op, lhs, rhs } => {
                        let lhs_unit = value_units.get(lhs);
                        let rhs_unit = value_units.get(rhs);

                        if let (Some(lhs_unit), Some(rhs_unit)) = (lhs_unit, rhs_unit) {
                            // Both operands have units - check compatibility
                            let check_result =
                                self.check_unit_compatibility(op, lhs_unit, rhs_unit);

                            match check_result {
                                UnitCompatibility::Compatible => {
                                    // No action needed
                                }
                                UnitCompatibility::ConvertibleFrom { .. } => {
                                    if self.auto_convert {
                                        // Insert conversion instruction
                                        // TODO: Implement actual conversion
                                        result.conversions_inserted += 1;
                                        result.modified = true;
                                    } else if self.strict {
                                        // Insert assertion
                                        self.insert_unit_assert(
                                            &mut new_instructions,
                                            *lhs,
                                            *rhs,
                                            lhs_unit,
                                            rhs_unit,
                                            format!("Unit mismatch in {:?}: incompatible scales", op),
                                        );
                                        result.checks_inserted += 1;
                                        result.modified = true;
                                    }
                                }
                                UnitCompatibility::Incompatible { reason } => {
                                    // Insert assertion that will fail
                                    self.insert_unit_assert(
                                        &mut new_instructions,
                                        *lhs,
                                        *rhs,
                                        lhs_unit,
                                        rhs_unit,
                                        format!("Unit mismatch in {:?}: {}", op, reason),
                                    );
                                    result.checks_inserted += 1;
                                    result.modified = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }

                // Keep original instruction
                new_instructions.push(inst.clone());
            }

            // Replace instructions if we inserted checks
            if result.modified {
                block.instructions = new_instructions;
            }
        }

        result
    }

    /// Check unit compatibility for an operation
    fn check_unit_compatibility(
        &self,
        op: &ArithOp,
        lhs_unit: &PhysicalUnit,
        rhs_unit: &PhysicalUnit,
    ) -> UnitCompatibility {
        match op {
            ArithOp::Add | ArithOp::Sub | ArithOp::FAdd | ArithOp::FSub => {
                // Addition/subtraction requires identical dimensions
                if lhs_unit.dimensions != rhs_unit.dimensions {
                    return UnitCompatibility::Incompatible {
                        reason: format!(
                            "Cannot add/subtract different dimensions: {:?} vs {:?}",
                            lhs_unit.dimensions, rhs_unit.dimensions
                        ),
                    };
                }

                // Same dimensions, check scale
                if (lhs_unit.scale - rhs_unit.scale).abs() > 1e-10 {
                    UnitCompatibility::ConvertibleFrom {
                        from: rhs_unit.clone(),
                        to: lhs_unit.clone(),
                        scale: rhs_unit.scale / lhs_unit.scale,
                    }
                } else {
                    UnitCompatibility::Compatible
                }
            }
            ArithOp::Mul | ArithOp::FMul => {
                // Multiplication combines units dimensionally
                // This is always valid - no check needed
                UnitCompatibility::Compatible
            }
            ArithOp::SDiv | ArithOp::UDiv | ArithOp::FDiv => {
                // Division combines units dimensionally
                // This is always valid - no check needed
                UnitCompatibility::Compatible
            }
            _ => {
                // Other operations (bitwise, shifts) on dimensioned quantities are suspicious
                // but we'll allow them for now
                UnitCompatibility::Compatible
            }
        }
    }

    /// Insert a unit assertion check
    fn insert_unit_assert(
        &self,
        instructions: &mut Vec<Instruction>,
        _lhs: ValueId,
        _rhs: ValueId,
        _lhs_unit: &PhysicalUnit,
        _rhs_unit: &PhysicalUnit,
        message: String,
    ) {
        // Create a constant false condition (will always fail)
        let false_val = ValueId::new(0xFFFF_FFFF); // Placeholder
        
        let assert_inst = SirInst::Assert {
            cond: false_val,
            message,
            failure_mode: if self.strict {
                FailureMode::Panic
            } else {
                FailureMode::DegradeConfidence(0.5)
            },
        };

        instructions.push(Instruction::void(assert_inst));
    }
}

impl Default for UnitCheckInsertion {
    fn default() -> Self {
        Self::new()
    }
}

/// Unit compatibility result
#[derive(Debug, Clone)]
enum UnitCompatibility {
    /// Units are compatible as-is
    Compatible,
    /// Units can be converted
    ConvertibleFrom {
        from: PhysicalUnit,
        to: PhysicalUnit,
        scale: f64,
    },
    /// Units are fundamentally incompatible
    Incompatible { reason: String },
}

/// Result of running the unit check pass
#[derive(Debug, Clone, Default)]
pub struct UnitCheckResult {
    /// Was the module modified?
    pub modified: bool,
    /// Number of runtime checks inserted
    pub checks_inserted: usize,
    /// Number of automatic conversions inserted
    pub conversions_inserted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_compatibility_add() {
        let pass = UnitCheckInsertion::new();
        
        // Same units: compatible
        let meter = PhysicalUnit::meter();
        let result = pass.check_unit_compatibility(&ArithOp::Add, &meter, &meter);
        assert!(matches!(result, UnitCompatibility::Compatible));
    }

    #[test]
    fn test_unit_compatibility_different_dimensions() {
        let pass = UnitCheckInsertion::new();
        
        let meter = PhysicalUnit::meter();
        let second = PhysicalUnit::second();
        let result = pass.check_unit_compatibility(&ArithOp::Add, &meter, &second);
        assert!(matches!(result, UnitCompatibility::Incompatible { .. }));
    }

    #[test]
    fn test_unit_multiplication_always_compatible() {
        let pass = UnitCheckInsertion::new();
        
        let meter = PhysicalUnit::meter();
        let second = PhysicalUnit::second();
        let result = pass.check_unit_compatibility(&ArithOp::Mul, &meter, &second);
        assert!(matches!(result, UnitCompatibility::Compatible));
    }
}
