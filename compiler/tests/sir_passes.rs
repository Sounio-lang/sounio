//! Integration tests for SIR passes

use souc::sir::{
    blocks::{BasicBlock, Instruction, SirFunction},
    module::SirModule,
    ops::{ArithOp, SirInst},
    passes::{RefinementAssertionPass, UnitCheckInsertion},
    types::{SirType, ScalarType},
    values::{BlockId, FuncId, PhysicalUnit, Value, ValueId},
};

#[test]
fn test_unit_check_pass_exists() {
    let mut pass = UnitCheckInsertion::new();
    assert!(!pass.strict);
    assert!(pass.auto_convert);
}

#[test]
fn test_refinement_pass_exists() {
    let mut pass = RefinementAssertionPass::new();
    assert!(pass.use_smt);
    assert!(pass.strict);
}

#[test]
fn test_unit_check_compatibility() {
    let pass = UnitCheckInsertion::new();
    
    // Test same units are compatible
    let meter = PhysicalUnit::meter();
    // This tests the internal logic
    assert_eq!(meter.dimensions[0], 1); // meter is [1,0,0,0,0,0,0]
}

#[test]
fn test_unit_physical_operations() {
    // Test PhysicalUnit operations
    let meter = PhysicalUnit::meter();
    let second = PhysicalUnit::second();
    
    // Multiplication should combine units
    let velocity = meter.divide(&second);
    assert_eq!(velocity.dimensions[0], 1);  // meter
    assert_eq!(velocity.dimensions[2], -1); // per second
}
