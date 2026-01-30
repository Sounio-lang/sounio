//! Integration tests for SIR passes

use sounio::sir::{
    blocks::{BasicBlock, Instruction, SirFunction},
    module::SirModule,
    ops::{ArithOp, SirInst},
    passes::{EpistemicInsertion, RefinementAssertionPass, UnitCheckInsertion},
    types::SirType,
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
    assert_eq!(velocity.dimensions[0], 1); // meter
    assert_eq!(velocity.dimensions[2], -1); // per second
}

#[test]
fn test_epistemic_insertion_pass_exists() {
    let pass = EpistemicInsertion::new();
    // Pass should be created successfully
    drop(pass);
}

#[test]
fn test_epistemic_insertion_on_epistemic_function() {
    use sounio::sir::blocks::Terminator;

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

    let mut module = SirModule::new("test");
    module.functions.push(func);

    let mut pass = EpistemicInsertion::new();
    let result = pass.run(&mut module);

    // Pass should transform the epistemic operation
    assert!(result.modified);
    assert_eq!(result.operations_transformed, 1);
}
