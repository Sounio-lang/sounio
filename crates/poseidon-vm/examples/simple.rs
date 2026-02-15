//! Simple example demonstrating Poseidon VM usage

use poseidon_vm::PoseidonVm;
use soir::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create a simple SOIR module that returns 42
    let mut main_fn = IrFunction::default();
    main_fn.name = Name::from("main");
    main_fn.instr_count = 2;
    main_fn.reg_count = 1;

    // Load immediate value 42 into register 0
    main_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 42,
        ..Default::default()
    };

    // Return value in register 0
    main_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 0,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 1;
    module.functions[0] = main_fn;

    // Serialize to bytecode
    let bytecode = serialize(&module)?;
    println!("SOIR bytecode size: {} bytes", bytecode.len());

    // Create VM and load bytecode
    let mut vm = PoseidonVm::new()?;
    vm.load(&bytecode)?;

    println!("Executing program...");

    // Execute and get result
    let exit_code = vm.execute()?;

    println!("Program exited with code: {}", exit_code);
    assert_eq!(exit_code, 42);

    Ok(())
}
