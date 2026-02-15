//! Integration tests for Poseidon VM wrapper

use poseidon_vm::{MAX_REGISTERS, PoseidonVm, VmError};
use soir::*;

/// Helper to create a simple instruction with defaults
fn make_instr(op: IrOpcode) -> IrInstr {
    IrInstr {
        op,
        ..Default::default()
    }
}

/// Helper to create a simple SOIR module with main function
fn create_simple_module() -> SoirModule {
    let mut main_fn = IrFunction::default();
    main_fn.name = Name::from("main");
    main_fn.instr_count = 2;
    main_fn.reg_count = 1;

    main_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 42,
        ..Default::default()
    };

    main_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 0,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 1;
    module.functions[0] = main_fn;
    module
}

/// Helper to create module that adds two numbers
fn create_add_module() -> SoirModule {
    let mut main_fn = IrFunction::default();
    main_fn.name = Name::from("main");
    main_fn.instr_count = 4;
    main_fn.reg_count = 3;

    main_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 10,
        ..Default::default()
    };

    main_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1,
        imm_i64: 32,
        ..Default::default()
    };

    main_fn.instrs[2] = IrInstr {
        op: IrOpcode::IrBinOp,
        dst: 2,
        src1: 0,
        src2: 1,
        bin_op: BinaryOp::OpAdd,
        ..Default::default()
    };

    main_fn.instrs[3] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 2,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 1;
    module.functions[0] = main_fn;
    module
}

/// Helper to create module with function call
fn create_call_module() -> SoirModule {
    // Add function at index 0
    let mut add_fn = IrFunction::default();
    add_fn.name = Name::from("add");
    add_fn.instr_count = 2;
    add_fn.reg_count = 3;
    add_fn.param_count = 2;
    add_fn.param_regs[0] = 0;
    add_fn.param_regs[1] = 1;

    add_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrBinOp,
        dst: 2,
        src1: 0,
        src2: 1,
        bin_op: BinaryOp::OpAdd,
        ..Default::default()
    };

    add_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 2,
        ..Default::default()
    };

    // Main function at index 1
    let mut main_fn = IrFunction::default();
    main_fn.name = Name::from("main");
    main_fn.instr_count = 4;
    main_fn.reg_count = 3;

    main_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 10,
        ..Default::default()
    };

    main_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1,
        imm_i64: 20,
        ..Default::default()
    };

    main_fn.instrs[2] = IrInstr {
        op: IrOpcode::IrCall,
        dst: 2,
        src1: 0,  // first arg in r0
        src2: 1,  // second arg in r1
        fn_id: 0, // call add function
        arg_count: 2,
        ..Default::default()
    };

    main_fn.instrs[3] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 2,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 2;
    module.functions[0] = add_fn;
    module.functions[1] = main_fn;
    module
}

/// Helper to create module with conditional branch
fn create_branch_module(condition_value: i64) -> SoirModule {
    let mut main_fn = IrFunction::default();
    main_fn.name = Name::from("main");
    main_fn.instr_count = 8;
    main_fn.reg_count = 2;
    main_fn.label_count = 3;

    main_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: condition_value,
        ..Default::default()
    };

    main_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrBranchTrue,
        src1: 0,
        label_id: 1,
        ..Default::default()
    };

    main_fn.instrs[2] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1,
        imm_i64: 100,
        ..Default::default()
    };

    main_fn.instrs[3] = IrInstr {
        op: IrOpcode::IrJump,
        label_id: 2,
        ..Default::default()
    };

    main_fn.instrs[4] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 1,
        ..Default::default()
    };

    main_fn.instrs[5] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1,
        imm_i64: 200,
        ..Default::default()
    };

    main_fn.instrs[6] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 2,
        ..Default::default()
    };

    main_fn.instrs[7] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 1,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 1;
    module.functions[0] = main_fn;
    module
}

#[test]
fn test_load_and_execute_simple() {
    let module = create_simple_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    let exit_code = vm.execute().unwrap();
    assert_eq!(exit_code, 42);
}

#[test]
fn test_arithmetic_operations() {
    let module = create_add_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    let exit_code = vm.execute().unwrap();
    assert_eq!(exit_code, 42); // 10 + 32 = 42
}

#[test]
fn test_function_call() {
    let module = create_call_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    let exit_code = vm.execute().unwrap();
    assert_eq!(exit_code, 30); // 10 + 20 = 30
}

#[test]
fn test_branch_true() {
    let module = create_branch_module(1); // condition is true
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    let exit_code = vm.execute().unwrap();
    assert_eq!(exit_code, 200); // Should take true branch
}

#[test]
fn test_branch_false() {
    let module = create_branch_module(0); // condition is false
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    let exit_code = vm.execute().unwrap();
    assert_eq!(exit_code, 100); // Should take false branch
}

#[test]
fn test_register_access() {
    let module = create_simple_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    // Before execution, registers should be zero
    assert_eq!(vm.register(0).unwrap(), 0);

    // Step once to execute load_imm
    vm.step().unwrap();

    // Now r0 should be 42
    assert_eq!(vm.register(0).unwrap(), 42);
}

#[test]
fn test_register_bounds_checking() {
    let vm = PoseidonVm::new().unwrap();

    // Valid register
    assert!(vm.register(0).is_ok());
    assert!(vm.register(MAX_REGISTERS - 1).is_ok());

    // Invalid register
    let result = vm.register(MAX_REGISTERS);
    assert!(matches!(result, Err(VmError::InvalidRegister(_, _))));
}

#[test]
fn test_vm_reset() {
    let module = create_simple_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    // Execute once
    let exit_code1 = vm.execute().unwrap();
    assert_eq!(exit_code1, 42);
    assert!(vm.is_halted());

    // Reset and execute again
    vm.reset().unwrap();
    assert!(!vm.is_halted());

    let exit_code2 = vm.execute().unwrap();
    assert_eq!(exit_code2, 42);
}

#[test]
fn test_execute_already_halted() {
    let module = create_simple_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    // Execute once
    vm.execute().unwrap();
    assert!(vm.is_halted());

    // Try to execute again (should error)
    let result = vm.execute();
    assert!(matches!(result, Err(VmError::AlreadyHalted(_))));
}

#[test]
fn test_execute_without_load() {
    let mut vm = PoseidonVm::new().unwrap();
    let result = vm.execute();
    assert!(matches!(result, Err(VmError::InvalidBytecode(_))));
}

#[test]
fn test_load_invalid_bytecode() {
    let mut vm = PoseidonVm::new().unwrap();
    let invalid_bytes = vec![0u8; 10];
    let result = vm.load(&invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_step_execution() {
    let module = create_add_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    // Step through execution
    assert!(vm.step().unwrap()); // load_imm r0
    assert_eq!(vm.register(0).unwrap(), 10);

    assert!(vm.step().unwrap()); // load_imm r1
    assert_eq!(vm.register(1).unwrap(), 32);

    assert!(vm.step().unwrap()); // binop r2 = r0 + r1
    assert_eq!(vm.register(2).unwrap(), 42);

    assert!(!vm.step().unwrap()); // return (halts)
    assert!(vm.is_halted());
    assert_eq!(vm.exit_code(), Some(42));
}

#[test]
fn test_max_steps_limit() {
    // Create infinite loop module
    let mut loop_fn = IrFunction::default();
    loop_fn.name = Name::from("main");
    loop_fn.instr_count = 3;
    loop_fn.reg_count = 1;
    loop_fn.label_count = 1;

    loop_fn.instrs[0] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 0,
        ..Default::default()
    };

    loop_fn.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 1,
        ..Default::default()
    };

    loop_fn.instrs[2] = IrInstr {
        op: IrOpcode::IrJump,
        label_id: 0,
        ..Default::default()
    };

    let mut module = SoirModule::default();
    module.fn_count = 1;
    module.functions[0] = loop_fn;

    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::with_max_steps(100).unwrap();
    vm.load(&bytecode).unwrap();

    let result = vm.execute();
    assert!(matches!(result, Err(VmError::Timeout(_))));
}

#[test]
fn test_stack_depth_tracking() {
    let module = create_call_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    // Initial depth should be 0
    assert_eq!(vm.stack_depth(), 0);

    // Step until we're in the called function
    vm.step().unwrap(); // load r0
    vm.step().unwrap(); // load r1
    vm.step().unwrap(); // call (should push frame)

    // Now we should be in the add function
    assert_eq!(vm.stack_depth(), 1);
}

#[test]
fn test_vm_state_introspection() {
    let module = create_simple_module();
    let bytecode = serialize(&module).unwrap();

    let mut vm = PoseidonVm::new().unwrap();
    vm.load(&bytecode).unwrap();

    assert!(!vm.is_halted());
    assert_eq!(vm.exit_code(), None);
    assert_eq!(vm.program_counter(), 0);

    vm.execute().unwrap();

    assert!(vm.is_halted());
    assert_eq!(vm.exit_code(), Some(42));
}

#[test]
fn test_multiple_loads() {
    let module1 = create_simple_module();
    let bytecode1 = serialize(&module1).unwrap();

    let module2 = create_add_module();
    let bytecode2 = serialize(&module2).unwrap();

    let mut vm = PoseidonVm::new().unwrap();

    // Load and execute first module
    vm.load(&bytecode1).unwrap();
    let exit1 = vm.execute().unwrap();
    assert_eq!(exit1, 42);

    // Load and execute second module (should replace first)
    vm.load(&bytecode2).unwrap();
    let exit2 = vm.execute().unwrap();
    assert_eq!(exit2, 42);
}
