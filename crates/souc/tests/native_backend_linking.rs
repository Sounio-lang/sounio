//! Test native backend external function linking
//!
//! This test verifies that the AArch64 emitter properly tracks external symbols
//! and generates correct relocation entries for linking.

use souc::backend::native::aarch64::{AArch64Emitter, AArch64Reg, Relocation, RelocationKind};
use souc::sir::emit::RelocKind;

#[test]
fn test_bl_external_generates_relocation() {
    let mut emitter = AArch64Emitter::new();

    // Emit a call to an external function
    emitter.bl_external("__sounio_dispatch_io_print");

    // Verify that a relocation was created
    let relocations = emitter.relocations();
    assert_eq!(relocations.len(), 1, "Should have exactly one relocation");

    let reloc = &relocations[0];
    assert_eq!(reloc.offset, 0, "Relocation should be at offset 0");
    assert_eq!(
        reloc.symbol, "__sounio_dispatch_io_print",
        "Symbol name should match"
    );
    assert_eq!(
        reloc.kind,
        RelocationKind::AArch64Call26,
        "Relocation kind should be AArch64Call26"
    );
    assert_eq!(reloc.addend, 0, "Addend should be 0 for function calls");

    // Verify that the external symbol was registered
    let symbols = emitter.external_symbols();
    assert!(
        symbols.contains_key("__sounio_dispatch_io_print"),
        "External symbol should be registered"
    );
}

#[test]
fn test_multiple_bl_external_calls() {
    let mut emitter = AArch64Emitter::new();

    // Emit multiple calls to different external functions
    emitter.bl_external("__sounio_dispatch_io_print");
    emitter.bl_external("__sounio_dispatch_mut_get");
    emitter.bl_external("__sounio_dispatch_div_div");

    // Verify that three relocations were created
    let relocations = emitter.relocations();
    assert_eq!(relocations.len(), 3, "Should have three relocations");

    // Verify offsets are correct (each BL is 4 bytes)
    assert_eq!(relocations[0].offset, 0);
    assert_eq!(relocations[1].offset, 4);
    assert_eq!(relocations[2].offset, 8);

    // Verify all symbols are registered
    let symbols = emitter.external_symbols();
    assert_eq!(symbols.len(), 3, "Should have three unique symbols");
    assert!(symbols.contains_key("__sounio_dispatch_io_print"));
    assert!(symbols.contains_key("__sounio_dispatch_mut_get"));
    assert!(symbols.contains_key("__sounio_dispatch_div_div"));
}

#[test]
fn test_duplicate_external_calls() {
    let mut emitter = AArch64Emitter::new();

    // Emit multiple calls to the same external function
    emitter.bl_external("__sounio_dispatch_io_print");
    emitter.bl_external("__sounio_dispatch_io_print");

    // Verify that two relocations were created
    let relocations = emitter.relocations();
    assert_eq!(relocations.len(), 2, "Should have two relocations");

    // But only one unique symbol
    let symbols = emitter.external_symbols();
    assert_eq!(symbols.len(), 1, "Should have one unique symbol");
    assert!(symbols.contains_key("__sounio_dispatch_io_print"));
}

#[test]
fn test_bl_external_emits_correct_instruction() {
    let mut emitter = AArch64Emitter::new();

    emitter.bl_external("test_func");

    // Get the generated code
    let code = emitter.finish().expect("Should finish successfully");

    // Verify that a BL instruction was emitted (4 bytes)
    assert_eq!(code.len(), 4, "Should emit exactly 4 bytes for BL");

    // BL instruction should have opcode 0b100101 in the top 6 bits
    let inst = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
    let opcode = inst >> 26;
    assert_eq!(opcode, 0b100101, "Should be a BL instruction");

    // The offset field should be 0 (to be patched by linker)
    let offset = inst & 0x03FFFFFF;
    assert_eq!(offset, 0, "Offset should be 0 (linker placeholder)");
}

#[test]
fn test_relocation_kind_conversion() {
    // Test that our RelocationKind converts correctly to sir::emit::RelocKind
    use souc::backend::native::elf;

    // Create a RelocKind and convert it
    let reloc_kind = RelocKind::AArch64Call26;
    let reloc_type = elf::reloc_kind_to_type(&reloc_kind);

    // Verify it converts to the correct ELF relocation type
    assert_eq!(
        reloc_type,
        elf::RelocationType::AArch64Call26,
        "Should convert to AArch64Call26"
    );
}

#[test]
fn test_effect_dispatch_with_external_linking() {
    use souc::backend::native::effects::NativeEffectDispatcher;

    let dispatcher = NativeEffectDispatcher::new();
    let mut emitter = AArch64Emitter::new();

    // Generate code for an effect dispatch
    let arg_regs = vec![AArch64Reg::V0];
    let result =
        dispatcher.emit_effect_dispatch(&mut emitter, "IO", "print", &arg_regs, AArch64Reg::V0);

    assert!(result.is_ok(), "Effect dispatch should succeed");

    // Verify relocation was created
    let relocations = emitter.relocations();
    assert!(
        !relocations.is_empty(),
        "Should have at least one relocation"
    );

    // Verify it's calling the right runtime function
    assert!(
        relocations
            .iter()
            .any(|r| r.symbol == "__sounio_dispatch_io_print"),
        "Should call __sounio_dispatch_io_print"
    );
}

#[test]
fn test_complete_effect_program() {
    use souc::backend::native::effects::NativeEffectDispatcher;

    let dispatcher = NativeEffectDispatcher::new();
    let mut emitter = AArch64Emitter::new();

    // Simulate a complete effect program:
    // 1. Push IO handler
    // 2. Perform IO.print
    // 3. Pop handler

    dispatcher
        .emit_push_handler(&mut emitter, "IO", 1)
        .expect("Push handler should succeed");

    let arg_regs = vec![AArch64Reg::V0];
    dispatcher
        .emit_effect_dispatch(&mut emitter, "IO", "print", &arg_regs, AArch64Reg::V0)
        .expect("Effect dispatch should succeed");

    dispatcher
        .emit_pop_handler(&mut emitter)
        .expect("Pop handler should succeed");

    // Verify all three runtime calls were made
    let relocations = emitter.relocations();
    assert_eq!(
        relocations.len(),
        3,
        "Should have three relocations (push, dispatch, pop)"
    );

    // Verify symbols
    let symbols = emitter.external_symbols();
    assert!(symbols.contains_key("__sounio_push_handler_io"));
    assert!(symbols.contains_key("__sounio_dispatch_io_print"));
    assert!(symbols.contains_key("__sounio_pop_handler"));
}
