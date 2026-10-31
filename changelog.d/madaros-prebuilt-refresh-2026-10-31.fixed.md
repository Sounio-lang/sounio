The prebuilt Madaros behind `bin/souc` / `bin/madaros` is rebuilt from `main` (82bfa7b3d), so it now ships these compiler fixes:
- enum variants with payloads as tagged cells (KL-20, E277–E280);
- generic struct method type arguments inferred from the receiver;
- nested struct fields of an uninitialized `var x: T` get storage instead of trapping;
- first-order sensitivities reset per function, removing the `malformed_ir_registers` refusal;
- IR and vreg caps of 65536;
- `read_f64 … as i64` converts the value instead of passing raw bits.
