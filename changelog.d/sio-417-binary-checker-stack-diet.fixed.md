Split `checker_check_binary_with_operand_types_inplace` into per-branch helpers. The router held every binary-operand path in one ~4.86 MB check-time frame, so nested calls such as `foo(bar(x * x))` crashed the checker on the default 8MB stack. Each branch now owns its temporaries in a separate helper frame, with unchanged checks and diagnostic order.

Second cut: split `checker_check_call_arg_refinement_boundary_inplace` (3.6 MB frame) into per-proof-arm helpers (int/float literal, provided-type, ident-const, ident-env + ontology conjunction, epistemic tail) behind a thin router with identical diagnostics.

Third cut (the actual d3 driver): split `checker_check_call_arg_unit_boundary_inplace` (3.6 MB frame, identified by its 2+3+1+1+3 call fingerprint in the faulting binary) into KL-13 / D-24 / addable refusal helpers behind a thin router with identical diagnostics.
