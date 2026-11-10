Split `checker_check_binary_with_operand_types_inplace` into per-branch helpers. The router held every binary-operand path in one ~4.86 MB check-time frame, so nested calls such as `foo(bar(x * x))` crashed the checker on the default 8MB stack. Each branch now owns its temporaries in a separate helper frame, with unchanged checks and diagnostic order.

Second cut: split `checker_check_call_arg_refinement_boundary_inplace` (3.6 MB frame) into per-proof-arm helpers (int/float literal, provided-type, ident-const, ident-env + ontology conjunction, epistemic tail) behind a thin router with identical diagnostics.

Third cut (the actual d3 driver): split `checker_check_call_arg_unit_boundary_inplace` (3.6 MB frame, identified by its 2+3+1+1+3 call fingerprint in the faulting binary) into KL-13 / D-24 / addable refusal helpers behind a thin router with identical diagnostics.

Fourth cut (the d5 driver): split `checker_check_call_expr_inplace` (1.27 MB frame per nesting level) into arm helpers — gpu-launch, metric builtins, native forms, bridge forms, path-qualifier validation, shadow forms, direct-sig call, indirect tail — behind a thin router preserving evaluation order and diagnostics.

Fifth cut (stride 1.6 to 1.31 MB per level; d5 passes at 8MB): hoisted the call router qualifier into `checker_resolve_call_direct_sig_id_inplace`, direct-sig generics substitution into `checker_check_call_direct_sig_generics_inplace`, split the 31-arm `checker_check_expr_inplace` dispatcher into `checker_check_expr_group_a/b_inplace` halves with a sequential tuple protocol, and extracted the call-arg tail (type-compat, five boundary checks, linear consume) into `checker_check_call_arg_post_inplace` so the loop recurses before post-check state materializes.

Sixth cut (the D killer; 8MB ceiling eliminated, verified to depth 72): the remaining 1.17 MB per-level frame was the call-expr shell itself (proven by fn-map off-to-address vote on the faulting binary); its pre-recursion phase (IO attributes, multitest/hypothesis state, sig resolution, lexical shadowing) moved to `checker_check_call_expr_preflight_inplace`, whose frame is gone before the argument walk recurses.
