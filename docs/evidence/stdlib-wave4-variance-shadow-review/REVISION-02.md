# Review revision 02: durable invalid-context refusal and wide guard

This revision changes only the unapplied proposal and review documents. The compiler source remains SHA-256 87ca50e53e3f6284db6c158eea40637e15a8a4ef6893660e49d18b1ac75bc24c.

The invalid-context branch of shadow_refuse_scope now calls lower_note_error(30790, ir_empty_name()) BEFORE IR_LOWER_HARD_ERROR=1. That entry records IR_LOWER_COMPILE_ERROR_LATCH when compilation scope is active and increments the diagnostic total without reading current_func. The old flag alone was insufficient: lower_hard_error_reset clears it and the diagnostic total between dependencies, while the compile-scope latch survives. The real gate must observe the recorded error before reset, then prove that reset/new-dependency cannot turn publication back into success. The existing reporter is not modified.

shadow_closed_untracked_integer now rejects expr_wide_bits_ref(e)>64 before its IntLit/Ident/binary rules, in addition to the existing per-local wide_bits guard and the declared word-integer parameter whitelist.

Readonly suffix evidence: parser/exprs.sio parse_int_literal writes only ExprIntLit plus tok.int_value; expr_wide_bits_ref currently returns0 for a bare IntLit. lexer/mod.sio skip_numeric_suffix explicitly handles i8/16/32/64 and u8/16/32/64, but not i128/u128. Therefore this review does NOT claim that wide suffixes normalize to Cast. Their actual token/AST/check representation must be preserved in controls; unsupported parsing is not a semantic rejection pass. Explicit casts and declared wide parameters are real controls of the guard. No parser/lexer change is included.

Required new latch and wide/suffix gates are appended to CONTROLS.md. They are pending: no candidate compiler or native job was run. Static patch applicability and unchanged source identity are the only validations of this revision.
