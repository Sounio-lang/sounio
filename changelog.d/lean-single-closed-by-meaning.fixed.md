The lean_single compiler no longer reads a variable outside the block that
declared it. Five such reads in its own source (`dovf`/`hsig` in the literal
lexer, `rhs_ty`/`rhs_ty_hash` in statement compilation, `q` in epistemic
inference) were lowered to whatever the stack held, so their value came from an
earlier call:

- an assignment to a global was type-checked against a stale type, which drew
  spurious "assignment type mismatch" warnings (684 of them in 170 of the 432
  repository programs that declare globals) and, on some inputs, a rejection;
- `measure`/`asserted`/`constant` tagged a confidence at a stale token index;
  they now tag the call's closing parenthesis;
- the suffix checks on integer literals read the other literal path's state.

On those 432 programs exit codes and every produced binary are unchanged. The
seed is refreshed (sha256 `bd0100dc0d89bc50…`), and Madaros now accepts
`self-hosted/compiler/lean_single.sio`, checked by the new
`lean_single_closed_by_meaning` gate.
