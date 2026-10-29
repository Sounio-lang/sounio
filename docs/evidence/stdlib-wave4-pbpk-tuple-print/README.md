# PBPK tuple projection loses scalar print metadata

This is a readonly compiler investigation with reduction fixtures and an unapplied checker-only proposal. No production source, original PBPK test or lowerer was changed. Raw04 remains `a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333`.

The original frontier is retained by root jobs19687/19691 and `docs/evidence/stdlib-wave4-pbpk-compile-frontier`: the unchanged PBPK checks0 but refuses three bare prints; replacing only those calls with `print_int` in a diagnostic clone advances to a separate dependency11/SIG139. That clone is not a production fix and the later crash is not classified here.

## Reductions and independent native controls

Slurm19743 completed the observational matrix in13s. Slurm19746 completed the corrected two-step nested cases in6s. Slurm19756 completed the nested scalar-only annotation control in4s. Every check/compile retained the original1200-second deadline, both LOOP_REGIONS modes and the exact raw. An observational collector result0 means the matrix/guards completed, not that all programs passed.

| Shape | Check | Compile | Native/C++23 |
| --- | --- | --- | --- |
| Explicit signed i64 scalar and f64 scalar | 0 | 0 | exact0 |
| Tuple literal `(i64,[f64;4],[f64;4])` destructured | 0 | 1, unresolved scalar kind | no ELF |
| Local typed tuple producer, implicit return | 0 | 1, same refusal | no ELF |
| Same producer with explicit return | 0 | 1, same refusal | no ELF |
| Imported typed tuple producer | 0 | 1, same refusal | no ELF |
| Whole tuple temporary explicitly annotated | 0 | 1, same refusal | no ELF |
| Only the extracted status explicitly i64 | 0 | 0 | exact0, arrays also checked |
| `(i64,i64)` destructured | 0 | 1 | no ELF |
| `(f64,f64)` destructured | 0 | 0 | exact0 |
| `(i64,f64)` destructured | 0 | 1 | no ELF |
| Nested tuple projected in two valid steps | 0 | 1 | no ELF |
| Nested pair annotated as a tuple, scalar leaves unannotated | 0 | 1 | no ELF |
| Nested primitive leaves explicitly i64/f64, including no intermediate tuple annotation | 0 | 0 | exact0 |
| Opaque aggregate projected then bare-printed | 0 | 1 | stays closed |

Results in this table are identical across both LOOP_REGIONS modes. The signed control includes -7000000001, crossing the32-bit range. The C++23 oracle parses signed integers and compares the exact double bits for binary-exact1.0/0.5/1.25; no Python/Rust or approximate tolerance is used.

The initial nested syntax `let ((a,b),v)=...` was not a valid lowering witness: the parser skips non-binding direct subpatterns and the checker reported undeclared names. Likewise direct `.0.0` spelling hit a separate parser/checker ambiguity. Those failed attempts are retained in the first matrix, not attributed to the scalar-kind bug. The committed nested fixtures use explicit intermediate projections and check0.

## Where the type information stops being available

Current lower.sio and the source executed by raw04 match byte-for-byte: SHA `87ca50e53e3f6284db6c158eea40637e15a8a4ef6893660e49d18b1ac75bc24c`.

1. `parser/stmts.sio::tplet_emit_elems` around100–126 desugars the tuple let into a temporary and `let binder=__tupN.N`. Each generated element has `ty=None`; all share the source tuple-let span.
2. The checker knows the slot type: its field-access path around10490–10508 reads the corresponding `TyTuple.tuple_elems` entry. `checker_check_let_stmt_inplace` around15209 computes and publishes the inferred binding type into the environment, but accepts a by-value Stmt and returns Unit; the original AST annotation stays absent.
3. Lowering retains scalar-f64 and f64-array tuple masks. The scalar mask's zero explicitly means both no tuple and no floating elements. It does not prove an integer slot.
4. `expr_result_scalar_kind_ref` around25973–26160 recognizes an integer FieldAccess positively through named struct/layout fields. It has no corresponding authoritative integer tuple-slot type here. `lower_let_stmt_type_metadata_ref` around29850–29923 therefore leaves the unannotated projected status unclassified, while the existing explicit i64 annotation path marks kind1.
5. A later `print(status)` sees the unresolved local and fails closed at recorded site9712 (`lower_note_error` around23127). The float masks still classify the f64 pair and array elements correctly.

This is not specific to ODE calculations, imported identities, implicit return syntax or loop-region mode: all have isolated controls above. The explicit primitive-slot control demonstrates that the value representation and existing annotation consumer work for the reduced case. It is diagnostic evidence, not a recommendation to annotate the PBPK test or replace its prints.

## Proposed repair boundary

Root selected transporting the checked primitive type into the existing original AST Stmt.ty, rather than adding a lowerer mask or treating every non-float as integer. `PROPOSED_CHANGE.md` gives the two checker insertion points and the Box, rechecker, metadata and Worker invariants. No implementation was made before root review.

The parent statement-list traversal retains the original node immediately after the check/environment update and before loans/scope pop. It can read the already established binding type without checking/evaluating the RHS again. Only a canonical primitive annotation can be attached; enriched or unknown types must remain untouched. The existing lowerer consumer then supplies scalar kind without changing the claimed lower file.

All fixture/module/oracle sources and exact recipes are retained. The original test and lower source are guarded before/after the jobs. A full original PBPK run still needs the compiler repair, and any subsequent SIG139 must remain separate.
