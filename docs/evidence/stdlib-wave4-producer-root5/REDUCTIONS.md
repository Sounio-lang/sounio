# ROOT5 producer reductions before repairs

Original full modern bootstrap19789 failed9 diagnostics and has no RAW_READY. These reductions use the qualified private legacy producer ccfa, original9cf and qualified modern raw04 a4f5; the failed raw05 ELF is not executed.

19801 paired main-source compilation:
- bare Unit return before comma (including nested match): old9cf/finalccfa both parse-refuse; modern raw04 accepts.
- bare nonUnit return: both legacy parse-refuse, modern rejects return type.
- braced Unit return and valued return: all accept.
- tuple Some/None scalar, recursive Box and nestedtuple: old9cf/modern accept, ccfa rejects strict arm join.
- both Some(Box<Node>) branches: all accept.
- finalccfa refuses distinct Box nominal identities and mixed int/f64 arms. Modern raw04 accepts the nominal-negative fixture, so it is a grammar reference only, not an identity oracle.
- missing return / duplicate variant coverage: old seed accepts, strict final rejects.

The first imported fixtures in19801 failed unreadable-import because the legacy loader uses cwd/self-hosted. Corrected layout19804 preserves the SAME source bytes and all input hashes: imported bare-return, missing-return, missing-variant and tuple-join errors return0 and can emit ELF. These invalid ELFs were never executed. This establishes the existing imported error suppression. The standalone equivalents return1. No global warning policy change is proposed.

19803 is an accidental duplicate setup retry; it failed mkdir sdk before any compiler execution. Preserved as technical failure, not semantic evidence.

19809 private diagnostic producer differs only by logging match join operand type/hash, expected return metadata and tuple cache rows, then calling unchanged ty_eq. It is not a production fix or canonical compiler. It shows:
 Some(Box<Node>),i64 row: [Option<Box<Node>>,i64], Option=2 cells;
 None,i64 row: [Option hash0,i64], Option=2 cells.
 Scalar and nested cases preserve exact leaf shape; distinct nominals and int/f64 differ concretely. Existing ty_slot_count represents a nestedtuple as one pointer cell, so native SRet/copy/clobber proof remains necessary before accepting nested joins.
 Full diagnostic source remains in scratch with SHA a74481e5e1fcc4476824f2d4f5eebf5a4bdc44942378b30ea3474b8e45bd670f; the checked-in patch is diagnostic-only. It must not enter final producer source.

Parent-approved repair scope: comma as nonoperand boundary only; sibling-concrete tuple rows with exact recursive kind/arity/layout, None inference only to explicitly allowed scalar-word or concrete Box payload; strict publication latch only for new structural x86 flow diagnostics and actual primary parse fallback. No ty_eq relaxation, no lower or AArch64 body edits.

Scratch /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-producer-reductions{01,02} and root5-join-debug01. Slurm19801/19804/19809 terminal COMPLETED0; semantic statuses are captured individually, not inferred from collector status.

