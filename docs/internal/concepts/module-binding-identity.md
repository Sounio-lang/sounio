<!-- docs:meta
topic_id: repo.docs.internal.concepts.module-binding-identity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.module-binding-identity
-->

# Module Binding Identity

Concept-ID: `SOUNIO-MODULE-BINDING-IDENTITY`

Status: hypothesis

An imported name is not identified by spelling alone. A `use` declaration
creates a candidate binding from a caller module and local name to one exact
definition in a target module. Accessibility is then decided by the existing
visibility rule for that definition.

## Preserved Distinctions

```text
local spelling            != definition identity
import binding            != visibility permission
public export registry    != checker binding authority
physical module file      != logical import provenance
frontend traversal        != modular semantic checking
clean compilation         != fixed-point seed evidence
```

The authority must fail closed for ambiguous or unresolved named bindings. A
definition local to the caller module wins over an imported spelling. The
authority must never use import order or a global first-match table to choose
between two candidate definitions. It may identify a private definition so the
checker can reject access with the correct diagnostic; it may not make that
definition accessible.

## Semantic Lane

```text
Semantic-Lane-ID: issue901-module-binding-authority-20260725
Owner: codex
Concept-IDs: SOUNIO-MODULE-BINDING-IDENTITY, SOUNIO-SECOND-ORDER-COMPILATION
Intent-Preserved: a self-hosted compilation preserves the authored module relation from use site through definition selection and visibility checking
Transformation: replace name-only fallback for declared imports with exact caller-local binding resolution before the existing visibility predicate; reject unresolved named imports; preserve local-over-import precedence; require generic specialization to pass an unflattened E175/E176/E177-only authority probe before generic type normalization
Types-Changed: ModuleAuthority and ModuleBinding are compiler-owned checker inputs; no source-language type changes
Effects-Changed: none
IR-Changed: none in the first vertical
Claims-Introduced: a bounded function/struct/enum import family can identify the exact definition before reporting E175, E176, or E177; an imported function used as a value still carries its exact signature and E175 predicate; accepted enum struct-variant constructors use the same E177 predicate
Claims-Forbidden: full ModuleGraph completion; alias/reexport semantics; pub(super)/pub(in) ancestry; tuple-variant grammar support; frontend/native parity; fixed-point bootstrap; scientific, physical, or clinical validity
Assumptions: the closure has a complete physical file set and the direct-import fallback resolves a target file already present in that set
Write-Set: self-hosted/resolve/imports.sio, self-hosted/compiler/module_frontend.sio, self-hosted/compiler/main.sio, self-hosted/check/mod.sio, self-hosted/check/check.sio, self-hosted/check/defs.sio, tests/compiler/madaros_visibility_context/*, scripts/ci/madaros_visibility_context_gate.sh
Read-Set: parser AST ItemUse, ModuleClosure, visibility predicates, FnSigTable, StructTable, EnumTable
Positive-Witness: duplicate_private_single_main.sio and duplicate_private_18_main.sio check clean and execute their exact markers; public_facade_main.sio imports a function, struct, and enum then prints its exact marker; local_binding_shadows_direct_import_main.sio proves its local 42 rather than the imported 11 and prints its exact marker; local_value_shadows_import_main.sio typechecks a local closure returning i64 against an imported function returning bool
Negative-Witness: unresolved_named_import_main.sio fails at authority construction; visibility_fn_private_main.sio=E175; visibility_struct_private_main.sio=E176; visibility_enum_private_main.sio=E177; private_generic_import_main.sio=E175; private_function_value_main.sio=E175; private_enum_struct_variant_main.sio=E177
Acceptance-Gate: MADAROS_RAW_BIN=<current-source-madaros> SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved bash scripts/ci/madaros_visibility_context_gate.sh
Integration-Target: Issue #901 source-fresh Madaros bootstrap path
Authoritative-Only-If: the acceptance gate proves both duplicate-name witnesses and the local-shadow witness execute, an unresolved named import fails closed, and all real private accesses remain exact failures with no fallback
```

## Boundary

This concept constrains compiler symbol resolution. It does not define a
scientific observation, change EISA numerical semantics, or turn an accepted
frontend probe into a semantic proof.
