<!-- docs:meta
topic_id: repo.docs.spec.2026-10.07-library-packages-and-tools
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.07-library-packages-and-tools
-->

# Libraries, packages, and tools

The language and its environment cooperate through explicit interfaces.
Syntactic acceptance, library lookup, callable implementations, and runtime
support are different parts of that interface. This chapter defines how to
read the repository's library and tooling contracts without extending a
fixture's scope to an entire directory.

Rule identifiers do not imply verified enforcement. Language and library
contracts, source observations, and requirements on an evidence claim have
different scopes; see the [standing definitions](00-foundations-and-conventions.md#1-scope-and-authority)
and this chapter's [evidence catalog](07-evidence.tsv).

## 1. Language and library boundary

`LIB-001` — A library declaration contributes its actual exported signature,
effects, representation requirements, and preconditions. Its location under
`stdlib/` does not establish availability on every engine or target. A public
API claim requires a reachable import and an executable or refusal witness
for the selected route.

Primitive types, compiler intrinsics, and compiler-recognized constructs are
defined by the language chapters and implementation lookup. A similarly named
library type is not automatically the compiler construct. In particular,
compiler `Knowledge<T>`, a library `Epistemic` record, and EISA value/error/
uncertainty records have separate representations and operations.

The core library sources include `stdlib/core/prelude.sio`,
`stdlib/core/option.sio`, and `stdlib/core/result.sio`. Whether a spelling is
predeclared, imported by a prelude, recognized specially by the checker, or
resolved through a module must be established for that spelling. An import
does not authorize unrestricted use of every internal declaration.

`LIB-002` — Data structures with different allocation strategies keep their
own contracts. A fixed array, a heap-backed vector, a growable collection API,
and a compiler-specialized sequence are not interchangeable merely because
each exposes indexing. Their element representation, bounds, allocation,
copying, and ownership behavior must be checked on the relevant route.

Useful implementation entry points are `stdlib/collections/vec.sio`,
`stdlib/collections/heap_vec.sio`, `stdlib/mem/box.sio`,
`stdlib/mem/arena.sio`, and the execution chapter. The source organization also
contains alternate or historical implementations. A production call path is
identified by its resolver and imported symbols, rather than the newest-looking
filename.

## 2. Library organization

The following families describe the repository's scope. They identify areas
to inspect, rather than certify every member as callable. The repository
coverage table additionally maps the actual library subdirectories to this
reference.

| Family | Representative source areas | Contract of interest |
|---|---|---|
| Basic types and functional operations | `core`, `types`, `cmp`, `iter`, `functional` | Representation, equality/order, iteration, and explicit effects. |
| Storage and containers | `mem`, `collections`, `queue`, `cache` | Ownership, capacity, initialization, aliasing, and failure. |
| Text and interchange | `str`, `string`, `text`, `encoding`, `regex`, `json`, `csv`, `yaml`, `toml`, `serial`, `serialization`, `msgpack`, `compress` | Encoding, parsing errors, serialization identity, bounds, and round trips. |
| Host services | `io`, `os`, `path`, `time`, `cli`, `net`, `http`, `web`, `ffi`, `interop` | Host resources, effect declarations, errors, and external dependencies. |
| Parallel and distributed systems | `async`, `sync`, `thread`, `distributed`, `systems`, `infra`, `coordination` | Actual scheduler/runtime reachability, capabilities, ordering, and failure. |
| Numerical foundations | `math`, `numerical`, `constants`, `complex`, `linalg`, `tensor`, `special`, `polynomial`, `roots`, `optimize`, `interpolation`, `integrate`, `ode`, `autodiff` | Mathematical domain, numeric format, approximation, error, and oracle scope. |
| Probability and inference | `prob`, `random`, `stats`, `bayes`, `stochastic`, `ot`, `analysis`, `causal` | Distribution assumptions, state/seeding, dependence, uncertainty, and model interpretation. |
| Scientific information | `epistemic`, `eisa`, `metrology`, `units`, `ontology`, `logic`, `verify`, `theorem` | Value/error/uncertainty distinctions, units, predicates, evidence, and proof boundaries. |
| Algebra and material execution | `algebra`, `hardware`, `gpu`, `physics`, `particle_physics` | Parenthesization, precision, material target, and observed execution. |
| Quantum interfaces | `quantum`, `qpu`, `qnn`, `quantnn` | State simulation, linear resources, circuit serialization, and device receipts. |
| Machine learning and signals | `ml`, `nn`, `onn`, `snn`, `spnn`, `learn`, `ssm`, `signal`, `image`, `audio`, `connectivity`, `hypercomplex_graph` | Supported operators, training/inference routes, data/layout, and empirical scope. |
| Domain models | `chemistry`, `clinical`, `medical`, `medlang`, `pbpk`, `darwin_pbpk`, `neuro`, `fmri`, `genomics`, `heliobiology`, `hydride`, `simulation`, `science` | Declared model, context of use, validation, and domain-specific constraints. |
| Visualization and interaction | `plot`, `viz`, `graphics`, `render`, `display`, `ui`, `document` | Host/runtime dependencies and observable output. |
| Governance and research | `privacy`, `safety`, `fairness`, `regulatory`, `research`, `experimental`, `profile`, `test`, `pkg` | Policy boundaries, experimental status, provenance, and named acceptance procedures. |

This grouping is editorial. Namespace identity and import resolution remain
the resolver's responsibility; a directory's family is not a type-system fact.

`LIB-003` — An inventory scan establishes file and entrypoint structure. It
does not establish successful import, type checking, execution, numeric
accuracy, or full-module support. `scripts/stdlib/scan_stdlib.sh` collects such
an inventory. `scripts/ci/sounio_stdlib_surface_support_gate.sh` combines
inventory checks with a bounded package workflow. Its selected workflows and
assertions define the result's scope.

## 3. Text, I/O, and foreign interfaces

`LIB-004` — A textual or I/O API is specified by its selected implementation's
encoding, length unit, ownership, failure result, and effect signature. A
source `string` value, a foreign string handle, a byte buffer, and an external
text file require explicit conversions. Fixed-decimal printing is an output
format, not an exact serialization of every floating value.

`stdlib/str/lib.sio` and `stdlib/io/io.sio` are useful library entrypoints;
compiler builtins and target runtimes can provide distinct operations. Foreign
ABI and string layout details belong to the execution chapter. Copying a
standard-library example from another language is insufficient to establish a
Sounio API.

An FFI declaration's accepted syntax does not establish that the selected
target can resolve the symbol or preserve every argument/result layout. The
foreign call requires its ABI, symbol resolution, dynamic-library policy,
effects, and runtime witness. Target refusal is part of that interface.

## 4. Local packages

`PKG-001` — The current compiler constructs local package paths from the
import spelling. The canonical example is `use epistemic_core::*`, whose
conventional package source is `packages/epistemic-core/src/lib.sio`. The
root segment's underscores are mapped to hyphens. The example's `[lib].name`
agrees with the import name, but the inspected resolver does not read that
manifest field to discover the path.

`PKG-002` — Package submodule mapping applies to the package root segment;
it is not a general transformation of every module component. A package
submodule then follows the resolver's source-file and module-entry fallback
rules. An unreadable import must refuse translation on the checked package
route, without producing a misleading runnable artifact.

`docs/compiler/PACKAGE_IMPORT_RESOLUTION.md` describes this contract.
`scripts/ci/package_import_science_gate.sh` checks a manifest, package tests,
a downstream import witness, and a missing-package refusal. The relevant
fixtures include `tests/packages/package_import_science_witness.sio` and
`tests/packages/package_import_missing_package.sio`.

`PKG-003` — `sounio.toml` describes a package and its declared targets and
dependencies. A field's accepted spelling does not automatically implement
every future action suggested by that field. The current manifest design is
in `docs/ecosystem/SOUNIO_TOML_SPEC.md`; parsing sources include
`stdlib/pkg/manifest.sio`.

The `[science]` record, where present, identifies context of use, ring,
evidence references, and permitted claim classes. Policy evaluation and
scientific validation are different outcomes. A metadata field or numerical
score does not confer clinical, regulatory, causal, or publication authority.
The existing science-boundary gate and contracts identify the applicable
policy and acceptance conditions.

`PKG-004` — The checked local package tools and local registry design do not
establish a hosted public package service. `tools/sounio-pkg/sounio-pkg`
provides the supported local command surface selected by its gate. The shell
implementation in `tools/pkg/` is a separately labeled legacy prototype.
`tools/registry/` contains registry service/design material whose launch state
must be established independently.

The package support gate explicitly selects a seed compiler for some
compiled fixtures and drives package commands for other steps. Its results
must retain those engine distinctions. No download, publish, login, or search
capability follows merely from a manifest or registry schema being present.

The local command names also need their actual operation contracts:
`tools/sounio-pkg/src/cmd_build.sio` generates a shell command that checks
`src/lib.sio`; it does not construct a distributable package artifact.
`cmd_check.sio` scans `src/*.sio`, and `cmd_test.sio` runs `tests/*.sio`.
The package-import gate separately reads the manifest's test list. Custom
`[lib].path`, arbitrary library-name aliases, dependency version selection,
and manifest-controlled module lookup remain outside the demonstrated
compiler resolution contract.

## 5. Compiler command interface

`TOOL-001` — The public launcher and selected engine together determine CLI
behavior. The principal commands are `check`, `compile`, `build`, `run`,
`info`, and `--version`. Compiler artifact selection occurs through the
repository resolvers and explicit overrides. A raw bootstrap positional CLI
is a different interface from the public verb-based launcher.

For every command, a reliable recipe identifies the host and target, library
root, output location, relevant flags, and expected result. A flag present in
an old guide may be ignored or rejected by a current engine. The checked
command handling in `bin/souc` and the compiler driver controls that behavior.

`check` requests checking; `compile` and `build` request artifact production;
`run` requests generation and execution through the selected route. Exact
aliasing and supported options are implementation details documented with
their driver and tests. Successful checking cannot supply a runtime result.

## 6. Editor and agent tools

`TOOL-002` — Formatter acceptance includes a specified output convention and
idempotency witness. A file-backed REPL that recompiles source is not an
interactive in-memory JIT merely because its command is named `repl`.

`TOOL-003` — LSP capability advertisement, protocol correctness, and semantic
completeness are separate obligations. `tools/lsp/sounio-lsp.sh` is the
documented preview route behind `souc lsp --stdio`. The pure-Sounio server
source is a separate implementation. Rebuilding that source and exercising
the resulting server requires its own receipt.

The checked editor support procedure is
`scripts/ci/sounio_editor_tooling_support_gate.sh`; it contains CLI, formatter,
REPL, protocol, and static editor-asset checks. The result is bounded by its
actual assertions and warnings. VS Code, Helix, and Neovim wiring, notebook
integration, and language-server implementation status must each be identified
on their own surfaces.

`TOOL-004` — The in-tree MCP server under `tools/mcp/` exposes local compiler
operations and normalized diagnostics. Its transport and source-path policy
belong to that server. An accepted target label in an MCP request is not proof
that the selected compiler supports the target. Compilation and execution
errors must retain their actual outcomes in the tool response.

The root Python package, ecosystem Python bindings, Jupyter integration,
container recipes, editor distributions, and website implementations are
separate consumers or distribution surfaces. They are not alternate sources
for the language's type or numeric semantics.

## 7. Source and evidence map

| Subject | Primary source or procedure | Evidence boundary |
|---|---|---|
| Library structure | `scripts/stdlib/scan_stdlib.sh` | Inventory and entrypoint classification. |
| Bounded library support | `scripts/ci/sounio_stdlib_surface_support_gate.sh` | Selected package workflows and structural requirements. |
| Package imports | `scripts/ci/package_import_science_gate.sh` | Named package/downstream success and missing-import refusal. |
| Local package commands | `scripts/ci/sounio_package_support_gate.sh` | Selected compiler profiles and local tool smoke. |
| Science policy | `scripts/ci/science_boundary_gate.sh` | Declared policy/claim-contract behavior. |
| Editor preview | `scripts/ci/sounio_editor_tooling_support_gate.sh` | Advertised and tested preview features. |
| Agent transport | `tools/mcp/tests/test_protocol.py` | MCP protocol tests, distinct from numeric or compiler conformance. |

These procedures were inspected as sources during specification research.
Their listing does not certify a fresh successful execution of the procedures.
