<!-- docs:meta
topic_id: repo.frontdoor.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.readme
-->

<picture>
  <source media="(max-width: 640px)" srcset="docs/assets/github/github-cover-mobile.png">
  <img src="docs/assets/github/github-cover.png" alt="Sounio — scientific meaning, control through compilation" width="1280">
</picture>

# Sounio

[English](README.md) · [Ελληνικά — ξεκινήστε εδώ](README.el.md)

**Scientific computing with explicit uncertainty, units and effects — powered by a self-hosted compiler.**

> **Status: research language (pre-1.0).** APIs may change; not recommended for production use. Open limits: [KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md).

Sounio is an epistemic systems language created by **Demetrios Chiuratto Agourakis**. Its scientific abstractions and compiler are developed together: measurements carry uncertainty, quantities carry dimensions, and computations declare effects. Explore the language, the compiler and a scientific library spanning eight research domains below.

[Explore the science](#a-scientific-library-across-disciplines) · [Language reference](docs/spec/LANGUAGE_SPECIFICATION.md) · [Explore the compiler](#architecture) · [Start here](#get-started)

## Science expressed in the program

A measurement has a value and an uncertainty. A quantity has a dimension. A computation has effects. Sounio makes these distinctions part of the program so that scientific assumptions can be inspected alongside the calculation.

Explore the [scientific examples](https://github.com/Sounio-lang/sounio/tree/main/examples), the [standard library](https://github.com/Sounio-lang/sounio/tree/main/stdlib) and the [science hub](https://www.souniolang.org/science). For the language-level guide, read [units and uncertainty in Sounio](https://www.souniolang.org/learn/language/units-and-uncertainty/) — dimensional analysis and GUM uncertainty propagation. Each application should be assessed against executable evidence for the current source revision.

## A scientific library across disciplines

Sounio's standard library connects scientific domains with the language's uncertainty, units and effect systems. Explore the implementations and examples directly:

| Domain | What you can explore | Start here |
|---|---|---|
| **Ontology & biomedical terminology** | Typed terminology, queries, reasoning, closure, EL+ and temporal models; SNOMED, GO, HPO and LOINC adapters. | [Library](stdlib/ontology/) · [Query example](examples/ontology/query_demo.sio) |
| **PBPK & pharmacology** | Physiological models with 14/28 compartments, adaptive ODE simulation, tissue partitioning, drug interactions, fitting and uncertainty-aware workflows. | [Library](stdlib/darwin_pbpk/) · [Simulation](examples/darwin_pbpk/simulation_demo.sio) |
| **Particle physics** | Lorentz, spinor and gauge foundations; parameters, propagators, vertices, amplitudes, decay and cross-section tools. | [Library](stdlib/particle_physics/) · [Uncertainty example](examples/particle_physics/a_mu_gum_split.sio) |
| **Quantum computing** | Linear qubit interfaces, statevector/VQE, Born-shot uncertainty, QASM tape and CHSH experiments, with C++23 crosschecks. | [Library](stdlib/quantum/) · [H₂ VQE](demos/quantum/h2_vqe_2q.sio) · [Device runner](demos/quantum/tools/device_run.cpp) |
| **Chemistry & hydrogen** | Kinetics, thermochemistry, equilibrium, combustion, surfaces and catalysis, including independent-oracle sources. | [Library](stdlib/chemistry/) · [Hydrogen uncertainty](examples/chemistry/h2_ignition_uq_demo.sio) |
| **Scientific numerics & learning** | Linear algebra, ODEs, statistics, automatic differentiation, optimization, Bayesian methods and tensor/neural components. | [Linear algebra](stdlib/linalg/) · [ODEs](stdlib/ode/) · [Autodiff](stdlib/autodiff/) · [Tensors](stdlib/tensor/) |
| **Biomedical data & imaging** | Genomics, neuroimaging, PET receptor kinetics and terminology-linked research examples. | [Genomics](stdlib/genomics/) · [fMRI](stdlib/fmri/) · [PET examples](examples/neuroreceptor_pet/) |
| **Algebra & operator exploration** | Hypercomplex structures, EISA and scoped GPU/operator work through Pireus. | [Algebra](stdlib/algebra/) · [EISA](stdlib/eisa/) · [Pireus](tools/pireus/) |

This map describes implemented source areas. Execution evidence belongs to the named example, compiler and target; the library is a research platform with coverage that varies by route. See [tests](tests/), [current CI](https://github.com/Sounio-lang/sounio/actions) and each domain's documentation for reproducible evidence.

## Architecture

The compiler lives in this repository alongside the language. Follow the implementation from parsing and checking through intermediate representations, lowering and emission. Backend support is described by route and by tests of the current implementation.

| Explore | Source |
|---|---|
| Compiler implementation | [self-hosted/compiler](https://github.com/Sounio-lang/sounio/tree/main/self-hosted/compiler) |
| Scientific library | [stdlib](https://github.com/Sounio-lang/sounio/tree/main/stdlib) |
| Examples and experiments | [examples](https://github.com/Sounio-lang/sounio/tree/main/examples) |
| Regression evidence | [tests](https://github.com/Sounio-lang/sounio/tree/main/tests) |
| Architecture | [Compiler documentation](https://github.com/Sounio-lang/sounio/tree/main/docs/compiler) |
| Language specification | [Language reference and grammar](docs/spec/LANGUAGE_SPECIFICATION.md) |

### Repository map

- `self-hosted/` and `bootstrap/`: compiler and bootstrap paths.
- `stdlib/`, `tests/`, `examples/` and `demos/`: libraries, regression evidence, examples and guided demonstrations.
- `docs/` and `artifacts/research/`: technical documentation and research assets.
- `website/` and `site-next/`: separate website implementations; `ecosystem/` contains distributable templates.

## Get started

The default `bin/souc` route selects Madaros; its committed prebuilt compiler is Linux x86-64. Apple Silicon also has a separate current-source native ARM64 build and real Metal execution route integrated in [#2978](https://github.com/Sounio-lang/sounio/pull/2978); see the [macOS ARM64 guide](docs/compiler/MADAROS_MACOS_ARM64.md) for its setup and limits. A clean checkout still needs the documented native artifact before the launcher can use that route. Inspect the selected artifact before running examples.

```sh
git clone https://github.com/Sounio-lang/sounio.git
cd sounio
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
./bin/souc --version
./bin/souc info
./bin/souc check examples/hello.sio
```

For setup and route details, read [INSTALL.md](INSTALL.md), [Madaros status](docs/MADAROS_STATUS.md) and the [launcher](bin/souc). Historical setup notes can describe different artifacts; the selected compiler, host and source revision determine the route.

## Read the evidence

Source, distributed binaries, individual test lanes and remote CI are separate evidence. When reporting a result, identify the source revision, selected compiler and test route.

- [Current CI](https://github.com/Sounio-lang/sounio/actions)
- [Releases](https://github.com/Sounio-lang/sounio/releases)
- [Open issues](https://github.com/Sounio-lang/sounio/issues)

## Honest Status

| Capability | Current evidence |
|---|---|
| Native Apple execution | Native ARM64 build and real Metal execution integrated in [#2978](https://github.com/Sounio-lang/sounio/pull/2978); see the [macOS ARM64 guide](docs/compiler/MADAROS_MACOS_ARM64.md). Linux parity and self-hosting remain separate claims. |
| Madaros raw payload handling | Inline-array copies through raw mutable payloads are repaired in [#2995](https://github.com/Sounio-lang/sounio/pull/2995) and covered by a dedicated regression; this does not certify every payload shape or target. |
| Scientific values through calls | Direct and stored returned `Knowledge` values preserve variance in the tested paths integrated by [#2878](https://github.com/Sounio-lang/sounio/pull/2878). |
| Derived-unit numerical representation | Scientific parameter representation repairs integrated by [#2878](https://github.com/Sounio-lang/sounio/pull/2878). |
| Quantum circuits and experiments | Born-law, QASM tape and CHSH routes integrated by [#2880](https://github.com/Sounio-lang/sounio/pull/2880), with classical execution evidence. |
| Borrow and unit binding reconciliation | Integrated in [#2882](https://github.com/Sounio-lang/sounio/pull/2882); route-specific execution evidence remains separate from the merge state. |

GPU/Pireus execution has evidence on scoped Spark routes. The generic GPU CLI path and each other backend require their own status; support is recorded per engine, target and source revision.

This status snapshot was reconciled against `main` at `06f927dbca57` on 22 October 2026. Follow the linked changes and current CI for subsequent source, target and execution evidence.

## Compiler evolution note

The 2026 preprint ([doi:10.22541/au.177067589.93635741/v1](https://doi.org/10.22541/au.177067589.93635741/v1)) describes a Rust-based compiler (v0.100.0) with Cranelift and LLVM backends. That compiler is now retired. The current compiler is **Madaros**, self-hosted in Sounio, emitting native x86-64 ELF without LLVM. Cranelift and the v0.100.0 architecture are not part of the current codebase. Version numbering restarted at v0.1.0 for the self-hosted implementation; the preprint remains a historical design record, not a description of the current system.

## Research and design

The scientific programme connects uncertainty, order and non-associative structure with executable models. Explore [the founder’s intent](FOUNDER_INTENT.md), [the language tour](TOUR.md), [research notes](docs/research/) and [the octonion associator example](examples/octonion_168_associators.sio). Formal proof, numerical verification and empirical validation remain distinct forms of evidence.

## Notion MCP in Warp

Warp agents can read this workspace's Notion pages through Notion's hosted MCP server. The config is per user, not part of the compiler, and it survives a new terminal because it lives outside the repo.

1. Write `~/.warp/.mcp.json`:

```json
{
  "mcpServers": {
    "notion": {
      "url": "https://mcp.notion.com/mcp"
    }
  }
}
```

2. Warp detects that file without a restart. The server shows under **Settings → Agents → MCP servers**, labeled detected from Warp.
3. The first start opens a browser sign-in. After that, credentials stay on the machine and are reused.
4. Confirm persistence from a new shell: the same file is still there, and a Notion search still returns workspace pages. `oz mcp list` stays empty; that command lists account-shared servers, not this file.

Checked on 24 October 2026: Warp connected to Notion MCP 1.2.0 over Streamable HTTP. A new shell still read `~/.warp/.mcp.json`, and a second search for `Sounio` still returned workspace pages.

## Develop with Sounio

Start with a reproducible issue, a focused example or a documented compiler limit. See [CONTRIBUTING.md](https://github.com/Sounio-lang/sounio/blob/main/CONTRIBUTING.md) for contribution guidance.

Created by **Demetrios Chiuratto Agourakis**. See the [project credits](https://www.souniolang.org/credits) for attribution and AI disclosure. For research citation, see [CITATION.cff](CITATION.cff). Licensed under [Apache-2.0](https://github.com/Sounio-lang/sounio/blob/main/LICENSE).
