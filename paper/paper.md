---
title: "Sounio: A Systems Programming Language for Epistemic Computing"
tags:
  - programming languages
  - scientific computing
  - uncertainty quantification
  - type systems
  - epistemic computing
authors:
  - name: Demetrios Chiuratto Agourakis
    affiliation: "1, 2"
    orcid: 0009-0008-2276-5847
    email: demetrios@agourakis.med.br
    corresponding: true
  - name: Marli Gerenutti
    affiliation: 1
    orcid: 0000-0001-7165-646X
affiliations:
  - name: "Faculdade de Ciências Médicas e da Saúde, Pontifícia Universidade Católica de São Paulo (PUC-SP), Brazil"
    index: 1
  - name: Sounio Project
    index: 2
date: 2026-02-16
bibliography: paper.bib
---

# Summary

Scientific and engineering software routinely mixes measured data, model assumptions, and computed results, but most general-purpose languages represent all values as if they were exact. This creates a recurring failure mode: uncertainty is recorded in comments, spreadsheets, or side documents rather than encoded in program semantics. Sounio addresses this by treating uncertainty as a first-class language concern. Its central abstraction, `Knowledge<T>`, couples a value with uncertainty and provenance information so computations can preserve epistemic context throughout pipelines.

Sounio is implemented as an open-source systems language and compiler (approximately 627,000 lines of Rust, with a 42,500-line self-hosted bootstrap) focused on scientific workflows where traceability and uncertainty propagation are required. The project combines static checking, domain-oriented standard-library modules (235,000 lines), and effect annotations for side-effect control in high-stakes code paths. The implementation includes native code generation and a scientific standard library covering autodiff, pharmacometric modeling, numerical methods, and dimensional analysis.

# Statement of need

In many domains, software correctness is not only about obtaining a numerically plausible value; it is also about reporting how that value is known, with what uncertainty, and under what assumptions. This is explicit in measurement science and metrology standards such as GUM [@jcgm2008gum], and it is practically important in regulated biomedical and clinical contexts [@fda21cfr11; @iso17025].

Current practice usually falls into one of three unsatisfactory patterns:

1. Uncertainty is ignored in implementation even when present in source data.
2. Uncertainty is tracked manually outside the type system, which is error-prone.
3. Library wrappers are used selectively, without language-level guarantees that uncertainty metadata is preserved.

Sounio is designed for researchers and engineers who need uncertainty-aware computation without abandoning systems-level tooling. The target audience includes scientific software developers in pharmacometrics, causal inference, and adjacent computational science workflows where provenance and confidence handling are part of the core task, not optional post-processing.

# State of the field

Software for uncertainty handling generally appears in two forms. The first is library-based numeric uncertainty propagation (for example Python and Julia ecosystem packages), which can be effective but relies on developer discipline for consistent use [@lebigot2024uncertainties; @giordano2016measurements]. The second is probabilistic programming environments that model uncertainty statistically but are typically domain-specific and not intended as general-purpose systems languages [@carpenter2017stan; @bingham2019pyro; @cusumano2019gen].

Sounio contributes a different point in the design space:

1. Uncertainty-aware values are encoded in core typing via `Knowledge<T>`.
2. Uncertainty propagation semantics are integrated with language tooling and compiler checks.
3. Effect annotations make non-local behaviors explicit, improving auditability and reviewability.
4. The standard library is organized for scientific workloads, including epistemic, units, and domain modules.

This approach is inspired by adjacent lines of work in types for physical units [@kennedy2009units], linear/resource-sensitive typing [@wadler1990linear], and algebraic effects [@plotkin2009handlers], while focusing specifically on uncertainty-preserving scientific programs.

# Software availability and documentation

Sounio is distributed under the Apache License 2.0 and developed in a public Git repository (<https://github.com/Sounio-lang/sounio>) with open issue tracking. The repository includes source code, examples, automated tests, and contributor documentation. Core documentation covers language usage, epistemic types, effects, and standard-library modules. Examples include uncertainty-aware epidemiological and pharmacometric workflows, along with focused compiler test fixtures.

# Acknowledgements

The author thanks contributors and early users who reviewed language behavior, diagnostics, and documentation during pre-release iterations. Community feedback on scientific use cases and test quality substantially improved the project’s submission readiness.

# References
