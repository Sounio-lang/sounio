# PBPK Interop Roadmap for Sounio (Proposal)

**Status:** Proposed  
**Date:** 2026-04-20  
**Target:** Sounio v1.1+  
**Scope:** Code-first PBPK engine, observed-data pipeline, and calibration/export interoperability inspired by OSP workflows

## Motivation

Sounio already contains real PBPK infrastructure, but it does not yet present it as a coherent, production-facing workflow for model construction, observed-data ingestion, and downstream fitting/export.

The practical opportunity is not to clone PK-Sim/MoBi as a GUI suite.
The practical opportunity is to turn Sounio into a **typed PBPK kernel and interoperability layer** that can:

- express organ-resolved PBPK models in repo-native code
- track uncertainty and provenance through the simulation path
- ingest observed concentration data in a stable schema
- emit measurement/problem tables for downstream calibration and sensitivity tools
- support rich transporter-aware scenarios, especially brain/plasma workflows

This proposal defines a roadmap for that path.

## Repo Truth Today

### What is already real

- Validated Darwin PBPK science lane:
  - `tests/stdlib/darwin_pbpk/test_pipeline_real_e2e.sio`
  - `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- 14-compartment adaptive solver kernel:
  - `stdlib/darwin_pbpk/tsit5_pbpk14.sio`
- Executable reference-case wrapper:
  - `stdlib/darwin_pbpk/simulation_real.sio`
- Epistemic PBPK lane:
  - `stdlib/darwin_pbpk/epistemic_pbpk14.sio`
- Scientific CSV export with uncertainty columns:
  - `stdlib/io/scientific.sio`
- Organ-specific compartment work already exists:
  - `stdlib/darwin_pbpk/compartments/brain.sio`
  - `stdlib/darwin_pbpk/compartments/liver.sio`
- Mechanistic DDI and population-related surfaces already exist:
  - `stdlib/darwin_pbpk/ddi/mechanistic_ddi.sio`
  - `stdlib/darwin_pbpk/population/pop_sim.sio`

### What is not yet productized

- No first-class observed-data import schema for PBPK fitting workflows
- No stable PEtab export surface
- No first-class AMICI or Pumas adapter
- No unified transporter configuration model spanning organs and directions
- No curated “blessed” public PBPK API that cleanly separates validated kernels from exploratory examples
- No GUI workflow comparable to PK-Sim/MoBi

### Important constraint

The stable foundation should be the validated `stdlib/darwin_pbpk` lane, not the older mixed-quality `examples/pbpk/` tree.
Some examples are valuable research material, but they should not define the public contract.

## Product Direction

The right target is:

**Sounio as a typed PBPK computation and export platform**

Not:

**Sounio as an immediate replacement for PK-Sim/MoBi’s full modeling desktop UI**

That means the first-class deliverable should be:

1. a stable simulation kernel
2. a stable data model
3. stable import/export surfaces
4. calibration/sensitivity interoperability

and only later:

5. a richer user-facing workflow layer

## Core Use Cases

### 1. Brain/plasma time-activity curve generation

User provides:

- compound parameters
- patient or population covariates
- transporter settings
- dosing scenario

Sounio returns:

- blood/plasma trajectory
- brain trajectory
- derived exposure metrics
- uncertainty-aware measurement table

### 2. Transporter-aware organ modeling

User specifies:

- organ-local transporters
- directionality
- localization fractions
- organ-specific overrides

Sounio simulates:

- altered tissue exposure
- BBB restriction or enhancement
- hepatic/renal uptake-efflux scenarios

### 3. Calibration-ready observed-data workflow

User imports:

- experimental concentration-time data
- metadata for subject, matrix, organ, molecule, assay, units

Sounio produces:

- normalized internal observation table
- mapped outputs
- residual-ready export tables

### 4. Downstream fitting/export

Sounio exports:

- PEtab-style measurement tables
- parameter tables
- observable definitions
- uncertainty-enriched CSV or JSON for custom toolchains

## Proposed Public Architecture

### Layer 1: Stable kernel

Keep and harden:

- `stdlib/darwin_pbpk/tsit5_pbpk14.sio`
- `stdlib/darwin_pbpk/simulation_real.sio`
- `stdlib/darwin_pbpk/epistemic_pbpk14.sio`

Role:

- canonical simulation engine
- canonical validated reference cases
- canonical uncertainty propagation path

### Layer 2: PBPK domain schema

Add:

- `stdlib/darwin_pbpk/schema/compound.sio`
- `stdlib/darwin_pbpk/schema/patient.sio`
- `stdlib/darwin_pbpk/schema/transporter.sio`
- `stdlib/darwin_pbpk/schema/observation.sio`
- `stdlib/darwin_pbpk/schema/units.sio`

Role:

- make model inputs explicit and typed
- separate raw data ingestion from simulation execution
- define a stable contract for import/export

### Layer 3: Observed-data ingestion

Add:

- `stdlib/darwin_pbpk/io/observed_csv.sio`
- `stdlib/darwin_pbpk/io/observed_tsv.sio`
- `stdlib/darwin_pbpk/io/normalization.sio`
- `stdlib/darwin_pbpk/io/mapping.sio`

Role:

- normalize heterogeneous experimental input
- resolve units and matrix names
- map observations to simulation outputs

### Layer 4: Interoperability/export

Add:

- `stdlib/darwin_pbpk/export/petab_measurements.sio`
- `stdlib/darwin_pbpk/export/petab_parameters.sio`
- `stdlib/darwin_pbpk/export/petab_observables.sio`
- `stdlib/darwin_pbpk/export/calibration_csv.sio`

Role:

- emit portable downstream fitting assets
- keep Sounio independent of any one fitting engine

### Layer 5: Calibration and sensitivity harness

Add:

- `stdlib/darwin_pbpk/fit/problem.sio`
- `stdlib/darwin_pbpk/fit/residuals.sio`
- `stdlib/darwin_pbpk/fit/sensitivity.sio`
- `stdlib/darwin_pbpk/fit/objective.sio`

Role:

- support internal objective calculation
- support local sensitivity and parameter ranking
- provide exportable problem definitions

## Transporter Roadmap

This is the main feature gap relative to the OSP-style workflow.

### Current state

The repo already contains transporter-related material, but it is fragmented:

- P-gp-related logic appears in example code
- liver compartment work references transporter abundances
- brain exposure constraints are tested in epistemic PBPK scenarios

### Target state

Define a first-class transporter model with:

- transporter identity
- organ
- localization
- direction
- abundance
- confidence/provenance

### Proposed transport types

- `Influx`
- `Efflux`
- `Bidirectional`
- `Pgp`

### Proposed localization model

- generic organ-local compartments
- explicit support for:
  - blood
  - interstitial
  - tissue
  - apical
  - basolateral
  - BBB

### Proposed data type

```sio
enum TransportDirection {
    Influx,
    Efflux,
    Bidirectional,
    Pgp
}

enum TransportLocalization {
    Blood,
    Interstitial,
    Tissue,
    Apical,
    Basolateral,
    BBB
}

struct TransporterSetting {
    name: string,
    organ_id: i32,
    direction: TransportDirection,
    localization: TransportLocalization,
    fraction: f64,
    abundance: f64,
    confidence: f64
}
```

The exact syntax may need adaptation to current compiler limits, but this is the shape of the contract.

## Observed Data Roadmap

### Goal

Make experimental time-concentration input a first-class PBPK artifact rather than ad hoc CSV handling.

### Internal canonical row

Each observation row should carry at least:

- `subject_id`
- `time`
- `time_unit`
- `matrix`
- `organ`
- `compartment`
- `molecule`
- `value`
- `value_unit`
- `uncertainty`
- `assay`
- `source`

### Why this matters

This is the bridge between:

- Sounio simulation outputs
- residual analysis
- PEtab-like measurement tables
- later AMICI/Pumas adapters

## Export Roadmap

### Minimum export target

Support a PEtab-style subset first:

- measurement table
- parameter table
- observable table

### Why “PEtab-style” first

A strict full PEtab implementation can come later.
The first milestone should be enough to feed external calibration code without blocking on total standards coverage.

### Existing leverage

`stdlib/io/scientific.sio` already supports:

- CSV matrix printing
- uncertainty-paired CSV output
- JSON matrix output

That means the export work is mostly:

- schema design
- naming conventions
- stable table emission

not low-level I/O invention.

## Milestones

### Milestone 0 — Curate the kernel

Deliverables:

- declare `stdlib/darwin_pbpk` as the canonical PBPK lane
- add a README that explicitly marks validated modules vs exploratory modules
- reduce dependence on `examples/pbpk/` for contract claims

Acceptance:

- all public PBPK claims point to validated stdlib/tests paths

### Milestone 1 — Brain/plasma reference workflow

Deliverables:

- canonical brain/plasma scenario in stdlib
- stable output table for blood and brain trajectories
- uncertainty-aware CSV export

Acceptance:

- one end-to-end `check` + `run-pass` reference case
- reproducible `brain_to_blood_ratio`
- stable output columns

### Milestone 2 — First-class transporter schema

Deliverables:

- transporter config types
- organ-local transporter settings
- BBB-focused directionality support
- liver/kidney/gut/brain localization support

Acceptance:

- transporter setting materially changes trajectory output
- BBB restriction case is covered by tests

### Milestone 3 — Observed-data import normalization

Deliverables:

- normalized observed-data row type
- CSV/TSV import path
- unit normalization and output mapping

Acceptance:

- imported data can be mapped to blood/brain outputs
- residual-ready normalized table is produced

### Milestone 4 — PEtab-style export

Deliverables:

- measurements export
- parameters export
- observables export

Acceptance:

- exported tables are stable and documented
- one golden-file test per table

### Milestone 5 — Calibration/sensitivity bridge

Deliverables:

- internal objective/residual calculation
- sensitivity ranking
- export-ready fitting problem bundle

Acceptance:

- at least one small calibration benchmark runs end to end
- parameter ranking is reproducible

### Milestone 6 — External adapters

Deliverables:

- AMICI-oriented export adapter
- Pumas-oriented export adapter or documented conversion script

Acceptance:

- one documented round-trip example per downstream tool

## Validation Strategy

Each phase should ship with:

- one executable reference case
- one golden artifact
- one invariant report
- one uncertainty-specific assertion

### Required invariant classes

- non-negative concentrations
- finite outputs
- mass-balance sanity
- transporter constraint sanity
- export row-count and column-schema stability

### Failure taxonomy

Classify failures as:

- PBPK kernel
- transporter model
- data normalization
- export contract
- calibration bridge

Do not collapse them into generic “PBPK failed”.

## Non-Goals

Not in the first implementation wave:

- desktop GUI parity with PK-Sim/MoBi
- full visual chart editor
- complete standards-compliant PEtab coverage on day one
- full automatic import support for every external pharmacometrics format
- replacing OSP’s entire qualification ecosystem

## Immediate Next Step

The next concrete milestone should be:

**Brain/plasma TAC + PEtab-style measurement export**

Specifically:

1. build a canonical brain/plasma scenario on top of `stdlib/darwin_pbpk`
2. define a normalized observation row schema
3. emit a stable measurements CSV with uncertainty columns
4. add golden tests for that output

This is small enough to implement incrementally and large enough to prove the direction.

## Decision

Sounio should pursue this as a **typed PBPK interop platform**, not as a direct UI clone of PK-Sim/MoBi.

That path is aligned with the repo’s actual strengths:

- typed scientific modeling
- explicit uncertainty/provenance
- validated PBPK kernels
- exportable scientific data products

If successful, Sounio can become the place where PBPK models are:

- authored as code
- validated with explicit uncertainty
- exported into downstream calibration ecosystems

rather than a partial imitation of an existing desktop suite.
