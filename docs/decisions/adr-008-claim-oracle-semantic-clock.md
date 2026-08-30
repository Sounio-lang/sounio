<!-- docs:meta
topic_id: repo.docs.decisions.adr-008-claim-oracle-semantic-clock
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-008-claim-oracle-semantic-clock
-->

# ADR-008: Single Semantic Clock for Language and Library Claims

**Status**: accepted  
**Date**: 2026-08-06  
**Supersedes in part**: informal practice of Python “mirrors” as expected-value sources  
**Related**: ADR-002 (truth layers), ADR-006 (fixed-point seed scope),  
Stage0/Stage1 compiler contract, `FOUNDER_INTENT.md`,  
research note *novel-compiler-research-2* (single semantic clock survey)

---

## Authority paragraph (binding)

Sounio is a from-scratch self-hosted language. **Executable meaning for
language and library claims is owned by Sounio under the default user
compiler (Madaros / `bin/souc`), and nowhere else.**

- **Pass/fail of a claim** (stdlib correctness, numeric identity, EL+
  closure edges, epistemic trust sentinels, algebra identities) must be
  decided by Sounio-compiled witnesses, closed forms in Sounio, or
  metamorphic/intramorphic relations implemented in Sounio—not by
  agreement with a peer language runtime.
- **Lean** is a formal / Witness export axis. It may prove properties of
  a model. It must not generate the sole numeric expected values for
  run-pass CI as a second product runtime.
- **C / stage0 / lean_single fixed-point** (ADR-006) is bootstrap and
  recovery integrity. It is not authority for library expected values
  and is not fixed-point coverage of modular Madaros.
- **Non-Sounio tools** (Python, mpmath, SciPy, shell `diff` against
  foreign goldens) may **measure**, **corroborate**, or **hunt bugs**
  (McKeeman differential testing). They may **not** define the meaning
  of a language or library claim. Hard-fail gates whose only expected
  source is a foreign runtime are **legacy anti-patterns** under
  migration, not the design.

One clock for product semantics. Bootstrap and formal axes are
non-competing integrity and proof layers—not rival executables.

---

## Context

The monorepo already demotes Stage0 to bootstrap/oracle and gates some
epistemic trust with pure Sounio sentinels (`epistemic_trust_gate.sh`).
Concurrently, many science-adjacent gates still hard-fail when
`souc` disagrees with a Python oracle (e.g. special-function mpmath
parity, bigrat digit-for-digit oracle, sedenion fiber identity). That
bifurcation is **oracle drift**: two claim-acceptance authorities for
one language.

For a language product, drift is existential: the user runs Sounio; CI
must not crown another language as the judge of what Sounio “should”
compute.

## Decision

### 1. Claim-oracle classes (machine vocabulary)

Every gate that can affect acceptance of a **language or library claim**
must be classifiable as exactly one of:

| `oracle_class` | Meaning | May set CI fail for a *claim*? |
|---|---|---|
| `sounio_native_expected` | Expected values / sentinels computed or hardcoded in Sounio under Madaros | **Yes** |
| `sounio_closed_form_twin` | Two independent Sounio implementations must agree (same language, dual path) | **Yes** |
| `external_corroboration_only` | Foreign tool measures; disagreement is report/warn unless override | **No** (default) |
| `forbidden_as_claim_oracle` | Known anti-pattern: foreign runtime is sole expected authority | Must not open new; migrate |
| `bootstrap_integrity` | Fixed-point, seed ELF, Stage0 recovery (ADR-006) | Yes for *bootstrap*, not library claims |
| `formal_only` | Lean lake / proof obligations; no numeric product expecteds | N/A for run-pass floats |
| `research_harness` | Pure-Python (or non-Sounio) research contract; **not** a language/library claim clock | No (outside product semantics) |

### 2. New work rule

- **New** claim-bearing gates and stdlib tests: `sounio_native_expected`
  or `sounio_closed_form_twin` only.
- **New** use of Python/mpmath/SciPy in a path that can fail CI on a
  language/library claim: **forbidden** without an explicit
  `external_corroboration_only` classification and non-failing default.
- Packed data / headers for research drivers: preferred path is
  **Sounio self-witness** (run produces numbers; replay or dual Sounio
  path checks). Foreign generators of “expected” headers are drafts
  until re-homed.

### 3. Legacy migration rule

Existing `forbidden_as_claim_oracle` / hard-fail foreign goldens are
**inventoried**, not deleted on day one (codegen walls and silent wrong
exits still make differential tools useful as *bug finders*). Migration
target: demote hard-fail to optional corroboration; add Sounio native
witnesses for the claim.

### 4. Inventory

Machine-readable inventory:

- Schema: `docs/decisions/claim_oracle_inventory.schema.md`
- Scanner: `scripts/dev/claim_oracle_inventory.sh`
- Snapshot: `artifacts/audit/claim_oracle_inventory.tsv`

CI may later fail on **new** rows classified `forbidden_as_claim_oracle`
without a grandfather exception.

## Consequences

- Agents and humans must not introduce Python expected-value judges for
  Sounio stdlib or language claims.
- Science path (`CLAUDE.md` principle 4) and this ADR agree: Sounio is
  the language of claims; foreign code is measurement or bootstrap only.
- Fixed-point and Lean remain; they are not expanded into “second
  product runtimes.”
- Research EL+ / sedenion / bigrat gates with foreign oracles become
  explicit migration debt, visible in the inventory TSV.

## Pilot demotion (2026-08-07)

Migrations to the ADR default:

| gate family | claim clock | foreign path |
|---|---|---|
| `scripts/special_scipy_parity_gate.sh` | Sounio `tests/parity/special_parity_*.sio` emit | mpmath report-only unless `SOUNIO_FOREIGN_ORACLE_HARD=1` |
| `scripts/bigrat_gate.sh` | Sounio `eq_or_fail` + `BIGRAT_STDLIB_OK` | Python print-diff report-only unless HARD |
| `scripts/ci/sedenion_*.sh` (16 gates) | Sounio sentinels / OK tokens / constants | Python/diff via `scripts/lib/lib_sounio_claim_oracle.sh` soft unless HARD |
| `bigrat_col` / `bigrat_ext` / `interval_rat` | Sounio OK tokens + eq paths | Python print-diff soft unless HARD |
| `linalg_parity` / `stats_dist_parity` | Sounio emitters | mpmath soft unless HARD |
| `furey_*` / `gresnigt_*` / `cd_tower_seam` | Sounio OK tokens | Python/diff soft unless HARD |
| `l8` / `l9` ZD census C contracts | C contract VERDICT PASS + hash emit | numpy FNV cross-hash soft unless HARD |
| `clinical_midazolam_ddi_e2e` | Sounio compile/run sentinels | optional Python AUCR band soft unless HARD |

Shared helper: `scripts/lib/lib_sounio_claim_oracle.sh`.

Legacy hard-fail: export `SOUNIO_FOREIGN_ORACLE_HARD=1`.

Scanner residual hygiene (2026-08-07): pure-Python research harnesses
(`suffering_aware_*`, `self_falsifying_*`, …) are classed `research_harness`
(not language claims). `native_v2_*` golden gates and `claim_ast` tooling are
`sounio_native_expected`. After reclassify, `foreign_hard_fail=yes` should
approach zero for dual claim judges; any remaining rows need manual review.

## Grounded in

- Stage0/Stage1 compiler contract; ADR-006 fixed-point seed scope
- `scripts/epistemic_trust_gate.sh` (class A pattern)
- Residual class B: sedenion fiber gates, other inventory rows
  (`artifacts/audit/claim_oracle_inventory.tsv`)
- Prior art: McKeeman differential testing; CompCert/CakeML formal clock;
  EMI / intramorphic oracles; JCGM/NPL model-derived references
- Session research: single semantic clock survey (2026-08-06)
