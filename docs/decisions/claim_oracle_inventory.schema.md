<!-- docs:meta
topic_id: repo.docs.decisions.claim-oracle-inventory.schema
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.claim-oracle-inventory.schema
-->

# Claim-oracle inventory schema

Companion to **ADR-008**. Rows describe scripts that can affect
acceptance of language/library claims (gates) or that supply expected
values (oracles).

## TSV columns (tab-separated, `#` comments allowed)

| column | type | description |
|---|---|---|
| `gate_id` | string | Path relative to repo root (stable id) |
| `kind` | enum | `gate` \| `oracle` \| `parity_ref` \| `other` |
| `oracle_class` | enum | See ADR-008 table (+ `research_harness`) |
| `foreign_hard_fail` | `yes`/`no`/`unknown` | Does foreign mismatch set fail/exit≠0 by default? |
| `sounio_witness` | `yes`/`no`/`partial` | Is there a Sounio sentinel / ALL PASS / native expected? |
| `foreign_runtimes` | string | Comma-separated: `python3`, `mpmath`, `scipy`, `none`, … |
| `ci_tier` | string | Heuristic: `ci` \| `root_scripts` \| `dev` \| `research` \| `selfhost` \| `other` |
| `notes` | string | Short free text (no tabs) |
| `migration` | enum | `none` \| `keep` \| `demote_corroboration` \| `rehome_sounio` \| `delete` |
| `scanned_utc` | ISO-8601 | Scanner timestamp |

## Classification heuristics (scanner defaults)

Automated classes are **provisional**. Human override may append
`artifacts/audit/claim_oracle_inventory.overrides.tsv` (same columns;
last wins).

| Signal in file | Provisional `oracle_class` |
|---|---|
| Invokes `python3` / `*.py` and `diff`/mpmath on fail path **without** Sounio/C witness | `forbidden_as_claim_oracle`, `foreign_hard_fail=yes` (demote next) |
| Same numeric foreign path **with** Sounio/C witness or optional markers | `external_corroboration_only`, `foreign_hard_fail=no` |
| Dual `souc`+Python without numeric foreign judge | `sounio_native_expected` (Python tooling) |
| `souc`/`ALL PASS`/`GUM_TRUST`/`check: OK` and no foreign judge on fail path | `sounio_native_expected` |
| Two `souc` engines or dual Sounio paths compared | `sounio_closed_form_twin` |
| Fixed-point / bootstrap / gen2 gen3 / `scripts/bootstrap/*` / `scripts/selfhost/*` | `bootstrap_integrity` |
| Lean lake / formal (optionally with tooling Python) | `formal_only` |
| Path `scripts/archive/*`, CI fixtures, pure C/CUDA receipt, Python-only meta | `research_harness` (not a language claim clock) |
| ADR-008 soft foreign markers (`lib_sounio_claim_oracle`, `SOUNIO_FOREIGN_ORACLE_HARD`) | `external_corroboration_only`, `foreign_hard_fail=no` |
| Shell meta with neither Sounio nor Python nor C contract | `research_harness` |
| Insufficient signal | `unknown` → notes must say so; treat as review debt |

## Migration priorities (ADR-008)

1. Rows with `foreign_hard_fail=yes` and `oracle_class=forbidden_as_claim_oracle`
2. Claim-bearing stdlib gates under `scripts/` and `scripts/ci/`
3. Research oracles that still define paper-facing numbers

## Commands

```bash
# from repo root
bash scripts/dev/claim_oracle_inventory.sh
# writes artifacts/audit/claim_oracle_inventory.tsv
# summary on stdout
```
