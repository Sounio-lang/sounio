# R3 Catalog-Bound Mapping Proposal Evidence

This directory preserves the reconfirmed five-row mapping selection, its
deterministic processing receipt, the emitted `proposed-not-approved` mapping,
and a downstream non-authorizing gap assessment.

The selection was submitted at `2026-07-20T03:03:29Z` against catalog
`7bc569476058987386b096336256eef08eb4f2ac56d6c693c02cdb8ee7e933d6`.
It authorizes only preparation and review of this proposal. It does not
authorize repository changes, source removal, production approval, or cutover.

## Identities

- catalog identity:
  `7bc569476058987386b096336256eef08eb4f2ac56d6c693c02cdb8ee7e933d6`
- source-response SHA-256:
  `b92601cfb96d21da48ae2fb7d4c0c3504ca9ce8eec584b08da988f72cb7fdccb`
- mapping-decision identity:
  `3e75c95bed5a4b7525fdb018c2224423d4865d657709b96f6254ab942924e743`
- processing-receipt identity:
  `33b49ef2b913a80222110bed3374716f2c3b7ddfc212bee4672627cc3990e1b7`
- mapping-proposal identity:
  `5214a0d49c9fffab408fcfa255360bd6cb872cf045f0b2df40e4aa3119fa6439`
- gap-assessment identity:
  `d387b614d2f9fab8c00d5f3e2cb563de3dd679563de83c4f43a5ff13335e7abe`
- post-review drift-observation identity:
  `d238a91529396ab48e932b16d82207688fd876fb0a08d0e9d40586ac08fed1db`
- exact source-inventory identity:
  `e26c4dbbc19d127c13051213a156f7e323c7d3c4a4424a2b0c2f40600309bb67`

## Proposed Mappings

| Target | Repository | Expected `main` |
|---|---|---|
| `distribution:epistemic-core` | `Sounio-lang/epistemic-core` | `3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1` |
| `distribution:sounio-formats` | `Sounio-lang/sounio-formats` | `c412c0d1e7ef276d3ad9d1e662d681369e3e384c` |
| `distribution:sounio-io-primitives` | `Sounio-lang/sounio-io-primitives` | `8e593615072e7ad9962ab27c0e316a8be521457d` |
| `distribution:sounio-research-examples` | `Sounio-lang/sounio-examples` | `a22f66e0060ba6d007b8b69012ecadee7e9345bd` |
| `distribution:sounio-units` | `Sounio-lang/sounio-units` | `229d310f676d2a3a1e183983764da2ddd63f6fe0` |

All five selections are `reuse-observed`. The emitted mapping remains
`proposed-not-approved`; execution authority is `none`; canonical production
is not approved; cutover is `not-executed`.

## Evidence Files

- `source-response.md`: exact body of issue comment
  [`5018371541`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5018371541).
- `repository-observation.graphql.json`: original organization-wide GitHub
  observation underlying the bound catalog.
- `repository-catalog.v1.json`: validated point-in-time catalog.
- `drift-observation.v1.json`: later read-only destination-head and governed
  source-tree continuity observation.
- `source-inventory.v1.json`: exact seven-unit source inventory at canonical
  source head `5cf8be05b96c0a5c2ab101e022b36019dd61ebef`.
- `mapping-decision.v1.json`: deterministic transcription of the selection.
- `mapping-decision-receipt.v1.json`: deterministic processor result.
- `mapping-proposal.v1.json`: five-row `proposed-not-approved` proposal.
- `canonical-production-gap-assessment.v1.json`: downstream readiness report.

Run `sha256sum -c SHA256SUMS` from this directory for a byte-level integrity
check.

## Verification

The processor ran against a clean independent local `main` at exactly
`5cf8be05b96c0a5c2ab101e022b36019dd61ebef`, matching the catalog. Both
`process` and `verify` reconstructed the receipt and proposal from the bound
catalog, source inventory, Git observation, and mapping decision.

The gap assessor's `assess` and `verify` modes reconstructed assessment
identity `d387b614...`. It found all five mapped destinations available at the
cataloged heads, then stopped at
`production-evidence-and-human-decision-required` with execution authority
`none`.

At `2026-07-20T12:26:31Z`, canonical `main` had advanced 47 commits to
`4620b28892a6d224498e2149cae51affdfc8223a`. The bound head remained its
ancestor, all five governed source-tree object IDs were unchanged, and all five
destination `main` heads still matched the catalog. This later read-only check
does not replace or widen the catalog-bound proposal.

## Remaining Boundary

This package does not create or modify repositories, materialize or remove
source files, create or update Git refs, authenticate responder or
organizational authority, approve destination owners, approve canonical
production, or approve or execute cutover. Any later catalog or governed-source
drift requires a new selection record before downstream use.
