# R3 Post-Reconciliation Mapping Reselection Evidence

This directory preserves the referential post-reconciliation reselection,
deterministic processing receipt, emitted `proposed-not-approved` mapping, and
downstream non-authorizing gap assessment.

The directory name records the start of the evidence phase. The public
selection comment was published at `2026-07-19T14:54:08Z`, and the processing
receipt was published at `2026-07-19T15:04:02Z`.

## Selection Boundary

The interactive instruction `faca o proximo passo` followed an explicit
statement that the next required action was a new selection record against
catalog
`243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`.
The public record transcribes that instruction referentially, incorporates the
same five mappings explicitly reconfirmed in earlier issue comment
`5014112002`, and updates only the catalog binding and the reconciled
`epistemic-core` head.

This is session-scoped selection evidence, not an independently authenticated
quotation of a newly repeated five-row statement. It authorizes only
preparation and review of a `proposed-not-approved` mapping.

## Identities

- catalog identity:
  `243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`
- selection body SHA-256:
  `d4dc5ba0ea581928d14a962e3cbaf1340be430641e9fbb1fd9b5a52765ebae9e`
- mapping decision identity:
  `dd9d8efde5ff256a7fdce36b163b9b9885a50f1d698946b23de5e1ad57f2c7c8`
- processing receipt identity:
  `a921d72dffd4669d021bd91c5d04c4561ff3cfe3a36ce854edf186ff549b1377`
- proposal identity:
  `44f3a2f91534ca17fc0cd8e6794a78989629e5660256375464f33e48b743e069`
- downstream gap-assessment identity:
  `3c17f8b9229f00f789afe0771e02755cf1650a2b75fe431d5ef06c326d65d784`
- exact source inventory identity:
  `d7afd27e5ee04625a0ee7d76444cf22d1651fabf314f2fbf7577268e1f597cb4`
- post-selection drift-observation identity:
  `86baf00fa4af2692b6c4edc4f326c936266fc383b32b57a6aca168355a2ac1e4`
- processing-receipt body SHA-256:
  `7f135470ddc2ebe8acfb5b1af34b69775bc8fc5fc3d12979253042cfc2da07a7`

## Proposed Mappings

| Target | Repository | Expected `main` |
|---|---|---|
| `distribution:epistemic-core` | `Sounio-lang/epistemic-core` | `3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1` |
| `distribution:sounio-formats` | `Sounio-lang/sounio-formats` | `c412c0d1e7cf276d3ad9d1e662d681369e3e384c` |
| `distribution:sounio-io-primitives` | `Sounio-lang/sounio-io-primitives` | `8e593615072e7ad9962ab27c0e316a8be521457d` |
| `distribution:sounio-research-examples` | `Sounio-lang/sounio-examples` | `a22f66e0060ba6d007b8b69012ecadee7e9345bd` |
| `distribution:sounio-units` | `Sounio-lang/sounio-units` | `229d310f676d2a3a1e183983764da2ddd63f6fe0` |

All five rows are `reuse-observed`. Proposal status remains
`proposed-not-approved`; execution authority is `none`; canonical production
is `not-approved`; cutover is `not-executed`.

## Public Evidence

- referential reselection:
  [issue comment `5016172474`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5016172474)
- deterministic processing receipt:
  [issue comment `5016204605`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5016204605)

Both local Markdown bodies are byte-identical to the API-returned comment
bodies.

## Evidence Files

- `repository-observation.graphql.json`: original organization-wide GitHub
  observation underlying the selected catalog.
- `repository-catalog.v1.json`: validated 14-row point-in-time catalog.
- `drift-observation.v1.json`: later read-only destination-ref and governed
  source-tree observation.
- `source-inventory.v1.json`: exact seven-unit inventory at the catalog-bound
  canonical source head.
- `issue-1122-post-reconciliation-mapping-reselection.md`: exact public
  selection body.
- `issue-1122-post-reconciliation-mapping-reselection.api.json`: API response
  preserving that body and metadata.
- `mapping-decision.v1.json`: deterministic selection transcription.
- `mapping-decision-receipt.v1.json`: processor result.
- `mapping-proposal.v1.json`: five-row `proposed-not-approved` proposal.
- `canonical-production-gap-assessment.v1.json`: downstream assessment.
- `issue-1122-post-reconciliation-mapping-processing-receipt.md`: exact public
  processing receipt.
- `issue-1122-post-reconciliation-mapping-processing-receipt.api.json`: API
  response preserving that body and metadata.

Run `sha256sum -c SHA256SUMS` from this directory for a fail-closed byte check.

## Verification

The processor ran from a clean independent clone whose local `main` was exactly
`e19af3279a040a6a707967d786be657bdf0d4203`, matching the catalog. Its
`process` and `verify` modes independently reconstructed receipt identity
`a921d72d...` and proposal identity `44f3a2f9...`. The inventory's `inventory`
and `verify` modes reconstructed identity `d7afd27e...`. The gap assessor's
`assess` and `verify` modes reconstructed assessment identity `3c17f8b9...`
and retained `production-evidence-and-human-decision-required`.

At `2026-07-19T15:05:26Z`, all five public destination heads still matched the
catalog. Canonical `origin/main` had advanced to
`c5d8f752cfbe0827cc649b76328ab61f283d2837`, but the five governed source
trees remained identical to those at the catalog-bound head.

The selection body, proposal bundle, and processing receipt were reviewed by
xAI/Grok 4.3 and Z.AI/GLM-5.2 with no BLOCKER or MAJOR finding. DeepSeek and
Gemini provider failures are recorded. An xAI minor note incorrectly described
a `sounio-examples` repository-ID mismatch; the executable artifacts all use
`repository_id: sounio-examples`, and the disagreement is retained in the
offload log rather than silently adopted.

## Remaining Boundary

This evidence does not modify repositories or Git refs, authenticate responder
or organizational authority, approve a destination owner, remove source files,
create teams or branch rules, publish tags, releases, or registry entries,
approve canonical production, or approve or execute cutover. The downstream
production-evidence set and any later explicit production/cutover decision are
separate interfaces and remain absent. Later catalog or governed-source drift
requires another selection record before downstream use.
