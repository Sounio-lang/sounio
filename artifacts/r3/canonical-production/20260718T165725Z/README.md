# R3 Canonical-Production Mapping Decision Evidence

This directory preserves the first real Sounio v1 mapping selection processed
by the non-authorizing canonical-production mapping-decision interface.

The source response is GitHub issue #1122 comment `5012124187`, authored by
`agourakis82` at `2026-07-18T16:57:25Z`. The response selects four
`request-new` public repositories and one catalog-exact `reuse-observed`
repository. It authorizes only draft mapping preparation.

The processor and independent verify mode both reconstructed:

- decision identity: `246cd09179f1b0a49aebb2d87d65d33f48dbdd68d17df76d245d34bca7a034de`
- receipt identity: `f74837d7d0ae83c6ba3d8d13a317c6024d0aab5c83bfda1069aa4b97f42567b3`
- status: `destination-repository-creation-required`
- proposal output: `not-emitted`
- execution authority: `none`

## File Digests

Run `sha256sum -c SHA256SUMS` from this directory for a fail-closed byte check.

| File | SHA-256 |
|---|---|
| `repository-observation.graphql.json` | `323a40959b167bedf7e56264a794498f513b7478b1306b6774b32d17d7a3f701` |
| `repository-catalog.v1.json` | `7997070b3f4006f02a6e959b770101fbbf22693ae783f85f78714b381be9b043` |
| `canonical-production-gap-assessment.v1.json` | `9f349121d7bc0517b7ed4fa6a180a3c7db9a7126c77691760327190199f8e10d` |
| `source-response.md` | `f2e01687686dfa09df5acf173e67aa9c9d73fe22988302a72926e5d16c39408b` |
| `mapping-decision.v1.json` | `558ba54028addbcf5f5bc2b109609c4601588f288c3aed97cb8d33fc69acce3e` |
| `mapping-decision-receipt.v1.json` | `5b6fe346ca2dc5b1a6cf9ccd141376636c9d1a1135c14897e6200f7356b439ff` |

## Bound State

- repository catalog observed at: `2026-07-18T16:53:04Z`
- catalog identity: `122ed8713f46286fc8ff9d46a0f44812207d37bf0473b524d87ef38dd6f0bcf8`
- canonical `main`: `32fed91bb01c2269af8edd802c2afaf17509adfa`
- source response URL: `https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5012124187`
- processing receipt URL: `https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5012139001`

The receipt does not authorize repository provisioning. Any catalog or source
drift requires a new observation and complete reconfirmation after the four
requested repositories are separately authorized and provisioned.
