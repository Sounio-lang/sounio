# R3 Canonical-Production Repository Provisioning Evidence

This directory preserves the separately authorized creation and immediate
post-creation observation of the four public repositories requested by the
mapping selection recorded on 2026-07-18.

The repositories were created with `auto_init=false`. They are public and
empty, report configured default branch `main`, and have no Git refs. No package
content, README, release, tag, source removal, or source rewrite was performed.

## Identities

- provisioning receipt identity:
  `93c70857c13f0e5572a7870689d3c57ec5cd004d48445e6af89007f206254569`
- post-creation catalog identity:
  `46ef6e4ecde6063e3a1c744a499bc3cdca905a7334405d955cd120171142f0c6`
- source mapping decision identity:
  `246cd09179f1b0a49aebb2d87d65d33f48dbdd68d17df76d245d34bca7a034de`
- source mapping processing receipt identity:
  `f74837d7d0ae83c6ba3d8d13a317c6024d0aab5c83bfda1069aa4b97f42567b3`

The fresh catalog was observed at `2026-07-19T00:39:21Z`, contains 14
repositories, and records canonical `Sounio-lang/sounio` `main` at
`ffbc1c4d10e0115ddecaf179b43f87221442ee6c`.

## Public Receipt

Issue comment
[`5013552386`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5013552386)
records the provisioning result. Its body is preserved as
`issue-1122-provisioning-receipt.md`; both local and remote bodies have SHA-256
`e4d548de0fd236a01ded6544df8277f6a6c6db4a1f82e04eb396e524d6152ab1`.

## Evidence Files

- `repository-observation.graphql.json`: organization-wide GraphQL observation.
- `provisioned-repositories.rest.json`: authenticated REST observation.
- `public-repositories.rest.json`: unauthenticated public REST observation.
- `git-ref-observation.json`: unauthenticated `git ls-remote` ref counts.
- `repository-catalog.v1.json`: validated 14-row canonical-production catalog.
- `repository-provisioning-receipt.v1.json`: deterministic provisioning receipt.
- `issue-1122-provisioning-receipt.md`: exact public receipt body.

Run `sha256sum -c SHA256SUMS` from this directory for a fail-closed byte check.

## Remaining Boundary

The four destinations are not eligible for `reuse-observed` while empty because
they have no valid HEAD. The requested target-owner team slug was not observed,
so no team was created or assigned. Content materialization, a later catalog,
an explicit five-target reconfirmation, mapping proposal review, production
approval, and cutover remain separate operations.
