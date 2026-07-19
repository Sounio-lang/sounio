## Canonical-production destination repository provisioning receipt

The four repositories requested by the recorded mapping selection were
separately authorized for public, empty creation and provisioned under
`Sounio-lang` on 2026-07-19:

| Repository | GitHub database ID | Created at (UTC) | Observed state |
|---|---:|---|---|
| [`epistemic-core`](https://github.com/Sounio-lang/epistemic-core) | `1305314507` | `2026-07-19T00:34:39Z` | public, empty, configured default `main`, no Git refs |
| [`sounio-formats`](https://github.com/Sounio-lang/sounio-formats) | `1305314517` | `2026-07-19T00:34:41Z` | public, empty, configured default `main`, no Git refs |
| [`sounio-io-primitives`](https://github.com/Sounio-lang/sounio-io-primitives) | `1305314527` | `2026-07-19T00:34:43Z` | public, empty, configured default `main`, no Git refs |
| [`sounio-units`](https://github.com/Sounio-lang/sounio-units) | `1305314539` | `2026-07-19T00:34:45Z` | public, empty, configured default `main`, no Git refs |

The create requests used `auto_init=false`. No package content, README, tag,
release, source removal, source rewrite, or Git ref operation was performed.
Authenticated REST observation, unauthenticated public REST observation, and
unauthenticated `git ls-remote` agree on the state above.

A fresh organization observation at `2026-07-19T00:39:21Z` contains 14
repositories and has catalog identity
`46ef6e4ecde6063e3a1c744a499bc3cdca905a7334405d955cd120171142f0c6`.
The canonical `Sounio-lang/sounio` `main` head in that observation is
`ffbc1c4d10e0115ddecaf179b43f87221442ee6c`.

Provisioning receipt identity:
`93c70857c13f0e5572a7870689d3c57ec5cd004d48445e6af89007f206254569`.

The requested governance team slug
`sounio-scientific-packages-maintainers` was not observed in the organization,
so no team was created and no team permission was assigned.

This receipt is not a post-provisioning mapping reconfirmation. The four new
catalog rows are empty and have no HEAD, so they are not yet eligible for the
processor's `reuse-observed` action. Content materialization, a later catalog
observation, an explicit five-target reconfirmation, mapping proposal review,
canonical-production approval, and cutover execution remain separate steps.
