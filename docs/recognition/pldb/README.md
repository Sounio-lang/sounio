# Sounio PLDB submission package

Prepared for SIO-610, under the SIO-608 recognition epic. This directory is a
submission package for the founder to review and submit. No PLDB submission,
upstream pull request, account creation or catalogue acceptance is implied.

Sounio is a **research language (pre-1.0); APIs may change; not recommended for production use.**

## Contents and factual sources

- `sounio.scroll`: the proposed `concepts/sounio.scroll` entry in PLDB's current Scroll format.
- `sources.json`: every field value, immutable Sounio source URL, source-file SHA-256 and textual witness.
- `validate.cjs`: checks source witnesses and duplicates, then parses the entry and two malformed controls with PLDB's actual parser. This checks data format and source-text witnesses; it does not mechanically establish that a paraphrase follows from those witnesses. Factual interpretation still requires review. It is not Sounio compilation or language-feature qualification.

Sounio evidence is pinned to
[`3765b7d024a46a67c7a13675ba1e92857cffb8bd`](https://github.com/Sounio-lang/sounio/tree/3765b7d024a46a67c7a13675ba1e92857cffb8bd).
The entry is intentionally small:

| Fields | Repository evidence |
| --- | --- |
| `id`, `name`, former name (`aka`), creator, programming-language tag | [README](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/README.md) |
| `appeared 2025` | [Changelog entry dated 2025-12-25](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/CHANGELOG.md) and [founder's start-date account](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/FOUNDER_INTENT.md). Independently, the public [v0.97.0 release](https://github.com/Sounio-lang/sounio/releases/tag/v0.97.0) has a GitHub publication timestamp of 2025-12-28. Only the year is proposed; no claim about the exact first announcement day. |
| Website | [CITATION.cff URL](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/CITATION.cff) |
| Repository | [README clone URL](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/README.md) |
| `fileExtensions sio` | [Editor language registration](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/tools/editors/vscode/package.json) |
| Documentation | [Language specification](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/docs/spec/LANGUAGE_SPECIFICATION.md) |
| Description and research-status warning | [README overview and status](https://github.com/Sounio-lang/sounio/blob/3765b7d024a46a67c7a13675ba1e92857cffb8bd/README.md) |

There is no proposed latest-version, DOI, user count, ranking, benchmark,
production-readiness, clinical-validity or independent-adoption field. Historical
citation/version metadata is not used to override the current research-status
warning. Existing recognition work remains with its respective card owners.

## Format and validation

PLDB was inspected at
[`22cf13b127a581c6d0a25c59f93301a39c99cf7e`](https://github.com/Programming-Language-DataBase/pldb/tree/22cf13b127a581c6d0a25c59f93301a39c99cf7e):
its [`concepts/gleam.scroll`](https://github.com/Programming-Language-DataBase/pldb/blob/22cf13b127a581c6d0a25c59f93301a39c99cf7e/concepts/gleam.scroll),
[`code/conceptPage.scroll`](https://github.com/Programming-Language-DataBase/pldb/blob/22cf13b127a581c6d0a25c59f93301a39c99cf7e/code/conceptPage.scroll),
[`code/measures.parsers`](https://github.com/Programming-Language-DataBase/pldb/blob/22cf13b127a581c6d0a25c59f93301a39c99cf7e/code/measures.parsers),
and [`ScrollSetCLI` usage](https://github.com/Programming-Language-DataBase/pldb/blob/22cf13b127a581c6d0a25c59f93301a39c99cf7e/cli.js)
define the format and parser-loading route. An older `.pldb` recipe would be stale.

In a separate PLDB checkout at that revision, install its declared dependencies
with lifecycle scripts disabled. Pin `scroll-cli@178.2.3` and `scrollsdk@107.0.1`,
which are the parser versions used for this package; retain the resulting lockfile
with the validation receipt. From the Sounio checkout, run:

```sh
node docs/recognition/pldb/validate.cjs /absolute/path/to/pldb
```

Expected: `PASS` for the entry, its source witnesses and duplicate search, with
both an invalid year and an unknown measure rejected. A nonzero exit is a failed
validation. The validator reports source/parser revisions, entry and lockfile
hashes. It reads Git objects for the pinned Sounio evidence and does not edit the
PLDB checkout, render its website, submit data or execute Sounio programs.

## Founder handoff

1. Review the entry and the source map. Reconcile any existing PLDB entry or
   changed contribution procedure against the then-current upstream before submission.
2. In the founder's PLDB working copy, place `sounio.scroll` at
   `concepts/sounio.scroll`; its relative template import is intentional.
3. Run the then-current upstream validation and review the one-file diff.
4. The founder submits through PLDB's contribution process. Record the actual
   submission URL and, when accepted, the public entry URL on SIO-610.

An internal Sounio merge delivers this package. It does not complete SIO-610's
public-entry acceptance or the SIO-608 recognition epic. Internal merges follow
the live Notion Merge order view; this documentation package does not bypass
higher-priority release or correctness work.
