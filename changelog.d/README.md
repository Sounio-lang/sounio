<!-- docs:meta
topic_id: repo.changelog-fragments
authority: repo_only
audience: contributors
last_validated: 2026-10-26
validated_by: scripts/ci/changelog_fragment_gate.sh --selftest
source_of_truth: docs/decisions/adr-018-release-cycle-and-versioning.md
-->

# Changelog fragments

Do not edit `CHANGELOG.md` in a pull request. Add a fragment here instead. Every PR adds its own file, so fragments never conflict in the merge queue. At each release candidate, `scripts/release/changelog_assemble.sh` turns the fragments into the release section and deletes them. The rules are in [ADR-018](../docs/decisions/adr-018-release-cycle-and-versioning.md) §3–§4.

## When a fragment is required

When the PR touches anything that the Impact classifier counts as compiler, runtime or stdlib. CI scripts count too. The `Changelog fragment gate` step of the Impact job enforces this for PRs opened after 2026-10-25T03:00Z.

For those PRs the gate also refuses a hand edit to `CHANGELOG.md`, regardless of when the PR was opened — the release manager assembles `CHANGELOG.md` from fragments at release time, and a release-assembly PR touches no compiler, runtime or stdlib path, so it is not bound here. A fragment added by rename or copy counts the same as one added outright, and every added fragment must carry content at the head commit: an empty file satisfies the census while saying nothing.

## Name

`changelog.d/<slug>.<kind>.md`

- `slug`: lowercase letters, digits, `.`, `_` and `-`. The branch name is a good choice, for example `fix-atan-reduction`.
- `kind`: exactly one of the kinds below. A PR with two kinds of change adds two fragments.

| Kind | Use it when | Release |
|------|-------------|---------|
| `breaking` | a program that compiled no longer does, or its meaning changes | major |
| `refusal` | a program now refuses instead of printing a wrong or fabricated value (the soundness exception); name the diagnostic code | minor |
| `semantics` | the path a program takes changes in a declared way, or an uncertainty budget grows | minor |
| `numerical-correction` | an output moves toward an exact oracle and fails the consistency test; cite the oracle receipt | patch |
| `numerical` | an output changes but stays epistemically consistent (ADR-018 §3) | patch |
| `added` | new language feature, stdlib item, flag or backend | minor |
| `changed` | other user-visible change that is not breaking | minor |
| `deprecated` | something is scheduled for removal | minor |
| `removed` | a deprecated item is removed after its window | major or minor, per the deprecation policy |
| `fixed` | a defect fixed with no numerical or semantic change for correct programs | patch |
| `performance` | faster or smaller, same output | patch |
| `security` | a vulnerability fixed | patch |
| `internal` | no user-visible effect (refactor, CI, tests); still say what changed | none |

## Content

Write one paragraph in Markdown, as it should read in the release notes. Start with the effect on the user, not the implementation. Do not include the PR number, because the assembler adds it.

```markdown
`atan(x)` for `0 < x < 1` used the sign of the base as a branch flag and returned `atan(2)` for `atan(0.5)`. It now agrees with the exact C++23 oracle.
```
