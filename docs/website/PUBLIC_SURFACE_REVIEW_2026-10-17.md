<!-- docs:meta
topic_id: repo.docs.website.public-surface-review-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.website.public-surface-review-2026-10-17
-->

# Public entry-surface review — 17 October 2026

## Scope and source identity

Prepared from main commit `92d70d1eb6c178895fc0f72429352b7d11d2db7c`,
after the Greek user journey and LLM context work in PR #2986. This change
addresses the seven homepages, shared navigation/footer/metadata, seven
learning hubs, and nine first-use pages in English, Portuguese and Greek.
It is not an audit or translation of every historical technical page.

The founder's two hero artworks, emblem, source examples, compiler, bootstrap
paths, scientific modules and research destinations remain intact. The
homepage no longer renders the previous unscoped clinical assurances,
feature counters and illustrative pseudo-programs. This is a presentation
boundary, not a rejection of the research direction or a scientific result.

## Delivered changes

- A single localized homepage heading and a first-program entry point for
  every audience; explicit paths to LLM assistance, research, evidence and
  contribution. Source destinations that remain English are marked as such.
- Platform-aware onboarding: committed Linux x86-64 prebuilt and separate
  source-built macOS ARM64 prerequisites, never an implied universal binary.
- Complete existing hello/struct programs in EN/PT/EL. Their bytes match the
  historical execution manifest; expected output is not described as a new run.
- Responsive navigation, a native seven-language disclosure, localized
  accessibility labels, keyboard dismissal and synchronized theme controls.
- Search isolates all canonical locale prefixes, including `zh-hk`; implements
  the advertised arrow/Enter navigation; suppresses stale requests; handles
  preload rejection; and collects locale matches before limiting visible hits.
- Translation notices describe partial coverage rather than falsely declaring
  an already-translated page untranslated. The English spec remains canonical.
- Shared structured metadata retains platform boundaries and correct authorship
  without inventing a release version. Existing historical artifact snapshots
  are not refreshed or promoted by this documentation change.

## Executed validation

On macOS ARM64 with Node `v26.9.0`, from the isolated worktree:

```sh
cd website
npm run check:quality
cd ..
bash scripts/dev/check_docs_registry.sh
bash scripts/ci/public_claims_gate.sh
git diff --check
```

The website quality command passed, including the existing i18n, docs-parity,
build, Pagefind, routes, redirects, brand, navigation, fallback, highlighting
and Astro checks. Astro reported 237 files with zero errors, warnings or hints.
Navigation checked 29,638 internal links; highlighting checked 136 pages;
the existing locale-fallback gate checked 77 pages. These counts are bounded
structural checks, not proof that every page is accurate or accessible.

**Native-render limitation:** the existing render-assets substep attempted the
Linux Madaros executable on macOS, received exit 126 and explicitly skipped
render regeneration using pre-rendered assets. The overall quality command's
success must not be reported as fresh native rendering or compiler execution.
The new public-surface gate itself invokes no compiler and changes no resolver.
The build's incidental tracked `artifactStatus.ts` snapshot changes are excluded.

The new gate, wired into `check:quality`, reports seven locales, seven learning
hubs, nine onboarding pages, fourteen mutation/parser controls, seventeen
search controls and no errors. It verifies generated content, selected local
route/source destinations, metadata boundaries and complete example hashes.
It does not validate arbitrary external HTTP endpoints or all anchor fragments.
The Greek source validator now checks the MDX frontmatter title that the docs
layout actually renders, rather than requiring a second in-body H1. Plain
Markdown still requires its Greek H1. English/missing MDX titles are refused;
twenty mutation controls and ten mocked preflight controls pass.

### Failing before, passing after

The first version of the new structural gate failed on the preceding production
build: duplicate homepage headings, untranslated hero text, absent language
control/`zh-hk` alternate, unscoped claims and inaccurate software metadata.
The repaired build passes. A direct probe of the original private search
predicate returned `true` for an English search result at
`/zh-hk/learn/getting-started/`, where the required result was `false`.
The canonical-locale regression now rejects it. Synthetic search controls also
exercise a Greek match ranked after twelve English hits and request invalidation.

The first exact-head CI run (`39039652983`, head `2713690da65a`) passed
the Website job but correctly failed Contracts Core at the selected-doc-snippet
check: the first-program contract still hashed the replaced illustrative struct.
The follow-up changes only that contract's hash to
`b795fedfd7bd16ab2f7edbccbef9229c59a8bc4985b3af250337cf65151435f8`,
the exact unchanged `examples/native/struct_basic.sio` bytes displayed on the
page. Its `standalone` kind and required `positive` compiler verdict remain
unchanged; no skip, contextual downgrade or gate bypass is introduced. The
subsequent exact-head CI must still execute that positive compiler check.

### Interactive checks

Browser testing of the built preview confirmed Greek search results, ArrowDown
focus and Enter navigation to Greek getting-started; corrected single-heading
onboarding; native language selection from Greek to Portuguese; menu and
language-picker Escape dismissal; and synchronized desktop/mobile theme state.
The Greek header was visually inspected at the 1280px breakpoint and the
Portuguese/Greek mobile surface at 390px. The temporary viewport and dark-theme
preference were restored. This is focused interaction testing, not a WCAG
conformance certification or a native-speaker translation acceptance.

## Independent LLM review

Reviews used the isolated remote `bin/llm-offload` wrapper, not the remote
development checkout. Raw reviews are operational temporary files; hashes below
identify the reviewed input and responses. A completed response is evidence of
review, not proof of correctness.

1. Public content and shared chrome, `/tmp/llm-offload-gcSb6C`:
   xAI `grok-4.3` and Moonshot `aws-bedrock/kimi-k3:global`, both `stop`.
   No supported major blocker. Fixed complementary image crossfade,
   language-picker dismissal, malformed-metadata reporting, actual onboarding
   counts and visible English-destination labels.
   - Input: `4a8499386cda097ea77d32a1716e5142d98c2670a14fd12992a8fc31e58b7512`
   - xAI: `597b8260b352b22950d78ece3b6c6601d0951152146fe9d7697b47f01e78ef73`
   - Kimi: `05db48db67674b0d308411ee5e6213ee451831a23a1f085426fbdeef81989449`
2. Interaction review, `/tmp/llm-offload-OtPLQB`, same two models, both `stop`.
   Kimi identified the missing `zh-hk` exclusion and nonfunctional keyboard
   hints as MAJOR; xAI independently flagged keyboard navigation. Both are
   fixed, with executable regressions and focused browser checks. Accepted
   stale-request, preload-rejection and duplicate-theme-control findings.
   Rejected the claim that audience controls lack synchronization: the existing
   `useAudience` hook already distributes `audience-change`; it is unchanged.
   - Input: `50b697d01b5bb2a6840de5004e30ef5f1c767dbe34716d30d30673fea767e21e`
   - xAI: `6dcace2cf19680dc1ba83c137f2a318a85acc491b417b242b5e0760e60bbce12`
   - Kimi: `d2db6d857278feb49c3094adf4dd3d52d06aa4b8815d237d488de72fb2721f0c`
3. Search repair review, `/tmp/llm-offload-xklIUV`, same two models, both `stop`.
   No supported MAJOR/BLOCKER. Both identified slice-before-filter as a remaining
   risk; the final helper now collects matching results in bounded batches
   before truncating. Focus containment also handles an unexpected outside focus.
   - Input: `6b346ec9ea63e296ce574397ddf2a6a1ed77323f9e49dc4be4fed31fe85dea62`
   - xAI: `f87edc6d035edb549b6b4d97170b9dd7149ce36d492f851c72be867aac7f068b`
   - Kimi: `2146c053f4c3a1a1d0f04bf38a5d5e0c9249c85df522d521ec73ffa1c4f1e20d`

4. Final-delta review, `/tmp/llm-offload-Jxb9BG`: xAI `grok-4.3` and Moonshot
   `aws-bedrock/kimi-k3:global`, both `stop`, completed with no supported
   BLOCKER/MAJOR in locale collection, cancellation,
   partial-translation notice and duplicate-heading removal. Provider identities
   and hashes are recorded below. No provider timeout is counted as a pass.
   - Input: `22399e9b867527024c63b9ccb3b2e77df0de9de9b00989e858f26b625d6e87cf`
   - xAI: `94d872aa04afaf5e162f1aa7560445c559fc99176e3466dae189995d8ecb7b85`
   - Kimi: `d2ad41002890a6d799b6a92cbe5c6ce1f8dace7982c4711adf0ec469ad684e48`
5. Greek heading-validator delta, `/tmp/llm-offload-zHt8kz`: same two models,
   both `stop`, no supported MAJOR/BLOCKER. The built-page gate enforces one
   H1; the source gate checks the localized title, not rendered HTML. The
   positive no-body-H1 assertion preserves the title-only case, while missing
   and English titles are independently rejected.
   - Input: `51a315725efbae579214bfa67e367515b7c6c5bd7c4e46b0779b944de9ac21fd`
   - xAI: `de599155c64da2089844449f2edd56d80ff744af3fa595db2d78b1a3b738c118`
   - Kimi: `555d885ab14095a52f8322e7b7e71fd0ac19cf9f2d6966a5eab18a8106f9c916`
6. CI-discovered snippet-contract repair, `/tmp/llm-offload-ExCCly`: same two
   models, both `stop`, no supported MAJOR/BLOCKER. Review verifies the
   hash-only positive-contract change, not compilation. A separate local
   byte-identity assertion confirms source, fence and contract agree.
   - Input: `001420edc0c501d31bc6dbf731d5a1c34fead6937b715d5d190ae1ace6b9d0b2`
   - xAI: `4c57f91841ab0f77e04a4df4595368acc31c326549e870d30ead1166c4adc5ed`
   - Kimi: `e07249db9d6ef836d58c69002b82214b1a2c56bbaa1d32487b89795a1ee8c00c`

## Publication and remaining boundaries

This receipt records preparation and local checks only. Exact-head CI, protected
merge queue, main ancestry and production deployment must be verified separately
and recorded on the associated pull request after execution.

No new Sounio program, scientific computation, theorem or clinical-pathway change
is introduced. Historical examples retain their existing evidence profile and
failures elsewhere in the corpus are not reclassified. No legacy compiler path
is deleted. Deeper scientific exhibits and the remaining technical translations
need separate review; this work does not claim universal implementation,
clinical readiness, complete translation or success with every LLM.
