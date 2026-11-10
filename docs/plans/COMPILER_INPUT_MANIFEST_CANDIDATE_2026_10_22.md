<!-- docs:meta
topic_id: repo.docs.plans.compiler-input-manifest-candidate-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.compiler-input-manifest-candidate-2026-10-22
-->

# Compiler input manifest: observational optimization candidate

Status: read-only audit and proposal; no cache behavior was changed.

The current science-sprint dispatch explicitly sets SOUNIO_MADAROS_NOCACHE=1 and
SOUNIO_MADAROS_CACHE_READONLY=1. Fresh builds in these receipts are forced by that
policy, not evidence that broad cache invalidation caused the observed builds.

Separately, scripts/dev/madaros-cache.sh hashes all eligible self-hosted/stdlib
files, including Markdown, and includes that tree hash in the modular key.
The seed key is already narrower. scripts/ci/build_modular_madaros.sh supplies the
modular key, with source-stability checks retained around construction.
When cache reads are enabled, unrelated scientific-module/docs changes can thus
invalidate the modular cache by design. No performance gain is measured here.

## Bounded next scope

Produce an observational dependency manifest while keeping the existing broad
key and all guards authoritative. Record:

- Relative paths and SHA of every actual consumed input.
- Seed, build recipe, environment, resolver identity and resulting ELF identity.
- Implicit imports, including f128 support, and external stdlib resolution.
- Lookup precedence, candidates and negative lookups or equivalent absence facts.

Resolved-file hashes alone are insufficient: a newly created file at a higher
priority search path may change resolution without modifying any prior resolved
file. Textual use scanning alone is also insufficient for implicit dependencies.
Inspect module_frontend.sio's SOUNIO_STDLIB_PATH handling, dependency walker and
implicit f128 imports, and lean_single.sio's search precedence/fallbacks.

Do not narrow by directory names such as stats/genomics or drop diagnostic,
artifact-hash, external-stdlib or source-stability guards. Any future cache reuse
requires a demonstrated complete input/resolution contract and a separate new
namespace. Observational manifest production is not authorization to treat an
incomplete manifest as sufficient for reuse.
