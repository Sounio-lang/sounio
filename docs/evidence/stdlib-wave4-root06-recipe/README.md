# ROOT06 reviewed recipe templates; no actual ROOT06 execution

Prepared from worktree baseb4263c22b45e5b726050a2736e99f55f5b8b4907.
That commit is provenance for this delivery, NOT a required future source HEAD.
Every future plan must supply its actual qualified HEAD/tree, producer-source SHA
and producer-ELF SHA. No current ROOT06 plan or readiness is provided here.

Files:
- scripts/stdlib/root06_build_template.sh: root-reviewed stage/preflight/build phases.
- scripts/stdlib/root06_build.sbatch.in: unsubmitted resource template.
- tests/fixtures/wave4_root06_recipe/metadata.mjs: read-only graph/receipt checks,
  terminal publication contract and safe NUL-delimited path transport.
- controls.mjs: synthetic metadata fixtures only.

Root owns final review, source sealing, authorization and submission. None of
--stage, --preflight or --build has been run on the actual Sounio source.
The only Slurm jobs in this delivery execute syntax and metadata controls.

The source9 producer/frontier failures remain a real waitpoint. The four required
qualification roles must come from actual matching final producer runs, after the
fixes. A historical pass cannot be rebound to a different ELF by an adapter.

The future run keeps complete source-git and source-build trees separate.
source-git is checked against the full Git archive commit/census/blob identities,
including docs/tools/runtime and literal symlinks. cp -a preserves targets and
modes; source-build is checked AGAIN against the original Git census and fullv2
manifest BEFORE overlays. Only exact capacity patcheaa and the two canonical
Worker opt-in additions are permitted. Complete post-overlay manifests are supplied
and verified, including every directory, file mode/content and literal link target.

Private producer, cache, compiler lock, recipe lock and outputs live outside both
source trees and every immutable SDK root. The recipe applies no global install,
uses no alternate-node claim and never edits ROOT05. GNUtime, CXX, Node, Bash,
Git, archive tools, private environment and magic database must be supplied/pinned.

The finalizer records all whole-graph EXIT guards and terminal status first.
RAW_READY/PREFLIGHT_READY publication is the final operation, never based only on
producer exit0 or ELF existence. Explicit diagnostics with exit0 refuse. All
failure artifacts remain. A build artifact remains a candidate after publication;
services, numerical tests and compiler proofs retain their independent gates.

See CONTRACT.md for required schema, REVIEW.md for ROOT05 observations, and the
metadata receipts for scope. The templates have not been qualified as an actual
staging/preflight/compiler build.
