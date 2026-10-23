<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-deep-science-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-deep-science-2026-10-23.readme
-->

# Checked scientific stdlib: native integration receipt

Slurm job16153 completed with exit0: RESULTS.tsv contains330 records and no
nonzero statuses. STATUS is0. The exact archived index is Git tree
efa57fc8d2f4d5033230148ccc3b0b09ba3a2a5f, committed as
74996a7119722486ce1e0a1ace52815bc8aa0407. This archive includes foundation,
scientific 2D and science changes on the ce3886de895fa669ab325cb66f23fa5133ef9e90
source baseline. Subsequent CI-only contract corrections and generated metadata
are separate changes; they do not change this numerical source acceptance.

The source-built protected compiler was:
1148eabd3905ec76911d027c578fb8d6b2bd2dde2c186ee3c02c4c3b97733029.
Compiler build and callback/borrow controls are recorded in the earlier compiler
receipt. This replay consumes that ELF; it does not rebuild or prove its freshness
by itself. Compiler and listed stdlib/tests input hashes were unchanged across
this run. The input hash manifest covers those listed files, not every possible
tool/environment dependency. It is an endpoint integrity check, not a proof that
every file was consumed.

## Included evidence

RESULTS.tsv records exact negative checks, typechecks, compiles, native statuses,
expected stdout markers, independent C++ compile/emitter/concordance results and
endpoint integrity checks. resources/ retains per-executable time and maximum
RSS, identities/ binds compiled executables and fixture sources, output/ retains
scientific stdout and independent oracle summaries. RAW_OUTPUTS.sha256 hashes
the full scratch transcripts and manifests, including outputs not duplicated
here. Renderer raw outputs and ELFs remain on the remote scratch surface:
 /scratch/sounio-stdlib-deep-20261023/final1/artifacts.
The dispatch/job files specify the frozen worker invocation and environment.

The gate independently exercises exact PCG state/counters, one-million uniform,
small-mean Poisson and PTRS draws; full NB2 likelihood including216 Big400 cases;
256 high-precision local likelihood increments; count GLM/Fisher/profile/
fixed-alpha bootstrap; normalization selection and26 Big400 precision cases;
incremental TSV; checked special functions; dynamic QR/LU/SVD/eigen kernels;
checked ODE/RODAS controls; histograms, summaries, KDE and actual Canvas/SVG
references. See source tests and module contracts for the individual scopes and
tolerances. A passing reference suite is not a universal numerical error bound.

## Failure controls and scope

An earlier integrated run16139 exposed an incorrect exact analytic -1 assertion
in the slot fixture. Diagnostic native output localized the assertion; the test
now checks exact snapshot preservation against its saved scalar, and a separate
C++ analytic comparison retains its pre-existing tolerance. The final replay
includes the corrected fixture and explicitly positive minimum-subnormal input.

The RODAS references retain the failing unmodified Boost1.83 nonautonomous
positive-d4 control and passing corrected/augmented controls. Selected exact
rational Lean coefficient consistency was separately built in job16124; this
native replay does not claim a full solver proof or repeat all Lean targets.

The legacy Stage2 render-suite regression found in PR3048 is a separate open
integration issue at this receipt checkpoint. This Madaros replay does not
establish compatibility with every engine. GitHub CI and actual main merge are
separate acceptance states. No finite-sample calibration, global profile
certificate, profile-refit bootstrap, full RNA pipeline or DESeq2 parity is
claimed.

CXX_TOOLCHAIN.txt and SLURM.txt normalize incidental trailing whitespace for
versioned review; their recorded values are unchanged. Original tool output
remains on scratch (the raw CXX transcript hash is in RAW_OUTPUTS.sha256).
