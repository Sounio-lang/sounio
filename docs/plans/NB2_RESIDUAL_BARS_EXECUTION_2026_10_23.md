<!-- docs:meta
topic_id: repo.docs.plans.nb2-residual-bars-execution-2026-10-23
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.nb2-residual-bars-execution-2026-10-23
-->

# Checked NB2 residual bars: native execution — 2026-10-23

## Source and scientific scope

Main basis94866e463d follows the completed scientific integration PR3024.
Branch: `sprint/nb2-residual-bars-main-94866e-20261023`.
Initial frozen example/compiler build source:
`99082d7bcc404ef308347c25b2e67954ddf4c793`.
Corrected consumer:
`13c825f7a7b3fbc78c7c25754d212d244b09a1d2`.

This adds a real consumer of existing checked statistics and production signed
bar rendering. It does not introduce a new fitting algorithm. Nine synthetic
counts are evaluated with supplied mean4 and alpha0.25; two panels show NB2
Pearson and signed deviance residuals. The observation index is categorical;
the count mapping is printed separately. Both panels use one symmetric y domain
computed in Sounio from available residuals and explicit zero references.

Residual status is checked before plotting. Squared-deviance availability is
reported independently and does not suppress an available signed residual.
The zero-height observation remains unpainted and is disclosed in the caption.
No CI, p-value, model acceptance or numerical-concordance claim is made.

The author consulted Darwin sounio_verificado first; its let-mut candidate failed
E040/parse_failed and was corrected before the real Slurm check. Its candidate
check does not validate this repository source. Root source review and an
independent reviewer checked statuses, scratch-array copying and display meaning.

## Slurm execution without the training quota

The training filesystem was at its3TiB quota with0bytes free. Source and all
explicit job input/output/temp/cache/lock paths were placed on the t560 ext4
/scratch mount shared into the cpuops worker. The login pod has a different
/scratch overlay; submission therefore uses worker stdin through sbatch as
uid/gid1000, pinned cpu-ops / cpuops-t560-proxmox.

The batch and dispatcher each enforce identity; --export=NIL plus env-i excludes
ambient compiler/seed overrides. Canonical scripts/ci/build_modular_madaros.sh
selects the bootstrap and derives the seed. There is no Forge corpus transport
label or test-suite replacement in this source packet. Control scripts live
outside the unmodified Git source archive. Long submissions run in t560 tmux.

This avoids training-quota source/output writes in the recipe. The scheduler's
separate Ceph PVC remains involved; it is not an all-Ceph independence claim.
No historical deletion or quota increase was performed. The direct lane is an
observed fallback, not a replacement for the normal Forge workflow.

### Attempt1: native rejection of an invalid consumer request

Slurm16017: FAILED1:0, elapsed6:58, TotalCPU6:53.967, batchRSS2712792K.
The canonical fresh compiler build itself completed0: elapsed400.89s,
user397.03s, system1.21s, maxRSS736380KiB.
All nine library/example typechecks completed0. The new native program then
returned1 with EXAMPLE_UNAVAILABLE: axes, before emitting SVG.

Root localized this to two horizontal axes requesting5 intervals in400pixels.
The existing viz_add_axis guard requires space for112pixels per interval.
The fix changes only those two interval arguments to2. Data, domains, residual
kernels and the axis guard are unchanged. Failure stdout/ELF identity and the
original frozen source archive remain preserved; this was real program execution,
not an added test or manufactured mutant.

### Attempt2: corrected consumer and existing native control

Slurm16019: COMPLETED0:0, elapsed17seconds, TotalCPU16.415seconds,
batchRSS2441628K. The correction's Git diff for bin/scripts/self-hosted/stdlib
is empty. The retry explicitly reuses the compiler freshly built in16017 and
rechecks its artifact hash; it does not claim a compiler rebuild from13c825f7a7.
The corrected consumer is separately frozen, hashed and typechecked.

Compiler SHA-256:
`238c7e210d82cdd8474f68c488a0aed2b22d3157f64536703f03e1d995dce734`.
Corrected bars source SHA-256:
`e9b1281d5911fdaacb7b8eafb41b457289c58275a612a548d6cb84cd2faa5eda`.

| Native program | Typecheck/compile/execute | Runtime RSS KiB | Reported wall s |
|---|---|---:|---:|
| viz_nb2_residual_bars | all0 | 1584 | 0.00 |
| existing viz_nb2_deviance | all0 | 1052 | 0.00 |

0.00 is centisecond reporting resolution, not zero runtime. Whole-job RSS includes
compilation and differs from native runtime RSS. No comparative speedup or global
memory reduction is claimed. The example reuses two scratch arrays and requests
no explicit Canvas allocation; this is not an imported-runtime allocation proof.

## Actual figure review

Both actual stdout streams were byte-extracted into report, HTML and SVG using
the retained recipe. Both SVGs parsed with xmllint exit0. Full-aspect sips PNG
rendering and root inspection completed. Independent inspection of the corrected
1380x620 bars figure found no blocking issue: common symmetric ticks, aligned zero
references, positive/negative rectangles with complete end footprints, intentional
zero-height gap, visible parameter disclosure and no CI/p-value claim. No pixel
or independent numerical concordance was performed.

## Source endpoints and compact custody

The endpoint checks prove equality of manifest-listed inputs, not full-tree
immutability or complete consumed inputs. The wrapper materializes an additional
bin/madaros-linux-x86_64 and verification marker from the bound gzip. After-inventory
records both additions; the materialized prebuilt ELF's hash/readback is separate
from the fresh raw compiler explicitly selected for checks and compilation.
No broader cache authority or narrower key is inferred.

Private custody is at
/home/devsounio/sounio-artifact-archive/nb2-residual-bars-20261023.
Source archive and compressed executables are retained with decompression hashes
matched against scratch originals. Raw runtime ELFs are not duplicated into that
custody; raw execution trees remain in scratch. This is host-local custody, not
an independent physical backup. Training free space was observed as0 before and
after the runs; no capacity reservation for future jobs follows.

[Public compact evidence](../evidence/nb2-residual-bars-2026-10-23/README.md)
includes both attempts, control recipes, identity/resource/status receipts,
manifest metadata and actual vector figures. [Consumer contract](../../examples/viz_nb2_residual_bars.md)
retains the scientific and storage limits. Complete corpus/CI, Canvas equivalence,
numerical concordance and merge of this new checkpoint remain separate states.
