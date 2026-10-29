# Read-only ROOT05 review and changes expressed by the templates

Reviewed root-integrated05/build.sbatch,preflight.sbatch,STAGING_NOTES.md and
capacity.patch on the mounted DL380. They were not edited. The capacity file was
rehashed before/after the synthetic metadata jobs.

ROOT05 already declares/records its capacity patch and two-line Worker opt-in.
ROOT06 preserves that declared provenance while separating pristine Git origin,
the copied snapshot and the post-overlay identity. No old overlay is reclassified
as an unknown change.

Concrete publication difference: the older recipes write Ready before returning
through the EXIT trap, which subsequently checks additional guards. The template
puts publication after every finalizer guard and terminal status. A consumer
cannot observe a new Ready marker while those guards are still pending.

Fullv2 guards add complete path/type/mode/link-target/content fidelity and detect
extra/missing directories or files, beyond rechecking the regular-file list and
separate symlink targets. Source-build is rechecked against the original Git census
before overlays, including a synthetic relative-link control.

The exact existing eaa capacity patch is required, never regenerated from guessed
replacements. Patch applicability must pass on the future actual source. Worker
postimage equality requires only the original two additions. Tracked bin files
remain intact; the qualified seed is copied privately.

The final producer must pass104/34/211 and additional frontier manifests on the
actual final source/ELF. The older ROOT05 producer identity is not baked into this
template. Compiler exit0 plus an ELF is insufficient: explicit diagnostic output
is refused, including known dropped writes, invalid registers and closure errors.
The recipe treats grep read errors as failures too.

Private compiler and recipe locks are distinct to avoid self-deadlocking the
existing modular build lock. Cache/output paths stay outside S. Logs and failed
ELFs are retained. No source deletion, package installation, global change,
ROOT05 mutation, second bootstrap or cross-node execution is performed here.
