# Required supplied plan and qualification contracts

No example plan with guessed accepted hashes is shipped. Required schema:
sounio.root06.recipe.v1. Root supplies all actual values after qualification.

Every pinned file descriptor is {file,sha256,mode}: absolute regular-file path,
SHA-256 and exact lstat permission bits as a decimal integer. Directory manifests
use sounio.sealed-tree.v2 from the existing graph helper; symlinks are literal
target bytes and are never followed. Manifests must live outside their own trees.

Required top-level fields:
- repository, head, tree: actual Git repository and full40-character identities.
  review/staging checks current HEAD; subsequent phases use the immutable supplied
  Git object and graph, allowing unrelated later metadata commits on root.
- producer.source and producer.elf: pinned actual source/ELF. The source hash must
  equal self-hosted/compiler/lean_single.sio in the supplied Git HEAD. ELF must
  already be executable and have ELF magic.
- run_root: fresh dedicated run directory outside repository/SDK roots and ROOT05.
- build_seconds<=10800 and preflight_seconds<=1200: explicit positive deadlines.
- tools: pinned node,bash,git,tar,time,timeout,cxx; additional tools are also pinned.
- tool_environment: {file:PIN,context_R:ABSOLUTE_PATH}. This reuses the qualified
  private environment, with its original R context, then restores the new run R.
- sdk: nonempty list of {root,manifest:PIN}, covering the actual immutable private
  closure. Full path/mode/link/content census is checked before and after.
- pristine_manifest/build_manifest: pinned complete expectedv2 tree manifests.
- inputs: pinned archive,git_census,capacity,worker_optin,preflight_recipe,
  preflight_cases,metadata_helper,graph_helper,build_template,magic.
  archive is the FULL Git archive; git_census is exact Git ls-tree -rz --full-tree.
  capacity must hash eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60.
  Worker patch scope is only main.sio and its postimage must be the canonical two
  additions at the unique parser import and main entry, with no other source change.
  Physical helper/template paths must match the actually invoked frozen bundle.
- qualifications: exactly the four roles below, each {manifest:PIN,receipt:PIN}.

Qualification roles:
producer-flow-base104, producer-bounds-base34, producer-metadata-base211,
producer-frontiers. Base manifests contain exactly104/34/211 unique ordered
original case IDs. The additional frontiers manifest is root-supplied and nonempty,
covering Optiontuple/bareReturn/importfatal/source9fixes; no count is guessed.

Case manifest schema: sounio.qualification.cases.v1 with role and ordered ids.
Receipt schema: sounio.producer.qualification-adapter.v1 with role,completed:true,
manifest_sha256,producer_source_sha256,producer_elf_sha256,required,passed,
failed:0,skipped:0,ordered cases[{id,status:pass}],adapter:PIN,originals:[PIN...],
job_id. Every case must match the original manifest exactly.

Adapters preserve original actual-run logs and are bound to the FINAL source/ELF
run that generated those cases. They cannot turn old legacy receipts into evidence
for a future seed, including rebinding an earlier211 run to ccfa or any new ELF.
This schema does not claim old receipts were already structured this way. Root
reviews the adapter and actual provenance; the reader checks identities/counts.

Short preflight recipe is supplied/pinned by root; this template does not invent
a new numerical oracle or repeat the full producer suite. preflight_cases uses
schema sounio.root06.preflight-cases.v1, ordered cases[{id,role}], including logical
roles runtime-pthread,seed-cli-scalar-oracle,binding-stamp. Actual original IDs and
any additional controls are root-supplied. The recipe writes PREFLIGHT_RESULT.json:
schema sounio.root06.preflight-result.v1; head/tree; final producer source/ELF SHA;
build_manifest_sha256; case_manifest_sha256; ordered sdk_manifest_sha256;
exact ordered cases[{id,status:pass}]; failed/skipped0; adapter and originals pins.
Independent original C++23/Sounio gates stay unchanged; a JSON pass is not a new
claim that a numerical oracle ran unless the actual pinned recipe/output proves it.

Manual authorization is EXTERNAL to the plan, avoiding a circular hash.
After the final plan is frozen, root provides a record with authorized_by:root,
scope:ROOT06_STAGING_PREFLIGHT_BUILD and plan_sha256. Environment variables:
ROOT06_AUTHORIZATION, ROOT06_AUTHORIZATION_SHA256, ROOT06_AUTHORIZATION_MODE.
These record a manual decision; no script mints authorization.

ROOT06_NODE_BIN and ROOT06_NODE_SHA256 identify the qualified bootstrap Node.
The sbatch template additionally requires ROOT06_BASH_BIN,ROOT06_RECIPE,
ROOT06_PLAN and ROOT06_PHASE. It has export=NIL, so root explicitly supplies these
values in the reviewed submission. Root chooses existing owned Slurm log paths
OUTSIDE the fresh run root (which does not exist until staging starts).

Freeze this recipe bundle preserving its scripts/stdlib and tests/fixtures layout
outside source/SDK/run-root. No helper path may silently resolve to mutable WIP.
--review performs identity/metadata checks only and publishes no readiness.
Root manually invokes future --stage,then --preflight,then ONE --build only after
reviewing each actual terminal result. No such invocation happened in this delivery.
