<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-source-removal-authorization
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-source-removal-authorization
-->

# Physical Extraction Source-Removal Authorization

Status: executable R3 temporary-copy authorization boundary; canonical source removal is not executed.

## Scope

The authorization interface consumes a fully verified R3 materialization and a
separately reviewed
`sounio.physical-extraction-source-removal-policy.v1`. It reconstructs the
entire source snapshot in a temporary external workspace, removes only the
exact `extract-planned` roots from that copy, applies declared byte-exact
repository repairs, runs declared post-removal gates without a shell, and
revalidates the original sources and materialized destinations before emitting
one deterministic authorization receipt.

The receipt states exactly:

```text
authorization_type = verified-post-removal-candidate-authorization
authorization_status = authorized-not-executed
source_removal_execution_status = not-executed
assurance_level = identity-only
```

The authorizer has no source-removal execution command. Actual deletion, if it
is ever approved, belongs to the separate
`r3-physical-extraction-source-removal-execution` interface.

## Required Inputs

Authorization requires all of the following:

1. The original repository, rings, ownership TSV, inventory, destination
   policy, destination markers and copies still reproduce the supplied
   materialization receipt exactly.
2. The removal policy binds the exact inventory file and identity plus the
   exact materialization file and identity.
3. Its removal scope equals all and only the inventory units whose disposition
   is `extract-planned`; every source root, target, tree digest, file count and
   byte count is bound by one scope identity.
4. At least two distinct repository-local review evidence records are bound by
   reviewer label, path, size and SHA-256. Distinct labels are evidence of
   separate records, not proof of organizational independence.
5. At least one retained-path repair binds its original byte identity, one
   repository-local replacement evidence file, and the exact expected result.
6. At least one post-removal gate binds its argument vector, candidate-relative
   working directory, timeout, expected zero exit, stdout digest and stderr
   digest.
7. The temporary workspace is a preexisting regular directory outside both
   the source repository and materialization destinations. The receipt is also
   written outside the source repository and must be absent.

Policy identity is a deterministic change detector. It is not a signature and
does not establish hosted-service, legal, organizational or ownership
authority.

## Authorize And Verify

```bash
python3 tools/science_boundary/source_removal_authorizer.py authorize \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory /approved/extraction-inventory.json \
  --destination-policy /approved/destination-policy.json \
  --destinations-root /approved/destinations \
  --materialization-receipt /approved/materialization-receipt.json \
  --removal-policy /approved/source-removal-policy.json \
  --workspace-root /node-local/authorization-workspace \
  --authorization-receipt /approved/source-removal-authorization.json

python3 tools/science_boundary/source_removal_authorizer.py verify \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory /approved/extraction-inventory.json \
  --destination-policy /approved/destination-policy.json \
  --destinations-root /approved/destinations \
  --materialization-receipt /approved/materialization-receipt.json \
  --removal-policy /approved/source-removal-policy.json \
  --workspace-root /node-local/authorization-workspace \
  --authorization-receipt /approved/source-removal-authorization.json
```

Both commands reconstruct and test a fresh temporary candidate. Verification
then compares the supplied receipt with the fully reconstructed expected
receipt. Candidate paths and workspace locations are excluded from the
receipt, so equivalent evidence is deterministic across different external
workspace roots.

## Candidate Semantics

The tool snapshots every regular source file outside `.git`, copies those
bytes into a new temporary directory, and refuses symbolic links or special
files. It deletes planned roots only inside that candidate. Retained and
blocked inventory roots remain exact except for policy-declared repairs; the
entire candidate tree must equal the source snapshot minus planned files plus
those exact repairs.

Post-removal commands execute directly from their declared argument vectors.
Their exit status and output hashes must match policy, and the candidate tree
must still be exact after every gate. The temporary candidate is deleted on
both pass and refusal. Original sources and materialized destinations are
reverified immediately before receipt promotion.

The receipt binds:

- inventory, materialization and removal-policy file and semantic identities;
- exact removal scope and aggregate unit, file and byte counts;
- review records and applied repair evidence;
- gate argv, cwd, timeout, exit, stdout and stderr evidence;
- original and resulting candidate tree identities.

## Evidence Boundary

A passing receipt establishes that one exact planned scope produced one
removed, repaired and gate-passing temporary candidate while the original
source snapshot remained unchanged. It authorizes only that bound scope for a
future, separately governed execution interface.

It does not establish:

- deletion, absence or movement of any canonical source file;
- a real production migration or production destination approval;
- remote repository creation, commit, push, ownership or publication;
- organizational independence from distinct local reviewer labels;
- complete execution-environment capture or independent replay;
- scientific truth, clinical validation or clinical authority.

The focused adversarial gate is
`scripts/ci/physical_extraction_source_removal_authorization_gate.py`. The
composed shell gate first executes R0-R2, R2.5, R2.6, inventory and
materialization acceptance against the same archived source and current-source
Madaros witness.

The current composed fixture witness is Slurm job `6527` on
`gpuorangefs-r770-proxmox`. It passed 178 R0-R2, 65 R2.5, 82 R2.6, 141
inventory, 167 materialization, and 527 authorization checks using one
archived source snapshot and one current-source Madaros. The authorization
identity is
`84f864551bcbb2265006fab62d7a19895c3deb59163a1967d380c0e027a90a28`;
the policy identity is
`efb3071bb11d220a25e4e279bd54323a8db406a1a324191486fe6001100f80a0`.
These are deterministic fixture identities, not production approvals.

## Canonical Repository Status

The repository contains no approved production destination policy,
materialization receipt, source-removal policy or authorization receipt.
Consequently, every canonical source root remains in place. The executable
fixture proves interface behavior only; it does not authorize or perform a
canonical migration.

## Next Interface

`r3-physical-extraction-source-removal-execution` now consumes an exact valid
authorization, a separate execution policy, retained approval evidence and
four exact CLI confirmations. It performs only the bound repair and removal
set in the explicitly marked local root and verifies the resulting repository.
The canonical repository still has no production execution policy. Its next
interface is `r3-physical-extraction-canonical-cutover-approval`.
