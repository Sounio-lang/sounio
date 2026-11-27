<!-- docs:meta
topic_id: repo.docs.evidence.gen2-fixpoint.3818dbb2cab47c11ea4a29eac7bae9baa3dcae9ecfe31f2a50daa4d280fd4174
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.gen2-fixpoint.3818dbb2cab47c11ea4a29eac7bae9baa3dcae9ecfe31f2a50daa4d280fd4174
-->

# Receipt: gen2 fixed-point gate for #3755 (trait bounds) @ c5e42bb866 on the R740 (2026-11-27)

- **Job:** Slurm 31701, `gen2-gate-tb2`, node `gpuorangefs-multi-r740-proxmox`, lane `trait-bounds`. Started 18:50 UTC, job wall 03:01:12. Job file: `docs/evidence/gen2-fixpoint/3818dbb2cab47c11ea4a29eac7bae9baa3dcae9ecfe31f2a50daa4d280fd4174.sbatch` (verbatim).
- **Commit:** `c5e42bb866de54d77b2a096b6d664b424e5e44ce`, the head of #3755 rebased on main `91f32ac85c` (#3743). `self-hosted/` tree `809a0c1b127135349b01050a59d00ae183e13d36`.
- **Inputs key:** `3818dbb2cab47c11ea4a29eac7bae9baa3dcae9ecfe31f2a50daa4d280fd4174`, recorded by the job as step `r-head-key-…`.
- **Why a second receipt:** the first receipt (`33c1136b8d991ce04c086b07a208e85769f8550a1764cc577e0f136a2bda40dd`, job 29741) was for the state before #3743 touched the checker. After the rebase the key changed, so the earlier receipt no longer applies to this state.

## Prediction (job header, written before submission)

Fixpoint (gen2 == gen3), rc 0, IR parity pass; gate MaxRSS within 1.15 x 39,038,888 K; wall within 1.3 x earlier receipts. Model: the change only adds refusals (E315/E316) for violated trait bounds; Madaros's own source violates no bound; the self-compile is deterministic given its inputs. gen2 differs from main's gen2 because the source differs.

## Result (OBSERVED, `sacct -j 31701`)

| Field | Value |
|---|---|
| gen1 sha256 | `35c522172175b0d4c2e1d8ee65220685e5c8ae8e2ac26194d1c38797ec2a2fa7` |
| gate rc / wall | 0 / 10,767 s |
| gen2 sha256 | `6516cc57f767987abfa83f827be8e2eb52da85cba0b4aab93ba54a5f97cdbb1c` |
| gen3 sha256 | `6516cc57f767987abfa83f827be8e2eb52da85cba0b4aab93ba54a5f97cdbb1c` |
| IR parity | PASS (6 programs) |
| gate MaxRSS | 39,484,072 K (1.011 x job 28684) |

The prediction held on every count (`MADAROS_FIXED_POINT_OK`). Same benign `couldn't chdir` log line as earlier gates.

## How to re-read the raw record

    slurm-live sacct -j 31701 -o JobID%14,JobName%120,Elapsed,MaxRSS,State,ExitCode
