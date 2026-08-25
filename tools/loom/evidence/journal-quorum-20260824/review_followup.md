# Follow-up: checkpoint binding finding

Reassess the previous MAJOR M1 against these exact implementation facts. State
whether the finding remains BLOCKER/MAJOR/MINOR or is withdrawn.

1. A quorum journal line is serialized as 16 tab-separated fields. Fields
   11-16 are the ordered principal IDs and signature-or-`-` slots for all three
   configured members. Thus the literal journal bytes contain every per-event
   signer subset and signature.
2. Measurement computes `semantic_journal_digest = sha256 semantic_text` and
   `guardian_journal_digest = sha256 guardian_text`, where each `*_text` is the
   complete literal journal file.
3. The quorum checkpoint is:

   ```text
   SHA256("loom-journal-authority-quorum-checkpoint-v1" NUL
          principal_set_id NUL epoch NUL required NUL
          semantic_journal_digest NUL guardian_journal_digest NUL
          descriptor_digest)
   ```

   Therefore the checkpoint transitively commits every literal certificate and
   signer subset through collision-resistant journal digests. A 2-of-3 policy
   intentionally permits different valid 2-member subsets on different events;
   it does not require one fixed pair for the whole stream.
4. Before these digests are computed, replay verifies every present Ed25519
   signature against the configured ordered principal, context, epoch,
   sequence, previous head, and event hash, and refuses any event below quorum.
5. The strengthened selftest now independently hashes both literal journals,
   compares those hashes with measurement attestation v3, and recomputes the
   checkpoint byte-for-byte. It passes end to end.
6. The prior unconditional nominal helper was replaced. It now checks all four
   decision/measurement digest equalities, three distinct positive journal
   tokens, `journal_quorum_is_satisfied`, positive epoch, and positive
   checkpoint token. The focused typestate gate passes.

The Sounio adapter remains a nominal host-trust boundary and does not itself
reverify Ed25519; that limitation is explicitly documented. Review the narrow
operational claim only: configured quorum mode refuses successor creation unless
the OCaml verifier found at least two distinct configured valid shares on every
event of the exact checkpoint-committed journals.
