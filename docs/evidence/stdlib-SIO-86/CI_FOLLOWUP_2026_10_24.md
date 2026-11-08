<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-86.ci-followup-2026-10-24
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-86.ci-followup-2026-10-24
-->

# SIO-86 CI follow-up: declare the boundary helper mutation effect

PR3095 Full Test Suite at head6a27daf97949455bc8bca631111de485611071d6
failed in `ontology_trust_boundary.sio`, line16: E035, effect not declared.
[Failed CI job](https://github.com/Sounio-lang/sounio/actions/runs/40136901863/job/121001424017).
`boundary_equal` mutates its local counter; its signature omitted Mut. The
retained d74 modular check accepted it, while the CI seed rejected it. This
change adds the required effect; the algebra/kernel and test assertions do not
change.

## Fresh executable evidence

Slurm17103 completed0:0. The frozen packet is
`/scratch/sounio-stdlib-inference-scale-20261024/SIO86/source07` and receipts are
`SIO86/ci07`, `SIO86/result07`, `SIO86/logs/17103.{stdout,stderr}` under that base.
UID1000, stack524288KiB, unchanged shared quota.

The checked-in lean_single seed SHA256 is
`53b64a871490a6daf64fb2e56d62482c590b00d9f0ec94f0d621cefc2b341fd2`.
Using its positional CLI and the same stdlib:

- Before: frozen source04 boundary helper without Mut; compile exits1,
  `error[E035]: effect not declared in function signature at line 16`, no ELF.
- After: source07 helper declares Mut; compile and native execute successfully,
  stdout `TRUST_BOUNDARY_PASS`.
- Full focused modular gate: all16 recorded phases exit0, including generator
  identity, scalar/EL+ concordance, boundary run, numerical/source mutants,
  source and compiler integrity. Final stdout `ONTOLOGY_TRUST_GATE_PASS`, followed
  by launcher marker `CI07_PASS`.

The modular compiler remains the identified retained d74 artifact, not a rebuild
of current main. The checked-in seed is not claimed to be the exact CI-generated
stage2 binary. Remote CI must rerun on the new PR head.

## Preserved unsuccessful attempts

17103 supersedes no old receipt. Job17093/ci05 called the raw seed using an
unsupported subcommand CLI and failed with E221/no main; it is not numerical
regression evidence. Job17097/ci06 reproduced the seed failure and passed the
repaired native boundary fixture, then its modular gate stopped because the
copied packet omitted the private-field compile-fail fixture. Source07 adds that
input and17103 completes the whole gate. Old directories remain unchanged.
