<!-- docs:meta
topic_id: repo.docs.ecosystem.registry-attestation-spec
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.registry-attestation-spec
-->

# Registry Attestation Specification

Status: executable R2.6 local policy contract; public registry publishing is disabled.

## Scope

R2.6 defines a deterministic statement that one verified R2.5 package release
bundle matches one explicit local registry policy. It is a precondition for the
future R3 physical extraction milestone; it does not perform that extraction.

The attestation type is exactly:

```text
unsigned-local-policy-evaluation
```

Its decision is exactly `POLICY_MATCH`. This means that the supplied bundle,
original sources, package policy, claim contract, compiler, and registry policy
passed the named local checks. It does not mean that a package was uploaded,
published, signed, endorsed, independently replayed, or shown to be
scientifically true.

## Inputs

Attestation requires all of the following:

1. a `sounio.package-release-bundle.v1` directory;
2. the original package root and `sounio.toml`;
3. the exact Madaros compiler bound by the release receipt; and
4. a `sounio.registry-attestation-policy.v1` TOML file.

The policy has two tables. `[registry]` names a local catalog namespace and is
fixed to `authority-scope = "local-catalog-index"` plus
`publication-status = "disabled"`. `[acceptance]` declares allowed conclusive
rings, visibilities, claim classes, and the sole v1 assurance level
`identity-only`. Strict mode and verdict `OK` are mandatory.

The checked example is
`docs/ecosystem/registry-attestation-policy.example.toml`. Its parsed structure
is described by
`schemas/sounio.registry-attestation-policy.v1.schema.json`.

## Emit And Verify

The R2.6 tool is deliberately separate from package publication; no publish
subcommand exists.

```bash
python3 tools/science_boundary/registry_attestation.py attest \
  --bundle target/release/example-0.1.0.sio-release \
  --root . \
  --compiler /path/to/current-source-madaros \
  --registry-policy docs/ecosystem/registry-attestation-policy.example.toml \
  --output target/release/example-0.1.0.registry-attestation.json

python3 tools/science_boundary/registry_attestation.py verify \
  --attestation target/release/example-0.1.0.registry-attestation.json \
  --bundle target/release/example-0.1.0.sio-release \
  --root . \
  --compiler /path/to/current-source-madaros \
  --registry-policy docs/ecosystem/registry-attestation-policy.example.toml
```

Emission first performs full R2.5 verification. It then checks the registry
policy against ring, visibility, requested claim class, assurance, boundary
mode, and boundary verdict. The output is written through a sibling staging
file and renamed only after its identity is complete. Existing outputs are not
overwritten.

Verification reconstructs the complete expected attestation from the current
inputs. Recomputing the attestation identity after changing a source root,
package, registry, claim, compiler, policy, or release binding does not
authorize the change.

## Attestation Bindings

`sounio.registry-attestation.v1` binds:

- registry ID, namespace, authority scope, publication status, and policy hash;
- package name, version, and R2.5 bundle identity;
- exact bundle manifest, artifact, boundary receipt, and claim contract hashes;
- boundary receipt identity, claim ID, claim class, source closure, package
  policy, and compiler hashes;
- ring, evidence status, context of use, and visibility; and
- the fixed checks and limitations associated with `POLICY_MATCH`.

No timestamp or absolute path contributes to identity. The JSON structure is
described by `schemas/sounio.registry-attestation.v1.schema.json`.

## Refusal

The tool emits `REGISTRY_ATTESTATION_REFUSED` and no new output when:

- a policy is malformed or attempts to enable publication;
- a ring is non-conclusive or outside the policy;
- visibility, claim class, or assurance is outside the policy;
- the R2.5 bundle or any original verification input no longer matches; or
- an output already exists.

The diagnostic families are `E-SRB-REGISTRY-001` for malformed policy/input,
`E-SRB-REGISTRY-002` for policy or release refusal,
`E-SRB-REGISTRY-003` for invalid attestations, and
`E-SRB-REGISTRY-004` for output promotion conflicts.

## Evidence Boundary

Every v1 attestation carries limitations stating that it does not assert
scientific truth, clinical authority, public registry status, namespace
ownership, issuer identity, remote signature, attested execution, or
independent replay. Full verification still requires the original bundle,
sources, policies, and compiler.

The acceptance command is:

```bash
SOUNIO_REGISTRY_ATTESTATION_MADAROS_BIN=<current-source-elf> \
  bash scripts/ci/registry_attestation_spec_gate.sh
```

This composes the R0-R2 gate, the R2.5 package release gate, and the R2.6
adversarial attestation gate against the same Madaros ELF.
