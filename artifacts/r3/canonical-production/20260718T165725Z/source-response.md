```yaml
mapping_decision_scope: proposal-only
decision_author: "@agourakis82"
catalog_observed_at: "2026-07-18T16:53:04Z"
catalog_identity_sha256: "122ed8713f46286fc8ff9d46a0f44812207d37bf0473b524d87ef38dd6f0bcf8"
repository_catalog_closed: true
catalog_repository_count: 10
canonical_main_head_oid: "32fed91bb01c2269af8edd802c2afaf17509adfa"
authorized_operations: [draft-proposed-not-approved-mapping]
prohibited_operations: [create-repository, materialize-content, remove-source, update-git-ref, approve-canonical-production, execute-canonical-cutover]

targets:
  - {source_path: packages/epistemic-core, action: request-new, repository_id: epistemic-core, repository_url: "https://github.com/Sounio-lang/epistemic-core.git", default_branch: main, visibility: PUBLIC, target_owner: sounio-scientific-packages-maintainers, rationale: "Request the separately governed epistemic-core distribution."}
  - {source_path: packages/sounio-formats, action: request-new, repository_id: sounio-formats, repository_url: "https://github.com/Sounio-lang/sounio-formats.git", default_branch: main, visibility: PUBLIC, target_owner: sounio-scientific-packages-maintainers, rationale: "Request the separately governed formats distribution."}
  - {source_path: packages/sounio-io-primitives, action: request-new, repository_id: sounio-io-primitives, repository_url: "https://github.com/Sounio-lang/sounio-io-primitives.git", default_branch: main, visibility: PUBLIC, target_owner: sounio-scientific-packages-maintainers, rationale: "Request the separately governed IO primitives distribution."}
  - {source_path: packages/sounio-units, action: request-new, repository_id: sounio-units, repository_url: "https://github.com/Sounio-lang/sounio-units.git", default_branch: main, visibility: PUBLIC, target_owner: sounio-scientific-packages-maintainers, rationale: "Request the separately governed units distribution."}
  - {source_path: examples, action: reuse-observed, repository_id: sounio-examples, repository_url: "https://github.com/Sounio-lang/sounio-examples.git", default_branch: main, visibility: null, target_owner: sounio-research-maintainers, rationale: "Explicitly reuse the observed curated-examples repository."}

acknowledgement: >-
  I authorize only the drafting and review of a proposed-not-approved target
  mapping from these exact selections, bound to catalog identity
  122ed8713f46286fc8ff9d46a0f44812207d37bf0473b524d87ef38dd6f0bcf8
  and canonical main head 32fed91bb01c2269af8edd802c2afaf17509adfa. I do
  not authorize repository creation, content materialization, source removal,
  Git ref updates, or canonical cutover execution.
```
