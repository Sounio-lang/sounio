<!-- docs:meta
topic_id: repo.docs.ecosystem.madaros-distribution
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.madaros-distribution
-->

# Madaros distribution contract

The compiler distribution is the first dependency for independently released
Python, editor, Jupyter, library-package, and website consumers. It binds a raw
ELF, the standard library, and canonical launchers to a committed source tree.
It does not rebuild the compiler or infer capabilities from packaged files.

## Build a reviewed candidate

Run with Python 3.11+ and Git:

```sh
python3 scripts/distribution/bundle.py \
  --source <full-source-commit> \
  --binary <validated-linux-x86_64-ELF> \
  --binary-sha256 <reviewed-build-SHA256> \
  --build-receipt <build-evidence-URL> \
  --version <distribution-version> \
  --out <new-archive.tar.gz>
```

All source inputs come from Git objects at the requested commit. Uncommitted
files are excluded. The ELF hash must match before and after copying. Existing
outputs, unsupported source entry types, and non-x86_64 ELF64 inputs refuse.
The archive has normalized ordering, ownership, timestamps, and permissions.
Its reported SHA256 identifies the full archive. The source/ELF association is
an explicit publisher declaration backed by the supplied build receipt; a hash
alone cannot prove which source produced a binary.

The relocatable prefix contains:

- `bin/souc` and `bin/madaros`: adapters that set the bundled stdlib default and
  invoke the unmodified canonical launchers under `runtime/bin`.
- `runtime/bin/madaros-linux-x86_64`: the exact validated ELF, without a fallback
  bootstrap seed.
- `runtime/stdlib`: the standard library from the same commit.
- canonical launcher helpers, science-boundary tools, formatter and LSP tools.
- source license, citation and known-limitations document when present.
- `distribution.json`: source/tree/stdlib identity, ELF hash, build receipt,
  per-file inventory and explicit capability evidence status.
- `verify-distribution.py`: standalone content and executable-mode verifier.

Runtime requirements: Linux x86_64, Bash, GNU coreutils including `timeout`, and
Python 3.11+ for verification and Python helpers. The bundled LSP additionally
requires `jq`. The canonical launcher manages stack reservation. Explicit
compiler or stdlib overrides remain effective and must be removed when testing
the packaged distribution itself.

## External consumer acceptance

Verify the publisher's archive checksum before extracting. After extraction:

```sh
python3 /absolute/prefix/verify-distribution.py --verify /absolute/prefix
export PATH="/absolute/prefix/bin:$PATH"
souc check example.sio
souc compile example.sio -o example
./example
```

Run `tests/distribution/external_consumer.py <extracted-prefix>` on Linux outside
any source checkout. It removes compiler/stdlib environment overrides, uses a
fresh working directory, checks/compiles/runs SAME-NAME, unsplit multi_call,
W1/W2, runs a stdlib import, and checks that invalid source produces no ELF.
Relocate the extracted prefix before this test, including a path with spaces.
Packaging-only tests run with:

```sh
python3 -m unittest discover -s tests/distribution -v
```

The manifest marks capabilities as requiring consumer validation. A separate
receipt records the executed commands and results; shipping an LSP or package
helper is not evidence that its entire protocol works. Full Madaros fixed point,
ARM64, Python backend equivalence and registry/package resolution are not
inferred from this bundle.

## Repository separation sequence

1. Publish the complete compiler distribution and consumer evidence together.
2. Reconcile the existing `Sounio-lang/sounio-py` repository with the in-tree
   implementation and tests; explicitly resolve native and pure-Python backends.
3. Consolidate the two editor implementations and grammar ownership.
4. Validate Jupyter and the existing Sounio packages as external consumers.
5. Export normative docs, artifact receipts and rendered assets before moving
   the website.

Use the existing science-boundary ownership and extraction plan. Retain each
in-tree source until its external consumer, tests and migration repairs pass.
Publish all assets and evidence before making an immutable release public.
