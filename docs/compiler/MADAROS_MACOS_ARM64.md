<!-- docs:meta
topic_id: repo.docs.compiler.madaros-macos-arm64
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.madaros-macos-arm64
-->

# Madaros on macOS arm64

The native host is built from `self-hosted/compiler/main.sio` by a current-source
`lean_single` Mach-O arm64 bootstrap seed. The seed is not substituted for Madaros
when compiling user programs: Madaros parses, checks, lowers to IR and invokes
`native/codegen_arm64_macos.sio` for its native output.

Prerequisites on Apple Silicon: Apple command-line tools (`codesign` and a
C++23-capable `clang++`), a `timeout` command (`brew install coreutils` if absent),
Boost headers for the exact wide oracle (`brew install boost`), and a real Metal
device for full publication. The build refuses a missing `timeout` before
compiling the candidate; it does not silently omit bounded execution checks.

Build and install the local artifact on Apple Silicon:

```sh
bash scripts/ci/build_madaros_macos_arm64.sh /absolute/path/to/current-source-seed
```

The build requires a signed, executable seed. It writes the candidate into a
temporary directory, signs it ad-hoc, executes bounded acceptance, wide-IR and
extra positive/negative regressions, then the full 200-member/real-Metal gate.
Only then does it install `artifacts/self-hosted/madaros-arm64-macos`. Passing an
explicit scratch output path runs bounded acceptance only; a scratch candidate
is not a full-corpus publication. The shared Madaros resolver
and public launchers prefer that artifact on Darwin arm64, after explicit binary
overrides. Linux resolution remains unchanged.

The bootstrap's Darwin entry reserves a separate 512 MiB virtual stack with a
16 KiB inaccessible guard page. This avoids the initial macOS stack's 64 MiB
limit during deep compiler item-list walks. Argument and environment pointers
remain on the original stack. `a64_macos_mapped_stack.sio` is a non-tail-recursive
regression: the old entry crashes at the stack guard; the mapped entry prints
`20000`.

```sh
./bin/souc --version
./bin/souc check tests/run-pass/a64_unit_literal_return.sio
./bin/souc compile tests/run-pass/a64_unit_literal_return.sio -o /tmp/unit-example
./bin/souc run tests/run-pass/a64_unit_literal_return.sio
```

`build --target aarch64-macos` explicitly selects the backend. `native` and
`native-auto` select the detected host; Darwin detection uses `hw.machine` and
does not depend on the shell exporting `HOSTTYPE` or `OSTYPE`.

The native output contains a Mach-O arm64 image, a dyld entry point and libSystem
load command. Runtime I/O and allocation use Darwin syscalls; mathematical calls
resolve through libSystem using dyld chained imports. `run` performs
ad-hoc signing, executes the output and propagates its exit status. `build`
produces an unsigned output, which callers executing it directly must sign and
make executable:

```sh
chmod +x /tmp/unit-example
codesign -s - -f /tmp/unit-example
/tmp/unit-example
```

The acceptance gate executes the compiler's identity and check commands, builds
the struct/reference/call/control-flow fixture, rejects missing bodies or a
non-Mach-O output, executes the artifact, compares stdout, tests the CLI's own
`run` path, and requires propagation of exit status 7 from a second program:

```sh
bash scripts/ci/madaros_macos_arm64_acceptance_gate.sh artifacts/self-hosted/madaros-arm64-macos
```

This gate is a bounded executable acceptance test, not a declaration of full
Linux backend parity or of a Madaros self-compilation fixed point. Unimplemented
IR operations and unresolved calls are refused. The ARM64 path covers scalar
integer/floating arithmetic, ordered comparisons, six-decimal output, branches,
direct/indirect calls, stack arguments and sret, object/array allocation and
indexing, references, strings, global initialization/relocation and file reads
beyond 16 MiB. Escaping closures use tagged, process-lifetime environments.
Nested float-array rank and function-valued floating return metadata survive
lowering. The dedicated wide-IR gate checks ten wide integer operations against
an independent C++23 exact oracle. Target-specific AVX/ZMM operations and arbitrary
external calls are not promised and must not silently fall back to another
engine or architecture.

GPU kernels supported by the target-scoped AST/ABI bridge execute on Metal via
`libsounio_metal.dylib`, built from C++23/Objective-C++. It compiles MSL, submits
GPU work and copies mutable buffers back after completion, with no CPU fallback.
Source f64 retains eight-byte binary64 representation and uses pinned licensed
MSL software arithmetic; this is not native GPU f64 or an f32 substitution.
The launcher exports an absolute default `SOUNIO_METAL_RUNTIME` path. Unsupported
kernel ASTs/parameter shapes, missing devices or GPU failures are hard errors.
See [corpus and Metal evidence](MADAROS_MACOS_CORPUS.md).

Allocation currently has process lifetime. The compiler-owned `__rgn_enter`
protocol returns its documented zero token (no region opened), selecting the
lowering's plain-return path. This is not a region-reuse implementation; a
nonzero-token region close traps rather than silently losing its semantics.

## Initial local verification, 2026-10-15

The current-source native host emitted 12,331 functions and a 69,894,244-byte
Mach-O image. The signed installed artifact SHA-256 is
`f73d21a578b74f1cb01afb9aa38ef5ffdecae1815ed922d0b2cd99bddc5f40f2`.
The native acceptance gate and launcher exit-status gate passed. Public
`bin/souc compile` and `run` generated/executed arm64 output, and `run` propagated
exit status 7. Two consecutive public runs passed after fixing BSD `mktemp`
templates to end in Xs.

The mapped-stack bootstrap raw generations are byte-identical with SHA-256
`03a2ebf5a3ba84563cb4a514ac7542db9de58f5e86fedb2df6b8ab628c19a1a7`.
The non-tail-recursion fixture exited 139 with the old entry and printed 20000
with the mapped entry. Checking all 129 modules of `compiler/main.sio` through
the public wrapper progressed beyond the former stack crash, but hit its
120-second timeout (exit 124); its complete typecheck verdict remains unverified.
That initial artifact passed only 41 of the pinned 200 programs. Subsequent
current-source native candidates and full fresh corpus measurements are recorded
in [MADAROS_MACOS_CORPUS.md](MADAROS_MACOS_CORPUS.md), including separate original
and approved-corrected fixture results. The original-source round 18 reached
199/200; correcting the invalid epistemic fixture yielded 200/200 with the same
compiler, including ten required real Metal launches. A Madaros self-hosting
fixed point, complete repository suite and Linux backend parity remain separate,
unverified claims.

## Installed round-20 host

The installed local artifact is now the round-20 current-source candidate,
SHA-256 `8277c7a27f4c84a273a4a1d84d59ea6692d8575e2f7236da9f845c38205c203f`.
It passed native acceptance, extra positive/negative port regressions, and the
fresh full corrected 200-member corpus with required real Metal receipts.
Public `bin/souc` was then verified for version/check, automatic native targeting,
exit-7 propagation and actual GPU vec_add execution. The original 199/200
receipts remain separate from the approved corrected 200/200 corpus. The
replaced local artifact is retained in the validation bundle described in
[the full evidence report](MADAROS_MACOS_CORPUS.md).
