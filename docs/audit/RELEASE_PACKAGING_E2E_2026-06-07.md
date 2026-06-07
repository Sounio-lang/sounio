# `scripts/release.sh` — end-to-end packaging test (2026-06-07)

Scope: build the distributable tarball, install it into a clean temp dir, and
prove whether a program actually runs **from the package** — exercising the
documented user path (README.txt / install.sh "Next steps"). Read-only of the
repo; all output to temp dirs. Branch `claude/release-e2e-eval`, worktree
`/workspace/sounio-rel-eval`, off `011b9c3dca`.

## Verdict

**The package is not a working distributable as documented.** Every documented
entrypoint is broken; only a self-contained (import-free) program compiles, and
only via an **undocumented** invocation. The tarball *is* well-formed and the
shipped binary *can* compile — so the artifact is salvageable — but a user
following the bundled instructions verbatim cannot get past `souc --version`.

Root cause: **there is no launcher.** The docs (`README.txt`, `install.sh`,
`CLAUDE.md` §4) describe a stable `souc <subcommand>` surface — `souc --version`,
`souc info`, `souc check file`, `souc run file` — but `install.sh` (line 150)
ships `bin/souc` *as* that launcher, and the committed `bin/souc` at this tip is
a **raw `mini_native` ELF** (verified: `git cat-file` shows a 2.19 MB ELF, not a
script). No launcher wrapper exists in the checkout that translates the
subcommand surface onto a compiler — `scripts/lib/resolve_souc.sh` is a
*build-time* gate helper (sets `SOUC_BIN`, cargo-skip), not a runtime launcher.

And shipping a "better" compiler would not, by itself, fix it: **neither
available binary implements the documented subcommand CLI.**

| Binary | `souc --version` | `souc check file` (subcommand) | stdlib resolver |
|---|---|---|---|
| `mini_native` (what is shipped) | usage line, exits ≠0 | treats `check` as a filename | CWD-relative `stdlib/` \| `self-hosted/` |
| modular (`souc main.sio`) | ✓ `Madáres v0.80.0 …` | ✗ — uses **flag** form `--check file`, treats `check` as a positional | (flag-style CLI) |

So the documented *subcommand* interface is implemented by no binary in the
repo; the modular compiler is flag-style (`--check`, `--native-compile`). The
gap is a missing launcher that maps `souc <subcommand>` onto the real compiler,
plus docs that promise a surface nothing provides.

## What PASSES

| Check | Result |
|---|---|
| Tarball builds, extracts cleanly | ✓ `sounio-<ver>-x86_64.tar.gz` + `.sha256` |
| Layout correct | ✓ `bin/`, `lib/sounio/stdlib/` (1143 `.sio`), `share/doc/sounio/{INSTALL,KNOWN_LIMITATIONS}.md`, `README.txt` |
| Import-free program compiles + **runs** from the package | ✓ `souc /tmp/h.sio /tmp/h.elf && /tmp/h.elf` → **exit 42** (`21*2`) |

So the package can compile and run trivial programs — but only with the raw
`souc <src> <out>` CLI, which appears nowhere in the docs.

## What FAILS — six bugs

**1. VERSION derivation produces a garbage, newline-containing string (default path).**
`release.sh` derives `VERSION` from
`bin/souc --version | awk 'NR==1{print $NF}' || echo 1.0.0-beta.5`.
`bin/souc` is mini_native: `--version` prints its *usage* line and **exits
non-zero**. So `awk` grabs the last field `x86_64-windows]`, **and** because the
pipeline exits non-zero under `set -o pipefail`, the `|| echo 1.0.0-beta.5`
**also** fires. `VERSION` becomes the two-line string `x86_64-windows]\n1.0.0-beta.5`,
yielding a malformed artefact name:
```
/tmp/rel_out/sounio-x86_64-windows]
1.0.0-beta.5-x86_64.tar.gz          # literal newline in the filename
```
Unusable unless the caller passes `--version` / `SOUNIO_RELEASE_VERSION`.

**2. `release.sh` exits 1 even when it produced the tarball.**
Its final step is the smoke check `"$TMP_VERIFY/.../souc" --version`. mini_native
`--version` exits non-zero, and `set -e` propagates it. Verified:
```
$ bash scripts/release.sh --out /tmp/rel_out2 --version 0.80.0-test ; echo $?
… (tarball + .sha256 written) …
1
```
CI calling `bash scripts/release.sh` sees FAILURE despite a good artefact.

**3. The shipped `souc` is a raw compiler, not the documented launcher.**
`install.sh` and README describe `bin/souc` as a "launcher; PATH-friendly" with
`--version` / `info` / `run` / `check`. In this checkout `bin/souc` *is* the
mini_native ELF (committed as a binary, not a script), so every subcommand is
treated as a *filename*:
```
$ souc --version → Usage: mini_native …
$ souc info      → source: info 0 bytes …
$ souc run h.sio → source: run 0 bytes …
```
This is not fixed by swapping in the modular compiler: it speaks a *flag* CLI
(`--check file`), so `souc check file` still fails there too (see the table
above). The documented subcommand surface needs a launcher; no binary provides
it directly.

**4. The built-in smoke check is hollow.**
`release.sh`'s only verification is "extract + `--version`". That never compiles
anything; it tells you nothing about whether the package works. (And per bug #2
it actually exits non-zero anyway.)

**5. `SOUNIO_STDLIB_PATH` is ignored — the documented stdlib mechanism does nothing.**
README.txt and install.sh "Next steps" both instruct
`export SOUNIO_STDLIB_PATH="…/lib/sounio/stdlib"`. The shipped compiler resolves
imports from a **CWD-relative `stdlib/` or `self-hosted/`** directory and never
consults the env var. Measured from the extracted package (`use collections::vec`):

| Setup | Result |
|---|---|
| `SOUNIO_STDLIB_PATH` set (README way) | ✗ `error: unreadable import: self-hosted/collections/vec.sio` |
| CWD-relative `ln -s lib/sounio/stdlib stdlib` | ✓ ELF emitted |
| CWD-relative `ln -s lib/sounio/stdlib self-hosted` | ✓ ELF emitted |

The package ships stdlib at `lib/sounio/stdlib/`, which matches *neither*
CWD-relative name. **So any program with a stdlib import fails when the package
is used as documented.** Undocumented workaround: `cd` into the package and
`ln -s lib/sounio/stdlib stdlib`.

**6. `install.sh` post-install guidance points at broken commands.**
It honestly prints `WARN  souc --version returned non-zero`, then still lists
`souc --version` and `souc check examples/hello.sio` as "Next steps" — neither
works against the installed binary.

## Reproduction

```bash
# default path — observe the broken VERSION
bash scripts/release.sh --out /tmp/rel_out           # filename has a newline; exit 1

# clean build, then prove what runs
bash scripts/release.sh --out /tmp/rel_out --version 0.80.0-test
tar -xzf /tmp/rel_out/sounio-0.80.0-test-x86_64.tar.gz -C /tmp/pkg
PKG=/tmp/pkg/sounio-0.80.0-test-x86_64
printf 'fn main()->i64{ let x=21\n x*2 }\n' > /tmp/h.sio
"$PKG/bin/souc" /tmp/h.sio /tmp/h.elf && /tmp/h.elf; echo $?      # → 42  (works)
"$PKG/bin/souc" --version                                         # → mini_native usage (broken)
( cd "$PKG" && SOUNIO_STDLIB_PATH="$PKG/lib/sounio/stdlib" \
    ./bin/souc <stdlib-importing>.sio out.elf )                   # → unreadable import (broken)
```

## Recommended fixes (not applied — bounded-scope test only)

Packaging-script level (safe, in `release.sh` / `install.sh`):
1. **VERSION:** stop parsing `souc --version`. Default to a real version
   constant (or require `--version`), and drop the `| awk … || echo` pattern
   that injects a newline. Guard against multi-line/whitespace VERSION before
   naming the artefact.
2. **Smoke check that means something:** compile *and run* a real program from
   the extracted tarball (using the working CLI), and assert the exit code —
   instead of `--version`. Don't let a hollow check gate the release, and don't
   let it `set -e`-fail a good artefact.

Toolchain level (the real blockers — out of packaging-script scope):
3. **Build/restore the launcher** that maps the documented `souc <subcommand>`
   surface onto the real compiler (the modular compiler already provides
   `--version` and `--check` in flag form — a thin wrapper could translate
   `check`→`--check`, etc.). Alternatively, document the actual CLI and stop
   advertising `--version`/`info`/`run`/`check`. Note: simply swapping in the
   modular binary is *not* sufficient — it is flag-style, so the subcommand docs
   would still be wrong.
4. **Honour `SOUNIO_STDLIB_PATH`** in the shipped compiler, or lay the stdlib
   down at the CWD-relative path the compiler actually searches — otherwise the
   1143 shipped stdlib files are unreachable as documented.

## Bottom line

`release.sh` produces a structurally valid tarball whose binary can compile and
run import-free programs (exit 42 confirmed), but **fails the "does the
distributable really work?" test as documented**: broken version string, a
release that self-reports failure, a missing launcher (so the documented
`souc <subcommand>` surface works on no binary in the repo), and a stdlib the
documented env var can't reach. This matches the standing note that the release
apparatus predates the current self-hosted toolchain.

(No source or script changes landed — this is an end-to-end verification.)
