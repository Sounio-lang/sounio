# Sounio Installation Guide

The committed Madaros prebuilt behind `bin/souc` is a static Linux x86-64 ELF; it does not run on macOS. Apple Silicon has a separate current-source native ARM64 build and real Metal route integrated in [#2978](https://github.com/Sounio-lang/sounio/pull/2978). Follow the [macOS ARM64 guide](docs/compiler/MADAROS_MACOS_ARM64.md) to build and install the native artifact before using the launcher on that platform.

The prebuilt binary and native source-built route are distinct. See the root [README.md](README.md) for the current repository map and status.

On Linux x86-64 the commands below are the whole first-contact path. The website guide [Getting Started](website/src/content/docs/en/getting-started.mdx) repeats them.

## Prerequisites

Linux x86-64 needs:

- `git`
- `bash`
- `gzip` — the first `./bin/souc` decompresses the committed prebuilt
- `sha256sum` from GNU coreutils
- about 2 GB of free disk

Measured on 27 October 2026, Linux x86-64, `main` at `75fd75c587a799169d59954cd50f2f327eed5f4e`: `git clone --depth 1` occupied 1.1 GB. The first `./bin/souc` then wrote `bin/madaros-linux-x86_64` (129,918,841 bytes). The tree was 1.3 GB after that. A clone without `--depth 1` also downloads history and is larger.

## Linux x86-64

```sh
git clone --depth 1 https://github.com/Sounio-lang/sounio.git
cd sounio
git rev-parse HEAD
uname -sm
```

`--depth 1` fetches the current commit and not the history. Keep the revision that `git rev-parse HEAD` prints.

From the repository root:

```sh
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
./bin/souc info
./bin/souc --version
```

The committed compiler is stored as `bin/madaros-linux-x86_64.gz`. The first `./bin/souc` decompresses it to `bin/madaros-linux-x86_64` and prints one line on stderr with a sha256 prefix and the byte count. Later runs reuse that file.

`bin/souc` is a launcher, not the compiler binary itself. Inspect the `raw_elf` path reported by `info`. `./bin/souc --version` prints provenance lines on stderr and the version on stdout. Keep that identity with your results. Do not switch engines to work around a platform or setup failure.

The checksum file names `madaros-linux-x86_64` with no directory. Run it from `bin/`, after the first `./bin/souc` invocation. Before that invocation the ELF is not on disk, and the check cannot open it. From the repository root, `sha256sum -c bin/madaros-linux-x86_64.sha256` looks for the ELF in the current directory and fails.

```sh
(cd bin && sha256sum -c madaros-linux-x86_64.sha256)
```

A match prints `madaros-linux-x86_64: OK`.

```sh
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

Run the second command only if the check succeeds. `souc --help` does not list `--quiet` or `--silent`. `run` prints compiler log lines and then the program's own output. `run` prints `Hello, Sounio`, with exit status zero. Those log lines are the compiler, not the program.

`./bin/souc compile examples/hello.sio -o hello.elf` writes an ELF and does not print `Hello, Sounio`. You do not need that step to see the message.

The program is `examples/hello.sio`:

```sounio
fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

Force the legacy bootstrap seed only when a page names it:

```bash
SOUNIO_SOUC_ENGINE=lean_single "$PWD/bin/souc" run demos/intro/demo_unidades.sio
```

Default Madaros rejects that units demo with `error[E001]` on `let distance: m = 100.0` (#2752).

`check --show-ast` and `check --show-types` are ignored. `compile` and `build` reject both flags with `error: madaros build: unsupported option` (exit 2).

## Other hosts

| Host | Setup path |
| --- | --- |
| macOS ARM64 (Apple Silicon) | Build and install the native artifact first, following the [macOS ARM64 guide](docs/compiler/MADAROS_MACOS_ARM64.md). The Linux ELF does not run natively on macOS. |
| Other hosts | This page does not provide a compatible prebuilt. |

The macOS source build requires a signed, executable current-source bootstrap seed, Apple command-line tools with C++23-capable `clang++` and `codesign`, `timeout`, and Boost headers. The full installation gate also requires a real Metal device.

Do not start from `cargo build` at the repo root.

## What this page does not claim

The April 22, 2026 counts in `docs/guide/MINIMUM_VIABLE_SOUNIO.md` were not re-measured here. Do not copy them. GPU emission was not re-run on 2026-10-15. `TOUR.md` §6 (2026-10-04) measured `souc build --backend gpu` as an empty-body PTX skeleton. The root README records scoped Spark evidence for GPU/Pireus and says the generic GPU CLI needs its own status.

## Next

- [docs/guide/getting-started.md](docs/guide/getting-started.md)
- [README.md](README.md)
- [TOUR.md](TOUR.md)
- [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)
