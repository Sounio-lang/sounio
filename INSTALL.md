# Sounio Installation Guide

The default compiler is the committed Madaros binary behind `bin/souc`. Measured 2026-10-15: `./bin/souc --version` prints `Madaros v0.80.0`. That binary is a static Linux x86-64 ELF. It does not run on macOS. On a Mac checkout the launcher materializes `bin/madaros-linux-x86_64` and cannot execute it.

Apple execution is tracked separately in [#2873](https://github.com/Sounio-lang/sounio/pull/2873). That record does not mean this launcher distributes a host-native macOS compiler. See the root [README.md](README.md).

## Check the checkout

```bash
cd /path/to/sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
```

On Linux x86-64:

```bash
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello.elf
"$SOUC_BIN" run examples/hello.sio
```

`run` prints `Hello, Sounio`. The program is `examples/hello.sio`:

```sounio
fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

Force the legacy bootstrap seed only when a page names it:

```bash
SOUNIO_SOUC_ENGINE=lean_single "$SOUC_BIN" run demos/intro/demo_unidades.sio
```

Default Madaros rejects that units demo with `error[E001]` on `let distance: m = 100.0` (#2752).

`check --show-ast` and `check --show-types` are ignored. `compile` and `build` reject both flags with `error: madaros build: unsupported option` (exit 2).

## What this page does not claim

The April 22, 2026 counts in `docs/guide/MINIMUM_VIABLE_SOUNIO.md` were not re-measured here. Do not copy them. GPU emission was not re-run on 2026-10-15. `TOUR.md` §6 (2026-10-04) measured `souc build --backend gpu` as an empty-body PTX skeleton. The root README records scoped Spark evidence for GPU/Pireus and says the generic GPU CLI needs its own status.

Do not start from `cargo build` at the repo root.

## Next

- [docs/guide/getting-started.md](docs/guide/getting-started.md)
- [README.md](README.md)
- [TOUR.md](TOUR.md)
- [docs/compiler/KNOWN_LIMITATIONS.md](docs/compiler/KNOWN_LIMITATIONS.md)
