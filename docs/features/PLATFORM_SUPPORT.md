<!-- docs:meta
topic_id: repo.docs.features.platform-support
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.features.platform-support
-->

# Platform Support

This document describes the platforms supported by the Sounio compiler and runtime, along with known limitations and platform-specific considerations.

## Supported Platforms

### Tier 1: Full Support

These platforms are tested in CI and guaranteed to work:

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Linux    | x86_64      | ✅     | Primary development platform |
| Linux    | aarch64     | ✅     | Full support, tested in CI |
| macOS    | aarch64 (M1+) | ✅   | Native ARM support |
| macOS    | x86_64      | ✅     | Intel Macs |

### Tier 2: Best Effort

These platforms should work but are not regularly tested:

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Windows  | x86_64      | ⚠️     | Requires MinGW/MSVC, limited testing |
| FreeBSD  | x86_64      | ⚠️     | Should work, not tested in CI |
| OpenBSD  | x86_64      | ⚠️     | Should work, not tested in CI |

### Unsupported

- 32-bit architectures (i686, armv7)
- Big-endian systems (except via QEMU testing)
- WebAssembly (planned future support)

## Build Requirements

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install build-essential gcc make

# Fedora/RHEL
sudo dnf install gcc make

# Build
cd bootstrap/poseidon
make
cd ../..
cargo build --release
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Build
cd bootstrap/poseidon
make
cd ../..
cargo build --release
```

### Windows

**Option 1: MinGW-w64**

```powershell
# Install via MSYS2
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make

# Build
cd bootstrap\poseidon
mingw32-make
cd ..\..
cargo build --release
```

**Option 2: MSVC**

```powershell
# Install Visual Studio Build Tools
# Then build with MSVC
cd bootstrap\poseidon
nmake /f Makefile.msvc  # (requires creation of Windows Makefile)
cd ..\..
cargo build --release
```

## Binary Format: SOIR v1

The Sounio Object IR (SOIR) binary format is **always little-endian**, regardless of host platform. The Poseidon VM automatically converts between little-endian and host byte order when loading bytecode.

### Endianness Handling

- **Little-endian hosts** (x86_64, aarch64): No conversion needed
- **Big-endian hosts** (POWER, SPARC): Automatic byte swapping via `platform.h`
- **Mixed-endian hosts**: Not supported

The platform abstraction layer in `bootstrap/poseidon/platform.h` provides:
- `platform_le64_to_host()` - Convert i64 from LE to host
- `platform_host_to_le64()` - Convert i64 from host to LE
- Similar functions for 32-bit and 16-bit values

## Native Backend

The native backend (`--backend=native`) produces ELF executables on Linux/BSD and Mach-O executables on macOS.

### Linux (ELF)

- **ABI**: System V AMD64 ABI
- **Entry point**: `_start`
- **Syscalls**: Direct via `syscall` instruction
- **Linking**: Static by default, no libc dependency
- **Tested on**: Ubuntu 24.04, Fedora 40, Arch Linux

### macOS (Mach-O)

- **ABI**: macOS x86_64/ARM64 ABI
- **Entry point**: `_main` (with leading underscore)
- **Syscalls**: Via macOS syscall convention
- **Linking**: Static linking not supported, uses libSystem.dylib
- **Tested on**: macOS 14 (Sonoma), macOS 15 (Sequoia)

### Windows (PE/COFF)

- **Status**: Planned, not yet implemented
- **Blocker**: Need PE/COFF linker support
- **Workaround**: Use Cranelift JIT or VM backend on Windows

## Poseidon VM

The Poseidon VM is a portable C-based bytecode interpreter for SOIR v1.

### Portability Features

- **Pure C99**: No platform-specific assembly (except for runtime syscalls)
- **Platform abstraction**: All OS-specific code isolated in `platform.h`
- **Endianness-safe**: Automatic conversion from little-endian SOIR format
- **File I/O**: Abstracted for POSIX vs Windows
- **Path handling**: Platform-specific separators (`/` vs `\`)

### Performance

Approximate execution speed relative to native code:

| Platform | Speed | Notes |
|----------|-------|-------|
| Linux x86_64 | ~5-10x slower | Baseline |
| Linux aarch64 | ~5-10x slower | Similar to x86_64 |
| macOS M1 | ~5-10x slower | Rosetta 2 not needed |
| Windows | ~10-15x slower | Some overhead from Windows I/O |

## Path Handling

Sounio uses Rust's `std::path` for all path operations, which provides cross-platform abstractions:

- **Path separators**: `/` on Unix, `\` on Windows
- **Drive letters**: Supported on Windows (`C:\path\to\file`)
- **UNC paths**: Supported on Windows (`\\server\share\file`)
- **Unicode paths**: Full UTF-8 support on all platforms
- **Spaces in paths**: Properly handled via shell quoting

### Examples

```bash
# Linux/macOS
souc build examples/hello.sio -o /tmp/hello

# Windows (both styles work)
souc build examples/hello.sio -o C:\temp\hello.exe
souc build examples/hello.sio -o C:/temp/hello.exe
```

## Known Limitations

### Per-Platform

**Linux**:
- ✅ No major limitations

**macOS**:
- ⚠️ Static linking not supported (requires libSystem.dylib)
- ⚠️ Code signing may be required on ARM Macs for JIT
- ⚠️ Notarization required for distribution

**Windows**:
- ❌ Native backend not yet implemented
- ⚠️ Poseidon VM requires MinGW or MSVC
- ⚠️ Some test scripts use Bash (requires WSL or Git Bash)

### Architecture-Specific

**x86_64**:
- ✅ Full support for all backends
- ✅ AVX-512 SIMD support (with `avx512` feature flag)

**aarch64**:
- ✅ Full support for all backends
- ✅ NEON SIMD support
- ⚠️ Some assembly optimizations pending

**Other architectures**:
- ❌ Not supported (use Poseidon VM or Cranelift)

## Performance Tuning

### Compilation Flags

```bash
# Maximum optimization
cargo build --release

# Platform-specific optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Link-time optimization
RUSTFLAGS="-C lto=fat" cargo build --release

# Profile-guided optimization (Linux only)
cargo pgo build -- --release
```

### Backend Selection

For best performance on each platform:

| Platform | Best Backend | Fallback |
|----------|-------------|----------|
| Linux x86_64 | Native (`--backend=native`) | Cranelift JIT |
| Linux aarch64 | Native (`--backend=native`) | Cranelift JIT |
| macOS x86_64 | Cranelift JIT | Poseidon VM |
| macOS ARM64 | Cranelift JIT | Poseidon VM |
| Windows | Cranelift JIT | Poseidon VM |

### Memory Usage

Typical memory usage for compilation:

| Operation | Memory | Notes |
|-----------|--------|-------|
| Parse small file (<1000 LOC) | ~10 MB | |
| Parse large file (10K LOC) | ~50 MB | |
| Typecheck stdlib | ~100 MB | One-time cost |
| Native codegen | ~20 MB per module | |
| Full build (self-hosted) | ~500 MB peak | Parallel compilation |

## Troubleshooting

### "Failed to allocate module" on Poseidon VM

- **Cause**: Out of memory or corrupt SOIR file
- **Fix**: Check file integrity, increase system memory

### "Permission denied" when running executables

**Linux/macOS**:
```bash
chmod +x ./myprogram
./myprogram
```

**Windows**:
- Check Windows Defender / antivirus
- Run as Administrator if needed

### "Cannot find -lc" on macOS

- **Cause**: Trying to statically link (not supported on macOS)
- **Fix**: Use dynamic linking (default) or use Poseidon VM

### Slow compilation on Windows

- **Cause**: Windows Defender scanning each file
- **Fix**: Add Sounio workspace to Defender exclusions

### Big-endian systems

While SOIR format is little-endian, the Poseidon VM will automatically convert byte order. To test on big-endian systems:

```bash
# Install QEMU
sudo apt-get install qemu-user

# Cross-compile for big-endian
cargo build --target powerpc64-unknown-linux-gnu

# Run via QEMU
qemu-ppc64 target/powerpc64-unknown-linux-gnu/debug/souc --version
```

## Reporting Platform Issues

When reporting platform-specific bugs, please include:

1. **Platform details**:
   - OS and version (`uname -a` on Unix, `systeminfo` on Windows)
   - Architecture (`uname -m` or `arch`)
   - Compiler version (`gcc --version` or `rustc --version`)

2. **Reproduction steps**:
   - Minimal test case
   - Exact commands run
   - Expected vs actual behavior

3. **Environment**:
   - `RUST_BACKTRACE=1` output
   - `SOUNIO_DEBUG=1` output (if relevant)
   - Any custom build flags

File issues at: https://github.com/demetrios-chiuratto/sounio/issues

## Future Platform Support

Planned platform additions:

- **WebAssembly**: VM backend targeting WASI
- **Windows native backend**: PE/COFF executable generation
- **RISC-V**: Native backend for RISC-V 64-bit
- **Mobile**: Android and iOS support via VM backend

## See Also

- [docs/compiler/ARCHITECTURE.md](../compiler/ARCHITECTURE.md) - Compiler architecture
- [docs/compiler/CODE_GENERATION_ARCHITECTURE.md](../compiler/CODE_GENERATION_ARCHITECTURE.md) - Backend implementation details
- [bootstrap/poseidon/README.md](../../bootstrap/poseidon/README.md) - Poseidon VM documentation
