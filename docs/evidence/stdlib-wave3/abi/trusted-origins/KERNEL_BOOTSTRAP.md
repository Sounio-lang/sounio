# Trusted origins: kernel identity and dual bootstrap

This supersedes the libc-based helper from 6b3d9dbc21. The public registry API and
physical device/inode policy are unchanged. The serial loader captures the two
authority identities once per fresh graph; each loaded path is compared with
those identities. All state is scalar or a bounded module-flag array.

The helper validates /proc/self/exe as ELF64, little endian, machine x86-64,
before issuing Linux newfstatat (262, AT_FDCWD, flags0). Other process ABIs and
missing/invalid authority files fail closed. Full C strings use named pointer
locals for every store. Neither an AST namespace nor copied bytes grants trust.
Links to the actual authority file are accepted, and aliases coexist.

Slurm19018 COMPLETED0 passed the exact portable gate:
- Modern raw f3ceba1af545a006ab61c66dafde4bbc5eb63c0b1463f63d6ed2f3dd853cb332 checks,
  compiles and executes both fixtures.
- Legacy seed9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a
  compiles and executes the SAME positive/negative fixtures, with no explicit
  error or dropped-store diagnostics. Positive tests exercise real identities,
  copied spoofs, hard/symbolic links, alias IDs, bounds and reset; negative
  execution uses a missing authority root.
- The C++23 oracle checks stat64 offsets/widths and compares the actual kernel
  newfstatat buffer's device/inode words with libc stat64 for both arguments,
  including the link. Kernel syscall number262 is also a compile-time assertion.
- Source, compiler and executable guards pass. CXX_POSTRUN.sha256 is explicitly
  post-run metadata for the installed C++ tools and resulting oracle executable.

Full artifacts: /scratch/sounio-stdlib-wave3-20261027/origin-kernel/result.
Small receipts are under kernel-bootstrap/.

Preserved failures explain this change: 18998 still reported legacy E001 after
the named-store refactor. 19007 compiled the word-shaped dynamic-loader boundary
without explicit errors, but native positive exited1. 19013's diagnostic showed
valid host header and C-string bytes, then exit1 at dlopen itself. Thus a modern
component pass did not establish bootstrap compatibility. 18991, which had begun
building with the superseded helper, was canceled. The dynamic-loader route is
no longer used by this registry. No failure was counted as a pass.

This gate does not establish source-context/cache admission, final Worker entry
certificates, or the whole rebuilt compiler. Those remain independent gates.
