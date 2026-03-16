---
name: JIT stdout warning corruption
description: JIT runtime builtins (file_size, etc.) emit warnings to stdout, corrupting SNIO binary protocol stream
type: feedback
---

Never call `file_size()` or `read_file()` on potentially missing files in code that runs under the SNIO IPC server (serve_entry.sio). The JIT runtime prints warnings like `warning: file_size('path') failed: No such file or directory` to stdout, which corrupts the binary protocol stream since stdout IS the SNIO channel.

**Why:** Discovered during Sprint embedding ABI work. The SNIO server uses stdout as its binary protocol channel (magic + msg_type + body). Any text output to stdout between SNIO messages causes parse failures in the host process ("Bad magic: b't of'").

**How to apply:** Source file validation in KERNEL_DESCRIBE must be deferred until either (a) the native-compiled server routes warnings to stderr, or (b) a silent file-exists check is added to the JIT runtime. For now, KERNEL_DESCRIBE registers kernels without validating the source file exists.
