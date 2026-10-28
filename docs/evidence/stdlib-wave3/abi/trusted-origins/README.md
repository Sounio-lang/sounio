# Trusted physical module origins — component qualification

The loader records the physical files selected by BFS under their module IDs.
Only files identical by Linux x86-64 libc stat64 device/inode to the configured
stdlib root's ffi/worker.sio or mem/typed_storage.sio gain the respective flag.
An AST module name, a basename, or a copied file confers no authority. All aliases
that identify the real file may be recorded; reset clears every flag.

The helper checks the running compiler's /proc/self/exe ELF class, endianness and
machine before admitting the stat64 layout. Missing procfs, libc.so.6, stat64,
configured files, or a possibly truncated environment root fail closed. The
configured SOUNIO_STDLIB_PATH is an authority supplied by the compiler operator.
With no configured root, the authority is the cwd-relative stdlib directory,
matching the loader's existing fallback. The helper does not attempt to protect
against concurrent edits to source files during compilation.

Qualification: Slurm 18940, completed exit 0. The modern raw c02a0d212761d1954959f53832689d44887da96a048749dbff4a3976c268c3dd
checks and compiles both original component fixtures. Native positives verify real
files, both kinds of links, simultaneous alias IDs, bounds and reset. Native
negatives verify missing authority; copied-file spoofs are refused in the positive
fixture. The C++23 oracle verifies the actual host stat64 offsets, widths and
physical identity. INPUTS, SEED and executable guards all pass. See small receipts
in this directory; full artifacts remain at
/scratch/sounio-stdlib-wave3-20261027/origin-final/result02.

Earlier harness results remain separate: 18910 used a legacy seed and produced
type diagnostics, 18913/18921 omitted the required 512 MiB compiler stack and
crashed; 18923 supplied that stack and passed its smaller native component.
18930 passed both check/compile/native pairs but its C++ step failed because
--export=NIL supplied no PATH. The final portable gate explicitly sets PATH and
ran unmodified in 18940. These failures are not hidden or counted as passes.

The hook in module_frontend_pfi_note_use_targets calls reset once for each fresh
loaded source graph and records the actual file_paths entries. The full rebuilt
compiler integration, capability certification, typed storage and whole-stdlib
acceptance remain separate gates; this component alone enables none of them.
