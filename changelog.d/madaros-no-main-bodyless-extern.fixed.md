Madaros no longer treats a bodyless `extern "C" fn main()` as an entry point.
An importing file whose only `main` is such a declaration is now refused with
`error[E221]: no main`; before, Madaros wrote an ELF whose entry was the empty
stub and which died with SIGILL (rc=132).
