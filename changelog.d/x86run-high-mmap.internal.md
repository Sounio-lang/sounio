The `Bootstrap x86 model` CI step now also builds the `x86run` executable
(`formal/Bootstrap/RunHex.lean`), so its examples are checked on every run.
`x86run` places mappings above 4 GiB (`X86Env.highAddrs`), with witnesses
that a pointer cut to 32 bits lands outside every mapping.
