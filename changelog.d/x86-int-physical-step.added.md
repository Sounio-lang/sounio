`formal/Bootstrap/X86Int.lean` gives the x86-64 instructions the Sporos interpreter is written in. They run over the X86Env process memory and kernel, dispatch through a jump table (`step_jtab`), and have a physical step in the PDist monad. The silicon may fail in four places. The fetch may decode a neighbouring encoding, and the register file, the mapped memory or the flags may suffer a single-bit upset once an instruction retires. Each place has its own rate.
- `faultFree_prun`: for every fault model, the zero-fault branch of the physical run is the architectural run.
- `prun_ok`: the physical run is a distribution.
- `baseline`: for rates anywhere in their calibration intervals, a run of `n` steps faults with probability at most `n` times the sum of the upper ends.
