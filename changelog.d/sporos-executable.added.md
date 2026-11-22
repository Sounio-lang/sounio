`sporos` is the Sporos reference interpreter, runnable as a program: `lake build sporos`, then `sporos PROGRAM.spb < input > output`.
- **Format.** It reads the binary bytecode of `formal/Sporos/Bytecode.lean`: one opcode byte per instruction, with little-endian immediates.
- **Checking.** It refuses a file that does not decode or that the checker rejects, then runs it on standard input.
- **Semantics.** The run uses the DAG semantics with hash-consing, which `drunC_init` proves equal to the reference semantics.
- **Examples.** `sporos-examples DIR` writes four example programs: echo, hello, upper and fib. Each must pass the checker and the bytecode round trip before it is written.
- **Speed.** The interpreter is correct but slow. Time grows roughly quadratically with input length, and the collector is off, because its node walk is exponential in DAG sharing. Both costs are measured in the file header.
