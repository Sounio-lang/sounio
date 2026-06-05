# Source → native ELF via the elected modular compiler

First working programs compiled from `.sio` SOURCE to a runnable native x86-64 ELF
by the modular self-hosted compiler (`--native-v2-compile`), with CORRECT return
values used as the process exit code.

    ./bin/souc self-hosted/compiler/main.sio /tmp/souc-modular.elf   # build modular compiler
    /tmp/souc-modular.elf --native-v2-compile examples/native_v2_source/fib.sio -o /tmp/fib
    /tmp/fib; echo $?     # => 55  (fib(10))

Verified working subset (exit code = program return value):
- Integer arithmetic: + - * / %, with operator precedence
- if/else, comparisons (< <= > >= == !=)
- Multi-function programs, function calls with parameters
- Recursion (fib)
- while loops with mutable `var` bindings, let bindings, casts

Milestone branch: feat/native-v2-source-bridge. See
docs/audit/g1_wip/BYVALUE_MODULE_COPY_MAP_2026-06-04.md for the conversion map.
