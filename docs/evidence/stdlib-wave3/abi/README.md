# Generic ownership and allocation-failure controls

The C++23 stdlib_heap_vec_alloc_fault.cpp harness launches and traces only its
own child. It never attaches to an existing process or exhausts machine memory.
After exactly one HEAP_VEC_INJECT_ARM marker, it suppresses the next eligible
4096-byte anonymous private read/write mmap and supplies ENOMEM at syscall exit.
The trace must observe one exec, one arm, one injection and one completed done
marker, and the child must exit successfully. Unexpected signals and protocol
violations fail. The syscall entry/exit distinction uses the documented
[PTRACE_GET_SYSCALL_INFO interface](https://man7.org/linux/man-pages/man2/ptrace.2.html),
with PTRACE_O_TRACESYSGOOD. Child-parent lifetime guards prevent an abandoned
traced process.

Slurm18518 compiled the harness with C++23 and -Wall -Wextra -Werror.
Its C++ child verified the injected failure, preservation of an existing mapped
page, successful retry, unchanged page contents after retry and explicit unmaps.
The positive selftest exited0; no-arm, wrong-size and bad-child-exit controls
each exited1 with explicit harness refusals. Source and executable hashes
verified. The prior18511 build failure (an uncast Linux request macro conflicting
with the C++ ptrace enum parameter) remains preserved; no warning was disabled.

This qualifies the deterministic fault-injection mechanism. It is not a
HeapVec implementation, generic-layout proof or ownership acceptance receipt.
The actual Sounio integration must additionally prove that failed growth returns
Err((3,the_same_input)), leaves the vector's length/capacity/old elements intact,
and permits explicit retry, typed removals and one close per resource.

Generic Option/Result payload cells, target layout hooks, typed moves and
owner-anchored borrow provenance remain compiler prerequisites. A missing API
or unrelated checker error cannot count as a successful ownership-negative test.
Every ownership refusal requires a passing positive counterpart and the actual
borrow/move diagnostic. The trusted raw typed-move boundary does not establish
arbitrary raw-pointer memory safety or initialization validity.
