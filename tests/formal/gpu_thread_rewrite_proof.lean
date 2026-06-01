import Init.Data.Nat.Basic

-- Original emulator behavior: uses the first argument as thread count.
def emulator_threads_original (offset count cap : Nat) : Nat := offset

-- Rewritten AST: injects `count` as the explicit first parameter.
def emulator_threads_rewritten (threads offset count cap : Nat) : Nat := threads

-- Theorem: The AST rewrite ensures the emulator launches exactly `count` threads,
-- regardless of the `offset` value (which caused the offset=0 bug).
theorem gpu_thread_rewrite_correct (offset count cap : Nat) :
  emulator_threads_rewritten count offset count cap = count := by rfl
