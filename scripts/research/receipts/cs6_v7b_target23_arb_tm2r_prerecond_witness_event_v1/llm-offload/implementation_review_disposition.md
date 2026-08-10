# Hostile implementation review disposition

Provider: xAI Grok 4.3, task `review`.

- BLOCKER missing-receipt read: accepted. Both frozen receipts are now checked
  with explicit `SystemExit` messages before either is read or hashed.
- BLOCKER module-global reconditioner: rejected for this single-threaded,
  one-shot Slurm worker. The imported TM2R implementation exposes
  `base.recondition` as its established policy hook. The diagnostic explicitly
  reinstalls the lineage-preserving function after the capture helper restores
  production policy, and a mandatory control verifies its identity before the
  event replay. There is no concurrent caller in the job process.
- MAJOR LIFO work list: rejected. LIFO `pending.pop()` plus paired half-step
  pushes is the exact frozen production time-refinement schedule being replayed
  and is deterministic for fixed inputs.
- MAJOR broad exception catch: rejected. Only mathematical `base.Refusal`
  outcomes may become diagnostic refusals. An unexpected Python or Arb error
  must abort the runner rather than be laundered into evidence.
- MINOR initial step: rejected. The extended budget recursively halves the
  same frozen `1/256` step from depth 10 through depth 18; changing the initial
  step would prevent an exact replay of the production boundary.
- NIT duplicated failure strings: retained because the imported modules expose
  no canonical failure-class constants. Exact strings are also frozen by the
  prior receipt and independent verifier.
