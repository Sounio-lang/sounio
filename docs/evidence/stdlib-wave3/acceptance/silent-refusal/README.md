# Explicit exact-silent native refusal contract

run-fail now requires exactly one explicit contract: a diagnostic substring or expect-silent-refusal. The silent variant accepts only successful check/compile followed by native exit1 and a raw output file of zero bytes. It examines raw bytes before Bash command substitution, so a newline or NUL cannot disappear into apparent silence. Conflicting or missing contracts refuse before compiler invocation. Existing diagnostic-mode behavior is preserved.

Slurm19368 passes14 new silent controls plus14 diagnostic controls,11 compile-fail controls, diagnostics custody and10 JSON/emitter controls. Newline, NUL, messages, exit0/139/124/122, checker/compiler failure and conflicting/missing contracts are refused. Historical JSON bytes remain unchanged; the reviewed runner projection is repinned.

Only two annotations were added to six original statistics probes. Slurm19379 checks, compiles and executes all six through the canonical runner with sealed raw826d9c18:6pass/0fail/0skip and zero selection/source identity errors. Their numerical bodies are unchanged. This verifies deliberate rejection, not the full statistical concordance gate.

Failed attempts19360 and19363 are preserved: a shell syntax defect prevented execution, the operational syntax guard refused, and the six-case collector recorded zero results plus six errors. After correction, new snapshots19368/19379 qualified; the failed attempt is never counted as native execution.
