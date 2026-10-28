# Stiff ODE example: type repair and independent contract counterexample

The active example has four narrow type/API repairs: explicitly typed arithmetic closures, a single reference (not &&PBox) for the identity RHS, and the declared Prob effect required by monte_carlo. Equations, parameters, sample count5000, steps80, tolerances and assertions are unchanged in this commit. This does not qualify the complete example.

Slurm19532 ran the unchanged vdp_final_y helper against an independent C++23 Euler reference (-ffp-contract=off): five cases, exact f64 bit agreement in both loop-region modes. Wrong bits, missing rows and a duplicate stream were refused8/9/9. The full example then checked0, but compilation refused1: its callback captures values that the fn-pointer interface cannot supply (#1542), plus three unresolved scalar print arguments. No full native result exists.

Initial19512 had a shell-function syntax error before tests.19523 compiled the helper source but refused five inline f64_to_bits print arguments; the producer was corrected to bind explicit i64 words. No mathematical body, input case or expected bit was changed to obtain19532.

C++23 counterexample19546 independently implements the actual PBox lift and mean-bound arithmetic with the original80 steps: input mean bounds5,5 with variance0.25 produce final mean bounds0.13673808454714045,0.13673808454714045 and width0. The GUM width is0.017602697585605842. Thus the original width>GUM assertion is false under this API. pb_from_knowledge explicitly creates no Knightian gap; variance is a separate field. Existing test_knightian_basic/test_knightian_stdlib already test that zero-gap contract. No ambiguity interval has been invented from a standard deviation.

The original example is preserved verbatim in fixtures. Selection of the corrected scientific contract is pending user preference; the active run-pass remains failing and is not reclassified as a negative or skipped. The private numerical helper result is not a proof of enclosure, a Monte Carlo-distribution qualification or a fix for captured callbacks.

Counterexample19542 failed to locate cc1plus under the unset-path Slurm environment.19546 uses /usr/bin/g++ and explicit PATH; source bytes are identical.
