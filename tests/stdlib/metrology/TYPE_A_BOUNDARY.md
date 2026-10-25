# Type-A uncertainty of a repeated-observation mean (SIO-94)

This is the isolated port of the numerical repair proposed in [PR #2567](https://github.com/Sounio-lang/sounio/pull/2567), onto main `be761447e1cb21fd0f9113dccf20c9e2aed8c7a4`. The original PR branch is unchanged. Current-main public `TypeAEval` fields remain public; read-only convenience accessors are additive and re-exported through `metrology::mod`.

For independent repeated observations, `type_a_eval` returns the sample mean, sample standard deviation `s`, and standard uncertainty of the mean `s / sqrt(n)`. This corrects the previous `s / n` calculation. The sampling assumptions come from the caller, and are not assessed by this function. See [JCGM 100:2008, sections 4.2.2–4.2.3, equations (4) and (5)](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).

The array capacity is 16 and the accepted active count is 2 through 16. Counts outside that range are rejected before any read. Only the active prefix is read; inactive nonfinite entries do not invalidate a valid prefix. Constant readings produce zero standard deviation and zero standard uncertainty. The square root reduces a positive finite variance to a bounded range before Newton iterations, rather than applying eight steps to an unscaled variance.

`valid=false` returns zero numeric payloads and `n=0` when the count is invalid, an active input/intermediate is nonfinite, accumulation overflows, or an observed nonzero squared deviation/variance is lost to zero underflow. It describes this implementation's inability to return a supported numerical result; it does not classify the physical measurement as invalid. Underflow is conservatively rejected instead of reporting false certainty. The final uncertainty divides standard deviations so that dividing a subnormal variance by `n` does not first erase a representable result.

The two-pass binary64 algorithm still has summation, cancellation and subnormal rounding limitations. It can reject mathematically representable answers when an intermediate is not representable. There is no arbitrary-input accuracy certificate, fitted statistical model assessment, confidence interval, coverage claim, metrological accreditation, or change to other calibration functions. Their independent numerical implementations remain outside this repair.

## Executed validation

All Sounio checks, compilation, native execution, and C++23 calculations below ran on the remote Slurm native Linux lane with stack limit 524288 KiB. The selected source-built Madaros executable SHA-256 was `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`. This work did not rebuild that compiler or claim validation against every compiler revision.

- Slurm **17064**: the unchanged main numerical library, using a consumer adapted to its existing public fields, checked and compiled, then exited **4** at the uncertainty-of-the-mean assertion. The corrected public-facade consumer checked, compiled and exited **0**. C++ compilation and actual-output concordance both exited **0**. The baseline consumer replaces getter calls with corresponding public-field reads; its numeric assertions are unchanged. It does not pretend that new accessors existed in the old library.
- Slurm **17065**: the exact committed replay script checked, compiled and executed the corrected consumer, found its completion marker, compiled the independent C++23 reference, compared the actual native scientific output and verified input hashes: every phase exited **0**. Mutated output with an incorrect value was rejected with **7**; output missing completion was rejected with **8**.
- Slurm **17063** is retained as a failed report-format attempt: the Sounio assertions passed, but requesting unsupported 17-digit scientific output produced formatter error lines, so C++ comparison rejected it. The corrected fixture uses the existing formatter's supported **15 significant digits**. Neither the numerical tolerance nor the production algorithm was relaxed.

The consumer checks symmetric, constant, small-spread, large-spread and represented-subnormal cases; invalid counts 0/1/17/-1; sum and squared-deviation overflow; nonzero squared-deviation underflow; NaN/Inf; inactive NaN; and direct public-field/getter compatibility. The independent `boost::multiprecision::cpp_dec_float_100` reference compares five actual result triples (15 values) with relative tolerance `1e-12`, requiring exact zero for zero expectations. Small/large input values are promoted from the same binary64 C++ literals used by the Sounio input, rather than interpreted as exact decimal data. The reference's own algebraic and invalid-count controls run before concordance. Finite decimal precision and 15-digit output are not round-trip or full-precision certification.

## Replay and compact custody

On an authorized native Linux worker, from the repository root, with an explicitly selected executable compiler:

```sh
MADAROS_RAW_BIN=/absolute/path/to/fresh/madaros.elf \
  bash tests/stdlib/metrology/type_a_gate.sh /absolute/fresh/output
```

The gate refuses an existing output directory, records compiler/source/ELF identities and per-phase exits, checks the completion marker, compares actual output using C++23, rejects two mutated-output controls, and verifies its source inputs. `scripts/stdlib/deep_science_gate.sh` invokes this focused gate and records failure in its overall result. A future CI execution is separate from the historical Slurm receipts above.

[evidence/receipts.tar.gz](evidence/receipts.tar.gz) contains compact original logs, per-phase results, native stdout/stderr, scheduler records, resource observations, hashes and the exact snapshot protocols for 17063/17064/17065. It contains no ELF, reference executable, full source snapshot or cache. Archive SHA-256: `18868e4d4062f117f9a94601d05ddeff3c1dca764948c914a8ec184e0eed1b01`.

```sh
sha256sum -c tests/stdlib/metrology/evidence/ARCHIVE.sha256
mkdir /tmp/type-a-receipts
tar -xzf tests/stdlib/metrology/evidence/receipts.tar.gz -C /tmp/type-a-receipts
(cd /tmp/type-a-receipts && sha256sum -c MANIFEST.sha256)
```

Native resource files are individual small-program observations, not benchmarks or a general performance claim. An executed focused gate does not establish full-corpus CI, merge, release, or human acceptance.
