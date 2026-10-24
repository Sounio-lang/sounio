# Checked dynamic radix-2 Fourier transforms

`math::fft_checked` provides a public complex and real-input DFT using caller-owned storage and an explicitly owned reusable plan. It has no fixed 4096-sample ceiling: a supported length is a positive power of two that fits the checked integer bounds and caller payload budget. This does not promise feasibility for every representable size.

The forward transform has exponent sign -1; the inverse has sign +1. The result is in frequency-index order. Normalization is an explicit argument on `fft_complex_into` and `fft_real_into`: **0** unnormalized, **1** divide by N, **2** divide by sqrt(N). `fft_forward_into` selects forward/0 and `fft_inverse_into` selects inverse/1, so their composition recovers the input up to floating-point error. These are the usual [DFT sign and unnormalized inverse conventions](https://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html); this module does not call or depend on FFTW.

## Ownership and storage

`fft_plan_new(n, max_bytes)` creates an opaque linear `FFTPlan`. The caller owns the returned plan and must call `fft_plan_close(plan)` on success or failure. Accessors are `fft_plan_status`, `fft_plan_len`, and `fft_plan_payload_bytes`. No permutation, twiddle pointer or mutable cache view is public.

Construction validates the radix and integer products before allocating. The cache contains N i64 bit-reversal indices and N f64 twiddle scalars: **16N requested payload bytes** for N >= 2; N=1 needs only **8 bytes**. Two payload allocations are made for N>=2, one for N=1. The budget describes requested cache payload, not allocator bookkeeping, managed-object headers, caller buffers, stack or process RSS. An allocation failure may retain the first allocation in an unavailable owner; `payload_bytes` reports that retained payload, and `close` releases it. Plan status 0 means available, 1 invalid size/radix or negative budget, 7 insufficient payload budget, 8 allocation failure, and 5 unavailable cached arithmetic.

Transforms borrow the plan, read shared `NumericBuffer` input(s), and exclusively borrow a scratch `NumericBuffer` of length at least 2N and distinct real/imaginary output buffers. The complex output lengths must be >= N. The real-input output lengths must be >= floor(N/2)+1, including both DC and Nyquist when N is even. **N=4096 real input therefore needs 2049 complex bins**. Extra input/output suffixes are ignored or preserved. Inputs and outputs are separate owners; this API does not introduce an aliased in-place mode.

The caller creates and closes these buffers using `collections::numeric_buffer`. No output or scratch resizing is implicit. Once a plan and buffers exist, transforms perform no heap allocation and evaluate no trigonometric functions. The bit-reversal and twiddle data are cached by construction; each call performs prefix validation and radix-2 butterflies with scalar buffer access.

## Result and failure contract

Each `_into` function returns an i64 status, without constructing a rich per-transform result:

| Status | Meaning |
|---|---|
| 0 | Output committed |
| 2 | Unavailable buffer or insufficient active length |
| 3 | Nonfinite active input |
| 4 | Unavailable plan |
| 5 | Nonfinite intermediate or normalized output |
| 6 | Normalization outside 0..2 |

Precedence is unavailable plan, normalization, buffer status/length, active inputs, then arithmetic. All count, capacity, mode and finite-input validation finishes before any scratch or public output write. Nonfinite inactive suffixes are ignored. Arithmetic failure may change scratch, but leaves both public output buffers unchanged. Successful output is committed only after all requested final values pass finite checks. Public outputs and scratch must remain disjoint owners as enforced by the borrow contract.

This is finite binary64 arithmetic. Underflow and rounding follow the underlying arithmetic; availability is not a bound on relative error, preservation of every tiny contribution, or proof of exact cancellation. A normalized transform can be rejected when an unscaled intermediate overflows even if a differently scaled algorithm could represent the final result. No automatic frequency units, sampling-rate metadata, windowing, density normalization or statistical uncertainty is invented.

## Legacy fixed-array safety

`math/fft.sio` retains its existing private fixed-array implementation and valid-call conventions. It preflights size, radix and finite active inputs before mutation. Its private real-input helper has only 2048 output elements, so it rejects N=4096 before allocating work or writing output. It does not silently drop Nyquist or truncate the spectrum; use the public dynamic facade for 4096 real samples. The legacy void signatures and declared Panic effect remain unchanged: an invalid request triggers an assertion before any supplied output is changed. A native assertion exits with status 1 on the verified compiler and emits no diagnostic text; the regression checks this explicitly, rather than pretending a quiet return is a successful transform. `next_power_of_2` returns 0 when the next positive power cannot fit i32, rather than overflowing its loop.

## Reproducible validation

The focused replay is `scripts/stdlib/math_fft_checked_gate.sh`. It requires an explicit executable `MADAROS_RAW_BIN` and a fresh absolute output directory. It checks and compiles the public caller and the legacy same-module safety witness; runs their native executables; checks completion markers; builds a C++23 independent reference; and compares actual native output. Mutated-output controls exercise numerical disagreement and missing completion. Its source-input and executable identities are recorded.

The optional second argument is a frozen pre-repair `fft.sio`. It adds the actual failing-before legacy witness and three ordered pairs of repeated-4096 process observations in AB/BA/AB order (A = legacy on-the-fly twiddles, B = cached checked facade). Each benchmark process performs 32 forward/inverse pairs on the same input. Cached-plan construction is included in the new process measurement; neither compile time nor timing ratios are represented as kernel-only measurements. The two algorithms have different storage/validation policies. Three observations do not establish a general speed or memory bound.

```sh
MADAROS_RAW_BIN=/absolute/selected/madaros.elf \
  bash scripts/stdlib/math_fft_checked_gate.sh /absolute/fresh/results \
  /absolute/frozen/pre-repair-fft.sio
```

C++23 computes direct long-double DFTs for N=1 and N=8 and an independently derived periodic-16-plus-impulse DFT for N=4096. It verifies full coefficient coverage, 2049 real bins, DC/Nyquist, Parseval, round-trip error, explicit unitary scaling, and small circular self-convolution. The comparator uses `2e-10*(1+abs(expected))`; the native round-trip check uses maximum component error <=1e-10. Native scalar output uses 15 significant digits. These bounded controls do not certify every input or full binary64 precision.

## Executed evidence (SIO-163)

Source baseline: main `a2a44df195bb8caf3fb07cbc9517275b29325e94`. Slurm **17115** executed the frozen final gate with the explicitly reused source-built Madaros SHA-256 `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`. This is not a current-main compiler rebuild. Library/public-caller check, compilation, actual execution, C++23 build, 6184-row actual-output concordance, valid legacy calls, six paired process executions and their numerical checks, and input integrity all passed; final gate STATUS was **0**.

The legacy capacity negative is deliberately nonzero: corrected source aborts **1** after its request marker, before its continuation marker; frozen old source continues and exits **42**. This is the failing-before/passing-after rejection contract, not an unexpected functional failure hidden by the gate. The valid legacy witness independently passes impulse FFT/IFFT, convolution, power spectrum, 2048-real unique bins with suffix preservation, and N=1. Oracle copies with a corrupted value and missing completion were rejected with **11** and **13**.

The public caller checks N=1,8,4096, invalid zero/non-radix/overflowing size and budget, exactly insufficient 4096-real output, invalid normalization, inactive NaN, active late NaN/Inf, arithmetic overflow, untouched public suffixes and transactional output/scratch behavior. Failure predicates use assertions so the caller closes its linear owners on all normal paths. Slurm **17116** separately checked the complete caller/import closure through the shipped lean_single wrapper, with compiler SHA-256 `53b64a871490a6daf64fb2e56d62482c590b00d9f0ec94f0d621cefc2b341fd2`, and exited **0**. That compatibility check compiled a disposable ELF; it did not execute that ELF or replace canonical Madaros execution.

Each row below measures a complete process containing 32 complex forward/inverse pairs on N=4096 with identical data, including plan construction for B. GNU time reports seconds and maximum RSS in KiB. These are three descriptive observations, with no population/performance inference and no claim of a general RSS reduction.

| Pair and order | A elapsed / RSS | B elapsed / RSS |
|---|---|---|
| 1: A then B | 2.44 s / 1076 KiB | 0.79 s / 1636 KiB |
| 2: B then A | 2.43 s / 1628 KiB | 0.78 s / 1584 KiB |
| 3: A then B | 2.43 s / 1084 KiB | 0.78 s / 1080 KiB |

[Compact native receipts](../../tests/stdlib/fft/evidence/receipts.tar.gz) preserve final results, exact input/ELF/compiler hashes, actual stdout, raw resources, ordering, controls and historical failed attempts. No ELF/full source tree/cache is embedded. Byte-identical receipt files are deduplicated with tar hard links. The bounded frozen baseline module is included as text for replay. Archive SHA-256: `a89097682fd7b7d519645c061a5519dfc7f39bbad229a88507b5b403e09cb733` (530435 bytes).

`SCOPE.txt` distinguishes initial constructor linearity errors, the raw seed CLI invocation mistake, rejected fixture early-return ownership, and the historical gate that incorrectly demanded assertion diagnostic text. Old AB/AB/AB measurements are historical only. Final acceptance uses 17115/17116 and their exact sources. The original worker snapshots remain separate and unmodified.

From the repository root, verify and extract to a fresh directory:

```sh
sha256sum -c tests/stdlib/fft/evidence/ARCHIVE.sha256
mkdir /tmp/sounio-fft-receipts
tar -xzf tests/stdlib/fft/evidence/receipts.tar.gz -C /tmp/sounio-fft-receipts
(cd /tmp/sounio-fft-receipts && sha256sum -c MANIFEST.sha256)
```

Focused native success and a seed compatibility check do not imply full-corpus CI, publication, merge, or human acceptance.
