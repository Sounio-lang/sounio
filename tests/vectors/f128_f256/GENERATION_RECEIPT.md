<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256.generation-receipt
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: minimax-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.vectors.f128-f256.generation-receipt
-->

# WS-G MPFR f128 / f256 Test-Vector Corpus — Generation Receipt

Wave 1, lane `ws-g-mpfr-vectors` (dispatched 2026-08-16 from
`MADAROS_FOCUS_PLAN_2026-08-16.md` §1 WS-G, §3 Wave 1 cold lanes).

## What this directory is

External-oracle test vectors for IEEE-754 binary128 and binary256 arithmetic
and comparison, produced by an out-of-tree generator that uses MPFR (the
Multiple Precision Floating-Point Reliable Library) as the ground truth. The
vectors are **the oracle**; this directory does not contain any Sounio
arithmetic implementation — per the wave-1 constraint, Sounio's softfloat
routines (WS-G V0-D) come in a later wave.

The vectors exist so that, once Sounio gains f128/f256 arithmetic, a test
gate can read these JSONL files and compare Sounio's output against MPFR's
output bit-for-bit on every case.

## Toolchain used

| Component | Version |
|---|---|
| GCC        | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| MPFR       | 4.2.1-1build1.1 (libmpfr-dev) |
| GMP        | 2:6.3.0+dfsg-2ubuntu6.1 (libgmp-dev) |
| OS         | Ubuntu 24.04.4 LTS (Noble Numbat) |

Build command (recorded for reproducibility):

```
gcc -O2 -Wall -Wextra -Wno-unused-function -Wno-unused-parameter \
    -Wno-shift-count-overflow \
    -o mpfr_vector_gen mpfr_vector_gen.c -lmpfr -lgmp
```

## Determinism

The generator uses **PCG-XSH-RR** with a fixed seed:

| Constant | Value |
|---|---|
| `pcg_state` | `0x853c49e6748fea9b` |
| `pcg_inc`   | `0xda3e39cb94b95bdb` |

No clock reads, no `/dev/urandom`, no environment variables are consulted.
Two runs on the same toolchain produce byte-identical output. Verified by
running `gen/run.sh` (or invoking the binary directly) twice and diffing.

**Cross-platform caveat:** different GMP/MPFR builds may produce different
last-bit rounding for some edge inputs, so byte-identical reproduction
across machines is not guaranteed. To re-validate after a toolchain change,
re-run `gen/run.sh` and update the hashes below.

## Format parameters

IEEE-754 binary128 and binary256 per IEEE 754-2008 §3.6:

| | binary128 | binary256 |
|---|---|---|
| storage bits | 128 | 256 |
| exponent bits (`K`) | 15 | 19 |
| precision bits (`P`) | 113 | 237 |
| trailing significand bits | 112 | 236 |
| exponent bias | 16383 | 262143 |
| `emax` | 16383 | 262143 |

Confirmed by `self-hosted/compiler/f128_f256_format_descriptor_probe.sio`
(the `binary_format_binary128_descriptor()` /
`binary_format_binary256_descriptor()` entries).

## Wire encoding (per vector)

Each operand and each result in the JSONL is encoded as a Sounio
IR-wide-numeric-payload-pool entry:

```jsonc
{
  "class":        "zero" | "subnormal" | "normal" | "inf" | "nan" | "snan",
  "sign":         0 | 1,
  "exponent":     <biased exponent, uint16 or uint32 depending on format>,
  "trailing_hex": "<hex of the P-1 trailing bits, big-endian>",
  "limbs":        [<LSW i64>, <next i64>, ...]   // little-endian, 2 limbs for f128, 4 limbs for f256
}
```

`limbs` is the wire format consumed by `IrWideNumericPayloadPool`
(`self-hosted/compiler/f128_f256_numeric_payload_wire_probe.sio`
references the same `numeric_format_binary128_id()` /
`numeric_format_binary256_id()` format IDs as this corpus).

The `class` field is **derived** from `(exponent, trailing)` and is
redundant with them — it's there so consumers can filter on kind without
parsing the hex. Rules:

| class      | exponent | trailing |
|------------|----------|----------|
| `zero`     | 0        | 0        |
| `subnormal`| 0        | nonzero  |
| `normal`   | 1..2*emax | any     |
| `inf`      | 2*emax+1 | 0        |
| `nan` (qNaN)| 2*emax+1| bit (P-2) set (qNaN marker) |
| `snan` (sNaN)| 2*emax+1| bit (P-2) clear, any other bit set |

## Operation coverage

11 ops per format. For both f128 and f256:

| Op | Meaning | Notes |
|---|---|---|
| `f{128,256}_add`    | binary + | incl. NaN, ±0, ±inf, subnormal + normal, overflow |
| `f{128,256}_sub`    | binary − | ditto |
| `f{128,256}_mul`    | binary × | incl. 0×inf → NaN, subnormal × subnormal |
| `f{128,256}_div`    | binary ÷ | incl. x/0 → ±inf, 0/0 → NaN |
| `f{128,256}_cmp_eq` | ==       | signed: NaN ≠ NaN, +0 == −0 |
| `f{128,256}_cmp_ne` | !=       | ditto |
| `f{128,256}_cmp_lt` | <        | unordered on NaN |
| `f{128,256}_cmp_le` | ≤        | ditto |
| `f{128,256}_cmp_gt` | >        | ditto |
| `f{128,256}_cmp_ge` | ≥        | ditto |
| `f{128,256}_cmp_unord` | unordered | true iff either operand is NaN |

Comparison results are JSON booleans; arithmetic results are the
`{class, sign, exponent, trailing_hex, limbs}` object described above.

Rounding mode for all arithmetic: **IEEE-754 round-to-nearest-even** with
gradual underflow. Recorded as `"rounding": "rne"` on every vector.

## Operand-class distribution

Counted across all 11 ops for each format (an operand is one slot of an
arithmetic op or either operand of a cmp):

| class       | f128 count | f256 count |
|-------------|-----------:|-----------:|
| normal      |       3362 |       3359 |
| subnormal   |        526 |        526 |
| nan         |        463 |        463 |
| inf         |         33 |         33 |
| zero        |         30 |         30 |

The generator deliberately over-samples NaN (signaling + quiet) and
subnormal cases because that's where most IEEE-754 implementation bugs
hide.

## Output hashes (this generation)

Recorded by `gen/run.sh` on 2026-08-16:

```
f128.jsonl  4414 lines  md5    = 267bf24904b035b964e47f2c4ec6d20d
                      sha256 = 4d999bae27a52459d6ee3d5c70a4c2c7d90f6c45298243c779fe04374511a729

f256.jsonl  4411 lines  md5    = fce8f18838b3b274b97b2c32c6a4c9ff
                      sha256 = 0bd0ec0b655ebb2110601b2c47096e9cca7a73fd00b6e775866c4ccbd7ecf17a
```

Re-check by running:

```
md5sum    tests/vectors/f128_f256/*.jsonl
sha256sum tests/vectors/f128_f256/*.jsonl
```

## Spot-check coverage

The generator's IEEE-754 RNE rounding path was exercised directly against
the boundary cases below during development (the test was an out-of-tree
binary that included the same `round_to_binaryN` function as the corpus
generator):

| Input | Expected (sign, exp, trailing) | Result |
|---|---|---|
| +0 | zero, 0, 0 | PASS |
| −0 | zero, 0, 0 | PASS |
| +inf | inf, 32767, 0 | PASS |
| −inf | inf, 32767, 0 (sign=1) | PASS |
| qNaN | nan, 32767, top bit of trailing | PASS |
| +1.0 | normal, 16383, 0 | PASS |
| +2.0 | normal, 16384, 0 | PASS |
| +0.5 | normal, 16382, 0 | PASS |
| +4.0 | normal, 16385, 0 | PASS |
| +π | normal, 16384, 0x0000921fb54442d18 | PASS (matches well-known binary128 π) |

(I did not include a cross-check script in the corpus itself; the same
algorithm appears in `gen/mpfr_vector_gen.c` and the spot tests above were
run against that exact copy during development.)

## How to regenerate

```
cd tests/vectors/f128_f256/gen
./run.sh                    # regenerates ../f128.jsonl and ../f256.jsonl
./run.sh /tmp/whatever      # writes to a different output dir
```

The script records tool versions and hashes to stderr; pipe to a file
when updating `GENERATION_RECEIPT.md`.

## Cross-validation against an independent oracle (future work)

The generator uses MPFR for the extended-precision computation and a
manual IEEE RNE round back to binary128/binary256 for the result. A
stronger external oracle is **another** independent binary128/binary256
library that does NOT use MPFR. Candidates:

- Intel's `libbid` (Binary Integer Decimal, but also handles binary IEEE)
- The `correctly-rounded` crate's IEEE-754 backend (Rust)
- `SoftFloat` by John Hauser (C reference implementation, public domain)

A wave-3 cross-validation script that diffs this corpus's results against
one of those libraries would be a high-value add — tracked under WS-G
V0-D followups, not wave 1.

## Lane claim (Wave 1 arithmetic)

The Sounio coordination bus records the original arithmetic corpus as claimed by
`minimax-cli3` for the intent `WS-G MPFR f128/f256 test vectors` against
file scope `tests/vectors/f128_f256/**`. See
`docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` §1 WS-G
and §3 Wave 1. Commit: `8ce767f33b`.

---

## Wave 3 extension — literal / boundary / double-rounding vectors

**Agent:** grok-cli1 · lane `ws-g-ref-vectors` · 2026-08-16  
**Disposition of Wave 1 corpus:** **REUSED** (not replaced). Arithmetic
`f128.jsonl` / `f256.jsonl` and `gen/mpfr_vector_gen.c` unchanged from
`8ce767f33b`. New files only.

### Why a second corpus

V0-B probes check that a **source literal** becomes the correct wide bit
pattern. The arithmetic corpus answers “what is a⊕b?” — necessary for V0-D,
insufficient for V0-B. In particular, a parser that widens through `f64`
is self-consistent against itself and still wrong on every
`double_rounds_differs=true` case.

### Toolchain (this generation)

| Component | Version |
|---|---|
| GCC | host `gcc` on Ubuntu 24.04 pod |
| MPFR | **4.2.1** (`pkg-config --modversion mpfr`) |
| GMP | linked via `-lgmp` |
| Rounding | `MPFR_RNDN` (IEEE round-ties-to-even) |
| Extended precision | 2048 bits before RNE to target |

Build / run:

```
cd tests/vectors/f128_f256/gen
gcc -O2 -Wall -Wextra -Wno-unused-function \
  -o literal_boundary_gen literal_boundary_gen.c -lmpfr -lgmp
./run_literal_boundary.sh
```

Exact invocation per vector (also inlined in each JSON `provenance` field):

```
mpfr_set_str(s, 0, MPFR_RNDN);           # parse source (decimal or hexfloat)
direct:  mpfr_set(y, x, MPFR_RNDN) at p=113 (f128) or p=237 (f256); encode wire
via_f64: mpfr_set(d, x, MPFR_RNDN) at p=53; encode that real as binaryN
double_rounds_differs := (direct wire != via_f64 wire)
```

### Coverage (literal corpora)

| Family | f128 | f256 | Notes |
|---|---:|---:|---|
| exactly_representable | 17 | 17 | 0, ±1, dyadics, hexfloat 1.0/2.0/0.5 |
| provably_not_representable | 6 | 6 | 0.1, 0.2, 0.3, π/e digit strings |
| subnormal | 4 | 3 | min subnormal 2^(emin−(p−1)), neighbors |
| min_normal | 2 | 2 | ±2^emin |
| max_finite | 2 | 1 | all-ones trailing at emax |
| ulp_neighbors | 4 | 2 | 1.0, 1+ulp, ulp(1.0) |
| double_rounding_trap | 10 | 10 | cases where via_f64 ≠ direct |
| literal_boundary | 8 | 8 | 1e0, 1E0, 0X1.0P+0, equiv spellings |
| **Total** | **53** | **49** | |

Measured `double_rounds_differs=true`: **18** (f128) + **16** (f256) = **34**
(includes traps plus some non-trap family rows that still differ, e.g. 0.1
under `provably_not_representable`).

### Spot checks (this session)

| Input | Expected property | Result |
|---|---|---|
| `1` → binary128 | exp=16383, trailing=0, limbs hi=`0x3fff000000000000` | PASS |
| `0x1p-16494` | min subnormal, limbs `[1,0]` | PASS |
| `0x1p-16382` | min normal, exp_biased=1 | PASS |
| `0.1` direct vs via_f64 | `double_rounds_differs=true`; trailing `…999a` vs `…a000…` | PASS |

### Output hashes (literal corpora, this generation)

```
literal_boundary_f128.jsonl  53 lines
  md5    = e8ab21e596603c7aa37ebb70fd14e673
  sha256 = 4b7804f0d70016770fb8bda4b67beb9226be1cebf9dc02f3771a4a7fcdbfa52d

literal_boundary_f256.jsonl  49 lines
  md5    = 7727492e8fc930bfdf304ace65f8e47d
  sha256 = 574161d42fe10379d42198c03474a7b0b1daa39235111c45c2e8e28a7017a4c3
```

Re-check: `md5sum tests/vectors/f128_f256/literal_boundary_*.jsonl`

### How grok-cli3 V0-B probes should consume these

1. For each vector with `source_kind` decimal/hexfloat, emit a Sounio literal
   (once V0-B accepts them) or a payload constant built from `expected.limbs`.
2. Assert the compiled/checked constant matches `expected` bit-for-bit.
3. **Must include** every `double_rounds_differs=true` row — these fail any
   widen-f64 shortcut that still passes pure dyadic cases.
4. Do **not** use Sounio output as the oracle for these files.

### What was not done here

- SoftFloat / libbid cross-check of the arithmetic corpora (still future).
- Regeneration of minimax arithmetic JSONL (left frozen at `8ce767f33b` hashes).
- Sounio probes or ladder gate scripts (grok-cli3 ownership).