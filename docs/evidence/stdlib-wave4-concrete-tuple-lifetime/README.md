# Concrete nested-tuple lifetime audit

The defect is reproducible without Option, None, generic inference, raw casts or
FFI. Original19818 sources remain unchanged. This package introduces an independent
fully concrete flat return (i64,i64) and nested return ((i64,i64),i64), both observed
before/after the same Sounio clobber512 call. Scalar reads are captured before any
printing, so the observation does not insert an IO call between construction and
the before/after measurements.

Slurm19835 COMPLETED0/7s/463744K. Exact C++23 oracle:
| Compiler artifact | Flat compile/native/oracle | Nested compile/native/oracle | assertfalse native |
| --- | --- | --- | --- |
|9cf seed |0/0/0 |0/0/1 |1 |
|ccfa producer |0/0/0 |0/0/1 |1 |
|raw04 LOOP_REGIONS0 + R05 stdlib |0/0/0 |0/0/0 |1 |
|raw04 LOOP_REGIONS1 + R05 stdlib |0/0/0 |0/0/0 |1 |

Raw04 check stages also return0. Legacy seeds integrate parser/checker into their
compile CLI; no separate successful check status is invented.

Both legacy nested outputs are:
0 42 1 7 0
1 602 603 7 182016
2 -9 0 8 0
3 603 604 8 182528

The original expected values remain42/1/7 and-9/0/8 after the clobber. Checksums
182016/182528 match the C++23 reference and ensure the clobber result is consumed.
Flat and raw04 outputs retain all values. The C++ oracle has6 controls including
wrong values, removed/extra/reordered rows and wrong checksum.

The concrete wrong behavior already exists in9cf and ccfa, independent of the
private19818 Option-join candidate.19847 confirms that their generated flat ELFs
are byte-identical and their generated nested ELFs are byte-identical. This does
not claim their compiler sources are identical.

This is a read-only audit. No compiler/lower/runtime source or ABI was modified.
Raw04 passes these exact witnesses; no general claim about every nested type or
the proposed producer is made. Shared lifetime repair and representation decisions
remain with root.
