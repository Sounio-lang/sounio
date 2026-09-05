// ADR-009 verified_foreign_reference pilot: F# reference for Sounio's
// stdlib/data/bigrat.sio exact rational arithmetic (BigRat = num/den in
// lowest terms, den > 0), independently authored using F#'s built-in
// System.Numerics.BigInteger rather than transliterating Sounio's
// from-scratch base-1e9-limb BigInt.
//
// Sounio's own module doc flags a real risk this addresses: "this
// compiler has a whole-program codegen capacity wall for struct-heavy
// BigInt-by-value code -- past a shape-sensitive op-count it can
// SILENTLY EMIT WRONG VALUES with a clean exit." The existing gate
// (scripts/bigrat_gate.sh) already cross-checks against a Python
// oracle, but ADR-008 classifies ANY Python-authority path as
// external_corroboration_only (report-only, cannot fail CI). This F#
// reference lets the same claim be checked as verified_foreign_reference
// (can fail CI) per ADR-009's admission criteria: independently
// authored, statically typed, exact (no untyped numeric coercion).

open System.Numerics

type Rat = { Num: BigInteger; Den: BigInteger }

let rec bgcd (a: BigInteger) (b: BigInteger) : BigInteger =
    if b = BigInteger.Zero then BigInteger.Abs a else bgcd b (a % b)

// make(n, d): normalize sign onto numerator, reduce to lowest terms.
// Mirrors bigrat_make's contract exactly: den > 0, gcd(num,den) = 1,
// and n=0 collapses to 0/1 (Sounio's zero-gcd branch is unreachable
// for any nonzero den, since gcd(0, d) = |d| != 0; both implementations
// agree 0/d reduces to 0/1 via ordinary gcd reduction).
let make (n: BigInteger) (d: BigInteger) : Rat =
    let n, d = if d.Sign < 0 then -n, -d else n, d
    let g = bgcd n d
    if g = BigInteger.Zero then { Num = BigInteger.Zero; Den = BigInteger.One }
    else { Num = n / g; Den = d / g }

let r (n: int64) (d: int64) : Rat = make (BigInteger n) (BigInteger d)

let add (a: Rat) (b: Rat) : Rat = make (a.Num * b.Den + b.Num * a.Den) (a.Den * b.Den)
let mul (a: Rat) (b: Rat) : Rat = make (a.Num * b.Num) (a.Den * b.Den)
let sub (a: Rat) (b: Rat) : Rat = make (a.Num * b.Den - b.Num * a.Den) (a.Den * b.Den)
let div (a: Rat) (b: Rat) : Rat = make (a.Num * b.Den) (a.Den * b.Num)
let cmp (a: Rat) (b: Rat) : int = compare (a.Num * b.Den) (b.Num * a.Den)

let show (label: string) (x: Rat) =
    printfn "%s: %s/%s" label (x.Num.ToString()) (x.Den.ToString())

let main () =
    show "add" (add (r 1L 2L) (r 1L 3L))
    show "mul" (mul (r 2L 3L) (r 3L 4L))
    show "sub" (sub (r 1L 2L) (r 1L 3L))
    show "div" (div (r 1L 2L) (r 1L 3L))
    show "cmp_lt" (r 1L 3L)
    show "cmp_eq" (r 1L 2L)

    let mutable acc = r 0L 1L
    for d in [2L; 3L; 5L; 7L; 11L; 13L] do
        acc <- add acc (r 1L d)
    show "prime_recip_sum" acc

    printfn "bigrat_cmp(1/3,1/2)=%d" (cmp (r 1L 3L) (r 1L 2L))
    printfn "bigrat_cmp(1/2,1/2)=%d" (cmp (r 1L 2L) (r 1L 2L))

main ()
