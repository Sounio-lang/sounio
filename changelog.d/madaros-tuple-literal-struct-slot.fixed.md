Madaros reads the fields of a struct held in a tuple-literal slot at that
struct's own layout. `let t = (n, B { .. })` typed no slot from a literal:
slot 0 only from a call result, later slots never. So `t.1.x` resolved `x` by
a vote across every struct with an `x`, and when the indices disagreed it read
the first-byte hash slot: `(B { y: 2, x: 11 }, 0).0.x` printed 0. lean_single
printed 11. A tuple local records two struct slots; a struct in a third slot
is still untyped, and `SOUNIO_STRICT_FIELD=1` refuses its field reads with
E311 instead of guessing.
