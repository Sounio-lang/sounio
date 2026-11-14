The macOS arm64 backend now fails closed on signed integer overflow, as the
x86-64 backend has since SIO-207 (ADR-031 R1/R2). `+`, `-` and `*` on i64
use `adds`/`subs` and a `smulh`-vs-`asr #63` check, unary minus uses `negs`,
and `MIN / -1` and `MIN % -1` are caught before `sdiv` (which returns MIN
without a flag). Each guard branches over a three-word `exit(1)` stub, the
same panic shape the x86 emitter uses. Before this change the same programs
wrapped silently on Apple silicon. `wrapping_add/sub/mul` keep the wrap.
Unsigned add/sub/mul are not distinguished on `main` yet (#3526); the
arm64 carry guards follow that PR.
