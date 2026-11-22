Fixed (ADR-015 row 5, Madaros): binary (`0b101`) and octal (`0o17`) integer literals. Madaros read `0b101` as the number `0` followed by a name `b101` (E137): the live lexer only scanned `0x`, and the scanner that did read `0b`/`0o` sat below `return tc`, unreachable. Upper-case prefixes, `_` between two digits and integer suffixes (`0b1111_1111u8`) work; probes `int_bin` and `int_oct` evaluate.

Refused with a new code, E289, instead of a silent split or the value 0: a prefix with no digit (`0o`), a decimal digit outside the base (`0b102`), and a float suffix (`0b1f32`). A value past 64 bits is E276.

Also fixed: the width check counted 22 octal digits as fitting 64 bits, so `0o2000000000000000000000` (2^65) was accepted and wrapped. 22 octal digits fit only when led by `1`.

Pins: `radix_literals_madaros` (run-pass); `radix_digit_outside_base_madaros`, `radix_no_digits_madaros`, `radix_float_suffix_madaros`, `radix_octal_too_wide_madaros`, `radix_suffix_range_madaros` (compile-fail).
