Fixed an 8-byte overread in the hand-written Sporos stage-1 hex: the
data-segment length field at `0x1a5` was loaded with a 64-bit load, so
a file whose length field ends the mapped buffer could fault instead
of being refused. It is now a 32-bit load (`mov eax, [rsi]`), the
exact width of the field, and such a file exits 1. Covered by the new
vector `bootstrap/roots/sporos/vectors/data-overread-65535.spb`. The
stage-1 differential gate now feeds the file itself to the committed
`x86run` with `@FILE`, replacing a generated RunHex variant.
