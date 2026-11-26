Madaros now prints an element of a `[string; N]` parameter as a string:
`print((*names)[i])` with `names: &[string; N]` (or `&![string; N]`),
`let s = (*names)[i]; print(s)`, `print(names[i])` on a by-value
`[string; N]` parameter, and `print((*r)[i])` after `let r = &names` printed
the element's address as an integer (for example 4207508) in place of the
text. The value was correct; the print dispatch lacked evidence that the
element is a string. Array parameters and `let r = &names` references now
record `string` as their element type, as `let names: [string; N]` locals
already did, and the string classifier accepts `(*r)[i]` through a
reference. The struct path `(*q).n[i]` was not affected.
