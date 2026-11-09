`defer_action()` no longer crashes `souc check`. Every call segfaulted (rc 139)
after its diagnostic had already printed, because the by-value bridge returned a
164 KB `Checker` whose six `*mut` fields came back null, which
`checker_store_from_value` then wrote over the live checker's pointers. The
handler is now transcribed to operate on `*mut Checker` and intercepted before
that bridge, the same remedy `measure()` already had. The help text for E097,
E098 and E099 is keyed by code, so the in-place and by-value paths print the
same message, help and note.
