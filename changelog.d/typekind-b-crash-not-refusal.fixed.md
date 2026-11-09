The family-B TypeKind archaeology gate no longer accepts a compiler crash as a
refusal. It greps the refuse log for the expected diagnostic, so a compiler that
printed the right diagnostic and *then* died (rc 139) satisfied
`refuse_named=yes` and was classified `Reserved`. Any check that exits
abnormally — timeout (124), exec failure (126/127) or a signal (128+N) — is now
reported as `TYPEKIND_B_CRASH kind=… phase=… rc=…`, derives `Hypothesis`
instead of `Reserved`/`Claim-ready`, and fails the gate through a new `crashed`
counter.
