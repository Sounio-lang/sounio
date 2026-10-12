# Root explorations

Exploratory `.sio` probes that used to sit at the repository root. They were
moved here on 2026-10-04 so that the root shows only what a visitor needs (see
`TOUR.md`). Nothing referenced them by path except one comment in
`self-hosted/native/codegen_plan.sio`, which has been updated. Their contents
are unchanged. Measured before and after the move: `bin/souc check` exits 1 on
them in both places, so they were not compiling at the root either.
