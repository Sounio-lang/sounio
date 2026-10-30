The Stdlib Reliability workflow, dispatched without a `ref` input, now evaluates the branch it was dispatched from. It used to default to `main`, so a run dispatched on `release/3.0` measured `main`.
