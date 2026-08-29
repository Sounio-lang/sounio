# ZD-SSM Dashboard

Static single-page UI that requests a surgical operation from the
ZD-SSM service and returns the Lean witness.

## Running locally

The dashboard is a single static `index.html`; you can open it
directly in a browser.  It stubs the server response for
demonstration purposes.

To wire it to a real ZD-SSM instance:

1. Start the Sounio inference driver (`souc run inference.sio`)
   behind a small HTTP surface (any micro-framework will do; a
   minimal Rust/axum skeleton is 30 lines).
2. The `POST /api/surgery` endpoint should accept the JSON body
   `{op, chunk, klass, subject}` and return a JSON envelope with
   fields `op`, `chunk`, `zd_class`, `lean_theorem`, `residual`,
   `verified`, and `witness_bytes`, plus a `Content-Disposition:
   attachment` body containing the `.lean` witness.

## Contract

The witness `.lean` is a self-contained file that imports
`SounioSurgicalInterventions` and exports one top-level
`#eval`'able term whose type is the Lean theorem named in the
envelope.  The auditor verifies by running

```
lean --run witness.lean
```

and observing `true` / exit-0 status.  Any tampering (wrong
chunk, wrong class, off-by-one in the algebraic check) is caught
by the `native_decide` gate.

## Safety notes

- The dashboard is read-only from the model's point of view; each
  surgical request is idempotent at the algebraic level (apply
  the kernel projection for that (chunk, class) pair).
- Revivable operations additionally require a Temporal window id
  to be passed through; that field will be added to the form when
  Paper~G-aligned reversibility is wired up in the server.
