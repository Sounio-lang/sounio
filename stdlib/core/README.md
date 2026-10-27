# stdlib/core

The canonical imports `use core::{Option,Result}`, explicit
`use core::lib::{Option,Result}` and direct module imports
`use core::option::{Option}` / `use core::result::{Result}` refer to the same declarations.
Generic instances preserve their defining module identity through re-exports.

Option<T> has Some(T) and None; Result<T,E> has Ok(T) and Err(E).
Payloads move when they own linear resources. Copy-only instances remain copyable.
The legacy concrete integer/float helper APIs remain available from the direct
option and result modules.

The generic APIs require the qualified generic-enum compiler stack.
Owner-bound returned references and borrowed payload carriers remain separate
compiler prerequisites; do not infer their safety from owned-payload support.
See tools/unicode/GENERIC_CORE.md for exact execution and compatibility receipts.
