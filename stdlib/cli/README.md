# stdlib/cli

Command-line argument parsing.

## Key Types
- `Args`: Fixed-capacity argument storage (32 args max)

## Key Functions
- `args_new()`: Create empty Args
- `args_push(args, s)`: Add string argument
- `args_get(args, idx)`: Get argument at index
- `args_count(args)`: Get argument count

## Tests

`tests/stdlib/cli/test_cli_core.sio` (check-only, Madaros gate)