# stdlib/queue

Fixed-capacity FIFO queue.

## Key Types
- `Queue`: FIFO queue with fixed 256-element capacity

## Key Functions
- `queue_new()`: Create empty queue
- `queue_push(q, value)`: Add element to back
- `queue_pop(q)`: Remove element from front (returns -1 if empty)
- `queue_size(q)`: Get element count
- `queue_is_empty(q)`: Check if empty

## Tests

- `tests/stdlib/queue/test_queue_core.sio` — FIFO + FFI stubs (check-only)
- `tests/stdlib_queue/test_queue_e2e.sio` — legacy run-pass harness