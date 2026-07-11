# stdlib/serial

Serial communication types and buffer management.

## Key Types
- `SerialConfig`: Serial port configuration (baud rate, data bits, stop bits, parity)
- `SerialBuffer`: TX/RX ring buffers (256 bytes each)
- `Parity`: Parity enum (None, Even, Odd)

## Key Functions
- `serial_config_new(rate)`: Create config with baud rate
- `serial_buffer_new()`: Create empty TX/RX buffers
- `serial_buffer_push_rx(buf, byte)`: Push byte to RX buffer
- `serial_buffer_pop_tx(buf)`: Pop byte from TX buffer

## Tests

`tests/stdlib/serial/test_serial_core.sio` (check-only, Madaros gate)