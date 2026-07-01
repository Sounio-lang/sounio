# stdlib/audio

Audio processing utilities with FFI wrappers.

## Key Types
- `AudioBuffer`: Fixed-capacity audio sample buffer (44100 Hz default)

## Key Functions
- `audio_buffer_new(sample_rate)`: Create new buffer
- `audio_buffer_push(buf, sample)`: Add sample to buffer
- `audio_buffer_get(buf, idx)`: Get sample at index
- `audio_buffer_size(buf)`: Get current size
- `audio_buffer_sample_rate(buf)`: Get sample rate

## Tests

`tests/stdlib/audio/test_audio_core.sio` (check-only, Madaros gate)