# stdlib/image

Image processing with fixed-capacity buffers.

## Key Types
- `Image`: Fixed-capacity image buffer (256×256 max)

## Key Functions
- `image_new(width, height)`: Create new image
- `image_get_pixel(img, x, y)`: Get pixel value
- `image_set_pixel(img, x, y, value)`: Set pixel value
- `image_width(img)`: Get image width
- `image_height(img)`: Get image height

## Tests

`tests/stdlib/image/test_image_core.sio` (check-only, Madaros gate)