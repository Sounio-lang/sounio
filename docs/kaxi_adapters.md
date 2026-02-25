# K-AXI Adapter Flags

`flags` is a 32-bit packed field with this layout:

- `op_kind`: bits `[3:0]`
- `failure_inc`: bits `[15:8]`
- `success_inc`: bits `[23:16]`
- `alpha_q8`: bits `[31:24]`

Packing formula:

`flags = (op_kind & 0xF) | ((failure_inc & 0xFF) << 8) | ((success_inc & 0xFF) << 16) | ((alpha_q8 & 0xFF) << 24)`
