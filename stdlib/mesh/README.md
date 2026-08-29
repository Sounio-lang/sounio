# stdlib/mesh

Fixed-capacity mesh for 3D graphics.

## Architecture

- `pure/types.sio` - Mesh, Vertex types and operations
- `lib.sio` - Public API

## Storage Model

- Fixed capacity: 256 vertices (× 3 coords) + 128 faces (× 3 indices)
- Vertices stored as flat `[f64; 768]` array
- Faces stored as flat `[i32; 384]` array

## Capabilities

- Create meshes with `mesh_new()`
- Add vertices with `mesh_add_vertex(&! mesh, x, y, z)`
- Add faces with `mesh_add_face(&! mesh, i0, i1, i2)`
- Query counts with `mesh_vertex_count`, `mesh_face_count`
- Access vertex data with `mesh_get_vertex(&mesh, idx) -> [f64; 3]`
- Access face indices with `mesh_get_face(&mesh, idx) -> [i32; 3]`
- Compute triangle area with `mesh_triangle_area(&mesh, face_idx) -> f64`
- Compute total surface area with `mesh_total_surface_area(&mesh) -> f64`

## Compiler Bug Workaround

Returning array literals with computed indices causes segfaults. Use explicit local variables:

```
// BROKEN (segfaults):
pub fn get_vertex(m: &Mesh, idx: i32) -> [f64; 3] {
    let i = idx as usize
    [m.vertices[i * 3], m.vertices[i * 3 + 1], m.vertices[i * 3 + 2]]
}

// WORKAROUND:
pub fn get_vertex(m: &Mesh, idx: i32) -> [f64; 3] {
    let i = idx as usize
    let x = m.vertices[i * 3]
    let y = m.vertices[i * 3 + 1]
    let z = m.vertices[i * 3 + 2]
    let result: [f64; 3] = [x, y, z]
    result
}
```

Functions returning arrays with `with Div` effect may cause typecheck errors. Avoid this combination.

## Tests

- `tests/stdlib/mesh/test_mesh_core.sio` — vertices, faces, area (check-only)
- `tests/stdlib_mesh/test_mesh_e2e.sio` — legacy run-pass harness

FFI render stubs: `mesh::ffi::wrapper::{mesh_render_gl, mesh_render_vulkan}` (no-op).