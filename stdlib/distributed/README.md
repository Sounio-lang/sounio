# stdlib/distributed

Distributed computing types and node registry.

## Key Types
- `NodeId`: Node identifier (host, port, id)
- `DistributedMessage`: Message with from/to nodes and payload
- `NodeRegistry`: Registry of nodes (max 16)

## Key Functions
- `node_id_new(host, port, id)`: Create node ID
- `registry_new()`: Create empty registry
- `registry_add(r, node)`: Add node to registry
- `registry_size(r)`: Get registry size

## Tests

`tests/stdlib/distributed/test_distributed_core.sio` (check-only, Madaros gate)