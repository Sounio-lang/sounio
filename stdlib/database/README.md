# database — SQL Database Interface

SQL database connectivity for Sounio with both pure Sounio and native FFI backends.

## Overview

The `database` module provides a unified SQL interface supporting:

- **Pure Sounio Engine** (`pure/engine.sio`) — In-memory SQLite-like engine with no external dependencies
- **FFI Backends** (`ffi/`) — Native libsqlite3 and libpq when available

## Epistemic Differentiators

- `EpistemicResultSet` — Query results with `Knowledge<f64>` uncertainty for aggregates
- Provenance tracking for query lineage
- Confidence-aware aggregation functions
- GUM-compliant uncertainty propagation

## Quickstart

```sio
use database::pure::engine

let mut db = in_memory_db_new()

// Create and populate
let sql = "CREATE TABLE users (id INTEGER, name TEXT, age REAL)".to_string()
engine_execute_sql(&mut db, sql)

let insert = "INSERT INTO users VALUES (1, 'Alice', 30.5)".to_string()
engine_execute_sql(&mut db, insert)

// Query
let select = "SELECT name, age FROM users WHERE age > 25".to_string()
let result = engine_execute_sql(&mut db, select)
```

## Module Structure

| File | Description |
|------|-------------|
| `pure/types.sio` | Core types: Database, Connection, Value, ResultSet |
| `pure/parser.sio` | SQL parser (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE) |
| `pure/engine.sio` | In-memory SQL execution engine |
| `ffi/bindings.sio` | FFI declarations for libsqlite3 and libpq |
| `ffi/wrapper.sio` | Sounio wrapper over FFI when available |
| `ffi/fallback.sio` | Graceful fallback when FFI unavailable |

## Supported SQL

- `SELECT` with WHERE, ORDER BY, GROUP BY, HAVING, LIMIT, OFFSET, DISTINCT
- `INSERT INTO table VALUES (...)`
- `UPDATE table SET col = val WHERE ...`
- `DELETE FROM table WHERE ...`
- `CREATE TABLE table_name (col_name TYPE, ...)`
- `DROP TABLE table_name`

## Benchmarks

See `../../benchmarks/README.md` for performance targets vs SQLite/libpq.

## Validation Status

- Parser: ✅ Tests passing for all supported SQL forms
- Engine: ✅ CREATE, INSERT, SELECT, UPDATE, DELETE working
- FFI: ⚠️ Bindings declared but require libsqlite3/libpq to be linked

## License

MIT / Apache-2.0 (same as Sounio)