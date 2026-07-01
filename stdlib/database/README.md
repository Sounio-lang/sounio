# stdlib/database

In-memory SQL-like database with relational algebra operations.

## Architecture

- `pure/types.sio` - Core types (InMemoryDB, Table)
- `pure/queries.sio` - Query execution engine
- `lib.sio` - Public API

## Storage Model

- Flat array storage: 4 tables × 16 rows × 4 columns = 256 slots
- Column types: STRING, I64, F64
- Null markers for empty slots

## Capabilities

- CREATE TABLE, INSERT, SELECT, UPDATE, DELETE
- WHERE clause filtering
- Aggregate functions: COUNT, SUM, AVG, MIN, MAX
- JOIN support (INNER)
- GROUP BY with aggregates

## Usage

```
use database::lib

var db = database_new()
execute(&! db, "CREATE TABLE users (name STRING, age I64)")
execute(&! db, "INSERT INTO users VALUES ('Alice', 30)")
execute(&! db, "INSERT INTO users VALUES ('Bob', 25)")
let result = query(&db, "SELECT name, age FROM users WHERE age > 25")
```

## Tests

- `tests/stdlib/database/test_database_core.sio` — CRUD + drop (check-only)
- `tests/stdlib_database/test_database_e2e.sio` — legacy run-pass harness

FFI (`database::ffi::*`) exposes SQLite/libpq stubs returning errors until wired.