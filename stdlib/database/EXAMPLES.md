# database — Examples

> **Fixed-capacity engine.** `database::pure::engine` is an in-memory engine
> over `InMemoryDB` (4 tables, 16 rows/table, 4 `i64` columns/row). It exposes
> only `engine_create_table`, `engine_drop_table`, `engine_insert_row`,
> `engine_get_cell`, and `engine_table_row_count` — there is **no** SQL string
> API such as `engine_execute_sql`. Mutating calls take a `&!` borrow; reads take
> `&`. Aggregate/query logic is written in Sounio by reading cells back.

## 1. In-Memory Database

```sio
use database::pure::engine
use database::pure::types

var db = in_memory_db_new()

// Create a 2-column table (InMemoryDB stores fixed i64 columns v0, v1).
let create_rc = engine_create_table(&!db, "products")
assert(create_rc == 1)

// Insert rows (v0 = id, v1 = price in cents).
let i1 = engine_insert_row(&!db, "products", 1, 2999)
let i2 = engine_insert_row(&!db, "products", 2, 4999)
assert(i1 == 1 && i2 == 1)

// Read a cell back (row 0, column 0 = id; column 1 = price).
let id = engine_get_cell(&db, "products", 0, 0)
let price = engine_get_cell(&db, "products", 0, 1)
println("product {} costs {} cents", id, price)

// Count rows in the table.
let count = engine_table_row_count(&db, "products")
assert(count == 2)
```

## 2. Aggregations

```sio
use database::pure::engine

var db = in_memory_db_new()

let create_rc = engine_create_table(&!db, "orders")
assert(create_rc == 1)

// Insert (v0 = amount in cents): Alice 10000, Alice 15000, Bob 20000.
engine_insert_row(&!db, "orders", 10000, 0)
engine_insert_row(&!db, "orders", 15000, 0)
engine_insert_row(&!db, "orders", 20000, 0)

// The fixed-capacity engine has no SQL, so aggregate by reading cells in Sounio.
let n = engine_table_row_count(&db, "orders")
var total: i64 = 0
var i = 0
while i < n {
    total = total + engine_get_cell(&db, "orders", i, 0)
    i = i + 1
}
println("total amount = {} cents", total)  // 45000
```

## 3. Table Operations

```sio
use database::pure::engine
use database::pure::types

var db = in_memory_db_new()

let create_rc = engine_create_table(&!db, "users")
assert(create_rc == 1)

let ins = engine_insert_row(&!db, "users", 1, 0)
assert(ins == 1)

// Count rows in the table.
let count = engine_table_row_count(&db, "users")
assert(count == 1)

// Read a cell back.
let id = engine_get_cell(&db, "users", 0, 0)
assert(id == 1)
```
