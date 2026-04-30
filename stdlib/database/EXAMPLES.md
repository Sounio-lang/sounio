# database — Examples

## 1. In-Memory Database

```sio
use database::pure::engine
use database::pure::types

let mut db = in_memory_db_new()

let create = "CREATE TABLE products (id INTEGER, name TEXT, price REAL)".to_string()
engine_execute_sql(&mut db, create)

let insert = "INSERT INTO products VALUES (1, 'Widget', 29.99)".to_string()
engine_execute_sql(&mut db, insert)

let insert2 = "INSERT INTO products VALUES (2, 'Gadget', 49.99)".to_string()
engine_execute_sql(&mut db, insert2)

let select = "SELECT name, price FROM products WHERE price < 40".to_string()
let result = engine_execute_sql(&mut db, select)

match result {
    Ok(qr) => match qr.result {
        Result::Rows(rs) => {
            print("Found {} products\n", rs.row_count)
        }
        _ => {}
    }
    Err(e) => print("Error: {:?}\n", e)
}
```

## 2. Aggregations

```sio
use database::pure::engine

let mut db = in_memory_db_new()

let create = "CREATE TABLE orders (customer TEXT, amount REAL)".to_string()
engine_execute_sql(&mut db, create)

let i1 = "INSERT INTO orders VALUES ('Alice', 100.0)".to_string()
let i2 = "INSERT INTO orders VALUES ('Alice', 150.0)".to_string()
let i3 = "INSERT INTO orders VALUES ('Bob', 200.0)".to_string()
engine_execute_sql(&mut db, i1)
engine_execute_sql(&mut db, i2)
engine_execute_sql(&mut db, i3)

let select = "SELECT customer, SUM(amount) FROM orders GROUP BY customer".to_string()
let result = engine_execute_sql(&mut db, select)
```

## 3. Table Operations

```sio
use database::pure::engine
use database::pure::types

let mut db = in_memory_db_new()

let create = "CREATE TABLE users (id INTEGER, name TEXT)".to_string()
engine_execute_sql(&mut db, create)

let insert = "INSERT INTO users VALUES (1, 'Alice')".to_string()
engine_execute_sql(&mut db, insert)

let tables = engine_list_tables(&db)
assert_eq!(tables.len(), 1)

let desc = engine_describe_table(&db, "users")
match desc {
    Ok(t) => assert_eq!(t.name, "users"),
    Err(_) => assert!(false),
}
```