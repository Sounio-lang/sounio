//! Chiuratto-specific runtime FFI bridge.
//!
//! These symbols are consumed by the generated Chiuratto Sounio entrypoint
//! (`extern "http_ffi"` / `extern "postgres_ffi"`).
//! The goal here is to provide non-stub behavior for UUID/time/Postgres paths
//! while the native HTTP transport ABI is still being hardened.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
pub struct PgPool {
    dsn: CString,
    _min_conn: i32,
    _max_conn: i32,
    _timeout_ms: i32,
}

#[repr(C)]
pub struct PgResult {
    row_count: i32,
}

#[repr(C)]
pub struct ValueArray {
    pub ptr: i64,
    pub count: i64,
}

static UUID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn leak_cstring(value: String) -> *const c_char {
    match CString::new(value) {
        Ok(s) => s.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok().map(ToOwned::to_owned) }
}

fn psql_output(dsn: &CStr, sql: &CStr) -> Option<std::process::Output> {
    Command::new("psql")
        .arg(dsn.to_str().ok()?)
        .arg("-v")
        .arg("ON_ERROR_STOP=1")
        .arg("-X")
        .arg("-A")
        .arg("-t")
        .arg("-c")
        .arg(sql.to_str().ok()?)
        .output()
        .ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn uuid_v4() -> *const c_char {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let ctr = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut x = nanos ^ (pid << 32) ^ ctr.rotate_left(17) ^ 0xA5A5_5A5A_DEAD_BEEFu64;

    // xorshift64* mix for stable pseudo-random UUID material.
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let y = x.wrapping_mul(0x2545_F491_4F6C_DD1Du64);

    let part_a = x;
    let part_b = y;
    let uuid = format!(
        "{:08x}-{:04x}-4{:03x}-a{:03x}-{:012x}",
        (part_a >> 32) as u32,
        ((part_a >> 16) & 0xffff) as u16,
        (part_a & 0x0fff) as u16,
        ((part_b >> 52) & 0x0fff) as u16,
        part_b & 0x000f_ffff_ffff_ffffu64
    );
    leak_cstring(uuid)
}

#[unsafe(no_mangle)]
pub extern "C" fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn pg_pool_new(
    uri: *const c_char,
    min_conn: i32,
    max_conn: i32,
    timeout_ms: i32,
) -> *mut PgPool {
    let dsn = match cstr_to_string(uri).and_then(|s| CString::new(s).ok()) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let pool = PgPool {
        dsn,
        _min_conn: min_conn,
        _max_conn: max_conn,
        _timeout_ms: timeout_ms,
    };
    Box::into_raw(Box::new(pool))
}

#[unsafe(no_mangle)]
pub extern "C" fn pg_pool_query(
    pool: *mut PgPool,
    sql: *const c_char,
    _params: *mut ValueArray,
) -> *mut PgResult {
    if pool.is_null() || sql.is_null() {
        return std::ptr::null_mut();
    }

    let pool_ref = unsafe { &*pool };
    let sql_cstr = unsafe { CStr::from_ptr(sql) };
    let out = match psql_output(&pool_ref.dsn, sql_cstr) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    if !out.status.success() {
        return std::ptr::null_mut();
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let row_count = stdout.lines().filter(|line| !line.trim().is_empty()).count() as i32;
    Box::into_raw(Box::new(PgResult { row_count }))
}

#[unsafe(no_mangle)]
pub extern "C" fn pg_pool_execute(
    pool: *mut PgPool,
    sql: *const c_char,
    _params: *mut ValueArray,
) -> i64 {
    if pool.is_null() || sql.is_null() {
        return -1;
    }

    let pool_ref = unsafe { &*pool };
    let sql_cstr = unsafe { CStr::from_ptr(sql) };
    let out = match psql_output(&pool_ref.dsn, sql_cstr) {
        Some(v) => v,
        None => return -1,
    };
    if !out.status.success() {
        return -1;
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn pg_result_row_count(result: *mut PgResult) -> i32 {
    if result.is_null() {
        return 0;
    }
    unsafe { (*result).row_count }
}

#[unsafe(no_mangle)]
pub extern "C" fn pg_result_free(result: *mut PgResult) -> i32 {
    if result.is_null() {
        return 0;
    }
    let _ = unsafe { Box::from_raw(result) };
    0
}

#[repr(C)]
pub struct Request {
    pub method: *const c_char,
    pub path: *const c_char,
    pub headers: *const c_void,
    pub body: *const c_char,
    pub query_params: *const c_void,
}

#[repr(C)]
pub struct Response {
    pub status: i64,
    pub body: *const c_char,
}

#[unsafe(no_mangle)]
pub extern "C" fn http_server_start(_port: i32) -> i32 {
    // Explicit non-success until the native HTTP ABI (arrays/headers/query)
    // is stabilized in backend lowering.
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn http_server_stop() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn http_next_request() -> Request {
    Request {
        method: std::ptr::null(),
        path: std::ptr::null(),
        headers: std::ptr::null(),
        body: std::ptr::null(),
        query_params: std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn http_send_response(_response: Response) -> i32 {
    0
}
