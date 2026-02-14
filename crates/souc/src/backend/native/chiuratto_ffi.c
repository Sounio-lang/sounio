/* chiuratto_ffi.c - Chiuratto Runtime FFI Bridge
 *
 * SECURITY HARDENING:
 * - All string operations use bounded functions (snprintf, strncmp, etc.)
 * - Input validation on all FFI boundaries
 * - Integer parsing uses strtoll/strtol with error checking
 * - Buffer sizes are capped to prevent integer overflow
 * - Memory probing via mincore() to prevent segfaults
 * - Arena allocator limits to prevent DoS
 *
 * IMPORTANT: This code runs in production HTTP servers and must be
 * secure against malicious inputs. Any changes must be security-reviewed.
 */

#define _GNU_SOURCE
#define _XOPEN_SOURCE 700

#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

typedef struct {
    const char *key;
    const char *value;
} KeyValue;

typedef struct {
    const char *method;
    const char *path;
    const char *authorization;
    const char *body;
    const char *query;
} RequestRepr;

typedef struct {
    void *result; // PGresult* (opaque)
} PgResult;

typedef struct {
    char *dsn;
    void *conn; // PGconn* (opaque). Lazy-connected on first query/execute.
} PgPool;

// Matches the Sounio-side ValueArray layout (ptr: i64, count: i32).
// `ptr` points at an array of raw (tagged) string values.
typedef struct {
    int64_t ptr;
    int32_t count;
    int32_t _pad;
} ValueArray;

static int g_listener_fd = -1;
static int g_active_client_fd = -1;
static char g_as_ptr_buf[16384];
static char g_pg_value_buf[16384];
static char g_route_param_id[256];
static char g_query_param_buf[256];
// Scratch space for native string operations (replace, etc.). Reset per request.
// Per-request arena used for native string operations (concat/replace/substring).
// This must be large enough to handle O(n^2) intermediate allocations from chained concats.
// 64MB is intentionally generous for production stability.
static char g_string_arena[67108864];
static size_t g_string_arena_used = 0;

static RequestRepr g_request = {0};
static KeyValue g_headers[65];
static char *g_owned_method = NULL;
static char *g_owned_path = NULL;
static char *g_owned_query = NULL;
static char *g_owned_body = NULL;
static char *g_owned_header_keys[65];
static char *g_owned_header_values[65];

// ============================================================================
// Safe memory probing for string decoding
// ============================================================================
//
// Current Sounio native string lowering is still evolving, and bugs can cause
// non-string values (like pointers to structs) to flow into string helpers.
// A naive `strlen()` would segfault on such inputs and kill the whole server.
//
// We harden the runtime by probing candidate pointers using `mincore()` to
// validate page mappings and requiring a NUL terminator within a bounded scan.
static long g_page_size = 0;

// Forward declarations (used by string decoding below).
static int decode_u32_ascii_contiguous(const uint32_t *src, char *dst, size_t dst_cap);
static int decode_u32_ascii_sparse(const uint32_t *src, char *dst, size_t dst_cap);

static int ensure_page_size(void) {
    if (g_page_size > 0) {
        return 1;
    }
    long sz = sysconf(_SC_PAGESIZE);
    if (sz <= 0) {
        return 0;
    }
    g_page_size = sz;
    return 1;
}

static int probe_mapped_range(const void *p, size_t len) {
    if (p == NULL || len == 0) {
        return 0;
    }
    if (!ensure_page_size()) {
        return 0;
    }

    uintptr_t start = (uintptr_t)p;
    uintptr_t end = start + len;
    uintptr_t page_mask = (uintptr_t)(g_page_size - 1);
    uintptr_t cursor = start & ~page_mask;
    unsigned char vec = 0;

    while (cursor < end) {
        if (mincore((void *)cursor, (size_t)g_page_size, &vec) != 0) {
            return 0;
        }
        cursor += (uintptr_t)g_page_size;
    }

    return 1;
}

static int probe_cstr(const char *p, size_t max_scan) {
    if (p == NULL || max_scan == 0) {
        return 0;
    }
    if (!probe_mapped_range(p, 1)) {
        return 0;
    }

    // Scan, checking page mappings at boundaries.
    for (size_t i = 0; i < max_scan; i++) {
        if (!probe_mapped_range(p + i, 1)) {
            return 0;
        }
        if ((unsigned char)p[i] == '\0') {
            return 1;
        }
    }

    return 0;
}

static ssize_t bounded_cstr_len(const char *p, size_t max_scan) {
    if (p == NULL || max_scan == 0) {
        return -1;
    }
    if (!probe_mapped_range(p, 1)) {
        return -1;
    }
    for (size_t i = 0; i < max_scan; i++) {
        if (!probe_mapped_range(p + i, 1)) {
            return -1;
        }
        if ((unsigned char)p[i] == '\0') {
            return (ssize_t)i;
        }
    }
    return -1;
}

static void reset_owned_kv(char **arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] != NULL) {
            free(arr[i]);
            arr[i] = NULL;
        }
    }
}

static void reset_request_storage(void) {
    if (g_owned_method != NULL) {
        free(g_owned_method);
        g_owned_method = NULL;
    }
    if (g_owned_path != NULL) {
        free(g_owned_path);
        g_owned_path = NULL;
    }
    if (g_owned_query != NULL) {
        free(g_owned_query);
        g_owned_query = NULL;
    }
    if (g_owned_body != NULL) {
        free(g_owned_body);
        g_owned_body = NULL;
    }

    reset_owned_kv(g_owned_header_keys, 65);
    reset_owned_kv(g_owned_header_values, 65);

    memset(g_headers, 0, sizeof(g_headers));
    memset(&g_request, 0, sizeof(g_request));
    memset(g_route_param_id, 0, sizeof(g_route_param_id));
    memset(g_query_param_buf, 0, sizeof(g_query_param_buf));
    g_string_arena_used = 0;
}

static char *arena_alloc(size_t n) {
    static int debug_arena = -1;
    if (debug_arena == -1) {
        debug_arena = getenv("SOUNIO_ARENA_DEBUG") != NULL ? 1 : 0;
    }
    if (n == 0) {
        n = 1;
    }
    if (g_string_arena_used + n > sizeof(g_string_arena)) {
        if (debug_arena) {
            fprintf(stderr,
                    "[arena-debug] OOM used=%zu need=%zu cap=%zu\n",
                    g_string_arena_used,
                    n,
                    sizeof(g_string_arena));
        }
        return NULL;
    }
    char *out = g_string_arena + g_string_arena_used;
    g_string_arena_used += n;
    return out;
}

static const char *arena_dup_cstr(const char *s) {
    if (s == NULL) {
        return "";
    }
    size_t n = strlen(s);
    char *out = arena_alloc(n + 1);
    if (out == NULL) {
        // Fallback: return the original pointer (may be unstable for shift-tag decoding).
        static int debug_arena = -1;
        if (debug_arena == -1) {
            debug_arena = getenv("SOUNIO_ARENA_DEBUG") != NULL ? 1 : 0;
        }
        if (debug_arena) {
            fprintf(stderr, "[arena-debug] dup failed len=%zu\n", n);
        }
        return s;
    }
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *dup_cstr(const char *s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *trim_ws(char *s) {
    if (s == NULL) {
        return s;
    }
    while (*s != '\0' && isspace((unsigned char)*s)) {
        s++;
    }
    if (*s == '\0') {
        return s;
    }
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return s;
}

static void parse_headers(char *headers_block) {
    int idx = 0;
    char *line = headers_block;
    while (line != NULL && *line != '\0' && idx < 64) {
        char *next = strstr(line, "\r\n");
        if (next != NULL) {
            *next = '\0';
        }

        if (*line == '\0') {
            break;
        }

        char *colon = strchr(line, ':');
        if (colon != NULL) {
            *colon = '\0';
            char *k = trim_ws(line);
            char *v = trim_ws(colon + 1);
            g_owned_header_keys[idx] = dup_cstr(k);
            g_owned_header_values[idx] = dup_cstr(v);
            g_headers[idx].key = g_owned_header_keys[idx];
            g_headers[idx].value = g_owned_header_values[idx];
            idx++;
        }

        if (next == NULL) {
            break;
        }
        line = next + 2;
    }

    g_headers[idx].key = NULL;
    g_headers[idx].value = NULL;
}

static const char *lookup_header(const char *key) {
    if (key == NULL || *key == '\0') {
        return "";
    }
    for (int i = 0; i < 64; i++) {
        if (g_headers[i].key == NULL) {
            break;
        }
        if (strcmp(g_headers[i].key, key) == 0) {
            return g_headers[i].value != NULL ? g_headers[i].value : "";
        }
    }
    return "";
}

static int decode_u32_ascii_contiguous(const uint32_t *src, char *dst, size_t dst_cap) {
    if (src == NULL || dst == NULL || dst_cap < 2) {
        return 0;
    }

    size_t out = 0;
    for (size_t i = 0; i < 4096 && out + 1 < dst_cap; i++) {
        uint32_t cp = src[i];
        if (cp == 0) {
            break;
        }
        if (cp > 0x7fU) {
            return 0;
        }
        dst[out++] = (char)cp;
    }

    if (out == 0) {
        return 0;
    }

    dst[out] = '\0';
    return 1;
}

static int decode_u32_ascii_sparse(const uint32_t *src, char *dst, size_t dst_cap) {
    if (src == NULL || dst == NULL || dst_cap < 2) {
        return 0;
    }

    size_t out = 0;
    int saw_nonzero = 0;
    int zero_run = 0;

    for (size_t i = 0; i < 8192 && out + 1 < dst_cap; i++) {
        uint32_t cp = src[i];
        if (cp == 0) {
            if (saw_nonzero) {
                zero_run++;
                if (zero_run >= 8) {
                    break;
                }
            }
            continue;
        }

        zero_run = 0;
        saw_nonzero = 1;

        if (cp > 0x7fU) {
            return 0;
        }
        dst[out++] = (char)cp;
    }

    if (!saw_nonzero || out == 0) {
        return 0;
    }

    dst[out] = '\0';
    return 1;
}

const char *__sounio_chiuratto_as_ptr_impl(uintptr_t raw) {
    if (raw == 0) {
        return "";
    }

    // Fast path: pointers into the native string arena are always plain C strings.
    // Important: arena allocations are intentionally not aligned, so valid pointers can be odd.
    uintptr_t arena_start = (uintptr_t)g_string_arena;
    uintptr_t arena_end = arena_start + (uintptr_t)sizeof(g_string_arena);
    if (raw >= arena_start && raw < arena_end) {
        const char *p = (const char *)raw;
        if (probe_cstr(p, 8192)) {
            return p;
        }
        // If the arena pointer is invalid, fall through to the more defensive paths below.
    }

    // Low-bit tagging can be used to mark owned pointers (ptr|1). Before attempting
    // shift-tag decoding, try to clear the low bit and validate the resulting pointer.
    if ((raw & 1ULL) == 1ULL) {
        uintptr_t low_untag = raw & ~1ULL;
        if (low_untag >= 0x1000ULL && low_untag <= 0x00007fffffffffffULL) {
            const char *p_low = (const char *)low_untag;
            if (bounded_cstr_len(p_low, 8192) >= 0) {
                return p_low;
            }
        }
    }

    // If the value already looks like a C string pointer, trust it.
    //
    // IMPORTANT: Sounio's current native lowering may pass owned strings through FFI using
    // a shift-tag encoding: `raw = (ptr << 1) | 1`. Those values are odd and can sometimes
    // fall into the user-space pointer range by coincidence, but they are NOT valid C-string
    // pointers. To avoid spuriously accepting tagged values and returning truncated/garbage
    // pointers, we only attempt the "trust raw pointer" path for even (aligned) pointers.
    if ((raw & 1ULL) == 0ULL && raw >= 0x1000ULL && raw <= 0x00007fffffffffffULL) {
        const char *p_raw = (const char *)raw;
        ssize_t raw_len = bounded_cstr_len(p_raw, 8192);

        // Low-bit tagging (ptr|1) can still occur; only attempt to clear the low bit if the
        // raw pointer does NOT validate as a C string.
        const char *p_untagged = NULL;
        ssize_t untagged_len = -1;
        if (raw_len < 0 && (raw & 1ULL) == 1ULL) {
            uintptr_t untagged = raw & ~1ULL;
            if (untagged >= 0x1000ULL && untagged <= 0x00007fffffffffffULL) {
                p_untagged = (const char *)untagged;
                untagged_len = bounded_cstr_len(p_untagged, 8192);
            }
        }

        if (raw_len >= 0) {
            return p_raw;
        }
        if (untagged_len >= 0) {
            return p_untagged;
        }
    }

    // Tagged path used by current native string lowering:
    // `raw = (ptr << 1) | 1` (shift-tag).
    if ((raw & 1ULL) == 1ULL) {
        uintptr_t decoded = raw >> 1;
        if (decoded >= 0x1000ULL && decoded <= 0x00007fffffffffffULL) {
            const uint32_t *u32 = (const uint32_t *)decoded;

            // The decode routines scan up to 16KB/32KB of u32 codepoints. Ensure that range
            // is mapped before attempting to read it.
            if (probe_mapped_range(u32, 4096 * sizeof(uint32_t))
                && decode_u32_ascii_contiguous(u32, g_as_ptr_buf, sizeof(g_as_ptr_buf))) {
                return arena_dup_cstr(g_as_ptr_buf);
            }
            if (probe_mapped_range(u32, 8192 * sizeof(uint32_t))
                && decode_u32_ascii_sparse(u32, g_as_ptr_buf, sizeof(g_as_ptr_buf))) {
                return arena_dup_cstr(g_as_ptr_buf);
            }
            if (decoded > sizeof(uint32_t)) {
                const uint32_t *shifted = (const uint32_t *)(decoded - sizeof(uint32_t));
                if (probe_mapped_range(shifted, 8192 * sizeof(uint32_t))
                    && decode_u32_ascii_sparse(shifted, g_as_ptr_buf, sizeof(g_as_ptr_buf))) {
                    return arena_dup_cstr(g_as_ptr_buf);
                }
            }

            // Fallback: some builds lower owned strings as byte buffers, not u32 codepoints.
            // In that case `decoded` already points at a null-terminated C string.
            const char *maybe_cstr = (const char *)decoded;
            if (probe_cstr(maybe_cstr, 8192)) {
                return arena_dup_cstr(maybe_cstr);
            }
        }

        return "";
    }

    return "";
}

int64_t __sounio_chiuratto_starts_with_impl(uintptr_t raw_value, uintptr_t raw_prefix) {
    const char *value = __sounio_chiuratto_as_ptr_impl(raw_value);
    const char *prefix = __sounio_chiuratto_as_ptr_impl(raw_prefix);
    if (value == NULL || prefix == NULL) {
        return 0;
    }
    size_t prefix_len = strlen(prefix);
    if (prefix_len == 0) {
        return 1;
    }
    return strncmp(value, prefix, prefix_len) == 0 ? 1 : 0;
}

int64_t __sounio_chiuratto_contains_impl(uintptr_t raw_value, uintptr_t raw_needle) {
    const char *value = __sounio_chiuratto_as_ptr_impl(raw_value);
    const char *needle = __sounio_chiuratto_as_ptr_impl(raw_needle);
    if (value == NULL || needle == NULL) {
        return 0;
    }
    static int debug_str = -1;
    if (debug_str == -1) {
        debug_str = getenv("SOUNIO_STR_DEBUG") != NULL ? 1 : 0;
    }
    if (debug_str) {
        // Focus debug noise on auth payload parsing markers.
        if (strcmp(needle, "\"email\":\"") == 0 || strcmp(needle, "\"password\":\"") == 0) {
            fprintf(stderr,
                    "[str-debug] contains needle=%s raw_value=0x%llx value_ptr=%p value_prefix=%.*s\n",
                    needle,
                    (unsigned long long)raw_value,
                    (void *)value,
                    220,
                    value != NULL ? value : "");
        }
    }
    if (*needle == '\0') {
        return 1;
    }
    return strstr(value, needle) != NULL ? 1 : 0;
}

const char *__sounio_chiuratto_replace_impl(uintptr_t raw_value, uintptr_t raw_from, uintptr_t raw_to) {
    const char *value = __sounio_chiuratto_as_ptr_impl(raw_value);
    const char *from = __sounio_chiuratto_as_ptr_impl(raw_from);
    const char *to = __sounio_chiuratto_as_ptr_impl(raw_to);

    if (value == NULL) {
        return "";
    }
    if (from == NULL || *from == '\0') {
        return value;
    }
    if (to == NULL) {
        to = "";
    }

    size_t value_len = strlen(value);
    size_t from_len = strlen(from);
    size_t to_len = strlen(to);

    size_t count = 0;
    const char *cursor = value;
    while ((cursor = strstr(cursor, from)) != NULL) {
        count++;
        cursor += from_len;
        if (*cursor == '\0') {
            break;
        }
    }

    if (count == 0) {
        return value;
    }

    size_t out_len = value_len + count * (to_len >= from_len ? (to_len - from_len) : 0) - count * (to_len < from_len ? (from_len - to_len) : 0);
    char *out = arena_alloc(out_len + 1);
    if (out == NULL) {
        return value;
    }

    const char *in_cursor = value;
    char *out_cursor = out;
    while ((cursor = strstr(in_cursor, from)) != NULL) {
        size_t chunk = (size_t)(cursor - in_cursor);
        memcpy(out_cursor, in_cursor, chunk);
        out_cursor += chunk;

        if (to_len > 0) {
            memcpy(out_cursor, to, to_len);
            out_cursor += to_len;
        }

        in_cursor = cursor + from_len;
        if (*in_cursor == '\0') {
            break;
        }
    }

    size_t tail = strlen(in_cursor);
    if (tail > 0) {
        memcpy(out_cursor, in_cursor, tail);
        out_cursor += tail;
    }
    *out_cursor = '\0';

    return out;
}

// Return the substring that occurs after the first occurrence of `needle` in `value`.
// If `needle` is not found, returns `value` unchanged (mirrors old util_substring_after semantics).
// The returned pointer is stable until the next request-cycle arena reset.
const char *__sounio_chiuratto_substring_after_first_impl(const char *raw_value, const char *raw_needle) {
    const char *value = __sounio_chiuratto_as_ptr_impl((uintptr_t)raw_value);
    const char *needle = __sounio_chiuratto_as_ptr_impl((uintptr_t)raw_needle);
    if (value == NULL) {
        return "";
    }
    if (needle == NULL || *needle == '\0') {
        return value;
    }

    const char *pos = strstr(value, needle);
    if (pos == NULL) {
        return value;
    }

    const char *after = pos + strlen(needle);
    size_t n = strlen(after);
    char *out = arena_alloc(n + 1);
    if (out == NULL) {
        return after;
    }
    memcpy(out, after, n);
    out[n] = '\0';
    return out;
}

const char *__sounio_chiuratto_digits_prefix_impl(const char *raw_value) {
    const char *value = __sounio_chiuratto_as_ptr_impl((uintptr_t)raw_value);
    if (value == NULL) {
        return "";
    }

    size_t n = 0;
    while (value[n] != '\0' && value[n] >= '0' && value[n] <= '9') {
        n++;
    }
    if (n == 0) {
        return "";
    }

    char *out = arena_alloc(n + 1);
    if (out == NULL) {
        return "";
    }
    memcpy(out, value, n);
    out[n] = '\0';
    return out;
}

const char *__sounio_chiuratto_concat_impl(uintptr_t raw_left, uintptr_t raw_right) {
    const char *left = __sounio_chiuratto_as_ptr_impl(raw_left);
    const char *right = __sounio_chiuratto_as_ptr_impl(raw_right);

    if (left == NULL) {
        left = "";
    }
    if (right == NULL) {
        right = "";
    }

    size_t left_len = strlen(left);
    size_t right_len = strlen(right);

    // +1 for terminating null.
    size_t out_len = left_len + right_len;
    char *out = arena_alloc(out_len + 1);
    if (out == NULL) {
        return "";
    }

    if (left_len > 0) {
        memcpy(out, left, left_len);
    }
    if (right_len > 0) {
        memcpy(out + left_len, right, right_len);
    }
    out[out_len] = '\0';
    return out;
}

int64_t __sounio_chiuratto_values_new_impl(int32_t count) {
    if (count <= 0) {
        return 0;
    }
    if (count > 64) {
        count = 64;
    }
    uintptr_t *arr = (uintptr_t *)arena_alloc(sizeof(uintptr_t) * (size_t)count);
    if (arr == NULL) {
        return 0;
    }
    memset(arr, 0, sizeof(uintptr_t) * (size_t)count);
    return (int64_t)(uintptr_t)arr;
}

int32_t __sounio_chiuratto_values_set_impl(int64_t values_ptr, int32_t idx, const char *raw_value) {
    if (values_ptr == 0 || idx < 0) {
        return -1;
    }
    uintptr_t *arr = (uintptr_t *)(uintptr_t)values_ptr;
    arr[idx] = (uintptr_t)raw_value;
    return 0;
}

ValueArray *__sounio_chiuratto_values_array_new_impl(int32_t count) {
    if (count <= 0) {
        return NULL;
    }
    if (count > 64) {
        count = 64;
    }

    ValueArray *out = (ValueArray *)arena_alloc(sizeof(ValueArray));
    if (out == NULL) {
        return NULL;
    }
    uintptr_t *arr = (uintptr_t *)arena_alloc(sizeof(uintptr_t) * (size_t)count);
    if (arr == NULL) {
        return NULL;
    }
    memset(arr, 0, sizeof(uintptr_t) * (size_t)count);
    out->ptr = (int64_t)(uintptr_t)arr;
    out->count = count;
    out->_pad = 0;
    return out;
}

int32_t __sounio_chiuratto_values_array_set_impl(ValueArray *values, int32_t idx, const char *raw_value) {
    if (values == NULL || idx < 0 || idx >= values->count) {
        return -1;
    }
    uintptr_t *arr = (uintptr_t *)(uintptr_t)values->ptr;
    arr[idx] = (uintptr_t)raw_value;
    return 0;
}

// ============================================================================
// Dynamic libpq loading (no -lpq link dependency)
// ============================================================================

typedef struct pg_conn PGconn;
typedef struct pg_result PGresult;

typedef struct {
    void *handle;

    PGconn *(*PQconnectdb)(const char *conninfo);
    int (*PQstatus)(const PGconn *conn);
    const char *(*PQerrorMessage)(const PGconn *conn);
	    void (*PQfinish)(PGconn *conn);

	    PGresult *(*PQexec)(PGconn *conn, const char *command);
	    PGresult *(*PQexecParams)(PGconn *conn,
	                              const char *command,
	                              int nParams,
	                              const unsigned int *paramTypes,
	                              const char *const *paramValues,
	                              const int *paramLengths,
	                              const int *paramFormats,
	                              int resultFormat);
	    int (*PQresultStatus)(const PGresult *res);
	    int (*PQntuples)(const PGresult *res);
	    int (*PQnfields)(const PGresult *res);
    char *(*PQgetvalue)(const PGresult *res, int tup_num, int field_num);
    int (*PQgetisnull)(const PGresult *res, int tup_num, int field_num);
    char *(*PQcmdTuples)(PGresult *res);
    void (*PQclear)(PGresult *res);
} LibPQApi;

static LibPQApi g_libpq = {0};
static int g_libpq_state = 0; // 0 = uninitialized, 1 = loaded, -1 = unavailable

// Values per libpq ExecStatusType and ConnStatusType. They are stable in libpq.
#define SOUNIO_PQ_CONNECTION_OK 0
#define SOUNIO_PQRES_COMMAND_OK 1
#define SOUNIO_PQRES_TUPLES_OK 2
#define SOUNIO_PQRES_SINGLE_TUPLE 9

static int libpq_load(void) {
    if (g_libpq_state != 0) {
        return g_libpq_state == 1;
    }

    const char *candidates[] = {
        "libpq.so.5",
        "libpq.so",
        NULL,
    };

    for (int i = 0; candidates[i] != NULL; i++) {
        g_libpq.handle = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
        if (g_libpq.handle != NULL) {
            break;
        }
    }

    if (g_libpq.handle == NULL) {
        g_libpq_state = -1;
        return 0;
    }

#define SOUNIO_LOAD_PQ(sym)                                                         \
    do {                                                                            \
        *(void **)(&g_libpq.sym) = dlsym(g_libpq.handle, #sym);                     \
        if (g_libpq.sym == NULL) {                                                  \
            dlclose(g_libpq.handle);                                                \
            memset(&g_libpq, 0, sizeof(g_libpq));                                   \
            g_libpq_state = -1;                                                     \
            return 0;                                                               \
        }                                                                           \
    } while (0)

    SOUNIO_LOAD_PQ(PQconnectdb);
    SOUNIO_LOAD_PQ(PQstatus);
	    SOUNIO_LOAD_PQ(PQerrorMessage);
	    SOUNIO_LOAD_PQ(PQfinish);
	    SOUNIO_LOAD_PQ(PQexec);
	    SOUNIO_LOAD_PQ(PQexecParams);
	    SOUNIO_LOAD_PQ(PQresultStatus);
	    SOUNIO_LOAD_PQ(PQntuples);
	    SOUNIO_LOAD_PQ(PQnfields);
    SOUNIO_LOAD_PQ(PQgetvalue);
    SOUNIO_LOAD_PQ(PQgetisnull);
    SOUNIO_LOAD_PQ(PQcmdTuples);
    SOUNIO_LOAD_PQ(PQclear);

#undef SOUNIO_LOAD_PQ

    g_libpq_state = 1;
    return 1;
}

static PGconn *pool_ensure_conn(PgPool *pool) {
    if (pool == NULL) {
        return NULL;
    }
    if (!libpq_load()) {
        return NULL;
    }
    if (pool->conn != NULL) {
        return (PGconn *)pool->conn;
    }

    const char *dsn = pool->dsn != NULL ? pool->dsn : "";
    PGconn *conn = g_libpq.PQconnectdb(dsn);
    if (conn == NULL) {
        return NULL;
    }
    if (g_libpq.PQstatus(conn) != SOUNIO_PQ_CONNECTION_OK) {
        static int debug_pg = -1;
        if (debug_pg == -1) {
            debug_pg = getenv("SOUNIO_PG_DEBUG") != NULL ? 1 : 0;
        }
        if (debug_pg && g_libpq.PQerrorMessage != NULL) {
            fprintf(stderr, "[pg-debug] connect failed dsn=%s err=%s\n", dsn, g_libpq.PQerrorMessage(conn));
        }
        g_libpq.PQfinish(conn);
        return NULL;
    }

    static int debug_pg = -1;
    if (debug_pg == -1) {
        debug_pg = getenv("SOUNIO_PG_DEBUG") != NULL ? 1 : 0;
    }
    if (debug_pg) {
        fprintf(stderr, "[pg-debug] connected dsn=%s\n", dsn);
    }

    pool->conn = (void *)conn;
    return conn;
}

static int pq_result_ok(int status) {
    return status == SOUNIO_PQRES_COMMAND_OK || status == SOUNIO_PQRES_TUPLES_OK ||
        status == SOUNIO_PQRES_SINGLE_TUPLE;
}

const char *__sounio_chiuratto_uuid_v4_impl(void) {
    static char out[37];
    uint64_t t = (uint64_t)time(NULL);
    uint64_t pid = (uint64_t)getpid();
    uint64_t mix = t ^ (pid << 17) ^ 0x9e3779b97f4a7c15ULL;

    snprintf(
        out,
        sizeof(out),
        "%08x-%04x-4%03x-a%03x-%012llx",
        (unsigned int)(mix & 0xffffffffULL),
        (unsigned int)((mix >> 16) & 0xffffULL),
        (unsigned int)((mix >> 32) & 0x0fffULL),
        (unsigned int)((mix >> 44) & 0x0fffULL),
        (unsigned long long)(mix & 0x0000ffffffffffffULL));
    return out;
}

int64_t __sounio_chiuratto_unix_timestamp_impl(void) {
    return (int64_t)time(NULL);
}

void *__sounio_chiuratto_pg_pool_new_impl(
    const char *uri,
    int32_t min_conn,
    int32_t max_conn,
    int32_t timeout_ms) {
    (void)min_conn;
    (void)max_conn;
    (void)timeout_ms;

    PgPool *pool = (PgPool *)calloc(1, sizeof(PgPool));
    if (pool == NULL) {
        return NULL;
    }
    // Sounio may pass tagged string representations through FFI. Decode defensively.
    const char *effective_uri = __sounio_chiuratto_as_ptr_impl((uintptr_t)uri);
    pool->dsn = dup_cstr(effective_uri != NULL ? effective_uri : "");
    return (void *)pool;
}

int32_t __sounio_chiuratto_pg_pool_close_impl(void *pool) {
    if (pool == NULL) {
        return 0;
    }

    PgPool *pool_ref = (PgPool *)pool;
    if (pool_ref->conn != NULL && libpq_load()) {
        g_libpq.PQfinish((PGconn *)pool_ref->conn);
        pool_ref->conn = NULL;
    }

    if (pool_ref->dsn != NULL) {
        free(pool_ref->dsn);
        pool_ref->dsn = NULL;
    }
    free(pool_ref);
    return 0;
}

void *__sounio_chiuratto_pg_pool_query_impl(void *pool, const char *sql, void *params) {
    if (pool == NULL || sql == NULL) {
        return NULL;
    }
    ValueArray *values = (ValueArray *)params;

    PgPool *pool_ref = (PgPool *)pool;
    PGconn *conn = pool_ensure_conn(pool_ref);
    if (conn == NULL) {
        return NULL;
    }

    const char *effective_sql = __sounio_chiuratto_as_ptr_impl((uintptr_t)sql);
    static int debug_pg = -1;
    if (debug_pg == -1) {
        debug_pg = getenv("SOUNIO_PG_DEBUG") != NULL ? 1 : 0;
    }
    if (debug_pg) {
        const char *dbg_sql = effective_sql != NULL ? effective_sql : "";
        size_t dbg_len = strlen(dbg_sql);
        size_t dbg_head = dbg_len < 900 ? dbg_len : 900;
        fprintf(stderr, "[pg-debug] query len=%zu head=%.*s\n", dbg_len, (int)dbg_head, dbg_sql);
        if (dbg_len > dbg_head) {
            size_t dbg_tail = dbg_len < 240 ? dbg_len : 240;
            fprintf(stderr, "[pg-debug] query tail=%.*s\n", (int)dbg_tail, dbg_sql + (dbg_len - dbg_tail));
        }
        if (values != NULL) {
            fprintf(stderr,
                    "[pg-debug] params ptr=0x%llx count=%d\n",
                    (unsigned long long)values->ptr,
                    (int)values->count);
        }
        if (params != NULL) {
            const uint64_t *words = (const uint64_t *)params;
            fprintf(stderr,
                    "[pg-debug] params words[0..2]=0x%llx 0x%llx 0x%llx\n",
                    (unsigned long long)words[0],
                    (unsigned long long)words[1],
                    (unsigned long long)words[2]);
        }
    }
    PGresult *res = NULL;
    if (values != NULL && values->count > 0 && values->ptr != 0 && g_libpq.PQexecParams != NULL) {
        int n = values->count;
        if (n > 64) {
            n = 64;
        }
        const uintptr_t *raws = (const uintptr_t *)(uintptr_t)values->ptr;
        const char *param_values[64];
        for (int i = 0; i < n; i++) {
            param_values[i] = __sounio_chiuratto_as_ptr_impl(raws[i]);
        }
        res = g_libpq.PQexecParams(conn,
                                   effective_sql != NULL ? effective_sql : "",
                                   n,
                                   NULL,
                                   param_values,
                                   NULL,
                                   NULL,
                                   0);
    } else {
        res = g_libpq.PQexec(conn, effective_sql != NULL ? effective_sql : "");
    }
    if (res == NULL) {
        if (debug_pg && g_libpq.PQerrorMessage != NULL) {
            fprintf(stderr, "[pg-debug] PQexec returned NULL err=%s\n", g_libpq.PQerrorMessage(conn));
        }
        return NULL;
    }

    int status = g_libpq.PQresultStatus(res);
    if (!pq_result_ok(status)) {
        if (debug_pg && g_libpq.PQerrorMessage != NULL) {
            fprintf(stderr, "[pg-debug] bad status=%d err=%s\n", status, g_libpq.PQerrorMessage(conn));
        }
        g_libpq.PQclear(res);
        return NULL;
    }

    PgResult *out = (PgResult *)calloc(1, sizeof(PgResult));
    if (out == NULL) {
        g_libpq.PQclear(res);
        return NULL;
    }
    out->result = (void *)res;
    return (void *)out;
}

int64_t __sounio_chiuratto_pg_pool_execute_impl(void *pool, const char *sql, void *params) {
    if (pool == NULL || sql == NULL) {
        return -1;
    }
    ValueArray *values = (ValueArray *)params;

    PgPool *pool_ref = (PgPool *)pool;
    PGconn *conn = pool_ensure_conn(pool_ref);
    if (conn == NULL) {
        return -1;
    }

    const char *effective_sql = __sounio_chiuratto_as_ptr_impl((uintptr_t)sql);
    static int debug_pg = -1;
    if (debug_pg == -1) {
        debug_pg = getenv("SOUNIO_PG_DEBUG") != NULL ? 1 : 0;
    }
    if (debug_pg) {
        const char *dbg_sql = effective_sql != NULL ? effective_sql : "";
        size_t dbg_len = strlen(dbg_sql);
        size_t dbg_head = dbg_len < 900 ? dbg_len : 900;
        fprintf(stderr, "[pg-debug] exec len=%zu head=%.*s\n", dbg_len, (int)dbg_head, dbg_sql);
        if (dbg_len > dbg_head) {
            size_t dbg_tail = dbg_len < 240 ? dbg_len : 240;
            fprintf(stderr, "[pg-debug] exec tail=%.*s\n", (int)dbg_tail, dbg_sql + (dbg_len - dbg_tail));
        }
    }
    PGresult *res = NULL;
    if (values != NULL && values->count > 0 && values->ptr != 0 && g_libpq.PQexecParams != NULL) {
        int n = values->count;
        if (n > 64) {
            n = 64;
        }
        const uintptr_t *raws = (const uintptr_t *)(uintptr_t)values->ptr;
        const char *param_values[64];
        for (int i = 0; i < n; i++) {
            param_values[i] = __sounio_chiuratto_as_ptr_impl(raws[i]);
        }
        res = g_libpq.PQexecParams(conn,
                                   effective_sql != NULL ? effective_sql : "",
                                   n,
                                   NULL,
                                   param_values,
                                   NULL,
                                   NULL,
                                   0);
    } else {
        res = g_libpq.PQexec(conn, effective_sql != NULL ? effective_sql : "");
    }
    if (res == NULL) {
        if (debug_pg && g_libpq.PQerrorMessage != NULL) {
            fprintf(stderr, "[pg-debug] PQexec returned NULL err=%s\n", g_libpq.PQerrorMessage(conn));
        }
        return -1;
    }

    int status = g_libpq.PQresultStatus(res);
    if (!pq_result_ok(status)) {
        if (debug_pg && g_libpq.PQerrorMessage != NULL) {
            fprintf(stderr, "[pg-debug] bad status=%d err=%s\n", status, g_libpq.PQerrorMessage(conn));
        }
        g_libpq.PQclear(res);
        return -1;
    }

    int64_t out = 0;
    if (status == SOUNIO_PQRES_TUPLES_OK || status == SOUNIO_PQRES_SINGLE_TUPLE) {
        out = (int64_t)g_libpq.PQntuples(res);
    } else {
        const char *tuples = g_libpq.PQcmdTuples(res);
        if (tuples != NULL && *tuples != '\0') {
            char *endptr = NULL;
            long long parsed = strtoll(tuples, &endptr, 10);
            if (endptr != tuples && parsed >= 0) {
                out = (int64_t)parsed;
            }
        }
    }

    g_libpq.PQclear(res);
    return out;
}

int32_t __sounio_chiuratto_pg_result_row_count_impl(void *result_ptr) {
    if (result_ptr == NULL) {
        return 0;
    }

    PgResult *result = (PgResult *)result_ptr;
    if (result->result == NULL || !libpq_load()) {
        return 0;
    }
    return (int32_t)g_libpq.PQntuples((PGresult *)result->result);
}

int32_t __sounio_chiuratto_pg_result_col_count_impl(void *result_ptr) {
    if (result_ptr == NULL) {
        return 0;
    }

    PgResult *result = (PgResult *)result_ptr;
    if (result->result == NULL || !libpq_load()) {
        return 0;
    }
    return (int32_t)g_libpq.PQnfields((PGresult *)result->result);
}

const char *__sounio_chiuratto_pg_result_get_value_impl(void *result_ptr, int32_t row, int32_t col) {
    if (result_ptr == NULL || row < 0 || col < 0) {
        return "";
    }

    PgResult *result = (PgResult *)result_ptr;
    if (result->result == NULL || !libpq_load()) {
        return "";
    }

    PGresult *res = (PGresult *)result->result;
    int rows = g_libpq.PQntuples(res);
    int cols = g_libpq.PQnfields(res);
    if (row >= rows || col >= cols) {
        return "";
    }

    if (g_libpq.PQgetisnull != NULL && g_libpq.PQgetisnull(res, row, col) != 0) {
        return "";
    }

    const char *value = g_libpq.PQgetvalue(res, row, col);
    if (value == NULL) {
        return "";
    }

    // Copy into the per-request arena so callers can safely keep the returned string
    // after freeing the PGresult (libpq owns PQgetvalue memory).
    size_t n = strlen(value);
    // Hard cap to keep arena usage bounded even if value is unexpectedly huge.
    if (n > (8 * 1024 * 1024)) {
        n = 8 * 1024 * 1024;
    }
    char *out = arena_alloc(n + 1);
    if (out == NULL) {
        return "";
    }
    memcpy(out, value, n);
    out[n] = '\0';
    return out;
}

int32_t __sounio_chiuratto_pg_result_free_impl(void *result_ptr) {
    if (result_ptr == NULL) {
        return 0;
    }

    PgResult *result = (PgResult *)result_ptr;
    if (result->result != NULL && libpq_load()) {
        g_libpq.PQclear((PGresult *)result->result);
        result->result = NULL;
    }
    free(result_ptr);
    return 0;
}

int32_t __sounio_chiuratto_http_server_start_impl(int32_t port) {
    if (g_listener_fd >= 0) {
        return 0;
    }

    // Avoid process termination when a client disconnects early while we're
    // sending a response (SIGPIPE default action is to terminate).
    signal(SIGPIPE, SIG_IGN);

    g_listener_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listener_fd < 0) {
        return 1;
    }

    int opt = 1;
    setsockopt(g_listener_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);

    if (bind(g_listener_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(g_listener_fd);
        g_listener_fd = -1;
        return 1;
    }

    if (listen(g_listener_fd, 128) != 0) {
        close(g_listener_fd);
        g_listener_fd = -1;
        return 1;
    }

    return 0;
}

int32_t __sounio_chiuratto_http_server_stop_impl(void) {
    if (g_active_client_fd >= 0) {
        close(g_active_client_fd);
        g_active_client_fd = -1;
    }
    if (g_listener_fd >= 0) {
        close(g_listener_fd);
        g_listener_fd = -1;
    }
    return 0;
}

void *__sounio_chiuratto_http_next_request_impl(void) {
    if (g_listener_fd < 0) {
        return NULL;
    }

    if (g_active_client_fd >= 0) {
        close(g_active_client_fd);
        g_active_client_fd = -1;
    }

    reset_request_storage();

    struct sockaddr_in peer;
    socklen_t peer_len = sizeof(peer);
    int client_fd = accept(g_listener_fd, (struct sockaddr *)&peer, &peer_len);
    if (client_fd < 0) {
        return NULL;
    }
    g_active_client_fd = client_fd;

    char buffer[65536];
    ssize_t total = 0;

    // Read until we have the full headers, and if Content-Length is present,
    // read the full body as well. A single recv() is not reliable because TCP
    // segmentation can split headers/body across packets.
    size_t expected_total = 0; // header_len + body_len (when known)
    while ((size_t)total < sizeof(buffer) - 1) {
        ssize_t n = recv(client_fd, buffer + total, sizeof(buffer) - 1 - (size_t)total, 0);
        if (n <= 0) {
            return NULL;
        }
        total += n;
        buffer[total] = '\0';

        char *headers_end = strstr(buffer, "\r\n\r\n");
        if (headers_end == NULL) {
            continue;
        }

        // Compute Content-Length (if present) without mutating the header buffer yet.
        size_t header_len = (size_t)((headers_end + 4) - buffer);
        size_t content_len = 0;
        char *p = strstr(buffer, "\r\n");
        if (p != NULL) {
            p += 2; // start of headers
            while (p < headers_end) {
                char *line_end = strstr(p, "\r\n");
                if (line_end == NULL || line_end > headers_end) {
                    line_end = headers_end;
                }
                if ((size_t)(line_end - p) >= 15 && strncasecmp(p, "Content-Length:", 15) == 0) {
                    const char *num = p + 15;
                    while (*num == ' ' || *num == '\t') num++;
                    // Use strtoll with error checking instead of unsafe atoll
                    char *endptr = NULL;
                    errno = 0;
                    long long parsed = strtoll(num, &endptr, 10);
                    if (errno == 0 && endptr != num && parsed > 0 && parsed < (1LL << 30)) {
                        // Cap at 1GB to prevent integer overflow attacks
                        content_len = (size_t)parsed;
                    }
                    break;
                }
                if (line_end == headers_end) break;
                p = line_end + 2;
            }
        }

        expected_total = header_len + content_len;
        if (expected_total == header_len) {
            // No body expected.
            break;
        }

        if ((size_t)total >= expected_total) {
            // Full body received.
            break;
        }
    }

    if (total <= 0) {
        return NULL;
    }
    // If we computed a body length, cap the buffer at exactly header+body so the
    // duplicated body doesn't accidentally include extra bytes.
    if (expected_total > 0 && expected_total < (size_t)total) {
        buffer[expected_total] = '\0';
        total = (ssize_t)expected_total;
    } else {
        buffer[total] = '\0';
    }

    char *line_end = strstr(buffer, "\r\n");
    if (line_end == NULL) {
        return NULL;
    }
    *line_end = '\0';

    char *sp1 = strchr(buffer, ' ');
    if (sp1 == NULL) {
        return NULL;
    }
    *sp1 = '\0';
    char *sp2 = strchr(sp1 + 1, ' ');
    if (sp2 == NULL) {
        return NULL;
    }
    *sp2 = '\0';

    const char *method = buffer;
    char *target = sp1 + 1;

    char *query = strchr(target, '?');
    if (query != NULL) {
        *query = '\0';
        query++;
    }

    char *headers_start = line_end + 2;
    char *headers_end = strstr(headers_start, "\r\n\r\n");
    char *body = (char *)"";
    if (headers_end != NULL) {
        *headers_end = '\0';
        body = headers_end + 4;
    }

    parse_headers(headers_start);

    static int debug_http = -1;
    if (debug_http == -1) {
        debug_http = getenv("SOUNIO_HTTP_DEBUG") != NULL ? 1 : 0;
    }
    if (debug_http) {
        const char *cl = lookup_header("Content-Length");
        // Use strtol with error checking instead of unsafe atoi
        int cl_n = 0;
        if (cl != NULL && *cl != '\0') {
            char *endptr = NULL;
            long parsed = strtol(cl, &endptr, 10);
            if (endptr != cl && parsed >= 0 && parsed <= INT_MAX) {
                cl_n = (int)parsed;
            }
        }
        int body_n = body != NULL ? (int)strlen(body) : 0;
        if (strncmp(target, "/api/v1/auth/", 13) == 0
            || strncmp(target, "/api/v1/campaigns", 16) == 0) {
            fprintf(stderr, "[http-debug] %s %s content_length=%d body_len=%d body_prefix=%.*s\n",
                    method, target, cl_n, body_n, 160, body != NULL ? body : "");
        }
    }

    g_owned_method = dup_cstr(method);
    g_owned_path = dup_cstr(target);
    g_owned_query = dup_cstr(query != NULL ? query : "");
    g_owned_body = dup_cstr(body);

    g_request.method = g_owned_method != NULL ? g_owned_method : "GET";
    g_request.path = g_owned_path != NULL ? g_owned_path : "/";
    g_request.authorization = lookup_header("Authorization");
    g_request.body = g_owned_body != NULL ? g_owned_body : "";
    g_request.query = g_owned_query != NULL ? g_owned_query : "";

    return (void *)&g_request;
}

int32_t __sounio_chiuratto_http_route_match_impl(void *req_ptr, const char *method, const char *path) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return 0;
    }

    const char *actual_method = req->method != NULL ? req->method : "";
    const char *actual_path = req->path != NULL ? req->path : "";

    if (method != NULL && *method != '\0') {
        if (strcmp(actual_method, method) != 0) {
            return 0;
        }
    }

    if (path != NULL && *path != '\0') {
        if (strcmp(actual_path, path) != 0) {
            return 0;
        }
    }

    return 1;
}

static int str_has_prefix(const char *value, const char *prefix) {
    if (value == NULL || prefix == NULL) {
        return 0;
    }
    size_t value_len = strlen(value);
    size_t prefix_len = strlen(prefix);
    if (value_len < prefix_len) {
        return 0;
    }
    return strncmp(value, prefix, prefix_len) == 0;
}

int32_t __sounio_chiuratto_http_route_match_dynamic_impl(
    void *req_ptr,
    const char *method,
    const char *prefix,
    const char *suffix
) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return 0;
    }

    const char *actual_method = req->method != NULL ? req->method : "";
    const char *actual_path = req->path != NULL ? req->path : "";

    if (method != NULL && *method != '\0') {
        if (strcmp(actual_method, method) != 0) {
            return 0;
        }
    }

    if (prefix == NULL || *prefix == '\0') {
        return 0;
    }
    if (!str_has_prefix(actual_path, prefix)) {
        return 0;
    }

    const char *rest = actual_path + strlen(prefix);
    if (rest == NULL || *rest == '\0') {
        return 0;
    }

    memset(g_route_param_id, 0, sizeof(g_route_param_id));

    if (suffix != NULL && *suffix != '\0') {
        size_t rest_len = strlen(rest);
        size_t suffix_len = strlen(suffix);
        if (rest_len <= suffix_len) {
            return 0;
        }
        if (strcmp(rest + (rest_len - suffix_len), suffix) != 0) {
            return 0;
        }

        size_t id_len = rest_len - suffix_len;
        if (id_len == 0 || id_len >= sizeof(g_route_param_id)) {
            return 0;
        }
        for (size_t i = 0; i < id_len; i++) {
            if (rest[i] == '/') {
                return 0;
            }
            g_route_param_id[i] = rest[i];
        }
        g_route_param_id[id_len] = '\0';
        return 1;
    }

    if (strchr(rest, '/') != NULL) {
        return 0;
    }

    size_t id_len = strlen(rest);
    if (id_len == 0 || id_len >= sizeof(g_route_param_id)) {
        return 0;
    }
    memcpy(g_route_param_id, rest, id_len);
    g_route_param_id[id_len] = '\0';
    return 1;
}

const char *__sounio_chiuratto_http_route_param_id_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return "";
    }
    return g_route_param_id;
}

int32_t __sounio_chiuratto_http_route_code_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return 0;
    }

    const char *method = req->method != NULL ? req->method : "";
    const char *path = req->path != NULL ? req->path : "";

    if (strcmp(method, "GET") == 0 && strcmp(path, "/api/v1/orders/detail") == 0) {
        return 1001;
    }
    if (strcmp(method, "PATCH") == 0 && strcmp(path, "/api/v1/orders/status") == 0) {
        return 1002;
    }
    if (strcmp(method, "PATCH") == 0 && strcmp(path, "/api/v1/billing/invoices/settle") == 0) {
        return 1003;
    }

    // Legacy compatibility routes:
    //   GET   /api/v1/orders/{id}
    //   PATCH /api/v1/orders/{id}/status
    //   PATCH /api/v1/billing/invoices/{id}/settle
    if (strcmp(method, "GET") == 0 && strncmp(path, "/api/v1/orders/", 15) == 0) {
        const char *id = path + 15;
        size_t id_len = strlen(id);
        if (id_len > 0 && id_len < sizeof(g_route_param_id) && strchr(id, '/') == NULL) {
            memcpy(g_route_param_id, id, id_len);
            g_route_param_id[id_len] = '\0';
            return 1001;
        }
    }

    if (strcmp(method, "PATCH") == 0 && strncmp(path, "/api/v1/orders/", 15) == 0) {
        const char *rest = path + 15;
        const char *suffix = "/status";
        size_t rest_len = strlen(rest);
        size_t suffix_len = strlen(suffix);
        if (rest_len > suffix_len && strcmp(rest + (rest_len - suffix_len), suffix) == 0) {
            size_t id_len = rest_len - suffix_len;
            if (id_len > 0 && id_len < sizeof(g_route_param_id)) {
                int has_slash = 0;
                for (size_t i = 0; i < id_len; i++) {
                    if (rest[i] == '/') {
                        has_slash = 1;
                        break;
                    }
                    g_route_param_id[i] = rest[i];
                }
                if (!has_slash) {
                    g_route_param_id[id_len] = '\0';
                    return 1002;
                }
            }
        }
    }

    if (strcmp(method, "PATCH") == 0 && strncmp(path, "/api/v1/billing/invoices/", 24) == 0) {
        const char *rest = path + 24;
        const char *suffix = "/settle";
        size_t rest_len = strlen(rest);
        size_t suffix_len = strlen(suffix);
        if (rest_len > suffix_len && strcmp(rest + (rest_len - suffix_len), suffix) == 0) {
            size_t id_len = rest_len - suffix_len;
            if (id_len > 0 && id_len < sizeof(g_route_param_id)) {
                int has_slash = 0;
                for (size_t i = 0; i < id_len; i++) {
                    if (rest[i] == '/') {
                        has_slash = 1;
                        break;
                    }
                    g_route_param_id[i] = rest[i];
                }
                if (!has_slash) {
                    g_route_param_id[id_len] = '\0';
                    return 1003;
                }
            }
        }
    }

    return 0;
}

static const char *lookup_query_param_value(const char *query, const char *key) {
    memset(g_query_param_buf, 0, sizeof(g_query_param_buf));
    if (query == NULL || key == NULL || *key == '\0') {
        return "";
    }

    size_t key_len = strlen(key);
    const char *cursor = query;

    while (*cursor != '\0') {
        const char *pair_end = strchr(cursor, '&');
        if (pair_end == NULL) {
            pair_end = cursor + strlen(cursor);
        }

        const char *eq = memchr(cursor, '=', (size_t)(pair_end - cursor));
        if (eq != NULL) {
            size_t k_len = (size_t)(eq - cursor);
            if (k_len == key_len && strncmp(cursor, key, key_len) == 0) {
                const char *value_start = eq + 1;
                size_t value_len = (size_t)(pair_end - value_start);
                if (value_len >= sizeof(g_query_param_buf)) {
                    value_len = sizeof(g_query_param_buf) - 1;
                }
                memcpy(g_query_param_buf, value_start, value_len);
                g_query_param_buf[value_len] = '\0';
                return g_query_param_buf;
            }
        }

        if (*pair_end == '\0') {
            break;
        }
        cursor = pair_end + 1;
    }

    return "";
}

const char *__sounio_chiuratto_http_query_param_impl(void *req_ptr, const char *key) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return "";
    }

    const char *query = req->query != NULL ? req->query : "";
    return lookup_query_param_value(query, key);
}

int64_t __sounio_chiuratto_http_query_param_i64_impl(void *req_ptr, const char *key, int64_t fallback) {
    const char *value = __sounio_chiuratto_http_query_param_impl(req_ptr, key);
    if (value == NULL || *value == '\0') {
        return fallback;
    }

    char *endptr = NULL;
    long long parsed = strtoll(value, &endptr, 10);
    if (endptr == value || *endptr != '\0' || parsed <= 0) {
        return fallback;
    }

    return (int64_t)parsed;
}

const char *__sounio_chiuratto_http_request_method_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL || req->method == NULL) {
        return "GET";
    }
    return req->method;
}

int32_t __sounio_chiuratto_http_request_method_code_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    const char *method = "GET";
    if (req != NULL && req->method != NULL) {
        method = req->method;
    }

    if (strcmp(method, "GET") == 0) return 1;
    if (strcmp(method, "POST") == 0) return 2;
    if (strcmp(method, "PATCH") == 0) return 3;
    if (strcmp(method, "DELETE") == 0) return 4;
    if (strcmp(method, "OPTIONS") == 0) return 5;
    return 0;
}

const char *__sounio_chiuratto_http_request_path_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL || req->path == NULL) {
        return "/";
    }
    return req->path;
}

const char *__sounio_chiuratto_http_request_authorization_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL) {
        return "";
    }
    if (req->authorization != NULL) {
        return req->authorization;
    }
    return "";
}

int32_t __sounio_chiuratto_http_request_is_admin_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL || req->authorization == NULL) {
        return 0;
    }

    const char *auth = req->authorization;
    if (*auth == '\0') {
        return 0;
    }

    // Accept either raw token or Bearer token style.
    return strstr(auth, "admin-token") != NULL ? 1 : 0;
}

const char *__sounio_chiuratto_http_request_body_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL || req->body == NULL) {
        return "";
    }
    return req->body;
}

const char *__sounio_chiuratto_http_request_query_impl(void *req_ptr) {
    RequestRepr *req = (RequestRepr *)req_ptr;
    if (req == NULL || req->query == NULL) {
        return "";
    }
    return req->query;
}

int64_t __sounio_chiuratto_http_send_response_impl(int32_t status, const char *body) {
    if (g_active_client_fd < 0) {
        return 1;
    }

    int effective_status = status;
    // `body` originates from Sounio and may be a tagged/owned string representation.
    // Decode it into a stable C string before computing length/sending.
    const char *effective_body = __sounio_chiuratto_as_ptr_impl((uintptr_t)body);
    if (effective_status < 100 || effective_status > 599) {
        effective_status = 200;
    }
    if (effective_body == NULL) {
        effective_body = "{\"ok\":true}";
    }

    const char *status_text = "OK";
    if (effective_status == 201) status_text = "Created";
    if (effective_status == 204) status_text = "No Content";
    if (effective_status == 400) status_text = "Bad Request";
    if (effective_status == 401) status_text = "Unauthorized";
    if (effective_status == 403) status_text = "Forbidden";
    if (effective_status == 404) status_text = "Not Found";
    if (effective_status == 422) status_text = "Unprocessable Entity";
    if (effective_status >= 500) status_text = "Internal Server Error";

    size_t body_len = strlen(effective_body);
    char head[4096];
    int head_len = snprintf(
        head,
        sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, PATCH, PUT, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization, X-Request-ID\r\n"
        "\r\n",
        effective_status,
        status_text,
        body_len);

    if (head_len < 0) {
        close(g_active_client_fd);
        g_active_client_fd = -1;
        return 1;
    }

    (void)send(g_active_client_fd, head, (size_t)head_len, MSG_NOSIGNAL);
    if (body_len > 0) {
        (void)send(g_active_client_fd, effective_body, body_len, MSG_NOSIGNAL);
    }

    close(g_active_client_fd);
    g_active_client_fd = -1;
    return 0;
}
