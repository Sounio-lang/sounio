/*
 * sounio_embed.h — Sounio Embedding ABI for Scientific/HPC hosts.
 *
 * This header defines the target C ABI that a native-compiled Sounio
 * runtime will export from libsounio.so.  Until the native dylib is
 * available, hosts can use the SNIO binary protocol (IPC over stdin/stdout)
 * which implements the same semantics.
 *
 * SNIO message type equivalence:
 *   sounio_session_create    ↔  MSG 14 (SESSION_CREATE)
 *   sounio_session_destroy   ↔  MSG 15 (SESSION_DESTROY)
 *   sounio_kernel_describe   ↔  MSG 16 (KERNEL_DESCRIBE)
 *   sounio_kernel_execute    ↔  MSG 17 (KERNEL_EXECUTE)
 *   sounio_kernel_output     ↔  MSG 18 (KERNEL_OUTPUT)
 *   sounio_kernel_diagnostics↔  MSG 19 (KERNEL_DIAGNOSTICS)
 *   sounio_kernel_artifacts  ↔  MSG 20 (KERNEL_ARTIFACTS)
 *
 * ABI version: 2  (extends version 1 which covers FFI + EXPORT + PROFILING)
 * Stability: EXPERIMENTAL — field order and sizes may change before 1.0
 */

#ifndef SOUNIO_EMBED_H
#define SOUNIO_EMBED_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * VERSION & CAPABILITY QUERY
 * ======================================================================== */

#define SOUNIO_ABI_VERSION      2
#define SOUNIO_MAX_SESSIONS     8
#define SOUNIO_MAX_KERNELS      32
#define SOUNIO_MAX_DIAGNOSTICS  64
#define SOUNIO_MAX_ARTIFACTS    16
#define SOUNIO_MAX_ARGS         8192

/* Capability bitmask (matches IPC_CAP_* constants) */
#define SOUNIO_CAP_FFI          0x01
#define SOUNIO_CAP_EXPORT       0x02
#define SOUNIO_CAP_PROFILING    0x04
#define SOUNIO_CAP_GC           0x08
#define SOUNIO_CAP_SIMD         0x10
#define SOUNIO_CAP_GPU          0x20
#define SOUNIO_CAP_KERNEL       0x40

/* Session creation flags */
#define SOUNIO_FLAG_OPTIMIZE            0x01
#define SOUNIO_FLAG_TRACE               0x02
#define SOUNIO_FLAG_PROFILE             0x04
#define SOUNIO_FLAG_STRICT_DIAGNOSTICS  0x08

/* Session states */
#define SOUNIO_STATE_FREE       0
#define SOUNIO_STATE_CREATED    1
#define SOUNIO_STATE_READY      2
#define SOUNIO_STATE_EXECUTING  3
#define SOUNIO_STATE_ERROR      4

/* Diagnostic severity levels */
#define SOUNIO_DIAG_NOTE        0
#define SOUNIO_DIAG_WARNING     1
#define SOUNIO_DIAG_ERROR       2
#define SOUNIO_DIAG_FATAL       3

/* Artifact kinds (match .snpkg entry kinds) */
#define SOUNIO_ARTIFACT_SOURCE      1
#define SOUNIO_ARTIFACT_OBJECT      2
#define SOUNIO_ARTIFACT_PROFILE     3
#define SOUNIO_ARTIFACT_PROVENANCE  4
#define SOUNIO_ARTIFACT_ELF         5
#define SOUNIO_ARTIFACT_SHARED_LIB  6

/* Error codes */
#define SOUNIO_OK                   0
#define SOUNIO_ERR_INVALID_SESSION  (-1)
#define SOUNIO_ERR_NO_FREE_SLOTS    (-2)
#define SOUNIO_ERR_INVALID_KERNEL   (-3)
#define SOUNIO_ERR_COMPILE_FAILED   (-4)
#define SOUNIO_ERR_EXEC_FAILED      (-5)
#define SOUNIO_ERR_MAX_KERNELS      (-6)
#define SOUNIO_ERR_INVALID_ARGS     (-7)

/* ========================================================================
 * TYPES
 * ======================================================================== */

typedef int64_t sounio_session_t;   /* opaque session handle */
typedef int64_t sounio_kernel_t;    /* opaque kernel handle  */

typedef struct {
    int64_t severity;       /* SOUNIO_DIAG_* */
    int64_t line;
    int64_t col;
    int64_t code;           /* E001..E010 */
    const char *message;
    const char *fix_suggestion;
} sounio_diagnostic_t;

typedef struct {
    sounio_diagnostic_t items[SOUNIO_MAX_DIAGNOSTICS];
    int64_t count;
} sounio_diagnostics_t;

typedef struct {
    int64_t kind;           /* SOUNIO_ARTIFACT_* */
    const char *path;
    int64_t size_bytes;
    int64_t hash;
} sounio_artifact_t;

typedef struct {
    sounio_artifact_t items[SOUNIO_MAX_ARTIFACTS];
    int64_t count;
} sounio_artifacts_t;

typedef struct {
    int64_t ok;             /* 1 = success, 0 = error */
    int64_t kernel_id;
    int64_t param_count;
    int64_t return_count;
    int64_t compile_strategy;
    int64_t error_code;
    const char *error_msg;
} sounio_kernel_info_t;

typedef struct {
    int64_t ok;             /* 1 = success, 0 = error */
    int64_t *values;        /* caller-owned buffer */
    int64_t value_count;
    int64_t error_code;
    const char *error_msg;
} sounio_exec_result_t;

typedef struct {
    int64_t session_id;
    int64_t state;          /* SOUNIO_STATE_* */
    int64_t execute_count;
    int64_t error_count;
    int64_t kernel_count;
} sounio_session_info_t;

typedef struct {
    int64_t source_hash;
    int64_t abi_version;
    int64_t optimize;
    int64_t function_count;
    int64_t ir_instruction_count;
    int64_t stdlib_hash;
} sounio_provenance_t;

/* ========================================================================
 * LIFECYCLE
 * ======================================================================== */

/**
 * Query ABI version.  Returns SOUNIO_ABI_VERSION.
 */
int64_t sounio_abi_version(void);

/**
 * Query capability bitmask.
 */
int64_t sounio_capabilities(void);

/* ========================================================================
 * SESSION MANAGEMENT
 * ======================================================================== */

/**
 * Create a new session.
 * @param flags  Bitmask of SOUNIO_FLAG_* values.
 * @return       Session handle (>0) or negative error code.
 */
sounio_session_t sounio_session_create(int64_t flags);

/**
 * Destroy a session and release all associated resources.
 * @return  SOUNIO_OK or negative error code.
 */
int64_t sounio_session_destroy(sounio_session_t session);

/**
 * Query session state.
 * @param out  Filled with session metadata on success.
 * @return     SOUNIO_OK or negative error code.
 */
int64_t sounio_session_info(sounio_session_t session, sounio_session_info_t *out);

/* ========================================================================
 * KERNEL MANAGEMENT
 * ======================================================================== */

/**
 * Describe (compile) a kernel from a .sio source file.
 *
 * This parses, type-checks, and optionally compiles the source.
 * On success, the kernel is registered in the session and can be
 * executed via sounio_kernel_execute().
 *
 * @param session      Session handle.
 * @param source_path  Path to .sio source file.
 * @param flags        Compilation flags (SOUNIO_FLAG_*).
 * @param out          Filled with kernel metadata on success.
 * @return             SOUNIO_OK or negative error code.
 */
int64_t sounio_kernel_describe(
    sounio_session_t session,
    const char *source_path,
    int64_t flags,
    sounio_kernel_info_t *out
);

/**
 * Execute a described kernel with the given arguments.
 *
 * @param session   Session handle.
 * @param kernel    Kernel handle from sounio_kernel_describe().
 * @param args      Array of i64 arguments.
 * @param arg_count Number of arguments.
 * @param out       Filled with execution results.
 * @return          SOUNIO_OK or negative error code.
 */
int64_t sounio_kernel_execute(
    sounio_session_t session,
    sounio_kernel_t kernel,
    const int64_t *args,
    int64_t arg_count,
    sounio_exec_result_t *out
);

/* ========================================================================
 * STRUCTURED OUTPUT RECOVERY
 * ======================================================================== */

/**
 * Retrieve diagnostics from the last operation on this session.
 */
int64_t sounio_kernel_diagnostics(
    sounio_session_t session,
    sounio_diagnostics_t *out
);

/**
 * Retrieve artifacts produced by the last kernel_describe.
 */
int64_t sounio_kernel_artifacts(
    sounio_session_t session,
    sounio_artifacts_t *out
);

/**
 * Retrieve provenance metadata for the session.
 */
int64_t sounio_kernel_provenance(
    sounio_session_t session,
    sounio_provenance_t *out
);

/**
 * Clear diagnostics for the session.
 */
int64_t sounio_diagnostics_clear(sounio_session_t session);

#ifdef __cplusplus
}
#endif

#endif /* SOUNIO_EMBED_H */
