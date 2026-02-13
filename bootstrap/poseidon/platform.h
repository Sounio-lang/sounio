/* platform.h - Cross-Platform Abstraction Layer for Poseidon VM */

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * PLATFORM DETECTION
 * ============================================================================ */

/* Operating System */
#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_WINDOWS 1
    #define PLATFORM_POSIX 0
#elif defined(__APPLE__) && defined(__MACH__)
    #define PLATFORM_MACOS 1
    #define PLATFORM_POSIX 1
    #define PLATFORM_WINDOWS 0
#elif defined(__linux__)
    #define PLATFORM_LINUX 1
    #define PLATFORM_POSIX 1
    #define PLATFORM_WINDOWS 0
#else
    #define PLATFORM_POSIX 1
    #define PLATFORM_WINDOWS 0
#endif

/* Architecture */
#if defined(__x86_64__) || defined(_M_X64)
    #define PLATFORM_ARCH_X64 1
    #define PLATFORM_ARCH_ARM64 0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define PLATFORM_ARCH_ARM64 1
    #define PLATFORM_ARCH_X64 0
#else
    #define PLATFORM_ARCH_X64 0
    #define PLATFORM_ARCH_ARM64 0
#endif

/* Endianness */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    #define PLATFORM_LITTLE_ENDIAN 1
    #define PLATFORM_BIG_ENDIAN 0
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define PLATFORM_LITTLE_ENDIAN 0
    #define PLATFORM_BIG_ENDIAN 1
#elif defined(_WIN32)
    /* Windows is always little-endian */
    #define PLATFORM_LITTLE_ENDIAN 1
    #define PLATFORM_BIG_ENDIAN 0
#else
    /* Assume little-endian for common platforms */
    #define PLATFORM_LITTLE_ENDIAN 1
    #define PLATFORM_BIG_ENDIAN 0
#endif

/* ============================================================================
 * PLATFORM-SPECIFIC INCLUDES
 * ============================================================================ */

#if PLATFORM_WINDOWS
    #include <windows.h>
    #include <io.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <errno.h>
#endif

/* ============================================================================
 * ENDIANNESS CONVERSION
 * ============================================================================ */

/* SOIR binary format is always little-endian */

static inline uint16_t platform_le16_to_host(uint16_t x) {
#if PLATFORM_LITTLE_ENDIAN
    return x;
#else
    return ((x & 0x00FF) << 8) | ((x & 0xFF00) >> 8);
#endif
}

static inline uint32_t platform_le32_to_host(uint32_t x) {
#if PLATFORM_LITTLE_ENDIAN
    return x;
#else
    return ((x & 0x000000FFU) << 24) |
           ((x & 0x0000FF00U) << 8) |
           ((x & 0x00FF0000U) >> 8) |
           ((x & 0xFF000000U) >> 24);
#endif
}

static inline uint64_t platform_le64_to_host(uint64_t x) {
#if PLATFORM_LITTLE_ENDIAN
    return x;
#else
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8) |
           ((x & 0x000000FF00000000ULL) >> 8) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
#endif
}

static inline int64_t platform_le64_to_host_signed(int64_t x) {
    union {
        int64_t s;
        uint64_t u;
    } conv;
    conv.s = x;
    conv.u = platform_le64_to_host(conv.u);
    return conv.s;
}

static inline uint16_t platform_host_to_le16(uint16_t x) {
    return platform_le16_to_host(x);
}

static inline uint32_t platform_host_to_le32(uint32_t x) {
    return platform_le32_to_host(x);
}

static inline uint64_t platform_host_to_le64(uint64_t x) {
    return platform_le64_to_host(x);
}

/* ============================================================================
 * FILE I/O ABSTRACTION
 * ============================================================================ */

#if PLATFORM_WINDOWS
    typedef HANDLE platform_file_t;
    #define PLATFORM_INVALID_FILE INVALID_HANDLE_VALUE
#else
    typedef int platform_file_t;
    #define PLATFORM_INVALID_FILE (-1)
#endif

/* Open file for reading */
static inline platform_file_t platform_file_open_read(const char *path) {
#if PLATFORM_WINDOWS
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#else
    return open(path, O_RDONLY);
#endif
}

/* Get file size */
static inline int64_t platform_file_size(platform_file_t file) {
#if PLATFORM_WINDOWS
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size)) {
        return -1;
    }
    return (int64_t)size.QuadPart;
#else
    struct stat st;
    if (fstat(file, &st) != 0) {
        return -1;
    }
    return (int64_t)st.st_size;
#endif
}

/* Read from file */
static inline int64_t platform_file_read(platform_file_t file, void *buf,
                                         size_t count) {
#if PLATFORM_WINDOWS
    DWORD read;
    if (!ReadFile(file, buf, (DWORD)count, &read, NULL)) {
        return -1;
    }
    return (int64_t)read;
#else
    ssize_t result = read(file, buf, count);
    return (int64_t)result;
#endif
}

/* Close file */
static inline void platform_file_close(platform_file_t file) {
#if PLATFORM_WINDOWS
    CloseHandle(file);
#else
    close(file);
#endif
}

/* ============================================================================
 * PATH HANDLING
 * ============================================================================ */

#if PLATFORM_WINDOWS
    #define PLATFORM_PATH_SEPARATOR '\\'
    #define PLATFORM_PATH_SEPARATOR_STR "\\"
#else
    #define PLATFORM_PATH_SEPARATOR '/'
    #define PLATFORM_PATH_SEPARATOR_STR "/"
#endif

/* ============================================================================
 * MEMORY ALLOCATION
 * ============================================================================ */

#include <stdlib.h>
#include <string.h>

static inline void *platform_malloc(size_t size) {
    return malloc(size);
}

static inline void *platform_calloc(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

static inline void *platform_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

static inline void platform_free(void *ptr) {
    free(ptr);
}

static inline void *platform_memset(void *s, int c, size_t n) {
    return memset(s, c, n);
}

static inline void *platform_memcpy(void *dest, const void *src, size_t n) {
    return memcpy(dest, src, n);
}

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

static inline int platform_get_errno(void) {
#if PLATFORM_WINDOWS
    return (int)GetLastError();
#else
    return errno;
#endif
}

/* ============================================================================
 * STDOUT/STDERR OUTPUT
 * ============================================================================ */

#include <stdio.h>

static inline void platform_write_stdout(const char *msg) {
    fprintf(stdout, "%s", msg);
    fflush(stdout);
}

static inline void platform_write_stderr(const char *msg) {
    fprintf(stderr, "%s", msg);
    fflush(stderr);
}

#endif /* PLATFORM_H */
