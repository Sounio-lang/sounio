/* library_tests.c - Poseidon library API tests */

#include "poseidon.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DIR "fixtures"
#define FIXTURE_PATH_SIZE 512

typedef struct {
    int64_t values[8];
    size_t count;
} CaptureState;

static bool build_fixture_path(const char *name, char *out, size_t out_size) {
    int written = snprintf(out, out_size, "%s/%s.soir", FIXTURE_DIR, name);
    return written > 0 && (size_t)written < out_size;
}

static bool read_file_bytes(const char *path, uint8_t **out_buf, size_t *out_size) {
    FILE *file;
    long file_size;
    uint8_t *buf;
    size_t nread;

    *out_buf = NULL;
    *out_size = 0;

    file = fopen(path, "rb");
    if (!file) {
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }
    file_size = ftell(file);
    if (file_size < 0) {
        fclose(file);
        return false;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }

    buf = malloc((size_t)file_size);
    if (!buf && file_size > 0) {
        fclose(file);
        return false;
    }
    nread = fread(buf, 1, (size_t)file_size, file);
    fclose(file);
    if (nread != (size_t)file_size) {
        free(buf);
        return false;
    }

    *out_buf = buf;
    *out_size = (size_t)file_size;
    return true;
}

static void capture_print_int(int64_t value, void *user_data) {
    CaptureState *state = (CaptureState *)user_data;
    if (state && state->count < 8) {
        state->values[state->count++] = value;
    }
}

static bool test_load_file_run_add(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunResult result;
    PoseidonStatus status;

    if (!build_fixture_path("add", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (status != POSEIDON_STATUS_OK) {
        return false;
    }
    result = poseidon_run_module(module, NULL);
    poseidon_module_free(module);
    return result.status == POSEIDON_STATUS_OK &&
           result.exit_code == 42 &&
           result.steps_executed > 0;
}

static bool test_load_bytes_run_branch(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunResult result;
    uint8_t *buf = NULL;
    size_t size = 0;
    bool ok = false;

    if (!build_fixture_path("branch_true_taken", path, sizeof(path))) {
        return false;
    }
    if (!read_file_bytes(path, &buf, &size)) {
        return false;
    }
    if (poseidon_module_load_bytes(buf, size, &module) != POSEIDON_STATUS_OK) {
        free(buf);
        return false;
    }
    result = poseidon_run_module(module, NULL);
    if (result.status == POSEIDON_STATUS_OK && result.exit_code == 42) {
        ok = true;
    }
    poseidon_module_free(module);
    free(buf);
    return ok;
}

static bool test_missing_main(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunResult result;
    PoseidonStatus status;

    if (!build_fixture_path("missing_main", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (status != POSEIDON_STATUS_OK) {
        return false;
    }
    result = poseidon_run_module(module, NULL);
    poseidon_module_free(module);
    return result.status == POSEIDON_STATUS_NO_MAIN_FUNCTION;
}

static bool test_timeout(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunConfig config = poseidon_default_config();
    PoseidonRunResult result;
    PoseidonStatus status;

    if (!build_fixture_path("timeout_loop", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (status != POSEIDON_STATUS_OK) {
        return false;
    }
    config.max_steps = 3;
    result = poseidon_run_module(module, &config);
    poseidon_module_free(module);
    return result.status == POSEIDON_STATUS_EXECUTION_TIMEOUT &&
           result.steps_executed == 3;
}

static bool test_callback_capture(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunConfig config = poseidon_default_config();
    PoseidonRunResult result;
    CaptureState capture;
    PoseidonStatus status;

    capture.count = 0;
    if (!build_fixture_path("call_builtin_print_int", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (status != POSEIDON_STATUS_OK) {
        return false;
    }
    config.print_int_cb = capture_print_int;
    config.user_data = &capture;
    result = poseidon_run_module(module, &config);
    poseidon_module_free(module);
    return result.status == POSEIDON_STATUS_OK &&
           result.exit_code == 0 &&
           capture.count == 1 &&
           capture.values[0] == 42;
}

static bool test_invalid_fn_call_contract(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonRunResult result;
    PoseidonStatus status;

    if (!build_fixture_path("invalid_fn_call_returns_zero", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (status != POSEIDON_STATUS_OK) {
        return false;
    }
    result = poseidon_run_module(module, NULL);
    poseidon_module_free(module);
    return result.status == POSEIDON_STATUS_OK &&
           result.exit_code == 0 &&
           result.steps_executed > 0;
}

static bool test_loader_invalid_bad_magic(void) {
    char path[FIXTURE_PATH_SIZE];
    PoseidonModule *module = NULL;
    PoseidonStatus status;

    if (!build_fixture_path("bad_magic", path, sizeof(path))) {
        return false;
    }
    status = poseidon_module_load_file(path, &module);
    if (module != NULL) {
        poseidon_module_free(module);
        return false;
    }
    return status == POSEIDON_STATUS_INVALID_BYTECODE;
}

static bool test_status_messages(void) {
    return poseidon_status_message(POSEIDON_STATUS_OK) != NULL &&
           poseidon_status_message(POSEIDON_STATUS_INVALID_BYTECODE) != NULL &&
           poseidon_status_message(POSEIDON_STATUS_EXECUTION_TIMEOUT) != NULL;
}

int main(void) {
    size_t passed = 0;
    size_t failed = 0;

    struct {
        const char *name;
        bool (*fn)(void);
    } tests[] = {
        { "load_file_run_add", test_load_file_run_add },
        { "load_bytes_run_branch", test_load_bytes_run_branch },
        { "missing_main", test_missing_main },
        { "timeout", test_timeout },
        { "callback_capture", test_callback_capture },
        { "invalid_fn_call_contract", test_invalid_fn_call_contract },
        { "loader_invalid_bad_magic", test_loader_invalid_bad_magic },
        { "status_messages", test_status_messages }
    };
    size_t i;

    printf("Running Poseidon library tests...\n");
    for (i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        printf("  %s... ", tests[i].name);
        if (tests[i].fn()) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
            failed++;
        }
    }

    printf("Library test results: %zu passed, %zu failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
