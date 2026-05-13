/*
 * souc_emit_llvm.c — Sounio LLVM 18 IR emitter bridge
 *
 * Reads a Sounio source file, performs minimal parsing (fn main detection,
 * print string literals, return value), and emits LLVM IR via the LLVM C API.
 *
 * Build:
 *   clang -I/usr/lib/llvm-18/include -L/usr/lib/llvm-18/lib \
 *         -o souc-emit-llvm souc_emit_llvm.c -lLLVM \
 *         -Wl,-rpath,/usr/lib/llvm-18/lib
 *
 * Usage:
 *   souc-emit-llvm input.sio output.ll      # emit LLVM IR
 *   souc-emit-llvm input.sio output.o       # emit object file (LLVM backend)
 *
 * Architecture:
 *   1. Scan source for top-level fn main() and print("...") calls
 *   2. Build LLVM IR: module, libc printf declaration, main function
 *   3. For each print() call: emit a GEP + call to printf
 *   4. Emit return value (integer literal or 0)
 *   5. Write LLVM IR textual representation to output.ll
 *
 * This is the bridge that wires self-hosted/llvm/ .sio semantics into
 * a real LLVM 18 backend. The .sio LLVM modules define the IR primitives;
 * this C file drives them from the CLI dispatch layer.
 *
 * LLVM 18.1.3 required (libLLVM-18.so).
 */

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ================================================================
 * Source scanner — minimal Sounio parser for LLVM IR emission
 * Handles: fn main() -> i32, print("str"), return <int>
 * ================================================================ */

#define MAX_SOURCE    (4 * 1024 * 1024)
#define MAX_PRINTS    256
#define MAX_STR_LEN   4096

typedef struct {
    char text[MAX_STR_LEN];
    int  len;
} PrintArg;

typedef struct {
    PrintArg prints[MAX_PRINTS];
    int      print_count;
    int      return_val;     /* default 0 */
    bool     has_main;
    char     module_name[256];
} SounioMinimalAST;

/* Skip whitespace */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Skip to end of line */
static const char *skip_line(const char *p) {
    while (*p && *p != '\n') p++;
    if (*p == '\n') p++;
    return p;
}

/* Match a keyword at p (p must point to start of token) */
static bool match_kw(const char *p, const char *kw) {
    size_t n = strlen(kw);
    if (strncmp(p, kw, n) != 0) return false;
    char next = p[n];
    return !(next >= 'a' && next <= 'z') &&
           !(next >= 'A' && next <= 'Z') &&
           !(next >= '0' && next <= '9') &&
           next != '_';
}

/* Extract string contents from a double-quoted literal. Returns end pointer. */
static const char *extract_string(const char *p, char *out, int maxlen) {
    /* p points at opening " */
    if (*p != '"') { out[0] = 0; return p; }
    p++; /* skip opening " */
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n':  out[i++] = '\n'; break;
                case 't':  out[i++] = '\t'; break;
                case 'r':  out[i++] = '\r'; break;
                case '\\': out[i++] = '\\'; break;
                case '"':  out[i++] = '"';  break;
                case '0':  out[i++] = '\0'; break;
                default:   out[i++] = *p;   break;
            }
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    if (*p == '"') p++; /* skip closing " */
    return p;
}

/* Scan the source and build a minimal AST */
static SounioMinimalAST scan_source(const char *src, const char *src_path) {
    SounioMinimalAST ast = {0};
    ast.return_val = 0;
    ast.has_main   = false;
    ast.print_count = 0;

    /* Module name from file basename */
    const char *base = strrchr(src_path, '/');
    base = base ? base + 1 : src_path;
    strncpy(ast.module_name, base, sizeof(ast.module_name) - 1);
    /* Strip .sio extension */
    char *dot = strrchr(ast.module_name, '.');
    if (dot) *dot = '\0';

    bool in_main = false;
    int  brace_depth = 0;
    const char *p = src;

    while (*p) {
        /* Skip line comments */
        if (p[0] == '/' && p[1] == '/') {
            p = skip_line(p);
            continue;
        }

        /* Detect fn main */
        if (!in_main && match_kw(p, "fn")) {
            const char *q = skip_ws(p + 2);
            if (match_kw(q, "main")) {
                /* Scan ahead for opening brace */
                while (*q && *q != '{' && *q != '\n') q++;
                if (*q == '{') {
                    in_main = true;
                    ast.has_main = true;
                    brace_depth = 1;
                    p = q + 1;
                    continue;
                }
            }
        }

        if (in_main) {
            if (*p == '{') { brace_depth++; p++; continue; }
            if (*p == '}') {
                brace_depth--;
                if (brace_depth == 0) { in_main = false; }
                p++;
                continue;
            }

            /* Detect print("...") or println("...") */
            if (match_kw(p, "println") || match_kw(p, "print")) {
                const char *q = p + (match_kw(p, "println") ? 7 : 5);
                q = skip_ws(q);
                if (*q == '(') {
                    q = skip_ws(q + 1);
                    if (*q == '"' && ast.print_count < MAX_PRINTS) {
                        PrintArg *pa = &ast.prints[ast.print_count];
                        q = extract_string(q, pa->text, MAX_STR_LEN - 2);
                        pa->len = (int)strlen(pa->text);
                        /* println adds newline if not already present */
                        if (match_kw(p, "println") &&
                            (pa->len == 0 || pa->text[pa->len - 1] != '\n')) {
                            pa->text[pa->len++] = '\n';
                            pa->text[pa->len]   = '\0';
                        }
                        ast.print_count++;
                        p = q;
                        /* skip closing ) */
                        while (*p && *p != ')') p++;
                        if (*p == ')') p++;
                        continue;
                    }
                }
            }

            /* Detect return <int> */
            if (match_kw(p, "return")) {
                const char *q = skip_ws(p + 6);
                if (*q >= '0' && *q <= '9') {
                    ast.return_val = atoi(q);
                    while (*q >= '0' && *q <= '9') q++;
                    p = q;
                    continue;
                }
            }
        }

        p++;
    }

    return ast;
}

/* ================================================================
 * LLVM IR builder — converts SounioMinimalAST to LLVM IR
 * ================================================================ */

static int emit_llvm_ir(const SounioMinimalAST *ast, const char *out_path, bool emit_obj) {
    /* Initialize all targets */
    LLVMInitializeAllTargetInfos();
    LLVMInitializeAllTargets();
    LLVMInitializeAllTargetMCs();
    LLVMInitializeAllAsmPrinters();
    LLVMInitializeAllAsmParsers();

    /* Create context and module */
    LLVMContextRef ctx = LLVMContextCreate();
    LLVMModuleRef  mod = LLVMModuleCreateWithNameInContext(ast->module_name, ctx);

    /* Set target triple and data layout for x86-64 Linux */
    char *triple = LLVMGetDefaultTargetTriple();
    LLVMSetTarget(mod, triple);

    /* Get data layout from target machine */
    char *err_msg = NULL;
    LLVMTargetRef target_ref = NULL;
    if (LLVMGetTargetFromTriple(triple, &target_ref, &err_msg) != 0) {
        fprintf(stderr, "souc-emit-llvm: failed to get target: %s\n",
                err_msg ? err_msg : "(unknown)");
        if (err_msg) LLVMDisposeMessage(err_msg);
        LLVMDisposeModule(mod);
        LLVMContextDispose(ctx);
        LLVMDisposeMessage(triple);
        return 1;
    }

    char *cpu      = LLVMGetHostCPUName();
    char *features = LLVMGetHostCPUFeatures();
    LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
        target_ref, triple, cpu, features,
        LLVMCodeGenLevelDefault,
        LLVMRelocPIC,
        LLVMCodeModelDefault
    );
    LLVMDisposeMessage(cpu);
    LLVMDisposeMessage(features);

    LLVMTargetDataRef td = LLVMCreateTargetDataLayout(tm);
    char *layout_str = LLVMCopyStringRepOfTargetData(td);
    LLVMSetDataLayout(mod, layout_str);
    LLVMDisposeMessage(layout_str);
    LLVMDisposeTargetData(td);

    /* ---- Types ---- */
    LLVMTypeRef i8_t   = LLVMInt8TypeInContext(ctx);
    LLVMTypeRef i32_t  = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64_t  = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr_t  = LLVMPointerTypeInContext(ctx, 0);  /* opaque ptr (LLVM 15+) */
    LLVMTypeRef void_t = LLVMVoidTypeInContext(ctx);

    /* ---- Declare printf ---- */
    LLVMTypeRef printf_param_types[] = { ptr_t };
    LLVMTypeRef printf_type = LLVMFunctionType(i32_t, printf_param_types, 1, /*vararg=*/1);
    LLVMValueRef printf_fn  = LLVMAddFunction(mod, "printf", printf_type);
    LLVMSetFunctionCallConv(printf_fn, LLVMCCallConv);

    /* ---- Declare exit (for non-zero returns if desired) ---- */
    LLVMTypeRef exit_param_types[] = { i32_t };
    LLVMTypeRef exit_type = LLVMFunctionType(void_t, exit_param_types, 1, 0);
    LLVMValueRef exit_fn  = LLVMAddFunction(mod, "exit", exit_type);
    LLVMSetFunctionCallConv(exit_fn, LLVMCCallConv);
    (void)exit_fn;  /* may use in future */

    /* ---- Create main function ---- */
    LLVMTypeRef main_param_types[] = { i32_t, ptr_t };
    LLVMTypeRef main_type = LLVMFunctionType(i32_t, main_param_types, 2, 0);
    LLVMValueRef main_fn  = LLVMAddFunction(mod, "main", main_type);
    LLVMSetFunctionCallConv(main_fn, LLVMCCallConv);
    LLVMSetLinkage(main_fn, LLVMExternalLinkage);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx, main_fn, "entry");
    LLVMBuilderRef builder  = LLVMCreateBuilderInContext(ctx);
    LLVMPositionBuilderAtEnd(builder, entry);

    /* ---- Emit string globals and printf calls ---- */
    for (int i = 0; i < ast->print_count; i++) {
        const PrintArg *pa = &ast->prints[i];
        int str_len = pa->len;

        /* Create a global constant for the string (null-terminated) */
        char gname[64];
        snprintf(gname, sizeof(gname), ".str.%d", i);

        LLVMTypeRef arr_ty = LLVMArrayType2(i8_t, (uint64_t)(str_len + 1));
        LLVMValueRef global_str = LLVMAddGlobal(mod, arr_ty, gname);
        LLVMSetLinkage(global_str, LLVMPrivateLinkage);
        LLVMSetGlobalConstant(global_str, 1);
        LLVMSetUnnamedAddress(global_str, LLVMGlobalUnnamedAddr);

        /* Build the constant array */
        LLVMValueRef *chars = malloc((size_t)(str_len + 1) * sizeof(LLVMValueRef));
        for (int j = 0; j < str_len; j++) {
            chars[j] = LLVMConstInt(i8_t, (unsigned char)pa->text[j], 0);
        }
        chars[str_len] = LLVMConstInt(i8_t, 0, 0);  /* null terminator */
        LLVMValueRef init = LLVMConstArray2(i8_t, chars, (uint64_t)(str_len + 1));
        LLVMSetInitializer(global_str, init);
        free(chars);

        /* GEP to get i8* pointer to string */
        LLVMValueRef zero = LLVMConstInt(i64_t, 0, 0);
        LLVMValueRef gep_indices[] = { zero, zero };
        LLVMValueRef str_ptr = LLVMBuildGEP2(
            builder, arr_ty, global_str, gep_indices, 2, "str_ptr"
        );

        /* Call printf(str_ptr, ...) */
        LLVMValueRef printf_args[] = { str_ptr };
        LLVMBuildCall2(builder, printf_type, printf_fn, printf_args, 1, "");
    }

    /* ---- Emit return ---- */
    LLVMValueRef ret_val = LLVMConstInt(i32_t, (unsigned long long)ast->return_val, 0);
    LLVMBuildRet(builder, ret_val);

    /* ---- Verify module ---- */
    err_msg = NULL;
    if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &err_msg) != 0) {
        fprintf(stderr, "souc-emit-llvm: module verification failed: %s\n",
                err_msg ? err_msg : "(unknown)");
        if (err_msg) LLVMDisposeMessage(err_msg);
        LLVMDisposeBuilder(builder);
        LLVMDisposeTargetMachine(tm);
        LLVMDisposeModule(mod);
        LLVMContextDispose(ctx);
        LLVMDisposeMessage(triple);
        return 1;
    }
    if (err_msg) LLVMDisposeMessage(err_msg);

    int result = 0;
    if (emit_obj) {
        /* Emit object file via LLVM */
        err_msg = NULL;
        /* Cast away const: LLVM API takes char* not const char* */
        char out_path_buf[4096];
        strncpy(out_path_buf, out_path, sizeof(out_path_buf) - 1);
        out_path_buf[sizeof(out_path_buf) - 1] = '\0';

        if (LLVMTargetMachineEmitToFile(tm, mod, out_path_buf,
                                        LLVMObjectFile, &err_msg) != 0) {
            fprintf(stderr, "souc-emit-llvm: emit to object file failed: %s\n",
                    err_msg ? err_msg : "(unknown)");
            if (err_msg) LLVMDisposeMessage(err_msg);
            result = 1;
        } else {
            fprintf(stderr, "souc-emit-llvm: wrote object: %s\n", out_path);
        }
    } else {
        /* Print LLVM IR to file */
        char *ir_str = LLVMPrintModuleToString(mod);
        FILE *fp = strcmp(out_path, "-") == 0 ? stdout : fopen(out_path, "w");
        if (!fp) {
            fprintf(stderr, "souc-emit-llvm: cannot open output: %s\n", out_path);
            LLVMDisposeMessage(ir_str);
            result = 1;
        } else {
            fputs(ir_str, fp);
            if (fp != stdout) {
                fclose(fp);
                fprintf(stderr, "souc-emit-llvm: wrote LLVM IR: %s\n", out_path);
            }
            LLVMDisposeMessage(ir_str);
        }
    }

    LLVMDisposeBuilder(builder);
    LLVMDisposeTargetMachine(tm);
    LLVMDisposeModule(mod);
    LLVMContextDispose(ctx);
    LLVMDisposeMessage(triple);
    return result;
}

/* ================================================================
 * main
 * ================================================================ */

static void usage(void) {
    fprintf(stderr,
        "Usage: souc-emit-llvm <input.sio> <output.ll|output.o> [--obj]\n"
        "\n"
        "  <input.sio>   Sounio source file\n"
        "  <output.ll>   Write LLVM IR (textual)\n"
        "  <output.o>    Write object file (with --obj)\n"
        "  --obj         Emit object file instead of LLVM IR text\n"
        "\n"
        "LLVM 18.1.3 backend bridge for the Sounio compiler.\n"
        "Wired via: souc build --backend llvm / souc build --emit-llvm\n"
    );
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 2;
    }

    const char *src_path = argv[1];
    const char *out_path = argv[2];
    bool emit_obj = false;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--obj") == 0) {
            emit_obj = true;
        } else if (strcmp(argv[i], "--ir") == 0) {
            emit_obj = false;
        }
    }

    /* Auto-detect from extension */
    size_t out_len = strlen(out_path);
    if (!emit_obj && out_len >= 2 &&
        strcmp(out_path + out_len - 2, ".o") == 0) {
        emit_obj = true;
    }

    /* Read source */
    FILE *fp = fopen(src_path, "rb");
    if (!fp) {
        fprintf(stderr, "souc-emit-llvm: cannot open source: %s\n", src_path);
        return 1;
    }
    char *src = malloc(MAX_SOURCE);
    if (!src) {
        fprintf(stderr, "souc-emit-llvm: out of memory\n");
        fclose(fp);
        return 1;
    }
    size_t src_len = fread(src, 1, MAX_SOURCE - 1, fp);
    fclose(fp);
    src[src_len] = '\0';

    /* Scan source */
    SounioMinimalAST ast = scan_source(src, src_path);
    free(src);

    if (!ast.has_main) {
        fprintf(stderr, "souc-emit-llvm: no fn main() found in %s — "
                        "emitting empty module\n", src_path);
        /* Still emit a valid (empty) module */
    }

    fprintf(stderr, "souc-emit-llvm: %s -> %s [%s, %d print(s), return %d]\n",
            src_path, out_path,
            emit_obj ? "object" : "LLVM IR",
            ast.print_count,
            ast.return_val);

    return emit_llvm_ir(&ast, out_path, emit_obj);
}
