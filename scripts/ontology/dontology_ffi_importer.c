// scripts/ontology/dontology_ffi_importer.c
//
// C ABI importer for Sounio .dontology bundles.
//
// This file intentionally stops at ingestion: it decodes the deterministic
// DONT envelope and emits Sounio source containing ontology declarations and
// numeric constants. Subclass/disjoint reasoning remains owned by the Sounio
// compiler ontology kernel.

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DONT_MAGIC "DONT"
#define DONT_VERSION 1u
#define DONT_MAX_CLASSES 64
#define DONT_MAX_PARENTS 8
#define DONT_MAX_DISJOINTS 64
#define DONT_NAME_CAP 128
#define DONT_ONTOLOGY_CAP 64

typedef struct DontologyClassRecord {
    char curie[DONT_NAME_CAP];
    char sounio_name[DONT_NAME_CAP];
    int64_t numeric_id;
    char parents[DONT_MAX_PARENTS][DONT_NAME_CAP];
    int64_t parent_count;
} DontologyClassRecord;

typedef struct DontologyTable {
    char ontology[DONT_ONTOLOGY_CAP];
    DontologyClassRecord classes[DONT_MAX_CLASSES];
    char disjoint_left[DONT_MAX_DISJOINTS][DONT_NAME_CAP];
    char disjoint_right[DONT_MAX_DISJOINTS][DONT_NAME_CAP];
    int64_t class_count;
    int64_t disjoint_count;
    int64_t truncated;
    int64_t disjoint_truncated;
} DontologyTable;

static uint32_t read_le_u32(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static char *read_payload(const char *path, uint32_t *payload_size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    unsigned char header[12];
    if (fread(header, 1, sizeof(header), f) != sizeof(header)) {
        fclose(f);
        errno = EINVAL;
        return NULL;
    }
    if (memcmp(header, DONT_MAGIC, 4) != 0) {
        fclose(f);
        errno = EINVAL;
        return NULL;
    }

    uint32_t version = read_le_u32(header + 4);
    uint32_t payload_size = read_le_u32(header + 8);
    if (version != DONT_VERSION || payload_size == 0) {
        fclose(f);
        errno = EINVAL;
        return NULL;
    }

    char *payload = (char *)calloc((size_t)payload_size + 1u, 1);
    if (!payload) {
        fclose(f);
        return NULL;
    }
    if (fread(payload, 1, payload_size, f) != payload_size) {
        free(payload);
        fclose(f);
        errno = EINVAL;
        return NULL;
    }

    fclose(f);
    *payload_size_out = payload_size;
    return payload;
}

static const char *find_after(const char *start, const char *needle) {
    const char *p = strstr(start, needle);
    if (!p) {
        return NULL;
    }
    return p + strlen(needle);
}

static int copy_json_string(const char *start, char *out, size_t out_cap, const char **end_out) {
    size_t n = 0;
    const char *p = start;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
        }
        if (n + 1 < out_cap) {
            out[n++] = *p;
        }
        p++;
    }
    if (*p != '"') {
        return -1;
    }
    out[n] = '\0';
    if (end_out) {
        *end_out = p + 1;
    }
    return 0;
}

static void curie_to_sounio_name(const char *curie, char *out, size_t out_cap) {
    size_t n = 0;
    for (const char *p = curie; *p && n + 1 < out_cap; p++) {
        unsigned char ch = (unsigned char)*p;
        if (isalnum(ch)) {
            out[n++] = (char)ch;
        } else {
            out[n++] = '_';
        }
    }
    out[n] = '\0';
}

static int64_t curie_numeric_tail(const char *curie) {
    const char *colon = strchr(curie, ':');
    const char *p = colon ? colon + 1 : curie;
    if (!*p) {
        return -1;
    }

    int64_t value = 0;
    while (*p) {
        if (!isdigit((unsigned char)*p)) {
            return -1;
        }
        value = value * 10 + (int64_t)(*p - '0');
        p++;
    }
    return value;
}

static void copy_cstr(char *dst, size_t dst_cap, const char *src) {
    if (dst_cap == 0) {
        return;
    }
    size_t n = 0;
    while (src[n] && n + 1 < dst_cap) {
        dst[n] = src[n];
        n++;
    }
    dst[n] = '\0';
}

static int parse_parents(const char *term_start, const char *term_end, DontologyClassRecord *rec) {
    const char *parents = strstr(term_start, "\"parents\":[");
    if (!parents || parents >= term_end) {
        return 0;
    }
    parents += strlen("\"parents\":[");

    while (parents < term_end && *parents && *parents != ']') {
        while (parents < term_end && (*parents == ' ' || *parents == ',')) {
            parents++;
        }
        if (*parents != '"') {
            parents++;
            continue;
        }
        parents++;

        char parent_curie[DONT_NAME_CAP];
        const char *next = NULL;
        if (copy_json_string(parents, parent_curie, sizeof(parent_curie), &next) != 0) {
            return -1;
        }
        if (rec->parent_count < DONT_MAX_PARENTS) {
            curie_to_sounio_name(
                parent_curie,
                rec->parents[rec->parent_count],
                sizeof(rec->parents[rec->parent_count])
            );
            rec->parent_count++;
        }
        parents = next;
    }
    return 0;
}

static int parse_disjoints(const char *payload, DontologyTable *out) {
    const char *disjoints = strstr(payload, "\"disjoints\":[");
    if (!disjoints) {
        return 0;
    }
    disjoints += strlen("\"disjoints\":[");
    const char *end = strstr(disjoints, "]]");
    if (!end) {
        return 0;
    }

    const char *p = disjoints;
    while (p < end) {
        const char *left_start = strstr(p, "[\"");
        if (!left_start || left_start >= end) {
            break;
        }
        left_start += 2;

        char left_curie[DONT_NAME_CAP];
        const char *after_left = NULL;
        if (copy_json_string(left_start, left_curie, sizeof(left_curie), &after_left) != 0) {
            return -1;
        }

        const char *right_start = after_left;
        while (right_start < end && (*right_start == ' ' || *right_start == ',')) {
            right_start++;
        }
        if (*right_start != '"') {
            return -1;
        }
        right_start++;

        char right_curie[DONT_NAME_CAP];
        const char *after_right = NULL;
        if (copy_json_string(right_start, right_curie, sizeof(right_curie), &after_right) != 0) {
            return -1;
        }

        if (out->disjoint_count >= DONT_MAX_DISJOINTS) {
            out->disjoint_truncated = 1;
            p = after_right;
            continue;
        }

        curie_to_sounio_name(
            left_curie,
            out->disjoint_left[out->disjoint_count],
            sizeof(out->disjoint_left[out->disjoint_count])
        );
        curie_to_sounio_name(
            right_curie,
            out->disjoint_right[out->disjoint_count],
            sizeof(out->disjoint_right[out->disjoint_count])
        );
        out->disjoint_count++;
        p = after_right;
    }
    return 0;
}

int dontology_load_sounio_table(const char *path, DontologyTable *out) {
    if (!path || !out) {
        errno = EINVAL;
        return -1;
    }
    memset(out, 0, sizeof(*out));

    uint32_t payload_size = 0;
    char *payload = read_payload(path, &payload_size);
    if (!payload) {
        return -1;
    }

    const char *ontology_start = find_after(payload, "\"ontology\":\"");
    if (ontology_start) {
        const char *ignored = NULL;
        if (copy_json_string(ontology_start, out->ontology, sizeof(out->ontology), &ignored) != 0) {
            free(payload);
            errno = EINVAL;
            return -1;
        }
    } else {
        copy_cstr(out->ontology, sizeof(out->ontology), "GeneratedOntology");
    }

    const char *p = payload;
    while ((p = find_after(p, "\"curie\":\"")) != NULL) {
        const char *curie_end = NULL;
        char curie[DONT_NAME_CAP];
        if (copy_json_string(p, curie, sizeof(curie), &curie_end) != 0) {
            free(payload);
            errno = EINVAL;
            return -1;
        }

        const char *next_term = strstr(curie_end, "{\"curie\":\"");
        const char *term_end = next_term ? next_term : payload + payload_size;

        if (out->class_count >= DONT_MAX_CLASSES) {
            out->truncated = 1;
            p = curie_end;
            continue;
        }

        DontologyClassRecord *rec = &out->classes[out->class_count];
        copy_cstr(rec->curie, sizeof(rec->curie), curie);
        curie_to_sounio_name(curie, rec->sounio_name, sizeof(rec->sounio_name));
        rec->numeric_id = curie_numeric_tail(curie);
        if (parse_parents(curie_end, term_end, rec) != 0) {
            free(payload);
            errno = EINVAL;
            return -1;
        }

        out->class_count++;
        p = curie_end;
    }

    if (parse_disjoints(payload, out) != 0) {
        free(payload);
        errno = EINVAL;
        return -1;
    }

    free(payload);
    return 0;
}

static int emit_table(FILE *out, const char *source_path, const char *ontology_name, const DontologyTable *table) {
    fprintf(out, "// Generated by scripts/ontology/dontology_ffi_importer.c\n");
    fprintf(out, "// Source bundle: %s\n", source_path);
    fprintf(out, "// Ingestion-only artifact: Sounio compiler owns ontology reasoning.\n");
    fprintf(out, "// Current importer surface: classes, subclass parents, disjoint pairs, numeric constants.\n\n");

    fprintf(out, "ontology %s {\n", ontology_name);
    for (int64_t i = 0; i < table->class_count; i++) {
        const DontologyClassRecord *rec = &table->classes[i];
        fprintf(out, "    class %s", rec->sounio_name);
        if (rec->parent_count > 0) {
            fprintf(out, " subclass_of ");
            for (int64_t j = 0; j < rec->parent_count; j++) {
                if (j > 0) {
                    fprintf(out, ", ");
                }
                fprintf(out, "%s", rec->parents[j]);
            }
        }
        fprintf(out, "\n");
    }
    for (int64_t i = 0; i < table->disjoint_count; i++) {
        fprintf(out, "    disjoint %s, %s\n", table->disjoint_left[i], table->disjoint_right[i]);
    }
    fprintf(out, "}\n\n");

    for (int64_t i = 0; i < table->class_count; i++) {
        const DontologyClassRecord *rec = &table->classes[i];
        if (rec->numeric_id >= 0) {
            fprintf(out, "const %s_ID: i64 = %lld\n", rec->sounio_name, (long long)rec->numeric_id);
        }
    }

    if (table->truncated) {
        fprintf(out, "\n// WARNING: source bundle exceeded %d classes and was truncated.\n", DONT_MAX_CLASSES);
    }
    if (table->disjoint_truncated) {
        fprintf(out, "\n// WARNING: source bundle exceeded %d disjoint pairs and was truncated.\n", DONT_MAX_DISJOINTS);
    }
    return 0;
}

int dontology_emit_sio(const char *bundle_path, const char *ontology_name, const char *output_path) {
    DontologyTable table;
    if (dontology_load_sounio_table(bundle_path, &table) != 0) {
        return -1;
    }

    char default_name[DONT_NAME_CAP];
    if (!ontology_name || ontology_name[0] == '\0') {
        snprintf(default_name, sizeof(default_name), "%sGenerated", table.ontology[0] ? table.ontology : "Ontology");
        ontology_name = default_name;
    }

    FILE *out = stdout;
    if (output_path && output_path[0]) {
        out = fopen(output_path, "w");
        if (!out) {
            return -1;
        }
    }

    int rc = emit_table(out, bundle_path, ontology_name, &table);
    if (out != stdout) {
        fclose(out);
    }
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <bundle.dontology> [ontology-name] [output.sio]\n", argv[0]);
        return 2;
    }

    const char *bundle_path = argv[1];
    const char *ontology_name = argc >= 3 ? argv[2] : "";
    const char *output_path = argc >= 4 ? argv[3] : "";

    if (dontology_emit_sio(bundle_path, ontology_name, output_path) != 0) {
        fprintf(stderr, "dontology importer failed for %s: %s\n", bundle_path, strerror(errno));
        return 1;
    }
    return 0;
}
