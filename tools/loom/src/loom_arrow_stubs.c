#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

#define LOOM_ARROW_FIELD_COUNT 13

static const char *loom_arrow_names[LOOM_ARROW_FIELD_COUNT] = {
    "agent",          "lane",                "instance_id",
    "session_state",  "journal",             "sequence",
    "observed_at_utc", "kind",                "payload",
    "previous_sha256", "event_sha256",        "journal_head_sha256",
    "verified"};

static const enum ArrowType loom_arrow_types[LOOM_ARROW_FIELD_COUNT] = {
    NANOARROW_TYPE_STRING,            NANOARROW_TYPE_STRING,
    NANOARROW_TYPE_STRING,            NANOARROW_TYPE_STRING,
    NANOARROW_TYPE_STRING,            NANOARROW_TYPE_UINT64,
    NANOARROW_TYPE_STRING,            NANOARROW_TYPE_STRING,
    NANOARROW_TYPE_BINARY,            NANOARROW_TYPE_FIXED_SIZE_BINARY,
    NANOARROW_TYPE_FIXED_SIZE_BINARY, NANOARROW_TYPE_FIXED_SIZE_BINARY,
    NANOARROW_TYPE_BOOL};

static const char *loom_arrow_formats[LOOM_ARROW_FIELD_COUNT] = {
    "u", "u", "u", "u", "u", "L", "u", "u", "z", "w:32", "w:32",
    "w:32", "b"};

static struct ArrowStringView loom_arrow_string(value input) {
  struct ArrowStringView view;
  view.data = String_val(input);
  view.size_bytes = caml_string_length(input);
  return view;
}

static struct ArrowBufferView loom_arrow_bytes(value input) {
  struct ArrowBufferView view;
  view.data.data = String_val(input);
  view.size_bytes = caml_string_length(input);
  return view;
}

static int loom_arrow_hex_nibble(char character) {
  if (character >= '0' && character <= '9') return character - '0';
  if (character >= 'a' && character <= 'f') return character - 'a' + 10;
  if (character >= 'A' && character <= 'F') return character - 'A' + 10;
  return -1;
}

static int loom_arrow_digest(value hex, uint8_t out[32]) {
  if (caml_string_length(hex) != 64) return EINVAL;
  const char *input = String_val(hex);
  for (int index = 0; index < 32; index++) {
    int high = loom_arrow_hex_nibble(input[index * 2]);
    int low = loom_arrow_hex_nibble(input[index * 2 + 1]);
    if (high < 0 || low < 0) return EINVAL;
    out[index] = (uint8_t)((high << 4) | low);
  }
  return NANOARROW_OK;
}

static void loom_arrow_error(char *message, size_t capacity, const char *operation,
                             int code, struct ArrowError *error) {
  const char *detail = error == NULL ? "" : ArrowErrorMessage(error);
  if (detail == NULL || detail[0] == '\0') {
    snprintf(message, capacity, "%s failed with code %d", operation, code);
  } else {
    snprintf(message, capacity, "%s failed with code %d: %s", operation, code,
             detail);
  }
}

static int loom_arrow_schema_init(struct ArrowSchema *schema,
                                  struct ArrowError *error) {
  int code = ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT);
  if (code != NANOARROW_OK) return code;
  code = ArrowSchemaAllocateChildren(schema, LOOM_ARROW_FIELD_COUNT);
  if (code != NANOARROW_OK) return code;

  for (int index = 0; index < LOOM_ARROW_FIELD_COUNT; index++) {
    if (loom_arrow_types[index] == NANOARROW_TYPE_FIXED_SIZE_BINARY) {
      ArrowSchemaInit(schema->children[index]);
      code = ArrowSchemaSetTypeFixedSize(schema->children[index],
                                         NANOARROW_TYPE_FIXED_SIZE_BINARY, 32);
    } else {
      code = ArrowSchemaInitFromType(schema->children[index],
                                     loom_arrow_types[index]);
    }
    if (code != NANOARROW_OK) return code;
    code = ArrowSchemaSetName(schema->children[index], loom_arrow_names[index]);
    if (code != NANOARROW_OK) return code;
    schema->children[index]->flags &= ~ARROW_FLAG_NULLABLE;
  }

  (void)error;
  return NANOARROW_OK;
}

static int loom_arrow_append_event(struct ArrowArray *array, value event,
                                   struct ArrowError *error) {
  int code;
  uint8_t digest[32];

#define LOOM_APPEND_STRING(column, field)                                      \
  do {                                                                          \
    code = ArrowArrayAppendString(array->children[column],                      \
                                  loom_arrow_string(Field(event, field)));      \
    if (code != NANOARROW_OK) return code;                                      \
  } while (0)

  LOOM_APPEND_STRING(0, 0);
  LOOM_APPEND_STRING(1, 1);
  LOOM_APPEND_STRING(2, 2);
  LOOM_APPEND_STRING(3, 3);
  LOOM_APPEND_STRING(4, 4);

  int64_t sequence = Int64_val(Field(event, 5));
  if (sequence < 0) return EINVAL;
  code = ArrowArrayAppendUInt(array->children[5], (uint64_t)sequence);
  if (code != NANOARROW_OK) return code;

  LOOM_APPEND_STRING(6, 6);
  LOOM_APPEND_STRING(7, 7);
  code = ArrowArrayAppendBytes(array->children[8],
                               loom_arrow_bytes(Field(event, 8)));
  if (code != NANOARROW_OK) return code;

  for (int column = 9; column <= 11; column++) {
    code = loom_arrow_digest(Field(event, column), digest);
    if (code != NANOARROW_OK) return code;
    struct ArrowBufferView view;
    view.data.data = digest;
    view.size_bytes = sizeof(digest);
    code = ArrowArrayAppendBytes(array->children[column], view);
    if (code != NANOARROW_OK) return code;
  }

  code = ArrowArrayAppendInt(array->children[12], Bool_val(Field(event, 12)));
  if (code != NANOARROW_OK) return code;
  code = ArrowArrayFinishElement(array);
  if (code != NANOARROW_OK) return code;

  (void)error;
  return NANOARROW_OK;
#undef LOOM_APPEND_STRING
}

CAMLprim value sounio_loom_arrow_encode(value events) {
  CAMLparam1(events);
  CAMLlocal1(result);
  struct ArrowSchema schema;
  struct ArrowArray array;
  struct ArrowArrayView array_view;
  struct ArrowBuffer output;
  struct ArrowIpcOutputStream output_stream;
  struct ArrowIpcWriter writer;
  struct ArrowError error;
  char failure[1024] = {0};
  int code = NANOARROW_OK;

  memset(&schema, 0, sizeof(schema));
  memset(&array, 0, sizeof(array));
  memset(&array_view, 0, sizeof(array_view));
  memset(&output_stream, 0, sizeof(output_stream));
  memset(&writer, 0, sizeof(writer));
  memset(&error, 0, sizeof(error));
  ArrowBufferInit(&output);

#define LOOM_ARROW_CHECK(operation, expression)                                \
  do {                                                                          \
    code = (expression);                                                        \
    if (code != NANOARROW_OK) {                                                 \
      loom_arrow_error(failure, sizeof(failure), operation, code, &error);      \
      goto cleanup;                                                             \
    }                                                                           \
  } while (0)

  LOOM_ARROW_CHECK("schema initialization", loom_arrow_schema_init(&schema, &error));
  LOOM_ARROW_CHECK("array initialization",
                   ArrowArrayInitFromSchema(&array, &schema, &error));
  LOOM_ARROW_CHECK("array append initialization", ArrowArrayStartAppending(&array));

  mlsize_t event_count = Wosize_val(events);
  for (mlsize_t index = 0; index < event_count; index++) {
    LOOM_ARROW_CHECK("event append",
                     loom_arrow_append_event(&array, Field(events, index), &error));
  }

  LOOM_ARROW_CHECK("array validation",
                   ArrowArrayFinishBuildingDefault(&array, &error));
  LOOM_ARROW_CHECK("array view initialization",
                   ArrowArrayViewInitFromSchema(&array_view, &schema, &error));
  LOOM_ARROW_CHECK("array view binding",
                   ArrowArrayViewSetArray(&array_view, &array, &error));
  LOOM_ARROW_CHECK("IPC output initialization",
                   ArrowIpcOutputStreamInitBuffer(&output_stream, &output));
  LOOM_ARROW_CHECK("IPC writer initialization",
                   ArrowIpcWriterInit(&writer, &output_stream));
  LOOM_ARROW_CHECK("IPC schema write",
                   ArrowIpcWriterWriteSchema(&writer, &schema, &error));
  LOOM_ARROW_CHECK("IPC record batch write",
                   ArrowIpcWriterWriteArrayView(&writer, &array_view, &error));
  LOOM_ARROW_CHECK("IPC stream finalization",
                   ArrowIpcWriterWriteArrayView(&writer, NULL, &error));

  result = caml_alloc_string((mlsize_t)output.size_bytes);
  memcpy(Bytes_val(result), output.data, (size_t)output.size_bytes);

cleanup:
  if (writer.private_data != NULL) ArrowIpcWriterReset(&writer);
  if (output_stream.release != NULL) output_stream.release(&output_stream);
  ArrowArrayViewReset(&array_view);
  if (array.release != NULL) ArrowArrayRelease(&array);
  if (schema.release != NULL) ArrowSchemaRelease(&schema);
  ArrowBufferReset(&output);

  if (code != NANOARROW_OK) caml_failwith(failure);
  CAMLreturn(result);
#undef LOOM_ARROW_CHECK
}

static int loom_arrow_schema_validate(const struct ArrowSchema *schema,
                                      char *failure, size_t capacity) {
  if (schema->n_children != LOOM_ARROW_FIELD_COUNT) {
    snprintf(failure, capacity, "schema child count mismatch: expected %d got %lld",
             LOOM_ARROW_FIELD_COUNT, (long long)schema->n_children);
    return EINVAL;
  }

  for (int index = 0; index < LOOM_ARROW_FIELD_COUNT; index++) {
    const struct ArrowSchema *child = schema->children[index];
    const char *name = child->name == NULL ? "" : child->name;
    const char *format = child->format == NULL ? "" : child->format;
    if (strcmp(name, loom_arrow_names[index]) != 0 ||
        strcmp(format, loom_arrow_formats[index]) != 0) {
      snprintf(failure, capacity,
               "schema field %d mismatch: expected %s/%s got %s/%s", index,
               loom_arrow_names[index], loom_arrow_formats[index], name, format);
      return EINVAL;
    }
    if ((child->flags & ARROW_FLAG_NULLABLE) != 0) {
      snprintf(failure, capacity, "schema field %s unexpectedly nullable", name);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

CAMLprim value sounio_loom_arrow_inspect(value bytes) {
  CAMLparam1(bytes);
  CAMLlocal1(result);
  struct ArrowBuffer input;
  struct ArrowIpcInputStream input_stream;
  struct ArrowArrayStream stream;
  struct ArrowSchema schema;
  struct ArrowArray array;
  struct ArrowError error;
  char failure[1024] = {0};
  char summary[256];
  int code = NANOARROW_OK;
  int64_t row_count = 0;
  int64_t batch_count = 0;

  ArrowBufferInit(&input);
  memset(&input_stream, 0, sizeof(input_stream));
  memset(&stream, 0, sizeof(stream));
  memset(&schema, 0, sizeof(schema));
  memset(&array, 0, sizeof(array));
  memset(&error, 0, sizeof(error));

#define LOOM_ARROW_READ_CHECK(operation, expression)                           \
  do {                                                                          \
    code = (expression);                                                        \
    if (code != NANOARROW_OK) {                                                 \
      loom_arrow_error(failure, sizeof(failure), operation, code, &error);      \
      goto cleanup;                                                             \
    }                                                                           \
  } while (0)

  LOOM_ARROW_READ_CHECK(
      "IPC input copy",
      ArrowBufferAppend(&input, String_val(bytes), caml_string_length(bytes)));
  LOOM_ARROW_READ_CHECK("IPC input initialization",
                        ArrowIpcInputStreamInitBuffer(&input_stream, &input));
  LOOM_ARROW_READ_CHECK("IPC reader initialization",
                        ArrowIpcArrayStreamReaderInit(&stream, &input_stream, NULL));
  LOOM_ARROW_READ_CHECK("IPC schema read",
                        ArrowArrayStreamGetSchema(&stream, &schema, &error));
  code = loom_arrow_schema_validate(&schema, failure, sizeof(failure));
  if (code != NANOARROW_OK) goto cleanup;

  while (1) {
    LOOM_ARROW_READ_CHECK("IPC record batch read",
                          ArrowArrayStreamGetNext(&stream, &array, &error));
    if (array.release == NULL) break;
    row_count += array.length;
    batch_count++;
    ArrowArrayRelease(&array);
  }

  snprintf(summary, sizeof(summary),
           "schema=loom-spectral-events-v1 rows=%lld batches=%lld",
           (long long)row_count, (long long)batch_count);
  result = caml_copy_string(summary);

cleanup:
  if (array.release != NULL) ArrowArrayRelease(&array);
  if (schema.release != NULL) ArrowSchemaRelease(&schema);
  if (stream.release != NULL) ArrowArrayStreamRelease(&stream);
  if (input_stream.release != NULL) input_stream.release(&input_stream);
  ArrowBufferReset(&input);

  if (code != NANOARROW_OK) {
    if (failure[0] == '\0') {
      snprintf(failure, sizeof(failure), "Arrow IPC validation failed with code %d",
               code);
    }
    caml_failwith(failure);
  }
  CAMLreturn(result);
#undef LOOM_ARROW_READ_CHECK
}
