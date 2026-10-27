// CLDR locale-native fixture domain, intentionally distinct from UCA root kind4.
#define SOUNIO_CHUNK_KIND_MIN 5
#define SOUNIO_CHUNK_KIND_MAX 5
#define SOUNIO_CHUNK_NAME_PATTERN "^locale-[0-9]{6}\\.bin$"
#define SOUNIO_CHUNK_FILE_DESTINATION "tests/stdlib/wave3-core/locale-data/current.bin"
#include "collect_chunks.cpp"
