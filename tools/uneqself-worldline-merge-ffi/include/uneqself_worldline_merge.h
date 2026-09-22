#ifndef UNEQSELF_WORLDLINE_MERGE_H
#define UNEQSELF_WORLDLINE_MERGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UNEQSELF_WORLDLINE_MERGE_ABI_V1 1u

typedef struct {
    const uint8_t *ptr;
    size_t count;
} uneqself_digest_slice_v1;

typedef struct {
    const uint8_t *conflict_id_32;
    const uint8_t *producer_id_utf8;
    size_t producer_id_len;
    const uint8_t *implementation_digest_32;
    uneqself_digest_slice_v1 observed_facts;
    uneqself_digest_slice_v1 inferences;
    uneqself_digest_slice_v1 contradictions;
    uneqself_digest_slice_v1 missing_evidence;
    uneqself_digest_slice_v1 open_obligations;
    uneqself_digest_slice_v1 alternative_proposals;
    uint32_t domain;
} uneqself_worldline_merge_analysis_input_v1;

/*
 * Emits canonical EpistemicMergeAnalysis v1 CBOR into out_buf.
 * Digest slices contain count consecutive 32-byte digests.
 * Returns the byte count or a negative refusal code. The artifact is advisory:
 * this ABI exposes no decision, signing, membership, or ledger mutation call.
 */
int32_t uneqself_worldline_merge_analyze_v1(
    const uneqself_worldline_merge_analysis_input_v1 *input,
    uint8_t *out_buf,
    size_t out_cap
);

#ifdef __cplusplus
}
#endif

#endif
