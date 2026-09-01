//! Stable C ABI for digest-only SYNC-004 epistemic analysis.
//!
//! The exported surface can construct the exact protocol artifact, but it has
//! no function capable of selecting a branch, signing a decision, mutating a
//! ledger, or observing protected plaintext.

use sha2::{Digest, Sha256};
use std::{slice, str};

const MAX_DIGESTS_PER_CATEGORY: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DigestSliceV1 {
    pub ptr: *const u8,
    pub count: usize,
}

#[repr(C)]
pub struct WorldlineMergeAnalysisInputV1 {
    pub conflict_id_32: *const u8,
    pub producer_id_utf8: *const u8,
    pub producer_id_len: usize,
    pub implementation_digest_32: *const u8,
    pub observed_facts: DigestSliceV1,
    pub inferences: DigestSliceV1,
    pub contradictions: DigestSliceV1,
    pub missing_evidence: DigestSliceV1,
    pub open_obligations: DigestSliceV1,
    pub alternative_proposals: DigestSliceV1,
    pub domain: u32,
}

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    NullPointer = -1,
    InvalidProducer = -2,
    InvalidDomain = -3,
    InvalidDigestSlice = -4,
    OutputTooSmall = -5,
    SizeOverflow = -6,
}

fn head(major: u8, value: u64, out: &mut Vec<u8>) {
    let prefix = major << 5;
    match value {
        0..=23 => out.push(prefix | value as u8),
        24..=0xff => out.extend_from_slice(&[prefix | 24, value as u8]),
        0x100..=0xffff => {
            out.push(prefix | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(prefix | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            out.push(prefix | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
}

fn unsigned(value: u64, out: &mut Vec<u8>) {
    head(0, value, out);
}
fn map(count: u64, out: &mut Vec<u8>) {
    head(5, count, out);
}
fn array(count: u64, out: &mut Vec<u8>) {
    head(4, count, out);
}

fn bytes(value: &[u8], out: &mut Vec<u8>) {
    head(2, value.len() as u64, out);
    out.extend_from_slice(value);
}

fn text(value: &str, out: &mut Vec<u8>) {
    head(3, value.len() as u64, out);
    out.extend_from_slice(value.as_bytes());
}

fn hexadecimal(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(value.len() * 2);
    for byte in value {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

unsafe fn digest_slice(value: DigestSliceV1) -> Result<&'static [u8], Refusal> {
    if value.count > MAX_DIGESTS_PER_CATEGORY {
        return Err(Refusal::InvalidDigestSlice);
    }
    let length = value.count.checked_mul(32).ok_or(Refusal::SizeOverflow)?;
    if length == 0 {
        return Ok(&[]);
    }
    if value.ptr.is_null() {
        return Err(Refusal::NullPointer);
    }
    Ok(unsafe { slice::from_raw_parts(value.ptr, length) })
}

fn digest_array(value: &[u8], out: &mut Vec<u8>) {
    array((value.len() / 32) as u64, out);
    for digest in value.chunks_exact(32) {
        bytes(digest, out);
    }
}

fn encode_body(
    conflict_id: &[u8],
    producer_id: &str,
    implementation_digest: &[u8],
    categories: &[&[u8]; 6],
    domain: &str,
) -> Vec<u8> {
    let mut out = Vec::new();
    map(10, &mut out);
    unsigned(2, &mut out);
    text(&hexadecimal(conflict_id), &mut out);
    unsigned(3, &mut out);
    text(producer_id, &mut out);
    unsigned(4, &mut out);
    bytes(implementation_digest, &mut out);
    for (offset, category) in categories.iter().enumerate() {
        unsigned(5 + offset as u64, &mut out);
        digest_array(category, &mut out);
    }
    unsigned(11, &mut out);
    text(domain, &mut out);
    out
}

fn encode_artifact(body: &[u8], analysis_id: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::new();
    map(12, &mut out);
    unsigned(0, &mut out);
    unsigned(1, &mut out);
    unsigned(1, &mut out);
    text(&hexadecimal(analysis_id), &mut out);
    // The body is already the canonical map header followed by keys 2..11.
    // Drop its one-byte map(10) header and splice its ordered pairs.
    out.extend_from_slice(&body[1..]);
    out
}

unsafe fn analyze(input: &WorldlineMergeAnalysisInputV1) -> Result<Vec<u8>, Refusal> {
    if input.conflict_id_32.is_null()
        || input.producer_id_utf8.is_null()
        || input.implementation_digest_32.is_null()
    {
        return Err(Refusal::NullPointer);
    }
    if input.producer_id_len == 0 || input.producer_id_len > 255 {
        return Err(Refusal::InvalidProducer);
    }
    let producer_raw =
        unsafe { slice::from_raw_parts(input.producer_id_utf8, input.producer_id_len) };
    let producer = str::from_utf8(producer_raw).map_err(|_| Refusal::InvalidProducer)?;
    if producer.chars().any(char::is_control) {
        return Err(Refusal::InvalidProducer);
    }
    let domain = match input.domain {
        0 => "general",
        1 => "protected",
        2 => "body",
        3 => "research",
        4 => "operations",
        _ => return Err(Refusal::InvalidDomain),
    };
    let conflict_id = unsafe { slice::from_raw_parts(input.conflict_id_32, 32) };
    let implementation = unsafe { slice::from_raw_parts(input.implementation_digest_32, 32) };
    let categories = [
        unsafe { digest_slice(input.observed_facts)? },
        unsafe { digest_slice(input.inferences)? },
        unsafe { digest_slice(input.contradictions)? },
        unsafe { digest_slice(input.missing_evidence)? },
        unsafe { digest_slice(input.open_obligations)? },
        unsafe { digest_slice(input.alternative_proposals)? },
    ];
    let body = encode_body(conflict_id, producer, implementation, &categories, domain);
    let analysis_id: [u8; 32] = Sha256::digest(&body).into();
    Ok(encode_artifact(&body, &analysis_id))
}

/// Emit canonical `EpistemicMergeAnalysis` v1 CBOR.
///
/// # Safety
/// Every non-null input pointer must remain valid for its declared length, and
/// `out_buf` must be writable for `out_cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn uneqself_worldline_merge_analyze_v1(
    input: *const WorldlineMergeAnalysisInputV1,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if input.is_null() || out_buf.is_null() {
        return Refusal::NullPointer as i32;
    }
    let artifact = match unsafe { analyze(&*input) } {
        Ok(value) => value,
        Err(error) => return error as i32,
    };
    if artifact.len() > out_cap || artifact.len() > i32::MAX as usize {
        return Refusal::OutputTooSmall as i32;
    }
    unsafe { std::ptr::copy_nonoverlapping(artifact.as_ptr(), out_buf, artifact.len()) };
    artifact.len() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated(value: u8) -> [u8; 32] {
        [value; 32]
    }

    #[test]
    fn public_protocol_vector_is_byte_exact() {
        let conflict =
            hex_to_32("93c353beeb17ff144cdb9e8522ee8e3a48e935ffa70a3f54ba91fa44dfb9ac56");
        let producer = b"sounio-epistemic-v1";
        let implementation = repeated(0x31);
        let categories = [
            repeated(0x41),
            repeated(0x42),
            repeated(0x43),
            repeated(0x44),
            repeated(0x45),
            repeated(0x46),
        ];
        let slices = [
            DigestSliceV1 {
                ptr: categories[0].as_ptr(),
                count: 1,
            },
            DigestSliceV1 {
                ptr: categories[1].as_ptr(),
                count: 1,
            },
            DigestSliceV1 {
                ptr: categories[2].as_ptr(),
                count: 1,
            },
            DigestSliceV1 {
                ptr: categories[3].as_ptr(),
                count: 1,
            },
            DigestSliceV1 {
                ptr: categories[4].as_ptr(),
                count: 1,
            },
            DigestSliceV1 {
                ptr: categories[5].as_ptr(),
                count: 1,
            },
        ];
        let input = WorldlineMergeAnalysisInputV1 {
            conflict_id_32: conflict.as_ptr(),
            producer_id_utf8: producer.as_ptr(),
            producer_id_len: producer.len(),
            implementation_digest_32: implementation.as_ptr(),
            observed_facts: slices[0],
            inferences: slices[1],
            contradictions: slices[2],
            missing_evidence: slices[3],
            open_obligations: slices[4],
            alternative_proposals: slices[5],
            domain: 1,
        };
        let artifact = unsafe { analyze(&input) }.unwrap();
        assert_eq!(artifact.len(), 420);
        assert_eq!(
            hexadecimal(&Sha256::digest(&artifact)),
            "029d3c7a97b98db8774622f6031607a877d3ac1cf16303621c08be03e4949785",
        );
        assert!(artifact
            .windows(64)
            .any(|window| window
                == b"3084f720fcc926ea0f4d2a8d6d0d85f9f24b6ea310b05fff65d99fde193a097d"));
    }

    #[test]
    fn malformed_inputs_fail_closed_without_partial_authority() {
        let mut output = [0xaa; 32];
        let result = unsafe {
            uneqself_worldline_merge_analyze_v1(std::ptr::null(), output.as_mut_ptr(), output.len())
        };
        assert_eq!(result, Refusal::NullPointer as i32);
        assert_eq!(output, [0xaa; 32]);
    }

    fn hex_to_32(raw: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&raw[index * 2..index * 2 + 2], 16).unwrap();
        }
        out
    }
}
