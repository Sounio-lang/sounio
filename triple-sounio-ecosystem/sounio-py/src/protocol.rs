/// SNIO binary wire protocol — Rust port of interop/test_protocol.py and Protocol.fs.
///
/// Frame format (little-endian throughout):
///   [4 bytes] magic = b"SNIO"
///   [1 byte]  msg_type
///   [4 bytes] body_len (u32 LE)
///   [body_len bytes] payload
///
/// CALL_FUNC payload:
///   [2 bytes] name_len (u16 LE)
///   [name_len bytes] func name (UTF-8)
///   [8 bytes] arg_count (i64 LE)
///   [arg_count * 8 bytes] args (i64 LE each)
///
/// RESULT payload:
///   [8 bytes] value_count (i64 LE)
///   [value_count * 8 bytes] values (i64 LE each)
///
/// ERROR payload:
///   [2 bytes] err_len (u16 LE)
///   [err_len bytes] error message (UTF-8)
use std::io::{self, Read, Write};

// ---- Message type constants ----

pub const MSG_CALL_FUNC: u8 = 1;
pub const MSG_RESULT: u8 = 2;
pub const MSG_ERROR: u8 = 3;
pub const MSG_SHUTDOWN: u8 = 4;
pub const MSG_INFO: u8 = 5;
pub const MSG_CAPABILITIES: u8 = 6;
pub const MSG_COMPILE: u8 = 7;
pub const MSG_DIAGNOSTICS: u8 = 9;
pub const MSG_HEALTH: u8 = 10;
pub const MSG_GPU_CAPS: u8 = 11;
pub const MSG_INIT: u8 = 12;
pub const MSG_STATS: u8 = 13;
pub const MSG_SESSION_CREATE: u8 = 14;
pub const MSG_SESSION_DESTROY: u8 = 15;
pub const MSG_KERNEL_DESCRIBE: u8 = 16;
pub const MSG_KERNEL_EXECUTE: u8 = 17;
pub const MSG_KERNEL_OUTPUT: u8 = 18;
pub const MSG_KERNEL_DIAGNOSTICS: u8 = 19;
pub const MSG_KERNEL_ARTIFACTS: u8 = 20;

pub const MAGIC: &[u8; 4] = b"SNIO";

// ---- Low-level read helpers ----

fn read_exact_bytes<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    reader.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16_le<R: Read>(reader: &mut R) -> io::Result<u16> {
    let buf = read_exact_bytes(reader, 2)?;
    Ok(u16::from_le_bytes(buf[..2].try_into().unwrap()))
}

fn read_u32_le<R: Read>(reader: &mut R) -> io::Result<u32> {
    let buf = read_exact_bytes(reader, 4)?;
    Ok(u32::from_le_bytes(buf[..4].try_into().unwrap()))
}

fn read_i64_le<R: Read>(reader: &mut R) -> io::Result<i64> {
    let buf = read_exact_bytes(reader, 8)?;
    Ok(i64::from_le_bytes(buf[..8].try_into().unwrap()))
}

// ---- Low-level write helpers ----

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_u16_le<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32_le<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_i64_le<W: Write>(w: &mut W, v: i64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_magic<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(MAGIC)
}

// ---- Response type ----

#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Result(Vec<i64>),
    Error(String),
    Shutdown,
    /// Unrecognized message type; body already consumed.
    Unknown(u8),
}

// ---- Write functions ----

/// Write a CALL_FUNC message.
pub fn write_call_func<W: Write>(w: &mut W, func_name: &str, args: &[i64]) -> io::Result<()> {
    let name_bytes = func_name.as_bytes();
    let name_len = name_bytes.len();
    let arg_count = args.len();
    // body = 2 (name_len) + name_len + 8 (arg_count) + arg_count * 8
    let body_len: u32 = (2 + name_len + 8 + arg_count * 8) as u32;

    write_magic(w)?;
    write_u8(w, MSG_CALL_FUNC)?;
    write_u32_le(w, body_len)?;
    write_u16_le(w, name_len as u16)?;
    w.write_all(name_bytes)?;
    write_i64_le(w, arg_count as i64)?;
    for &a in args {
        write_i64_le(w, a)?;
    }
    w.flush()
}

/// Write a simple message with empty payload (SHUTDOWN, HEALTH, INFO, etc.).
pub fn write_simple<W: Write>(w: &mut W, msg_type: u8) -> io::Result<()> {
    write_magic(w)?;
    write_u8(w, msg_type)?;
    write_u32_le(w, 0)?;
    w.flush()
}

pub fn write_shutdown<W: Write>(w: &mut W) -> io::Result<()> {
    write_simple(w, MSG_SHUTDOWN)
}

pub fn write_health<W: Write>(w: &mut W) -> io::Result<()> {
    write_simple(w, MSG_HEALTH)
}

pub fn write_info<W: Write>(w: &mut W) -> io::Result<()> {
    write_simple(w, MSG_INFO)
}

pub fn write_capabilities<W: Write>(w: &mut W) -> io::Result<()> {
    write_simple(w, MSG_CAPABILITIES)
}

/// Write a SESSION_CREATE message: payload = [0i64 placeholder][flags: i64].
pub fn write_session_create<W: Write>(w: &mut W, flags: i64) -> io::Result<()> {
    write_magic(w)?;
    write_u8(w, MSG_SESSION_CREATE)?;
    write_u32_le(w, 16)?; // 2 * 8 bytes
    write_i64_le(w, 0)?;  // session_id placeholder
    write_i64_le(w, flags)?;
    w.flush()
}

/// Write a KERNEL_EXECUTE message.
pub fn write_kernel_execute<W: Write>(
    w: &mut W,
    session_id: i64,
    kernel_id: i64,
    args: &[i64],
) -> io::Result<()> {
    let body_len: u32 = (8 + 8 + 8 + args.len() * 8) as u32;
    write_magic(w)?;
    write_u8(w, MSG_KERNEL_EXECUTE)?;
    write_u32_le(w, body_len)?;
    write_i64_le(w, session_id)?;
    write_i64_le(w, kernel_id)?;
    write_i64_le(w, args.len() as i64)?;
    for &a in args {
        write_i64_le(w, a)?;
    }
    w.flush()
}

// ---- Read function ----

/// Read one SNIO response message from the stream.
/// Returns `Response::Result`, `Response::Error`, or `Response::Shutdown`.
pub fn read_response<R: Read>(r: &mut R) -> io::Result<Response> {
    // Validate magic
    let magic = read_exact_bytes(r, 4)?;
    if magic.as_slice() != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Bad SNIO magic: {:?}", magic),
        ));
    }

    let msg_type = read_u8(r)?;
    let body_len = read_u32_le(r)?;

    match msg_type {
        MSG_RESULT => {
            let count = read_i64_le(r)? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_i64_le(r)?);
            }
            Ok(Response::Result(values))
        }
        MSG_ERROR => {
            let err_len = read_u16_le(r)? as usize;
            let bytes = read_exact_bytes(r, err_len)?;
            let msg = String::from_utf8_lossy(&bytes).into_owned();
            Ok(Response::Error(msg))
        }
        MSG_SHUTDOWN => Ok(Response::Shutdown),
        other => {
            // Consume unknown body to keep stream aligned
            let _ = read_exact_bytes(r, body_len as usize)?;
            Ok(Response::Unknown(other))
        }
    }
}

// ---- Unit tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Round-trip: write CALL_FUNC then parse what souc would reply (mock RESULT).
    #[test]
    fn test_write_call_func_bytes() {
        let mut buf = Vec::new();
        write_call_func(&mut buf, "ping", &[]).unwrap();

        // magic(4) + type(1) + body_len(4) + name_len(2) + "ping"(4) + arg_count(8)
        assert_eq!(&buf[..4], b"SNIO");
        assert_eq!(buf[4], MSG_CALL_FUNC);
        // body_len = 2 + 4 + 8 + 0 = 14
        let bl = u32::from_le_bytes(buf[5..9].try_into().unwrap());
        assert_eq!(bl, 14);
    }

    #[test]
    fn test_read_result_response() {
        // Build a synthetic RESULT message: [SNIO][0x02][body_len=16][count=1][value=1]
        let mut msg = Vec::new();
        msg.extend_from_slice(b"SNIO");
        msg.push(MSG_RESULT);
        msg.extend_from_slice(&16u32.to_le_bytes()); // body_len (count i64 + 1 value i64 = 16)
        msg.extend_from_slice(&1i64.to_le_bytes());  // count = 1
        msg.extend_from_slice(&1i64.to_le_bytes());  // value = 1

        let mut cursor = Cursor::new(msg);
        let resp = read_response(&mut cursor).unwrap();
        assert_eq!(resp, Response::Result(vec![1]));
    }

    #[test]
    fn test_read_error_response() {
        let err_text = b"unknown function";
        let err_len = err_text.len() as u16;
        let body_len = (2 + err_text.len()) as u32;

        let mut msg = Vec::new();
        msg.extend_from_slice(b"SNIO");
        msg.push(MSG_ERROR);
        msg.extend_from_slice(&body_len.to_le_bytes());
        msg.extend_from_slice(&err_len.to_le_bytes());
        msg.extend_from_slice(err_text);

        let mut cursor = Cursor::new(msg);
        let resp = read_response(&mut cursor).unwrap();
        assert_eq!(resp, Response::Error("unknown function".to_string()));
    }

    #[test]
    fn test_read_shutdown_response() {
        let mut msg = Vec::new();
        msg.extend_from_slice(b"SNIO");
        msg.push(MSG_SHUTDOWN);
        msg.extend_from_slice(&0u32.to_le_bytes());

        let mut cursor = Cursor::new(msg);
        let resp = read_response(&mut cursor).unwrap();
        assert_eq!(resp, Response::Shutdown);
    }

    #[test]
    fn test_bad_magic() {
        let mut cursor = Cursor::new(b"XXXX\x02\x00\x00\x00\x00");
        let result = read_response(&mut cursor);
        assert!(result.is_err());
    }
}
