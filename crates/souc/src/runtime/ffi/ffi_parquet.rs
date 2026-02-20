//! Minimal Parquet Column Reader for Sounio
//!
//! Provides C-callable functions for reading Parquet files without external
//! dependencies. Implements the minimal subset of the Parquet format needed
//! to read flat (non-nested) DOUBLE and INT64 columns.
//!
//! Feature-gated behind `parquet` to keep the default build lean.
//!
//! # Parquet Format Overview
//!
//! Parquet files have a footer containing the schema and row group metadata.
//! The footer length is stored in the last 8 bytes: [4-byte length][PAR1].
//! We read the footer to find column chunk offsets, then read column data
//! using PLAIN encoding (the simplest encoding).
//!
//! # Limitations
//!
//! - Only reads PLAIN-encoded columns (no dictionary, RLE, or delta encoding)
//! - Only supports DOUBLE and INT64 physical types
//! - No nested types (only flat schemas)
//! - No page-level compression (Snappy/Gzip/LZ4 not decoded)
//! - No predicate pushdown or column pruning at the page level
//!
//! For full Parquet support, link against Apache Arrow C++ via FFI.
//!
//! # References
//!
//! - Apache Parquet format spec: <https://parquet.apache.org/documentation/latest/>

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

/// Parquet magic bytes
const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Parquet physical types (from the Thrift schema)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ParquetType {
    Boolean = 0,
    Int32 = 1,
    Int64 = 2,
    Int96 = 3,
    Float = 4,
    Double = 5,
    ByteArray = 6,
    FixedLenByteArray = 7,
}

/// Parquet encoding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Encoding {
    Plain = 0,
    PlainDictionary = 2,
    Rle = 3,
    BitPacked = 4,
    DeltaBinaryPacked = 5,
    DeltaLengthByteArray = 6,
    DeltaByteArray = 7,
    RleDictionary = 8,
}

/// A column of data read from a Parquet file.
#[derive(Debug, Clone)]
pub enum ParquetColumn {
    Double(Vec<f64>),
    Int64(Vec<i64>),
    Float(Vec<f32>),
    Int32(Vec<i32>),
}

impl ParquetColumn {
    pub fn len(&self) -> usize {
        match self {
            ParquetColumn::Double(v) => v.len(),
            ParquetColumn::Int64(v) => v.len(),
            ParquetColumn::Float(v) => v.len(),
            ParquetColumn::Int32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            ParquetColumn::Double(v) => v.clone(),
            ParquetColumn::Int64(v) => v.iter().map(|&x| x as f64).collect(),
            ParquetColumn::Float(v) => v.iter().map(|&x| x as f64).collect(),
            ParquetColumn::Int32(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
}

/// Metadata for a column in a Parquet file.
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub physical_type: ParquetType,
    pub num_values: usize,
    pub data_offset: u64,
    pub data_size: usize,
}

/// Parquet file reader (minimal implementation).
#[derive(Debug)]
pub struct ParquetReader {
    pub columns: Vec<ColumnMeta>,
    pub num_rows: usize,
}

/// Error type for Parquet operations.
#[derive(Debug)]
pub enum ParquetError {
    Io(io::Error),
    InvalidMagic,
    InvalidFooter,
    UnsupportedEncoding(u8),
    UnsupportedType(u8),
    ColumnNotFound(String),
}

impl From<io::Error> for ParquetError {
    fn from(e: io::Error) -> Self {
        ParquetError::Io(e)
    }
}

impl std::fmt::Display for ParquetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParquetError::Io(e) => write!(f, "I/O error: {e}"),
            ParquetError::InvalidMagic => write!(f, "Not a Parquet file (invalid magic)"),
            ParquetError::InvalidFooter => write!(f, "Invalid Parquet footer"),
            ParquetError::UnsupportedEncoding(e) => write!(f, "Unsupported encoding: {e}"),
            ParquetError::UnsupportedType(t) => write!(f, "Unsupported physical type: {t}"),
            ParquetError::ColumnNotFound(n) => write!(f, "Column not found: {n}"),
        }
    }
}

impl std::error::Error for ParquetError {}

/// Read Parquet footer metadata from a file.
///
/// The footer is at the end of the file:
/// `[... data ...][footer bytes][4-byte footer length]["PAR1"]`
pub fn read_parquet_metadata<R: Read + Seek>(reader: &mut R) -> Result<ParquetReader, ParquetError> {
    // Check file magic at start
    reader.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != PARQUET_MAGIC {
        return Err(ParquetError::InvalidMagic);
    }

    // Read footer length from last 8 bytes
    reader.seek(SeekFrom::End(-8))?;
    let mut footer_buf = [0u8; 8];
    reader.read_exact(&mut footer_buf)?;

    let footer_len = u32::from_le_bytes([footer_buf[0], footer_buf[1], footer_buf[2], footer_buf[3]]) as usize;

    if &footer_buf[4..8] != PARQUET_MAGIC {
        return Err(ParquetError::InvalidMagic);
    }

    // Read footer bytes (Thrift-encoded FileMetaData)
    reader.seek(SeekFrom::End(-(8 + footer_len as i64)))?;
    let mut footer = vec![0u8; footer_len];
    reader.read_exact(&mut footer)?;

    // Parse minimal Thrift: extract schema elements and row group column chunks
    // This is a simplified parser for the most common Parquet layout
    parse_thrift_metadata(&footer, reader)
}

/// Read a specific column by name from a Parquet file.
pub fn read_parquet_column<R: Read + Seek>(
    reader: &mut R,
    meta: &ParquetReader,
    column_name: &str,
) -> Result<ParquetColumn, ParquetError> {
    let col = meta
        .columns
        .iter()
        .find(|c| c.name == column_name)
        .ok_or_else(|| ParquetError::ColumnNotFound(column_name.to_string()))?;

    read_column_data(reader, col)
}

/// Read all columns from a Parquet file into a map.
pub fn read_parquet_all<R: Read + Seek>(
    reader: &mut R,
    meta: &ParquetReader,
) -> Result<HashMap<String, ParquetColumn>, ParquetError> {
    let mut result = HashMap::new();
    for col in &meta.columns {
        let data = read_column_data(reader, col)?;
        result.insert(col.name.clone(), data);
    }
    Ok(result)
}

fn read_column_data<R: Read + Seek>(
    reader: &mut R,
    col: &ColumnMeta,
) -> Result<ParquetColumn, ParquetError> {
    reader.seek(SeekFrom::Start(col.data_offset))?;
    let mut buf = vec![0u8; col.data_size];
    reader.read_exact(&mut buf)?;

    // Skip page header (simplified: assume data page with PLAIN encoding)
    // A real implementation would parse the Thrift page header first
    let data = skip_page_header(&buf);

    match col.physical_type {
        ParquetType::Double => {
            let values: Vec<f64> = data
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]))
                .collect();
            Ok(ParquetColumn::Double(values))
        }
        ParquetType::Int64 => {
            let values: Vec<i64> = data
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]))
                .collect();
            Ok(ParquetColumn::Int64(values))
        }
        ParquetType::Float => {
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]))
                .collect();
            Ok(ParquetColumn::Float(values))
        }
        ParquetType::Int32 => {
            let values: Vec<i32> = data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]))
                .collect();
            Ok(ParquetColumn::Int32(values))
        }
        other => Err(ParquetError::UnsupportedType(other as u8)),
    }
}

/// Skip a Thrift-encoded page header.
///
/// Parquet data pages start with a Thrift-encoded PageHeader.
/// For PLAIN-encoded data pages, we need to skip past this header
/// to get to the raw data. This is a heuristic skip.
fn skip_page_header(data: &[u8]) -> &[u8] {
    // Simplified: scan for the end of the Thrift struct (field type 0 = STOP)
    // In practice, we'd parse the Thrift compact protocol properly
    let mut pos = 0;
    let len = data.len();

    // Try to find the data after the page header
    // Page header fields: page_type, uncompressed_size, compressed_size, ...
    // After the header, data begins
    while pos < len.min(256) {
        // Look for a run of consecutive bytes that could be IEEE 754 doubles
        // This is a heuristic — a proper implementation parses the Thrift header
        if pos + 8 <= len {
            let candidate = f64::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
                data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
            ]);
            if candidate.is_finite() && candidate.abs() < 1e30 {
                // Likely found the start of data
                return &data[pos..];
            }
        }
        pos += 1;
    }

    // Fallback: return all data
    data
}

/// Parse minimal Thrift metadata from a Parquet footer.
///
/// This is a simplified parser that handles the most common case:
/// a single row group with flat (non-nested) columns using PLAIN encoding.
fn parse_thrift_metadata<R: Read + Seek>(
    footer: &[u8],
    _reader: &mut R,
) -> Result<ParquetReader, ParquetError> {
    // The Parquet FileMetaData is Thrift-encoded. We use a minimal parser.
    let mut parser = ThriftParser::new(footer);

    let mut num_rows = 0u64;
    let mut schema_names: Vec<String> = Vec::new();
    let mut schema_types: Vec<ParquetType> = Vec::new();
    let mut columns: Vec<ColumnMeta> = Vec::new();

    // Parse top-level FileMetaData struct
    loop {
        let (field_type, field_id) = match parser.read_field_header() {
            Some(h) => h,
            None => break,
        };

        match field_id {
            1 => {
                // version (i32)
                parser.skip_value(field_type);
            }
            2 => {
                // schema (list of SchemaElement)
                let (elem_type, count) = parser.read_list_header();
                for _ in 0..count {
                    let (name, ptype) = parser.read_schema_element(elem_type);
                    if let Some(t) = ptype {
                        schema_names.push(name);
                        schema_types.push(t);
                    }
                    // Skip root element (has no type)
                }
            }
            3 => {
                // num_rows (i64)
                num_rows = parser.read_i64() as u64;
            }
            4 => {
                // row_groups (list of RowGroup)
                let (_elem_type, rg_count) = parser.read_list_header();
                for _ in 0..rg_count {
                    let rg_columns = parser.read_row_group();
                    for (i, (offset, size, num_values)) in rg_columns.into_iter().enumerate() {
                        if i < schema_names.len() {
                            columns.push(ColumnMeta {
                                name: schema_names[i].clone(),
                                physical_type: if i < schema_types.len() {
                                    schema_types[i]
                                } else {
                                    ParquetType::Double
                                },
                                num_values,
                                data_offset: offset,
                                data_size: size,
                            });
                        }
                    }
                }
            }
            _ => {
                parser.skip_value(field_type);
            }
        }
    }

    Ok(ParquetReader {
        columns,
        num_rows: num_rows as usize,
    })
}

/// Minimal Thrift compact protocol parser.
struct ThriftParser<'a> {
    data: &'a [u8],
    pos: usize,
    last_field_id: i16,
}

impl<'a> ThriftParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            last_field_id: 0,
        }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    fn read_varint(&mut self) -> u64 {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            if self.remaining() == 0 {
                break;
            }
            let b = self.read_byte();
            result |= ((b & 0x7F) as u64) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        result
    }

    fn read_zigzag_i32(&mut self) -> i32 {
        let n = self.read_varint() as u32;
        ((n >> 1) as i32) ^ -((n & 1) as i32)
    }

    fn read_i64(&mut self) -> i64 {
        let n = self.read_varint();
        ((n >> 1) as i64) ^ -((n & 1) as i64)
    }

    fn read_string(&mut self) -> String {
        let len = self.read_varint() as usize;
        if self.pos + len > self.data.len() {
            self.pos = self.data.len();
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).to_string();
        self.pos += len;
        s
    }

    fn read_field_header(&mut self) -> Option<(u8, i16)> {
        if self.remaining() == 0 {
            return None;
        }

        let b = self.read_byte();
        if b == 0 {
            return None; // STOP
        }

        let delta = (b >> 4) as i16;
        let field_type = b & 0x0F;

        let field_id = if delta != 0 {
            self.last_field_id + delta
        } else {
            self.read_zigzag_i32() as i16
        };

        self.last_field_id = field_id;
        Some((field_type, field_id))
    }

    fn read_list_header(&mut self) -> (u8, usize) {
        let b = self.read_byte();
        let size_high = (b >> 4) as usize;
        let elem_type = b & 0x0F;

        let count = if size_high == 0x0F {
            self.read_varint() as usize
        } else {
            size_high
        };

        (elem_type, count)
    }

    fn skip_value(&mut self, field_type: u8) {
        match field_type {
            1 | 2 => {} // bool true/false
            3 | 4 => { self.read_varint(); } // i8/i16
            5 => { self.read_varint(); } // i32
            6 => { self.read_varint(); } // i64
            7 => { self.pos += 8.min(self.remaining()); } // double
            8 => { let len = self.read_varint() as usize; self.pos += len.min(self.remaining()); } // binary/string
            9 => { // list
                let (et, count) = self.read_list_header();
                for _ in 0..count { self.skip_value(et); }
            }
            11 => { // map
                let b = self.read_byte();
                let count = if b == 0 { 0 } else {
                    let n = self.read_varint() as usize;
                    let _kt = (b >> 4) & 0x0F;
                    let _vt = b & 0x0F;
                    n
                };
                for _ in 0..count {
                    self.skip_value((b >> 4) & 0x0F);
                    self.skip_value(b & 0x0F);
                }
            }
            12 => { // struct
                self.last_field_id = 0;
                loop {
                    match self.read_field_header() {
                        Some((ft, _)) => self.skip_value(ft),
                        None => break,
                    }
                }
            }
            _ => {}
        }
    }

    fn read_schema_element(&mut self, _elem_type: u8) -> (String, Option<ParquetType>) {
        let saved_last = self.last_field_id;
        self.last_field_id = 0;

        let mut name = String::new();
        let mut ptype: Option<ParquetType> = None;

        loop {
            let (field_type, field_id) = match self.read_field_header() {
                Some(h) => h,
                None => break,
            };

            match field_id {
                1 => {
                    // type (enum/i32)
                    let t = self.read_varint() as u8;
                    ptype = match t {
                        0 => Some(ParquetType::Boolean),
                        1 => Some(ParquetType::Int32),
                        2 => Some(ParquetType::Int64),
                        3 => Some(ParquetType::Int96),
                        4 => Some(ParquetType::Float),
                        5 => Some(ParquetType::Double),
                        6 => Some(ParquetType::ByteArray),
                        7 => Some(ParquetType::FixedLenByteArray),
                        _ => None,
                    };
                }
                4 => {
                    // name (string)
                    name = self.read_string();
                }
                _ => {
                    self.skip_value(field_type);
                }
            }
        }

        self.last_field_id = saved_last;
        (name, ptype)
    }

    fn read_row_group(&mut self) -> Vec<(u64, usize, usize)> {
        let saved_last = self.last_field_id;
        self.last_field_id = 0;
        let mut chunk_infos = Vec::new();

        loop {
            let (field_type, field_id) = match self.read_field_header() {
                Some(h) => h,
                None => break,
            };

            match field_id {
                1 => {
                    // columns (list of ColumnChunk)
                    let (_et, count) = self.read_list_header();
                    for _ in 0..count {
                        let info = self.read_column_chunk();
                        chunk_infos.push(info);
                    }
                }
                _ => {
                    self.skip_value(field_type);
                }
            }
        }

        self.last_field_id = saved_last;
        chunk_infos
    }

    fn read_column_chunk(&mut self) -> (u64, usize, usize) {
        let saved_last = self.last_field_id;
        self.last_field_id = 0;

        let mut offset = 0u64;
        let mut size = 0usize;
        let mut num_values = 0usize;

        loop {
            let (field_type, field_id) = match self.read_field_header() {
                Some(h) => h,
                None => break,
            };

            match field_id {
                2 => {
                    // meta_data (ColumnMetaData struct)
                    let saved2 = self.last_field_id;
                    self.last_field_id = 0;

                    loop {
                        let (ft, fid) = match self.read_field_header() {
                            Some(h) => h,
                            None => break,
                        };

                        match fid {
                            6 => { num_values = self.read_i64() as usize; }
                            9 => { offset = self.read_i64() as u64; }
                            10 => { size = self.read_i64() as usize; }
                            _ => { self.skip_value(ft); }
                        }
                    }

                    self.last_field_id = saved2;
                }
                _ => {
                    self.skip_value(field_type);
                }
            }
        }

        self.last_field_id = saved_last;
        (offset, size, num_values)
    }
}

// =============================================================================
// C-callable FFI for Sounio runtime
// =============================================================================

/// Opaque handle for a Parquet reader.
pub struct ParquetHandle {
    reader: ParquetReader,
    file: std::fs::File,
}

/// Open a Parquet file and read its metadata.
///
/// Returns a handle (pointer) for subsequent column reads, or null on failure.
///
/// # Safety
///
/// `path_ptr` must point to `path_len` valid UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_open(
    path_ptr: *const u8,
    path_len: usize,
) -> *mut ParquetHandle {
    if path_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let path_slice = std::slice::from_raw_parts(path_ptr, path_len);
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let mut file = match std::fs::File::open(path_str) {
        Ok(f) => f,
        Err(_) => return std::ptr::null_mut(),
    };

    let reader = match read_parquet_metadata(&mut file) {
        Ok(r) => r,
        Err(_) => return std::ptr::null_mut(),
    };

    Box::into_raw(Box::new(ParquetHandle { reader, file }))
}

/// Get number of rows in a Parquet file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_num_rows(handle: *const ParquetHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).reader.num_rows
}

/// Get number of columns in a Parquet file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_num_cols(handle: *const ParquetHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).reader.columns.len()
}

/// Read a column as f64 values.
///
/// Returns a heap-allocated buffer. Caller must free with `__sounio_parquet_free`.
///
/// # Safety
///
/// `name_ptr`/`name_len` must be a valid UTF-8 string. `out_len` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_read_f64(
    handle: *mut ParquetHandle,
    name_ptr: *const u8,
    name_len: usize,
    out_len: *mut usize,
) -> *mut f64 {
    if handle.is_null() || name_ptr.is_null() || out_len.is_null() {
        return std::ptr::null_mut();
    }

    let h = &mut *handle;
    let name_slice = std::slice::from_raw_parts(name_ptr, name_len);
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let column = match read_parquet_column(&mut h.file, &h.reader, name_str) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };

    let mut values = column.as_f64();
    *out_len = values.len();
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

/// Close a Parquet handle and free its resources.
///
/// # Safety
///
/// `handle` must have been returned by `__sounio_parquet_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_close(handle: *mut ParquetHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Free a buffer allocated by `__sounio_parquet_read_f64`.
///
/// # Safety
///
/// `ptr` must have been returned by a Parquet read function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_parquet_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_magic() {
        assert_eq!(PARQUET_MAGIC, b"PAR1");
    }

    #[test]
    fn test_thrift_varint() {
        let data = [0x96, 0x01]; // 150 in varint
        let mut parser = ThriftParser::new(&data);
        assert_eq!(parser.read_varint(), 150);
    }

    #[test]
    fn test_thrift_zigzag() {
        // zigzag(0) = 0, zigzag(1) = -1, zigzag(2) = 1, zigzag(3) = -2
        let data = [0x00];
        let mut parser = ThriftParser::new(&data);
        assert_eq!(parser.read_zigzag_i32(), 0);

        let data = [0x01];
        let mut parser = ThriftParser::new(&data);
        assert_eq!(parser.read_zigzag_i32(), -1);

        let data = [0x02];
        let mut parser = ThriftParser::new(&data);
        assert_eq!(parser.read_zigzag_i32(), 1);
    }

    #[test]
    fn test_thrift_string() {
        let data = [0x05, b'h', b'e', b'l', b'l', b'o'];
        let mut parser = ThriftParser::new(&data);
        assert_eq!(parser.read_string(), "hello");
    }

    #[test]
    fn test_invalid_magic() {
        let data = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut cursor = std::io::Cursor::new(data);
        let result = read_parquet_metadata(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_as_f64() {
        let col = ParquetColumn::Int64(vec![1, 2, 3]);
        let f = col.as_f64();
        assert_eq!(f, vec![1.0, 2.0, 3.0]);

        let col = ParquetColumn::Double(vec![1.5, 2.5]);
        assert_eq!(col.len(), 2);
        assert!(!col.is_empty());
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            assert!(__sounio_parquet_open(std::ptr::null(), 0).is_null());
            assert_eq!(__sounio_parquet_num_rows(std::ptr::null()), 0);
            assert_eq!(__sounio_parquet_num_cols(std::ptr::null()), 0);
            assert!(
                __sounio_parquet_read_f64(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut()
                )
                .is_null()
            );
        }
    }
}
