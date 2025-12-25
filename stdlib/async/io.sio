/// Async I/O Operations
///
/// Provides asynchronous file and I/O operations that don't block
/// the async runtime.

module async::io

import async::future::{Future, Poll, Context}

/// Result type for I/O operations
pub type IoResult<T> = Result<T, IoError>

/// I/O error type
pub struct IoError {
    kind: IoErrorKind,
    message: string,
}

impl IoError {
    /// Creates a new I/O error
    pub fn new(kind: IoErrorKind, message: string) -> IoError {
        IoError { kind, message }
    }

    /// Returns the error kind
    pub fn kind(&self) -> IoErrorKind {
        self.kind
    }

    /// Returns the error message
    pub fn message(&self) -> &string {
        &self.message
    }
}

/// Kinds of I/O errors
pub enum IoErrorKind {
    /// File or resource not found
    NotFound,
    /// Permission denied
    PermissionDenied,
    /// Connection refused
    ConnectionRefused,
    /// Connection reset
    ConnectionReset,
    /// Connection aborted
    ConnectionAborted,
    /// Not connected
    NotConnected,
    /// Address already in use
    AddrInUse,
    /// Address not available
    AddrNotAvailable,
    /// Broken pipe
    BrokenPipe,
    /// Resource already exists
    AlreadyExists,
    /// Would block (for non-blocking I/O)
    WouldBlock,
    /// Invalid input
    InvalidInput,
    /// Invalid data
    InvalidData,
    /// Timed out
    TimedOut,
    /// Write zero bytes
    WriteZero,
    /// Interrupted
    Interrupted,
    /// Unexpected end of file
    UnexpectedEof,
    /// Other error
    Other,
}

/// Async trait for reading bytes
pub trait AsyncRead {
    /// Attempts to read bytes into the buffer
    /// Returns the number of bytes read
    async fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> with Async, IO

    /// Reads exactly the requested number of bytes
    async fn read_exact(&mut self, buf: &mut [u8]) -> IoResult<()> with Async, IO {
        let mut filled = 0;
        while filled < buf.len() {
            match self.read(&mut buf[filled..]).await {
                Ok(0) => return Err(IoError::new(
                    IoErrorKind::UnexpectedEof,
                    "unexpected end of file".to_string()
                )),
                Ok(n) => filled += n,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Reads all bytes until EOF
    async fn read_to_end(&mut self, buf: &mut Vec<u8>) -> IoResult<usize> with Async, IO {
        let start_len = buf.len();
        let mut temp = [0u8; 1024];

        loop {
            match self.read(&mut temp).await {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&temp[..n]);
                }
                Err(e) if e.kind() == IoErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }

        Ok(buf.len() - start_len)
    }

    /// Reads all bytes until EOF into a string
    async fn read_to_string(&mut self, buf: &mut string) -> IoResult<usize> with Async, IO {
        let mut bytes = Vec::new();
        let n = self.read_to_end(&mut bytes).await?;

        match String::from_utf8(bytes) {
            Ok(s) => {
                buf.push_str(&s);
                Ok(n)
            }
            Err(_) => Err(IoError::new(
                IoErrorKind::InvalidData,
                "stream did not contain valid UTF-8".to_string()
            )),
        }
    }
}

/// Async trait for writing bytes
pub trait AsyncWrite {
    /// Attempts to write bytes from the buffer
    /// Returns the number of bytes written
    async fn write(&mut self, buf: &[u8]) -> IoResult<usize> with Async, IO

    /// Flushes the output stream
    async fn flush(&mut self) -> IoResult<()> with Async, IO

    /// Shuts down the output stream
    async fn shutdown(&mut self) -> IoResult<()> with Async, IO {
        self.flush().await
    }

    /// Writes all bytes from the buffer
    async fn write_all(&mut self, buf: &[u8]) -> IoResult<()> with Async, IO {
        let mut written = 0;
        while written < buf.len() {
            match self.write(&buf[written..]).await {
                Ok(0) => return Err(IoError::new(
                    IoErrorKind::WriteZero,
                    "failed to write whole buffer".to_string()
                )),
                Ok(n) => written += n,
                Err(e) if e.kind() == IoErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Async trait for seeking
pub trait AsyncSeek {
    /// Seeks to a position in the stream
    async fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> with Async, IO

    /// Returns the current position
    async fn stream_position(&mut self) -> IoResult<u64> with Async, IO {
        self.seek(SeekFrom::Current(0)).await
    }

    /// Rewinds to the beginning
    async fn rewind(&mut self) -> IoResult<()> with Async, IO {
        self.seek(SeekFrom::Start(0)).await?;
        Ok(())
    }
}

/// Position for seeking
pub enum SeekFrom {
    /// From the beginning of the stream
    Start(u64),
    /// From the end of the stream
    End(i64),
    /// From the current position
    Current(i64),
}

/// Async trait for buffered reading
pub trait AsyncBufRead: AsyncRead {
    /// Returns the contents of the internal buffer
    async fn fill_buf(&mut self) -> IoResult<&[u8]> with Async, IO

    /// Consumes bytes from the buffer
    fn consume(&mut self, amt: usize)

    /// Reads until a delimiter byte is found
    async fn read_until(&mut self, delim: u8, buf: &mut Vec<u8>) -> IoResult<usize> with Async, IO {
        let start_len = buf.len();

        loop {
            let available = self.fill_buf().await?;
            if available.is_empty() {
                break;
            }

            // Search for delimiter
            if let Some(i) = available.iter().position(|&b| b == delim) {
                buf.extend_from_slice(&available[..=i]);
                self.consume(i + 1);
                break;
            } else {
                buf.extend_from_slice(available);
                let len = available.len();
                self.consume(len);
            }
        }

        Ok(buf.len() - start_len)
    }

    /// Reads a line (until newline)
    async fn read_line(&mut self, buf: &mut string) -> IoResult<usize> with Async, IO {
        let mut bytes = Vec::new();
        let n = self.read_until(b'\n', &mut bytes).await?;

        match String::from_utf8(bytes) {
            Ok(s) => {
                buf.push_str(&s);
                Ok(n)
            }
            Err(_) => Err(IoError::new(
                IoErrorKind::InvalidData,
                "stream did not contain valid UTF-8".to_string()
            )),
        }
    }
}

/// An async file handle
pub struct File {
    /// Internal file descriptor or handle
    fd: i32,
    /// File path (for debugging)
    path: string,
}

impl File {
    /// Opens a file for reading
    pub async fn open(path: &str) -> IoResult<File> with Async, IO {
        // Would use actual async file open
        Ok(File {
            fd: 0,
            path: path.to_string(),
        })
    }

    /// Creates a new file for writing
    pub async fn create(path: &str) -> IoResult<File> with Async, IO {
        // Would use actual async file create
        Ok(File {
            fd: 0,
            path: path.to_string(),
        })
    }

    /// Opens a file with the given options
    pub async fn open_with(path: &str, options: &OpenOptions) -> IoResult<File> with Async, IO {
        // Would use actual async file open with options
        Ok(File {
            fd: 0,
            path: path.to_string(),
        })
    }

    /// Returns metadata about the file
    pub async fn metadata(&self) -> IoResult<Metadata> with Async, IO {
        // Would query actual file metadata
        Ok(Metadata {
            len: 0,
            is_dir: false,
            is_file: true,
            is_symlink: false,
            modified: None,
            accessed: None,
            created: None,
        })
    }

    /// Sets the length of the file
    pub async fn set_len(&self, size: u64) -> IoResult<()> with Async, IO {
        // Would truncate/extend the file
        Ok(())
    }

    /// Syncs all data to disk
    pub async fn sync_all(&self) -> IoResult<()> with Async, IO {
        // Would sync file data
        Ok(())
    }

    /// Syncs file data (but not metadata) to disk
    pub async fn sync_data(&self) -> IoResult<()> with Async, IO {
        // Would sync file data
        Ok(())
    }
}

impl AsyncRead for File {
    async fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        // Would perform actual async read
        Ok(0)
    }
}

impl AsyncWrite for File {
    async fn write(&mut self, buf: &[u8]) -> IoResult<usize> with Async, IO {
        // Would perform actual async write
        Ok(buf.len())
    }

    async fn flush(&mut self) -> IoResult<()> with Async, IO {
        Ok(())
    }
}

impl AsyncSeek for File {
    async fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> with Async, IO {
        // Would perform actual seek
        Ok(0)
    }
}

/// Options for opening files
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

impl OpenOptions {
    /// Creates a new set of options with all flags disabled
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
        }
    }

    /// Sets the read flag
    pub fn read(mut self, read: bool) -> OpenOptions {
        self.read = read;
        self
    }

    /// Sets the write flag
    pub fn write(mut self, write: bool) -> OpenOptions {
        self.write = write;
        self
    }

    /// Sets the append flag
    pub fn append(mut self, append: bool) -> OpenOptions {
        self.append = append;
        self
    }

    /// Sets the truncate flag
    pub fn truncate(mut self, truncate: bool) -> OpenOptions {
        self.truncate = truncate;
        self
    }

    /// Sets the create flag
    pub fn create(mut self, create: bool) -> OpenOptions {
        self.create = create;
        self
    }

    /// Sets the create_new flag
    pub fn create_new(mut self, create_new: bool) -> OpenOptions {
        self.create_new = create_new;
        self
    }

    /// Opens a file with these options
    pub async fn open(&self, path: &str) -> IoResult<File> with Async, IO {
        File::open_with(path, self).await
    }
}

/// File metadata
pub struct Metadata {
    len: u64,
    is_dir: bool,
    is_file: bool,
    is_symlink: bool,
    modified: Option<SystemTime>,
    accessed: Option<SystemTime>,
    created: Option<SystemTime>,
}

impl Metadata {
    /// Returns the file size
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns true if the file is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if this is a directory
    pub fn is_dir(&self) -> bool {
        self.is_dir
    }

    /// Returns true if this is a regular file
    pub fn is_file(&self) -> bool {
        self.is_file
    }

    /// Returns true if this is a symbolic link
    pub fn is_symlink(&self) -> bool {
        self.is_symlink
    }
}

/// Placeholder for system time
struct SystemTime {}

/// A buffered async reader
pub struct BufReader<R: AsyncRead> {
    inner: R,
    buf: Vec<u8>,
    pos: usize,
    cap: usize,
}

impl<R: AsyncRead> BufReader<R> {
    /// Creates a new buffered reader with default buffer size
    pub fn new(inner: R) -> BufReader<R> {
        BufReader::with_capacity(8192, inner)
    }

    /// Creates a new buffered reader with the given capacity
    pub fn with_capacity(capacity: usize, inner: R) -> BufReader<R> {
        BufReader {
            inner,
            buf: Vec::with_capacity(capacity),
            pos: 0,
            cap: 0,
        }
    }

    /// Returns a reference to the underlying reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Returns a mutable reference to the underlying reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consumes the BufReader, returning the underlying reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Returns the number of bytes available in the buffer
    pub fn buffer(&self) -> &[u8] {
        &self.buf[self.pos..self.cap]
    }
}

impl<R: AsyncRead> AsyncRead for BufReader<R> {
    async fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        // If buffer is empty, read more data
        if self.pos >= self.cap {
            self.cap = self.inner.read(&mut self.buf).await?;
            self.pos = 0;
        }

        // Copy from buffer to output
        let available = &self.buf[self.pos..self.cap];
        let amt = available.len().min(buf.len());
        buf[..amt].copy_from_slice(&available[..amt]);
        self.pos += amt;

        Ok(amt)
    }
}

impl<R: AsyncRead> AsyncBufRead for BufReader<R> {
    async fn fill_buf(&mut self) -> IoResult<&[u8]> with Async, IO {
        if self.pos >= self.cap {
            self.cap = self.inner.read(&mut self.buf).await?;
            self.pos = 0;
        }
        Ok(&self.buf[self.pos..self.cap])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.cap);
    }
}

/// A buffered async writer
pub struct BufWriter<W: AsyncWrite> {
    inner: W,
    buf: Vec<u8>,
}

impl<W: AsyncWrite> BufWriter<W> {
    /// Creates a new buffered writer with default buffer size
    pub fn new(inner: W) -> BufWriter<W> {
        BufWriter::with_capacity(8192, inner)
    }

    /// Creates a new buffered writer with the given capacity
    pub fn with_capacity(capacity: usize, inner: W) -> BufWriter<W> {
        BufWriter {
            inner,
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of bytes in the buffer
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Returns a reference to the underlying writer
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Returns a mutable reference to the underlying writer
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Consumes the BufWriter, returning the underlying writer
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: AsyncWrite> AsyncWrite for BufWriter<W> {
    async fn write(&mut self, buf: &[u8]) -> IoResult<usize> with Async, IO {
        // If buffer is full or data is large, flush first
        if self.buf.len() + buf.len() > self.buf.capacity() {
            self.flush().await?;
        }

        // If data fits in buffer, buffer it
        if buf.len() < self.buf.capacity() {
            self.buf.extend_from_slice(buf);
            Ok(buf.len())
        } else {
            // Write directly
            self.inner.write(buf).await
        }
    }

    async fn flush(&mut self) -> IoResult<()> with Async, IO {
        if !self.buf.is_empty() {
            self.inner.write_all(&self.buf).await?;
            self.buf.clear();
        }
        self.inner.flush().await
    }
}

/// Convenience functions for common file operations

/// Reads an entire file into a string
pub async fn read_to_string(path: &str) -> IoResult<string> with Async, IO {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}

/// Reads an entire file into a byte vector
pub async fn read(path: &str) -> IoResult<Vec<u8>> with Async, IO {
    let mut file = File::open(path).await?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).await?;
    Ok(contents)
}

/// Writes bytes to a file, creating it if it doesn't exist
pub async fn write(path: &str, contents: &[u8]) -> IoResult<()> with Async, IO {
    let mut file = File::create(path).await?;
    file.write_all(contents).await
}

/// Writes a string to a file, creating it if it doesn't exist
pub async fn write_string(path: &str, contents: &str) -> IoResult<()> with Async, IO {
    write(path, contents.as_bytes()).await
}

/// Copies one file to another
pub async fn copy(from: &str, to: &str) -> IoResult<u64> with Async, IO {
    let mut src = File::open(from).await?;
    let mut dst = File::create(to).await?;

    let mut buf = [0u8; 8192];
    let mut copied = 0u64;

    loop {
        let n = src.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).await?;
        copied += n as u64;
    }

    Ok(copied)
}
