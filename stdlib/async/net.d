/// Async Networking
///
/// Provides asynchronous TCP and UDP networking primitives.

module async::net

import async::future::{Future, Poll, Context}
import async::io::{AsyncRead, AsyncWrite, IoResult, IoError, IoErrorKind}

/// An IPv4 or IPv6 address
pub enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}

impl IpAddr {
    /// Returns true if this is an IPv4 address
    pub fn is_ipv4(&self) -> bool {
        match self {
            IpAddr::V4(_) => true,
            IpAddr::V6(_) => false,
        }
    }

    /// Returns true if this is an IPv6 address
    pub fn is_ipv6(&self) -> bool {
        !self.is_ipv4()
    }

    /// Returns true if this is a loopback address
    pub fn is_loopback(&self) -> bool {
        match self {
            IpAddr::V4(addr) => addr.is_loopback(),
            IpAddr::V6(addr) => addr.is_loopback(),
        }
    }
}

/// An IPv4 address
pub struct Ipv4Addr {
    octets: [u8; 4],
}

impl Ipv4Addr {
    /// Creates a new IPv4 address
    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr { octets: [a, b, c, d] }
    }

    /// The localhost address (127.0.0.1)
    pub fn localhost() -> Ipv4Addr {
        Ipv4Addr::new(127, 0, 0, 1)
    }

    /// The unspecified address (0.0.0.0)
    pub fn unspecified() -> Ipv4Addr {
        Ipv4Addr::new(0, 0, 0, 0)
    }

    /// The broadcast address (255.255.255.255)
    pub fn broadcast() -> Ipv4Addr {
        Ipv4Addr::new(255, 255, 255, 255)
    }

    /// Returns true if this is the loopback address
    pub fn is_loopback(&self) -> bool {
        self.octets[0] == 127
    }

    /// Returns true if this is a private address
    pub fn is_private(&self) -> bool {
        match self.octets[0] {
            10 => true,
            172 => self.octets[1] >= 16 && self.octets[1] <= 31,
            192 => self.octets[1] == 168,
            _ => false,
        }
    }

    /// Returns the octets
    pub fn octets(&self) -> [u8; 4] {
        self.octets
    }
}

/// An IPv6 address
pub struct Ipv6Addr {
    segments: [u16; 8],
}

impl Ipv6Addr {
    /// Creates a new IPv6 address
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Ipv6Addr {
        Ipv6Addr { segments: [a, b, c, d, e, f, g, h] }
    }

    /// The localhost address (::1)
    pub fn localhost() -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)
    }

    /// The unspecified address (::)
    pub fn unspecified() -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)
    }

    /// Returns true if this is the loopback address
    pub fn is_loopback(&self) -> bool {
        self.segments == [0, 0, 0, 0, 0, 0, 0, 1]
    }

    /// Returns the segments
    pub fn segments(&self) -> [u16; 8] {
        self.segments
    }
}

/// A socket address (IP + port)
pub struct SocketAddr {
    ip: IpAddr,
    port: u16,
}

impl SocketAddr {
    /// Creates a new socket address
    pub fn new(ip: IpAddr, port: u16) -> SocketAddr {
        SocketAddr { ip, port }
    }

    /// Returns the IP address
    pub fn ip(&self) -> &IpAddr {
        &self.ip
    }

    /// Returns the port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Sets the port
    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }
}

/// An async TCP listener
///
/// Listens for incoming TCP connections on a local address.
///
/// # Example
/// ```
/// async fn server() {
///     let listener = TcpListener::bind("127.0.0.1:8080").await.unwrap()
///
///     loop {
///         let (stream, addr) = listener.accept().await.unwrap()
///         spawn handle_client(stream, addr)
///     }
/// }
/// ```
pub struct TcpListener {
    /// Internal socket descriptor
    fd: i32,
    /// Local address
    local_addr: SocketAddr,
}

impl TcpListener {
    /// Binds to the given address
    pub async fn bind(addr: &str) -> IoResult<TcpListener> with Async, IO {
        // Would parse address and create socket
        let local_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            8080
        );

        Ok(TcpListener {
            fd: 0,
            local_addr,
        })
    }

    /// Accepts a new connection
    pub async fn accept(&self) -> IoResult<(TcpStream, SocketAddr)> with Async, IO {
        // Would perform actual async accept
        let peer_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            12345
        );

        let stream = TcpStream {
            fd: 0,
            local_addr: self.local_addr.clone(),
            peer_addr: peer_addr.clone(),
        };

        Ok((stream, peer_addr))
    }

    /// Returns the local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        Ok(self.local_addr.clone())
    }

    /// Sets the TTL (time to live)
    pub fn set_ttl(&self, ttl: u32) -> IoResult<()> {
        Ok(())
    }

    /// Returns an iterator over incoming connections
    pub fn incoming(&self) -> Incoming {
        Incoming { listener: self }
    }
}

/// Iterator over incoming TCP connections
pub struct Incoming<'a> {
    listener: &'a TcpListener,
}

impl<'a> Incoming<'a> {
    /// Accepts the next connection
    pub async fn next(&mut self) -> Option<IoResult<TcpStream>> with Async, IO {
        match self.listener.accept().await {
            Ok((stream, _)) => Some(Ok(stream)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// An async TCP stream
///
/// A connected TCP socket for reading and writing.
pub struct TcpStream {
    /// Internal socket descriptor
    fd: i32,
    /// Local address
    local_addr: SocketAddr,
    /// Peer address
    peer_addr: SocketAddr,
}

impl TcpStream {
    /// Connects to a remote address
    pub async fn connect(addr: &str) -> IoResult<TcpStream> with Async, IO {
        // Would parse address and connect
        let local_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            0
        );
        let peer_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            8080
        );

        Ok(TcpStream {
            fd: 0,
            local_addr,
            peer_addr,
        })
    }

    /// Returns the local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        Ok(self.local_addr.clone())
    }

    /// Returns the peer address
    pub fn peer_addr(&self) -> IoResult<SocketAddr> {
        Ok(self.peer_addr.clone())
    }

    /// Shuts down the read half of the connection
    pub async fn shutdown_read(&self) -> IoResult<()> with Async, IO {
        Ok(())
    }

    /// Shuts down the write half of the connection
    pub async fn shutdown_write(&self) -> IoResult<()> with Async, IO {
        Ok(())
    }

    /// Sets the TCP nodelay option
    pub fn set_nodelay(&self, nodelay: bool) -> IoResult<()> {
        Ok(())
    }

    /// Gets the TCP nodelay option
    pub fn nodelay(&self) -> IoResult<bool> {
        Ok(false)
    }

    /// Sets the TTL
    pub fn set_ttl(&self, ttl: u32) -> IoResult<()> {
        Ok(())
    }

    /// Splits the stream into read and write halves
    pub fn split(&mut self) -> (ReadHalf, WriteHalf) {
        (
            ReadHalf { stream: self },
            WriteHalf { stream: self }
        )
    }

    /// Peeks at incoming data without consuming it
    pub async fn peek(&self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        // Would peek at socket data
        Ok(0)
    }
}

impl AsyncRead for TcpStream {
    async fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        // Would perform actual async read
        Ok(0)
    }
}

impl AsyncWrite for TcpStream {
    async fn write(&mut self, buf: &[u8]) -> IoResult<usize> with Async, IO {
        // Would perform actual async write
        Ok(buf.len())
    }

    async fn flush(&mut self) -> IoResult<()> with Async, IO {
        Ok(())
    }

    async fn shutdown(&mut self) -> IoResult<()> with Async, IO {
        self.shutdown_write().await
    }
}

/// Read half of a TCP stream
pub struct ReadHalf<'a> {
    stream: &'a TcpStream,
}

impl<'a> AsyncRead for ReadHalf<'a> {
    async fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        // Would read from stream
        Ok(0)
    }
}

/// Write half of a TCP stream
pub struct WriteHalf<'a> {
    stream: &'a TcpStream,
}

impl<'a> AsyncWrite for WriteHalf<'a> {
    async fn write(&mut self, buf: &[u8]) -> IoResult<usize> with Async, IO {
        Ok(buf.len())
    }

    async fn flush(&mut self) -> IoResult<()> with Async, IO {
        Ok(())
    }
}

/// An async UDP socket
///
/// A UDP socket for sending and receiving datagrams.
pub struct UdpSocket {
    /// Internal socket descriptor
    fd: i32,
    /// Local address
    local_addr: SocketAddr,
}

impl UdpSocket {
    /// Binds to the given address
    pub async fn bind(addr: &str) -> IoResult<UdpSocket> with Async, IO {
        let local_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::unspecified()),
            0
        );

        Ok(UdpSocket {
            fd: 0,
            local_addr,
        })
    }

    /// Connects to a remote address
    ///
    /// After connecting, send() and recv() can be used instead of
    /// send_to() and recv_from().
    pub async fn connect(&self, addr: &str) -> IoResult<()> with Async, IO {
        Ok(())
    }

    /// Returns the local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        Ok(self.local_addr.clone())
    }

    /// Sends data to the connected address
    pub async fn send(&self, buf: &[u8]) -> IoResult<usize> with Async, IO {
        Ok(buf.len())
    }

    /// Receives data from the connected address
    pub async fn recv(&self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        Ok(0)
    }

    /// Sends data to the given address
    pub async fn send_to(&self, buf: &[u8], addr: &SocketAddr) -> IoResult<usize> with Async, IO {
        Ok(buf.len())
    }

    /// Receives data and returns the sender's address
    pub async fn recv_from(&self, buf: &mut [u8]) -> IoResult<(usize, SocketAddr)> with Async, IO {
        let addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            12345
        );
        Ok((0, addr))
    }

    /// Peeks at incoming data without consuming it
    pub async fn peek(&self, buf: &mut [u8]) -> IoResult<usize> with Async, IO {
        Ok(0)
    }

    /// Peeks and returns the sender's address
    pub async fn peek_from(&self, buf: &mut [u8]) -> IoResult<(usize, SocketAddr)> with Async, IO {
        let addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::localhost()),
            12345
        );
        Ok((0, addr))
    }

    /// Sets the broadcast flag
    pub fn set_broadcast(&self, broadcast: bool) -> IoResult<()> {
        Ok(())
    }

    /// Gets the broadcast flag
    pub fn broadcast(&self) -> IoResult<bool> {
        Ok(false)
    }

    /// Sets the TTL
    pub fn set_ttl(&self, ttl: u32) -> IoResult<()> {
        Ok(())
    }

    /// Joins a multicast group
    pub fn join_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> IoResult<()> {
        Ok(())
    }

    /// Leaves a multicast group
    pub fn leave_multicast_v4(&self, multiaddr: &Ipv4Addr, interface: &Ipv4Addr) -> IoResult<()> {
        Ok(())
    }
}

/// DNS resolution

/// Resolves a hostname to IP addresses
pub async fn lookup_host(host: &str) -> IoResult<Vec<IpAddr>> with Async, IO {
    // Would perform actual DNS lookup
    Ok(vec![IpAddr::V4(Ipv4Addr::localhost())])
}

/// Resolves an IP address to hostnames
pub async fn lookup_addr(addr: &IpAddr) -> IoResult<Vec<string>> with Async, IO {
    // Would perform actual reverse DNS lookup
    Ok(vec!["localhost".to_string()])
}

/// Parses an address string
pub fn parse_addr(s: &str) -> Result<SocketAddr, string> {
    // Would parse "host:port" format
    Ok(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::localhost()),
        8080
    ))
}

/// Trait for types that can be converted to socket addresses
pub trait ToSocketAddrs {
    /// Returns an iterator of socket addresses
    fn to_socket_addrs(&self) -> IoResult<Vec<SocketAddr>>
}

impl ToSocketAddrs for str {
    fn to_socket_addrs(&self) -> IoResult<Vec<SocketAddr>> {
        match parse_addr(self) {
            Ok(addr) => Ok(vec![addr]),
            Err(e) => Err(IoError::new(IoErrorKind::InvalidInput, e)),
        }
    }
}

impl ToSocketAddrs for SocketAddr {
    fn to_socket_addrs(&self) -> IoResult<Vec<SocketAddr>> {
        Ok(vec![self.clone()])
    }
}

impl ToSocketAddrs for (IpAddr, u16) {
    fn to_socket_addrs(&self) -> IoResult<Vec<SocketAddr>> {
        Ok(vec![SocketAddr::new(self.0.clone(), self.1)])
    }
}

impl ToSocketAddrs for (Ipv4Addr, u16) {
    fn to_socket_addrs(&self) -> IoResult<Vec<SocketAddr>> {
        Ok(vec![SocketAddr::new(IpAddr::V4(self.0.clone()), self.1)])
    }
}
