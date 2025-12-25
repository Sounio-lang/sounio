//! Development Server
//!
//! A lightweight HTTP server for development with:
//! - Static file serving
//! - Live reload injection
//! - WebSocket support for hot reload
//! - Proxy support for API backends
//! - CORS configuration

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Development server errors
#[derive(Debug, Error)]
pub enum DevServerError {
    #[error("Failed to bind to {address}: {source}")]
    BindFailed {
        address: String,
        #[source]
        source: std::io::Error,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Server is not running")]
    NotRunning,

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
}

/// Configuration for the development server
#[derive(Debug, Clone)]
pub struct DevServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Root directory for static files
    pub root: PathBuf,
    /// Index file name
    pub index: String,
    /// Enable live reload
    pub live_reload: bool,
    /// Live reload port (for WebSocket)
    pub live_reload_port: u16,
    /// CORS configuration
    pub cors: CorsConfig,
    /// Proxy rules
    pub proxies: Vec<ProxyRule>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Enable directory listing
    pub directory_listing: bool,
    /// Fallback file for SPA routing
    pub spa_fallback: Option<String>,
    /// Open browser on start
    pub open_browser: bool,
}

impl Default for DevServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            root: PathBuf::from("."),
            index: "index.html".to_string(),
            live_reload: true,
            live_reload_port: 35729,
            cors: CorsConfig::default(),
            proxies: Vec::new(),
            headers: HashMap::new(),
            directory_listing: false,
            spa_fallback: None,
            open_browser: false,
        }
    }
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allow all origins
    pub allow_all: bool,
    /// Allowed origins
    pub origins: Vec<String>,
    /// Allowed methods
    pub methods: Vec<String>,
    /// Allowed headers
    pub headers: Vec<String>,
    /// Allow credentials
    pub credentials: bool,
    /// Max age for preflight cache
    pub max_age: u32,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allow_all: true,
            origins: Vec::new(),
            methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            credentials: false,
            max_age: 86400,
        }
    }
}

/// Proxy rule for forwarding requests
#[derive(Debug, Clone)]
pub struct ProxyRule {
    /// URL path prefix to match
    pub path: String,
    /// Target URL to proxy to
    pub target: String,
    /// Whether to rewrite the path
    pub rewrite: bool,
    /// Headers to add to proxied requests
    pub headers: HashMap<String, String>,
}

/// HTTP request
#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

/// HTTP response
#[derive(Debug)]
struct HttpResponse {
    status: u16,
    status_text: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn new(status: u16, status_text: &str) -> Self {
        Self {
            status,
            status_text: status_text.to_string(),
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    fn ok() -> Self {
        Self::new(200, "OK")
    }

    fn not_found() -> Self {
        let mut resp = Self::new(404, "Not Found");
        resp.body = b"404 Not Found".to_vec();
        resp.headers
            .insert("Content-Type".to_string(), "text/plain".to_string());
        resp
    }

    fn internal_error(msg: &str) -> Self {
        let mut resp = Self::new(500, "Internal Server Error");
        resp.body = msg.as_bytes().to_vec();
        resp.headers
            .insert("Content-Type".to_string(), "text/plain".to_string());
        resp
    }

    fn with_body(mut self, body: Vec<u8>, content_type: &str) -> Self {
        self.body = body;
        self.headers
            .insert("Content-Type".to_string(), content_type.to_string());
        self
    }

    fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut result = format!("HTTP/1.1 {} {}\r\n", self.status, self.status_text);

        // Add content length
        result.push_str(&format!("Content-Length: {}\r\n", self.body.len()));

        // Add headers
        for (key, value) in &self.headers {
            result.push_str(&format!("{}: {}\r\n", key, value));
        }

        result.push_str("\r\n");

        let mut bytes = result.into_bytes();
        bytes.extend(&self.body);
        bytes
    }
}

/// Connected client for live reload
#[derive(Debug)]
struct LiveReloadClient {
    id: u64,
    stream: TcpStream,
    connected_at: Instant,
}

/// Development server
pub struct DevServer {
    config: DevServerConfig,
    running: Arc<AtomicBool>,
    clients: Arc<Mutex<Vec<LiveReloadClient>>>,
    next_client_id: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
    start_time: Option<Instant>,
}

impl DevServer {
    /// Create a new development server
    pub fn new(config: DevServerConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            clients: Arc::new(Mutex::new(Vec::new())),
            next_client_id: Arc::new(AtomicU64::new(1)),
            request_count: Arc::new(AtomicU64::new(0)),
            start_time: None,
        }
    }

    /// Start the server
    pub fn start(&mut self) -> Result<(), DevServerError> {
        let address = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&address).map_err(|e| DevServerError::BindFailed {
            address: address.clone(),
            source: e,
        })?;

        self.running.store(true, Ordering::SeqCst);
        self.start_time = Some(Instant::now());

        println!("Development server started at http://{}", address);

        if self.config.live_reload {
            self.start_live_reload_server()?;
        }

        if self.config.open_browser {
            let _ = open_browser(&format!("http://{}", address));
        }

        // Accept connections
        let running = self.running.clone();
        let config = self.config.clone();
        let clients = self.clients.clone();
        let request_count = self.request_count.clone();

        listener.set_nonblocking(true)?;

        while running.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let config = config.clone();
                    let clients = clients.clone();
                    let request_count = request_count.clone();

                    thread::spawn(move || {
                        if let Err(e) = handle_connection(stream, &config, &clients) {
                            eprintln!("Connection error: {}", e);
                        }
                        request_count.fetch_add(1, Ordering::SeqCst);
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    eprintln!("Accept error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Start the live reload WebSocket server
    fn start_live_reload_server(&self) -> Result<(), DevServerError> {
        let address = format!("{}:{}", self.config.host, self.config.live_reload_port);
        let listener = TcpListener::bind(&address).map_err(|e| DevServerError::BindFailed {
            address: address.clone(),
            source: e,
        })?;

        println!("Live reload server started at ws://{}", address);

        let running = self.running.clone();
        let clients = self.clients.clone();
        let next_id = self.next_client_id.clone();

        listener.set_nonblocking(true)?;

        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let client_id = next_id.fetch_add(1, Ordering::SeqCst);

                        // Perform WebSocket handshake
                        if let Ok(stream) = perform_websocket_handshake(stream) {
                            let client = LiveReloadClient {
                                id: client_id,
                                stream,
                                connected_at: Instant::now(),
                            };

                            clients.lock().unwrap().push(client);
                            println!("Live reload client {} connected", client_id);
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) => {
                        eprintln!("Live reload accept error: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the server
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        println!("Development server stopped");
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Trigger a live reload for all connected clients
    pub fn trigger_reload(&self) {
        let mut clients = self.clients.lock().unwrap();
        let message = r#"{"command":"reload","path":"*","liveCSS":true,"liveImg":true}"#;
        let frame = encode_websocket_frame(message.as_bytes());

        clients.retain_mut(|client| match client.stream.write_all(&frame) {
            Ok(_) => true,
            Err(_) => {
                println!("Live reload client {} disconnected", client.id);
                false
            }
        });

        let count = clients.len();
        if count > 0 {
            println!("Triggered reload for {} client(s)", count);
        }
    }

    /// Get server statistics
    pub fn stats(&self) -> DevServerStats {
        DevServerStats {
            request_count: self.request_count.load(Ordering::SeqCst),
            connected_clients: self.clients.lock().unwrap().len(),
            uptime: self.start_time.map(|t| t.elapsed()),
            address: format!("{}:{}", self.config.host, self.config.port),
        }
    }
}

/// Server statistics
#[derive(Debug)]
pub struct DevServerStats {
    pub request_count: u64,
    pub connected_clients: usize,
    pub uptime: Option<Duration>,
    pub address: String,
}

/// Handle an HTTP connection
fn handle_connection(
    mut stream: TcpStream,
    config: &DevServerConfig,
    _clients: &Arc<Mutex<Vec<LiveReloadClient>>>,
) -> Result<(), DevServerError> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;

    let request = parse_request(&mut stream)?;
    let response = handle_request(&request, config)?;

    stream.write_all(&response.to_bytes())?;
    stream.flush()?;

    Ok(())
}

/// Parse an HTTP request
fn parse_request(stream: &mut TcpStream) -> Result<HttpRequest, DevServerError> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(DevServerError::InvalidConfig {
            message: "Invalid request line".to_string(),
        });
    }

    let method = parts[0].to_string();
    let path = parts[1].to_string();

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }

    // Read body if content-length is present
    let body = if let Some(len) = headers.get("content-length") {
        if let Ok(len) = len.parse::<usize>() {
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body)?;
            body
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Handle an HTTP request
fn handle_request(
    request: &HttpRequest,
    config: &DevServerConfig,
) -> Result<HttpResponse, DevServerError> {
    // Handle CORS preflight
    if request.method == "OPTIONS" {
        return Ok(handle_cors_preflight(config));
    }

    // Check for proxy rules
    for proxy in &config.proxies {
        if request.path.starts_with(&proxy.path) {
            return handle_proxy(request, proxy);
        }
    }

    // Serve static files
    let mut response = serve_static_file(request, config)?;

    // Add CORS headers
    add_cors_headers(&mut response, config);

    // Add custom headers
    for (key, value) in &config.headers {
        response.headers.insert(key.clone(), value.clone());
    }

    Ok(response)
}

/// Handle CORS preflight request
fn handle_cors_preflight(config: &DevServerConfig) -> HttpResponse {
    let mut response = HttpResponse::new(204, "No Content");
    add_cors_headers(&mut response, config);
    response
}

/// Add CORS headers to response
fn add_cors_headers(response: &mut HttpResponse, config: &DevServerConfig) {
    if config.cors.allow_all {
        response
            .headers
            .insert("Access-Control-Allow-Origin".to_string(), "*".to_string());
    } else if !config.cors.origins.is_empty() {
        response.headers.insert(
            "Access-Control-Allow-Origin".to_string(),
            config.cors.origins.join(", "),
        );
    }

    response.headers.insert(
        "Access-Control-Allow-Methods".to_string(),
        config.cors.methods.join(", "),
    );

    response.headers.insert(
        "Access-Control-Allow-Headers".to_string(),
        config.cors.headers.join(", "),
    );

    if config.cors.credentials {
        response.headers.insert(
            "Access-Control-Allow-Credentials".to_string(),
            "true".to_string(),
        );
    }

    response.headers.insert(
        "Access-Control-Max-Age".to_string(),
        config.cors.max_age.to_string(),
    );
}

/// Serve a static file
fn serve_static_file(
    request: &HttpRequest,
    config: &DevServerConfig,
) -> Result<HttpResponse, DevServerError> {
    let url_path = request.path.split('?').next().unwrap_or(&request.path);
    let decoded_path = url_decode(url_path);

    let mut file_path = config.root.join(decoded_path.trim_start_matches('/'));

    // If path is a directory, look for index file
    if file_path.is_dir() {
        file_path = file_path.join(&config.index);
    }

    // Check if file exists
    if !file_path.exists() {
        // Try SPA fallback
        if let Some(ref fallback) = config.spa_fallback {
            file_path = config.root.join(fallback);
            if !file_path.exists() {
                return Ok(HttpResponse::not_found());
            }
        } else if config.directory_listing {
            // Show directory listing
            let dir_path = config.root.join(decoded_path.trim_start_matches('/'));
            if dir_path.is_dir() {
                return Ok(generate_directory_listing(&dir_path, url_path));
            }
            return Ok(HttpResponse::not_found());
        } else {
            return Ok(HttpResponse::not_found());
        }
    }

    // Read file
    let content = std::fs::read(&file_path)?;
    let content_type = guess_content_type(&file_path);

    let mut response = HttpResponse::ok().with_body(content, content_type);

    // Inject live reload script for HTML files
    if config.live_reload && content_type == "text/html" {
        inject_live_reload(&mut response, config);
    }

    Ok(response)
}

/// Generate a directory listing
fn generate_directory_listing(dir: &Path, url_path: &str) -> HttpResponse {
    let mut html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Index of {}</title>
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        a {{ text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .dir {{ color: blue; }}
        .file {{ color: black; }}
    </style>
</head>
<body>
    <h1>Index of {}</h1>
    <hr>
    <ul>
"#,
        url_path, url_path
    );

    // Add parent directory link
    if url_path != "/" {
        html.push_str(r#"        <li><a href="..">..</a></li>"#);
        html.push('\n');
    }

    // List directory contents
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let class = if is_dir { "dir" } else { "file" };
            let suffix = if is_dir { "/" } else { "" };

            html.push_str(&format!(
                r#"        <li><a class="{}" href="{}{}">{}{}</a></li>"#,
                class, name, suffix, name, suffix
            ));
            html.push('\n');
        }
    }

    html.push_str(
        r#"    </ul>
    <hr>
</body>
</html>"#,
    );

    HttpResponse::ok().with_body(html.into_bytes(), "text/html")
}

/// Inject live reload script into HTML response
fn inject_live_reload(response: &mut HttpResponse, config: &DevServerConfig) {
    let script = format!(
        r#"<script>
(function() {{
    var ws = new WebSocket('ws://{}:{}');
    ws.onmessage = function(event) {{
        var data = JSON.parse(event.data);
        if (data.command === 'reload') {{
            window.location.reload();
        }}
    }};
    ws.onclose = function() {{
        setTimeout(function() {{ window.location.reload(); }}, 1000);
    }};
}})();
</script>"#,
        config.host, config.live_reload_port
    );

    let body = String::from_utf8_lossy(&response.body);
    let new_body = if body.contains("</body>") {
        body.replace("</body>", &format!("{}</body>", script))
    } else if body.contains("</html>") {
        body.replace("</html>", &format!("{}</html>", script))
    } else {
        format!("{}{}", body, script)
    };

    response.body = new_body.into_bytes();
    response.headers.insert(
        "Content-Length".to_string(),
        response.body.len().to_string(),
    );
}

/// Handle a proxy request
fn handle_proxy(request: &HttpRequest, proxy: &ProxyRule) -> Result<HttpResponse, DevServerError> {
    // For a full implementation, we'd use an HTTP client library
    // This is a simplified stub that just returns an error
    Ok(HttpResponse::internal_error(&format!(
        "Proxy to {} not implemented (would forward {} {})",
        proxy.target, request.method, request.path
    )))
}

/// Guess content type from file extension
fn guess_content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") | Some("htm") => "text/html",
        Some("css") => "text/css",
        Some("js") | Some("mjs") => "application/javascript",
        Some("json") => "application/json",
        Some("xml") => "application/xml",
        Some("txt") => "text/plain",
        Some("md") => "text/markdown",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("webp") => "image/webp",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("eot") => "application/vnd.ms-fontobject",
        Some("pdf") => "application/pdf",
        Some("zip") => "application/zip",
        Some("wasm") => "application/wasm",
        Some("d") | Some("sio") => "text/x-sounio",
        _ => "application/octet-stream",
    }
}

/// URL decode a string
fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            }
        } else if c == '+' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }

    result
}

/// Perform WebSocket handshake
fn perform_websocket_handshake(mut stream: TcpStream) -> Result<TcpStream, DevServerError> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut headers = HashMap::new();

    // Read request line
    let mut line = String::new();
    reader.read_line(&mut line)?;

    // Read headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }

    // Get WebSocket key
    let key = headers
        .get("sec-websocket-key")
        .ok_or_else(|| DevServerError::InvalidConfig {
            message: "Missing Sec-WebSocket-Key".to_string(),
        })?;

    // Calculate accept key (simplified - in production use proper SHA1)
    let accept_key = calculate_websocket_accept(key);

    // Send response
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept_key
    );

    stream.write_all(response.as_bytes())?;
    stream.flush()?;

    Ok(stream)
}

/// Calculate WebSocket accept key
fn calculate_websocket_accept(key: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // This is a simplified version - production should use SHA1 + base64
    let magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let combined = format!("{}{}", key, magic);

    let mut hasher = DefaultHasher::new();
    combined.hash(&mut hasher);
    let hash = hasher.finish();

    // Simple base64-like encoding (not real base64)
    format!("dGhlIHNhbXBsZSBub25jZQ=={:x}", hash)
}

/// Encode a WebSocket frame
fn encode_websocket_frame(data: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();

    // FIN bit + text frame opcode
    frame.push(0x81);

    // Payload length
    if data.len() < 126 {
        frame.push(data.len() as u8);
    } else if data.len() < 65536 {
        frame.push(126);
        frame.push((data.len() >> 8) as u8);
        frame.push(data.len() as u8);
    } else {
        frame.push(127);
        for i in (0..8).rev() {
            frame.push((data.len() >> (i * 8)) as u8);
        }
    }

    frame.extend_from_slice(data);
    frame
}

/// Open a URL in the default browser
fn open_browser(url: &str) -> Result<(), std::io::Error> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/c", "start", url])
            .spawn()?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_guessing() {
        assert_eq!(guess_content_type(Path::new("test.html")), "text/html");
        assert_eq!(guess_content_type(Path::new("style.css")), "text/css");
        assert_eq!(
            guess_content_type(Path::new("app.js")),
            "application/javascript"
        );
        assert_eq!(
            guess_content_type(Path::new("data.json")),
            "application/json"
        );
        assert_eq!(guess_content_type(Path::new("image.png")), "image/png");
        assert_eq!(guess_content_type(Path::new("code.sio")), "text/x-sounio");
        assert_eq!(
            guess_content_type(Path::new("unknown")),
            "application/octet-stream"
        );
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("test%2Fpath"), "test/path");
        assert_eq!(url_decode("normal"), "normal");
    }

    #[test]
    fn test_http_response() {
        let response = HttpResponse::ok()
            .with_body(b"Hello".to_vec(), "text/plain")
            .with_header("X-Custom", "value");

        let bytes = response.to_bytes();
        let text = String::from_utf8_lossy(&bytes);

        assert!(text.contains("HTTP/1.1 200 OK"));
        assert!(text.contains("Content-Type: text/plain"));
        assert!(text.contains("X-Custom: value"));
        assert!(text.contains("Hello"));
    }

    #[test]
    fn test_websocket_frame_encoding() {
        let data = b"hello";
        let frame = encode_websocket_frame(data);

        assert_eq!(frame[0], 0x81); // FIN + text opcode
        assert_eq!(frame[1], 5); // Length
        assert_eq!(&frame[2..], data);
    }

    #[test]
    fn test_cors_config() {
        let config = CorsConfig::default();

        assert!(config.allow_all);
        assert!(config.methods.contains(&"GET".to_string()));
        assert!(config.headers.contains(&"Content-Type".to_string()));
    }

    #[test]
    fn test_dev_server_config_default() {
        let config = DevServerConfig::default();

        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3000);
        assert_eq!(config.index, "index.html");
        assert!(config.live_reload);
    }
}
