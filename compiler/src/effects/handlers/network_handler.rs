//! Network Effect Handler
//!
//! This module implements the `Network` effect handler for network I/O operations
//! through algebraic effects. The Network effect enables HTTP requests, WebSocket
//! connections, and other network operations with explicit effect tracking.
//!
//! # Effect Operations
//!
//! | Operation           | Arguments                           | Return Type | Confidence |
//! |--------------------|-------------------------------------|-------------|------------|
//! | `fetch`            | `(url: String)`                     | `String`    | 0.90       |
//! | `get`              | `(url: String, headers: Array)`     | `Response`  | 0.90       |
//! | `post`             | `(url: String, body: String)`       | `Response`  | 0.90       |
//! | `put`              | `(url: String, body: String)`       | `Response`  | 0.90       |
//! | `delete`           | `(url: String)`                     | `Response`  | 0.90       |
//! | `send`             | `(url: String, data: String)`       | `Unit`      | 1.0        |
//! | `websocket_connect`| `(url: String)`                     | `ConnId`    | 1.0        |
//! | `websocket_send`   | `(conn: ConnId, message: String)`   | `Unit`      | 1.0        |
//! | `websocket_receive`| `(conn: ConnId)`                    | `String`    | 0.90       |
//! | `websocket_close`  | `(conn: ConnId)`                    | `Unit`      | 1.0        |
//! | `resolve_dns`      | `(hostname: String)`                | `String`    | 0.95       |
//! | `is_connected`     | `()`                                | `Bool`      | 1.0        |
//!
//! # Epistemic Impact
//!
//! Network operations have significant epistemic impact because data from
//! external sources carries inherent uncertainty:
//! - Network data may be stale, corrupted, or manipulated
//! - External services may return inconsistent results
//! - Network conditions affect data reliability
//!
//! The default confidence factor of 0.90 (10% degradation) reflects this
//! uncertainty. Send operations don't degrade confidence since they don't
//! introduce external data into the computation.
//!
//! # Example
//!
//! ```text
//! // Sounio code using Network effect:
//! fn fetch_data(url: string) -> string with Network {
//!     fetch(url)
//! }
//!
//! fn post_json(url: string, data: string) -> Response with Network {
//!     post(url, data)
//! }
//!
//! fn realtime_chat() with Network, Async {
//!     let conn = websocket_connect("wss://chat.example.com")
//!     websocket_send(conn, "hello")
//!     let response = websocket_receive(conn)
//!     websocket_close(conn)
//!     response
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

/// Confidence factor for network fetch operations (10% degradation)
const NETWORK_FETCH_CONFIDENCE_FACTOR: f64 = 0.90;

/// Confidence factor for DNS resolution (5% degradation)
const DNS_CONFIDENCE_FACTOR: f64 = 0.95;

/// Key for storing WebSocket connection counter
const WS_CONNECTION_COUNTER_KEY: &str = "__network_ws_counter";

/// Key for storing active WebSocket connections
const WS_ACTIVE_CONNECTIONS_KEY: &str = "__network_ws_active";

/// Key for storing pending requests
const PENDING_REQUESTS_KEY: &str = "__network_pending_requests";

/// Key for storing request counter
const REQUEST_COUNTER_KEY: &str = "__network_request_counter";

/// Key for storing simulated connectivity state
const CONNECTIVITY_KEY: &str = "__network_connected";

/// Key for storing total bytes sent
const BYTES_SENT_KEY: &str = "__network_bytes_sent";

/// Key for storing total bytes received
const BYTES_RECEIVED_KEY: &str = "__network_bytes_received";

/// Static operation specifications for NetworkHandler
static NETWORK_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_network_operations() -> &'static [OperationSpec] {
    NETWORK_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("fetch", "String")
                .with_params(vec!["String"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("get", "Response")
                .with_params(vec!["String", "Array"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("post", "Response")
                .with_params(vec!["String", "String"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("put", "Response")
                .with_params(vec!["String", "String"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("delete", "Response")
                .with_params(vec!["String"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("send", "Unit")
                .with_params(vec!["String", "String"]),
            OperationSpec::new("websocket_connect", "ConnId")
                .with_params(vec!["String"]),
            OperationSpec::new("websocket_send", "Unit")
                .with_params(vec!["Int", "String"]),
            OperationSpec::new("websocket_receive", "String")
                .with_params(vec!["Int"])
                .with_confidence_factor(NETWORK_FETCH_CONFIDENCE_FACTOR),
            OperationSpec::new("websocket_close", "Unit")
                .with_params(vec!["Int"]),
            OperationSpec::new("resolve_dns", "String")
                .with_params(vec!["String"])
                .with_confidence_factor(DNS_CONFIDENCE_FACTOR),
            OperationSpec::new("is_connected", "Bool")
                .with_params(vec![]),
            OperationSpec::new("set_timeout", "Unit")
                .with_params(vec!["Int"]),
            OperationSpec::new("get_status_code", "Int")
                .with_params(vec!["Response"]),
        ]
    })
}

/// Handler for the Network effect
///
/// NetworkHandler provides network I/O operations through algebraic effects.
/// It simulates HTTP requests, WebSocket connections, and other network
/// operations for testing and interpretation purposes.
///
/// # State Tracking
///
/// The handler tracks:
/// - Active WebSocket connections
/// - Pending HTTP requests
/// - Total bytes sent/received
/// - Simulated connectivity state
///
/// # Real Implementation
///
/// In a real interpreter/runtime, this handler would delegate to actual
/// HTTP clients (reqwest, hyper) and WebSocket libraries (tungstenite).
/// The current implementation provides simulation for testing.
#[derive(Debug)]
pub struct NetworkHandler {
    /// Simulated latency in milliseconds (for testing)
    simulated_latency_ms: u64,
    /// Whether to simulate network failures
    simulate_failures: bool,
}

impl Default for NetworkHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkHandler {
    /// Create a new NetworkHandler with default settings
    pub fn new() -> Self {
        Self {
            simulated_latency_ms: 0,
            simulate_failures: false,
        }
    }

    /// Create a NetworkHandler with simulated latency
    pub fn with_latency(latency_ms: u64) -> Self {
        Self {
            simulated_latency_ms: latency_ms,
            simulate_failures: false,
        }
    }

    /// Create a NetworkHandler that simulates failures
    pub fn with_failures() -> Self {
        Self {
            simulated_latency_ms: 0,
            simulate_failures: true,
        }
    }

    /// Get WebSocket connection counter
    fn get_ws_counter(state: &mut HandlerState) -> i64 {
        match state.named_state.get(WS_CONNECTION_COUNTER_KEY) {
            Some(Value::Int(n)) => *n,
            _ => {
                state.named_state.insert(WS_CONNECTION_COUNTER_KEY.to_string(), Value::Int(1));
                1
            }
        }
    }

    /// Increment WebSocket connection counter
    fn next_ws_connection_id(state: &mut HandlerState) -> i64 {
        let id = Self::get_ws_counter(state);
        state.named_state.insert(WS_CONNECTION_COUNTER_KEY.to_string(), Value::Int(id + 1));
        id
    }

    /// Get request counter
    fn get_request_counter(state: &mut HandlerState) -> i64 {
        match state.named_state.get(REQUEST_COUNTER_KEY) {
            Some(Value::Int(n)) => *n,
            _ => {
                state.named_state.insert(REQUEST_COUNTER_KEY.to_string(), Value::Int(1));
                1
            }
        }
    }

    /// Increment request counter
    fn next_request_id(state: &mut HandlerState) -> i64 {
        let id = Self::get_request_counter(state);
        state.named_state.insert(REQUEST_COUNTER_KEY.to_string(), Value::Int(id + 1));
        id
    }

    /// Check if connected
    fn is_connected_internal(state: &HandlerState) -> bool {
        match state.named_state.get(CONNECTIVITY_KEY) {
            Some(Value::Bool(b)) => *b,
            _ => true, // Default to connected
        }
    }

    /// Update bytes sent
    fn add_bytes_sent(state: &mut HandlerState, bytes: i64) {
        let current = match state.named_state.get(BYTES_SENT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        state.named_state.insert(BYTES_SENT_KEY.to_string(), Value::Int(current + bytes));
    }

    /// Update bytes received
    fn add_bytes_received(state: &mut HandlerState, bytes: i64) {
        let current = match state.named_state.get(BYTES_RECEIVED_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        state.named_state.insert(BYTES_RECEIVED_KEY.to_string(), Value::Int(current + bytes));
    }

    /// Validate URL format (basic validation)
    fn validate_url(url: &str, op: &str) -> Result<(), HandlerResult> {
        if url.is_empty() {
            return Err(HandlerResult::Abort(HandlerError::new(
                "Network",
                op,
                "URL cannot be empty",
            )));
        }

        // Basic URL validation
        if !url.starts_with("http://") && !url.starts_with("https://")
           && !url.starts_with("ws://") && !url.starts_with("wss://") {
            return Err(HandlerResult::Abort(HandlerError::new(
                "Network",
                op,
                format!("invalid URL scheme: {}. Expected http://, https://, ws://, or wss://", url),
            )));
        }

        Ok(())
    }

    /// Extract a string from a Value
    fn extract_string(value: &Value, op: &str, param: &str) -> Result<String, HandlerResult> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Network",
                op,
                format!("expected String for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract an integer from a Value
    fn extract_int(value: &Value, op: &str, param: &str) -> Result<i64, HandlerResult> {
        match value {
            Value::Int(i) => Ok(*i),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Network",
                op,
                format!("expected Int for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Create a simulated HTTP response
    fn create_response(status: i64, body: &str, headers: Vec<(String, String)>) -> Value {
        let header_values: Vec<Value> = headers
            .into_iter()
            .map(|(k, v)| Value::Tuple(vec![Value::String(k), Value::String(v)]))
            .collect();

        Value::Tuple(vec![
            Value::Int(status),
            Value::String(body.to_string()),
            Value::Array(Rc::new(RefCell::new(header_values))),
        ])
    }

    /// Handle fetch operation
    fn handle_fetch(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "fetch",
                "fetch requires a URL argument",
            ));
        }

        let url = match Self::extract_string(&args[0], "fetch", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "fetch") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "fetch",
                "network is disconnected",
            ));
        }

        if self.simulate_failures {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "fetch",
                format!("simulated network failure for {}", url),
            ));
        }

        Self::next_request_id(state);

        // Simulate response
        let response_body = format!("{{\"url\": \"{}\", \"status\": \"ok\"}}", url);
        Self::add_bytes_received(state, response_body.len() as i64);

        HandlerResult::Resume(Value::String(response_body))
    }

    /// Handle get operation (with headers)
    fn handle_get(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "get",
                "get requires a URL argument",
            ));
        }

        let url = match Self::extract_string(&args[0], "get", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "get") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "get",
                "network is disconnected",
            ));
        }

        Self::next_request_id(state);

        let response_body = format!("{{\"url\": \"{}\", \"method\": \"GET\"}}", url);
        Self::add_bytes_received(state, response_body.len() as i64);

        let response = Self::create_response(
            200,
            &response_body,
            vec![
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
        );

        HandlerResult::Resume(response)
    }

    /// Handle post operation
    fn handle_post(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "post",
                "post requires URL and body arguments",
            ));
        }

        let url = match Self::extract_string(&args[0], "post", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        let body = match Self::extract_string(&args[1], "post", "body") {
            Ok(b) => b,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "post") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "post",
                "network is disconnected",
            ));
        }

        Self::next_request_id(state);
        Self::add_bytes_sent(state, body.len() as i64);

        let response_body = format!("{{\"url\": \"{}\", \"method\": \"POST\", \"received\": {}}}", url, body.len());
        Self::add_bytes_received(state, response_body.len() as i64);

        let response = Self::create_response(
            201,
            &response_body,
            vec![
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
        );

        HandlerResult::Resume(response)
    }

    /// Handle put operation
    fn handle_put(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "put",
                "put requires URL and body arguments",
            ));
        }

        let url = match Self::extract_string(&args[0], "put", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        let body = match Self::extract_string(&args[1], "put", "body") {
            Ok(b) => b,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "put") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "put",
                "network is disconnected",
            ));
        }

        Self::next_request_id(state);
        Self::add_bytes_sent(state, body.len() as i64);

        let response_body = format!("{{\"url\": \"{}\", \"method\": \"PUT\"}}", url);
        Self::add_bytes_received(state, response_body.len() as i64);

        let response = Self::create_response(200, &response_body, vec![]);

        HandlerResult::Resume(response)
    }

    /// Handle delete operation
    fn handle_delete(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "delete",
                "delete requires a URL argument",
            ));
        }

        let url = match Self::extract_string(&args[0], "delete", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "delete") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "delete",
                "network is disconnected",
            ));
        }

        Self::next_request_id(state);

        let response = Self::create_response(204, "", vec![]);

        HandlerResult::Resume(response)
    }

    /// Handle send operation (fire and forget)
    fn handle_send(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "send",
                "send requires URL and data arguments",
            ));
        }

        let url = match Self::extract_string(&args[0], "send", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        let data = match Self::extract_string(&args[1], "send", "data") {
            Ok(d) => d,
            Err(result) => return result,
        };

        if let Err(result) = Self::validate_url(&url, "send") {
            return result;
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "send",
                "network is disconnected",
            ));
        }

        Self::add_bytes_sent(state, data.len() as i64);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle websocket_connect operation
    fn handle_websocket_connect(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_connect",
                "websocket_connect requires a URL argument",
            ));
        }

        let url = match Self::extract_string(&args[0], "websocket_connect", "url") {
            Ok(u) => u,
            Err(result) => return result,
        };

        // Validate WebSocket URL
        if !url.starts_with("ws://") && !url.starts_with("wss://") {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_connect",
                format!("invalid WebSocket URL: {}. Expected ws:// or wss://", url),
            ));
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_connect",
                "network is disconnected",
            ));
        }

        let conn_id = Self::next_ws_connection_id(state);

        // Track active connection
        let key = format!("{}_{}", WS_ACTIVE_CONNECTIONS_KEY, conn_id);
        state.named_state.insert(key, Value::String(url));

        HandlerResult::Resume(Value::Int(conn_id))
    }

    /// Handle websocket_send operation
    fn handle_websocket_send(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_send",
                "websocket_send requires connection ID and message arguments",
            ));
        }

        let conn_id = match Self::extract_int(&args[0], "websocket_send", "conn_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        let message = match Self::extract_string(&args[1], "websocket_send", "message") {
            Ok(m) => m,
            Err(result) => return result,
        };

        // Check if connection exists
        let key = format!("{}_{}", WS_ACTIVE_CONNECTIONS_KEY, conn_id);
        if !state.named_state.contains_key(&key) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_send",
                format!("WebSocket connection {} not found", conn_id),
            ));
        }

        Self::add_bytes_sent(state, message.len() as i64);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle websocket_receive operation
    fn handle_websocket_receive(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_receive",
                "websocket_receive requires a connection ID argument",
            ));
        }

        let conn_id = match Self::extract_int(&args[0], "websocket_receive", "conn_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        // Check if connection exists
        let key = format!("{}_{}", WS_ACTIVE_CONNECTIONS_KEY, conn_id);
        if !state.named_state.contains_key(&key) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_receive",
                format!("WebSocket connection {} not found", conn_id),
            ));
        }

        // Simulate receiving a message
        let message = format!("{{\"conn_id\": {}, \"message\": \"simulated\"}}", conn_id);
        Self::add_bytes_received(state, message.len() as i64);

        HandlerResult::Resume(Value::String(message))
    }

    /// Handle websocket_close operation
    fn handle_websocket_close(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "websocket_close",
                "websocket_close requires a connection ID argument",
            ));
        }

        let conn_id = match Self::extract_int(&args[0], "websocket_close", "conn_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        // Remove connection
        let key = format!("{}_{}", WS_ACTIVE_CONNECTIONS_KEY, conn_id);
        state.named_state.remove(&key);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle resolve_dns operation
    fn handle_resolve_dns(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "resolve_dns",
                "resolve_dns requires a hostname argument",
            ));
        }

        let hostname = match Self::extract_string(&args[0], "resolve_dns", "hostname") {
            Ok(h) => h,
            Err(result) => return result,
        };

        if hostname.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "resolve_dns",
                "hostname cannot be empty",
            ));
        }

        if !Self::is_connected_internal(state) {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "resolve_dns",
                "network is disconnected",
            ));
        }

        // Simulate DNS resolution
        // In real implementation, would use std::net::ToSocketAddrs
        let ip = match hostname.as_str() {
            "localhost" => "127.0.0.1",
            _ => "93.184.216.34", // example.com's IP
        };

        HandlerResult::Resume(Value::String(ip.to_string()))
    }

    /// Handle is_connected operation
    fn handle_is_connected(&self, state: &HandlerState) -> HandlerResult {
        let connected = Self::is_connected_internal(state);
        HandlerResult::Resume(Value::Bool(connected))
    }

    /// Handle set_timeout operation
    fn handle_set_timeout(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "set_timeout",
                "set_timeout requires a timeout value in milliseconds",
            ));
        }

        let timeout_ms = match Self::extract_int(&args[0], "set_timeout", "timeout_ms") {
            Ok(t) => t,
            Err(result) => return result,
        };

        if timeout_ms < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "set_timeout",
                format!("timeout must be non-negative, got {}", timeout_ms),
            ));
        }

        state.named_state.insert("__network_timeout".to_string(), Value::Int(timeout_ms));

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_status_code operation
    fn handle_get_status_code(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Network",
                "get_status_code",
                "get_status_code requires a response argument",
            ));
        }

        // Response is a tuple of (status, body, headers)
        match &args[0] {
            Value::Tuple(elems) if !elems.is_empty() => {
                match &elems[0] {
                    Value::Int(status) => HandlerResult::Resume(Value::Int(*status)),
                    _ => HandlerResult::Abort(HandlerError::new(
                        "Network",
                        "get_status_code",
                        "invalid response format",
                    )),
                }
            }
            _ => HandlerResult::Abort(HandlerError::new(
                "Network",
                "get_status_code",
                format!("expected Response tuple, got {:?}", args[0].type_name()),
            )),
        }
    }

    /// Get total bytes sent (public accessor for testing)
    pub fn bytes_sent(state: &HandlerState) -> i64 {
        match state.named_state.get(BYTES_SENT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Get total bytes received (public accessor for testing)
    pub fn bytes_received(state: &HandlerState) -> i64 {
        match state.named_state.get(BYTES_RECEIVED_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Set connectivity state (for testing)
    pub fn set_connected(state: &mut HandlerState, connected: bool) {
        state.named_state.insert(CONNECTIVITY_KEY.to_string(), Value::Bool(connected));
    }
}

impl HandlerCapability for NetworkHandler {
    fn effect_name(&self) -> &str {
        "Network"
    }

    fn handler_name(&self) -> &str {
        "NetworkHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_network_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "fetch" => self.handle_fetch(args, state),
            "get" => self.handle_get(args, state),
            "post" => self.handle_post(args, state),
            "put" => self.handle_put(args, state),
            "delete" => self.handle_delete(args, state),
            "send" => self.handle_send(args, state),
            "websocket_connect" => self.handle_websocket_connect(args, state),
            "websocket_send" => self.handle_websocket_send(args, state),
            "websocket_receive" => self.handle_websocket_receive(args, state),
            "websocket_close" => self.handle_websocket_close(args, state),
            "resolve_dns" => self.handle_resolve_dns(args, state),
            "is_connected" => self.handle_is_connected(state),
            "set_timeout" => self.handle_set_timeout(args, state),
            "get_status_code" => self.handle_get_status_code(args),
            _ => HandlerResult::Abort(HandlerError::new(
                "Network",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Network operations have side effects on external systems
        false
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            // Operations that receive data from external sources
            "fetch" | "get" | "post" | "put" | "delete" | "websocket_receive" => {
                EpistemicImpact::with_confidence(NETWORK_FETCH_CONFIDENCE_FACTOR)
            }
            // DNS resolution has some uncertainty
            "resolve_dns" => {
                EpistemicImpact::with_confidence(DNS_CONFIDENCE_FACTOR)
            }
            // Send operations don't introduce external data
            _ => EpistemicImpact::none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = NetworkHandler::new();
        assert_eq!(handler.effect_name(), "Network");
        assert_eq!(handler.handler_name(), "NetworkHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = NetworkHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"fetch"));
        assert!(names.contains(&"get"));
        assert!(names.contains(&"post"));
        assert!(names.contains(&"put"));
        assert!(names.contains(&"delete"));
        assert!(names.contains(&"send"));
        assert!(names.contains(&"websocket_connect"));
        assert!(names.contains(&"websocket_send"));
        assert!(names.contains(&"websocket_receive"));
        assert!(names.contains(&"websocket_close"));
        assert!(names.contains(&"resolve_dns"));
    }

    #[test]
    fn test_no_multi_shot() {
        let handler = NetworkHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_fetch() {
        let handler = NetworkHandler::new();
        let impact = handler.epistemic_impact("fetch");
        assert!((impact.confidence_factor - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_impact_send() {
        let handler = NetworkHandler::new();
        let impact = handler.epistemic_impact("send");
        assert!((impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fetch_basic() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "fetch",
            &[Value::String("https://example.com".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::String(body)) => {
                assert!(body.contains("example.com"));
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }

        assert!(NetworkHandler::bytes_received(&state) > 0);
    }

    #[test]
    fn test_fetch_invalid_url() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "fetch",
            &[Value::String("not-a-url".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("invalid URL"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_fetch_empty_url() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "fetch",
            &[Value::String("".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("empty"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_fetch_disconnected() {
        let handler = NetworkHandler::new();
        let mut state = new_state();
        NetworkHandler::set_connected(&mut state, false);

        let result = handler.handle(
            "fetch",
            &[Value::String("https://example.com".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("disconnected"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_get_with_headers() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let headers = Value::Array(Rc::new(RefCell::new(vec![
            Value::Tuple(vec![
                Value::String("Authorization".to_string()),
                Value::String("Bearer token".to_string()),
            ]),
        ])));

        let result = handler.handle(
            "get",
            &[Value::String("https://api.example.com".to_string()), headers],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(elems)) => {
                assert_eq!(elems.len(), 3);
                match &elems[0] {
                    Value::Int(status) => assert_eq!(*status, 200),
                    _ => panic!("expected status code"),
                }
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_post_basic() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "post",
            &[
                Value::String("https://api.example.com".to_string()),
                Value::String("{\"key\": \"value\"}".to_string()),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(elems)) => {
                match &elems[0] {
                    Value::Int(status) => assert_eq!(*status, 201),
                    _ => panic!("expected status code"),
                }
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }

        assert!(NetworkHandler::bytes_sent(&state) > 0);
    }

    #[test]
    fn test_put_basic() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "put",
            &[
                Value::String("https://api.example.com/resource/1".to_string()),
                Value::String("{\"updated\": true}".to_string()),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(_)) => {}
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_delete_basic() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "delete",
            &[Value::String("https://api.example.com/resource/1".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(elems)) => {
                match &elems[0] {
                    Value::Int(status) => assert_eq!(*status, 204),
                    _ => panic!("expected status code"),
                }
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_send_basic() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "send",
            &[
                Value::String("https://telemetry.example.com".to_string()),
                Value::String("event data".to_string()),
            ],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        assert_eq!(NetworkHandler::bytes_sent(&state), 10); // "event data".len()
    }

    #[test]
    fn test_websocket_lifecycle() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        // Connect
        let result = handler.handle(
            "websocket_connect",
            &[Value::String("wss://chat.example.com".to_string())],
            cont(),
            &mut state,
        );

        let conn_id = match result {
            HandlerResult::Resume(Value::Int(id)) => id,
            other => panic!("expected Resume(Int), got {:?}", other),
        };

        // Send
        let result = handler.handle(
            "websocket_send",
            &[Value::Int(conn_id), Value::String("hello".to_string())],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        // Receive
        let result = handler.handle(
            "websocket_receive",
            &[Value::Int(conn_id)],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::String(_)) => {}
            other => panic!("expected Resume(String), got {:?}", other),
        }

        // Close
        let result = handler.handle(
            "websocket_close",
            &[Value::Int(conn_id)],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        // Verify closed (send should fail)
        let result = handler.handle(
            "websocket_send",
            &[Value::Int(conn_id), Value::String("test".to_string())],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("not found"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_websocket_invalid_url() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "websocket_connect",
            &[Value::String("https://not-websocket.com".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("invalid WebSocket URL"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_dns() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "resolve_dns",
            &[Value::String("localhost".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::String(ip)) => {
                assert_eq!(ip, "127.0.0.1");
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_dns_external() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "resolve_dns",
            &[Value::String("example.com".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::String(ip)) => {
                assert!(!ip.is_empty());
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }
    }

    #[test]
    fn test_is_connected() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle("is_connected", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }

        NetworkHandler::set_connected(&mut state, false);

        let result = handler.handle("is_connected", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_set_timeout() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle("set_timeout", &[Value::Int(5000)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_get_status_code() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let response = Value::Tuple(vec![
            Value::Int(404),
            Value::String("Not Found".to_string()),
            Value::Array(Rc::new(RefCell::new(vec![]))),
        ]);

        let result = handler.handle("get_status_code", &[response], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(status)) => {
                assert_eq!(status, 404);
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = NetworkHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = NetworkHandler::default();
        assert_eq!(handler.effect_name(), "Network");
    }

    #[test]
    fn test_with_failures() {
        let handler = NetworkHandler::with_failures();
        let mut state = new_state();

        let result = handler.handle(
            "fetch",
            &[Value::String("https://example.com".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("simulated network failure"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }
}
