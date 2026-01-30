//! Resource Types for Linear Type System
//!
//! This module defines common resource types that benefit from linearity:
//! - File handles
//! - Network connections
//! - Database connections
//! - Memory allocations
//! - Locks and synchronization primitives
//! - GPU resources
//!
//! Each resource type has a corresponding "capability" that represents
//! the permission to perform certain operations.
//!
//! # Design Philosophy
//!
//! Resources follow a capability-based security model:
//!
//! ```text
//! linear struct FileHandle {
//!     capabilities: Capability<Read, Write>,
//!     ...
//! }
//!
//! fn read(handle: &FileHandle where Read) -> Data
//! fn write(handle: &!FileHandle where Write, data: Data)
//! fn close(handle: FileHandle) -> ()  // Consumes the handle
//! ```

use std::collections::HashSet;
use std::fmt;

use super::kind::Linearity;
use super::typed::Linear;

// ============================================================================
// Capability System
// ============================================================================

/// A capability representing permission to perform an operation.
///
/// Capabilities are used with resource types to control what operations
/// are allowed. They form a lattice where more specific capabilities
/// are subtypes of more general ones.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Read capability
    Read,
    /// Write capability
    Write,
    /// Execute capability
    Execute,
    /// Close/finalize capability
    Close,
    /// Full access (all capabilities)
    Full,
    /// No capabilities
    None,
    /// Custom capability
    Custom(String),
    /// Conjunction of capabilities
    And(Vec<Capability>),
    /// Disjunction of capabilities
    Or(Vec<Capability>),
}

impl Capability {
    /// Check if this capability implies another.
    pub fn implies(&self, other: &Capability) -> bool {
        match (self, other) {
            // Full implies everything
            (Capability::Full, _) => true,
            // None implies nothing (except None)
            (Capability::None, Capability::None) => true,
            (Capability::None, _) => false,
            // Reflexivity
            _ if self == other => true,
            // And implies all components
            (Capability::And(caps), other) => caps.iter().any(|c| c.implies(other)),
            // Or requires any component to imply
            (Capability::Or(caps), other) => caps.iter().all(|c| c.implies(other)),
            // Check if other is in And
            (cap, Capability::And(caps)) => caps.iter().all(|c| cap.implies(c)),
            // Check if any in Or
            (cap, Capability::Or(caps)) => caps.iter().any(|c| cap.implies(c)),
            _ => false,
        }
    }

    /// Combine two capabilities (conjunction).
    pub fn and(self, other: Capability) -> Capability {
        match (self, other) {
            (Capability::Full, _) | (_, Capability::Full) => Capability::Full,
            (Capability::None, x) | (x, Capability::None) => x,
            (Capability::And(mut caps), Capability::And(other_caps)) => {
                caps.extend(other_caps);
                Capability::And(caps)
            }
            (Capability::And(mut caps), other) => {
                caps.push(other);
                Capability::And(caps)
            }
            (this, Capability::And(mut caps)) => {
                caps.push(this);
                Capability::And(caps)
            }
            (a, b) => Capability::And(vec![a, b]),
        }
    }

    /// Create a read + write capability.
    pub fn read_write() -> Capability {
        Capability::And(vec![Capability::Read, Capability::Write])
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Capability::Read => write!(f, "Read"),
            Capability::Write => write!(f, "Write"),
            Capability::Execute => write!(f, "Execute"),
            Capability::Close => write!(f, "Close"),
            Capability::Full => write!(f, "Full"),
            Capability::None => write!(f, "None"),
            Capability::Custom(name) => write!(f, "{}", name),
            Capability::And(caps) => {
                let names: Vec<_> = caps.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", names.join(" & "))
            }
            Capability::Or(caps) => {
                let names: Vec<_> = caps.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", names.join(" | "))
            }
        }
    }
}

// ============================================================================
// Resource Kind
// ============================================================================

/// Classification of resource types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// File system resource
    File,
    /// Network connection
    Network,
    /// Database connection
    Database,
    /// Memory allocation
    Memory,
    /// Lock or mutex
    Lock,
    /// GPU resource (buffer, texture, etc.)
    Gpu,
    /// Generic handle
    Handle,
    /// Session or channel
    Session,
    /// Custom resource kind
    Custom(String),
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceKind::File => write!(f, "File"),
            ResourceKind::Network => write!(f, "Network"),
            ResourceKind::Database => write!(f, "Database"),
            ResourceKind::Memory => write!(f, "Memory"),
            ResourceKind::Lock => write!(f, "Lock"),
            ResourceKind::Gpu => write!(f, "GPU"),
            ResourceKind::Handle => write!(f, "Handle"),
            ResourceKind::Session => write!(f, "Session"),
            ResourceKind::Custom(name) => write!(f, "{}", name),
        }
    }
}

// ============================================================================
// Resource Type
// ============================================================================

/// A resource type with linearity and capabilities.
///
/// This represents the type-level information about a resource:
/// - What kind of resource it is
/// - What linearity it has
/// - What capabilities it provides
/// - What state it's in
#[derive(Clone, Debug)]
pub struct ResourceType {
    /// The kind of resource
    pub kind: ResourceKind,
    /// The linearity of this resource
    pub linearity: Linearity,
    /// Available capabilities
    pub capabilities: HashSet<Capability>,
    /// Current state (for session-typed resources)
    pub state: Option<String>,
    /// Resource parameters (e.g., file path, connection string)
    pub parameters: Vec<(String, String)>,
}

impl ResourceType {
    /// Create a new resource type.
    pub fn new(kind: ResourceKind) -> Self {
        Self {
            kind,
            linearity: Linearity::Linear, // Resources are linear by default
            capabilities: HashSet::new(),
            state: None,
            parameters: Vec::new(),
        }
    }

    /// Create a file resource type.
    pub fn file() -> Self {
        let mut res = Self::new(ResourceKind::File);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res.capabilities.insert(Capability::Close);
        res
    }

    /// Create a network resource type.
    pub fn network() -> Self {
        let mut res = Self::new(ResourceKind::Network);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res.capabilities.insert(Capability::Close);
        res
    }

    /// Create a database resource type.
    pub fn database() -> Self {
        let mut res = Self::new(ResourceKind::Database);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res.capabilities.insert(Capability::Execute);
        res.capabilities.insert(Capability::Close);
        res
    }

    /// Create a GPU resource type.
    pub fn gpu() -> Self {
        let mut res = Self::new(ResourceKind::Gpu);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res
    }

    /// Create a memory resource type.
    pub fn memory() -> Self {
        let mut res = Self::new(ResourceKind::Memory);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res
    }

    /// Create a lock resource type.
    pub fn lock() -> Self {
        let mut res = Self::new(ResourceKind::Lock);
        res.capabilities.insert(Capability::Read);
        res.capabilities.insert(Capability::Write);
        res
    }

    /// Set the linearity.
    pub fn with_linearity(mut self, linearity: Linearity) -> Self {
        self.linearity = linearity;
        self
    }

    /// Make this resource affine.
    pub fn affine(mut self) -> Self {
        self.linearity = Linearity::Affine;
        self
    }

    /// Add a capability.
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Set the state.
    pub fn with_state(mut self, state: impl Into<String>) -> Self {
        self.state = Some(state.into());
        self
    }

    /// Add a parameter.
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.push((name.into(), value.into()));
        self
    }

    /// Check if this resource has a capability.
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.iter().any(|c| c.implies(cap))
    }

    /// Check if this resource can be discarded.
    pub fn can_discard(&self) -> bool {
        self.linearity.can_discard()
    }

    /// Check if this resource must be used.
    pub fn must_use(&self) -> bool {
        self.linearity.must_use()
    }
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if !self.capabilities.is_empty() {
            let caps: Vec<_> = self.capabilities.iter().map(|c| c.to_string()).collect();
            write!(f, "<{}>", caps.join(", "))?;
        }
        if let Some(state) = &self.state {
            write!(f, "[{}]", state)?;
        }
        if self.linearity != Linearity::Linear {
            write!(f, " @ {}", self.linearity)?;
        }
        Ok(())
    }
}

// ============================================================================
// Resource Handle (Runtime)
// ============================================================================

/// A runtime resource handle.
///
/// This is the actual handle that wraps an underlying resource value.
/// It tracks the resource's state at runtime.
#[derive(Debug)]
pub struct ResourceHandle<T> {
    /// The underlying resource
    inner: Option<T>,
    /// The resource type
    resource_type: ResourceType,
    /// Whether the resource has been closed/finalized
    finalized: bool,
}

impl<T> ResourceHandle<T> {
    /// Create a new resource handle.
    pub fn new(resource: T, resource_type: ResourceType) -> Self {
        Self {
            inner: Some(resource),
            resource_type,
            finalized: false,
        }
    }

    /// Get a reference to the resource.
    pub fn get(&self) -> Option<&T> {
        if self.finalized {
            None
        } else {
            self.inner.as_ref()
        }
    }

    /// Get a mutable reference to the resource.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.finalized {
            None
        } else {
            self.inner.as_mut()
        }
    }

    /// Consume the resource, returning the inner value.
    pub fn consume(mut self) -> Option<T> {
        if self.finalized {
            None
        } else {
            self.inner.take()
        }
    }

    /// Finalize the resource.
    pub fn finalize(&mut self) {
        self.finalized = true;
        self.inner = None;
    }

    /// Check if the resource has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Get the resource type.
    pub fn resource_type(&self) -> &ResourceType {
        &self.resource_type
    }

    /// Check if the resource has a capability.
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.resource_type.has_capability(cap)
    }
}

// ============================================================================
// Linear Resource - Type-safe linear resource
// ============================================================================

/// A linear resource that must be consumed.
///
/// This combines the `Linear<T>` wrapper with resource tracking.
pub struct LinearResource<T> {
    /// The linear handle
    handle: Linear<ResourceHandle<T>>,
}

impl<T> LinearResource<T> {
    /// Create a new linear resource.
    pub fn new(resource: T, resource_type: ResourceType) -> Self {
        Self {
            handle: Linear::new(ResourceHandle::new(resource, resource_type)),
        }
    }

    /// Create a file resource.
    pub fn file(file: T) -> Self {
        Self::new(file, ResourceType::file())
    }

    /// Create a network resource.
    pub fn network(conn: T) -> Self {
        Self::new(conn, ResourceType::network())
    }

    /// Create a database resource.
    pub fn database(conn: T) -> Self {
        Self::new(conn, ResourceType::database())
    }

    /// Get a reference to the resource.
    pub fn get(&self) -> Option<&T> {
        self.handle.as_ref().get()
    }

    /// Get a mutable reference to the resource.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.handle.as_mut().get_mut()
    }

    /// Consume the resource.
    pub fn consume(self) -> Option<T> {
        self.handle.consume().consume()
    }

    /// Close/finalize the resource.
    ///
    /// This consumes the linear resource and runs the finalizer.
    pub fn close(self, finalizer: impl FnOnce(T)) {
        if let Some(resource) = self.consume() {
            finalizer(resource);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for LinearResource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearResource")
            .field("handle", &self.handle)
            .finish()
    }
}

// ============================================================================
// Resource Pool
// ============================================================================

/// A pool of resources that can be borrowed linearly.
///
/// The pool tracks which resources are in use and ensures they're returned.
#[derive(Debug)]
pub struct ResourcePool<T> {
    /// Available resources
    available: Vec<T>,
    /// Number currently borrowed
    borrowed: usize,
    /// Resource type
    resource_type: ResourceType,
}

impl<T> ResourcePool<T> {
    /// Create a new empty pool.
    pub fn new(resource_type: ResourceType) -> Self {
        Self {
            available: Vec::new(),
            borrowed: 0,
            resource_type,
        }
    }

    /// Add a resource to the pool.
    pub fn add(&mut self, resource: T) {
        self.available.push(resource);
    }

    /// Borrow a resource from the pool.
    ///
    /// Returns a linear handle that must be returned to the pool.
    pub fn borrow(&mut self) -> Option<PooledResource<T>> {
        self.available.pop().map(|resource| {
            self.borrowed += 1;
            PooledResource {
                resource: Some(resource),
                pool_id: self as *const _ as usize,
            }
        })
    }

    /// Return a resource to the pool.
    pub fn return_resource(&mut self, mut pooled: PooledResource<T>) {
        if let Some(resource) = pooled.resource.take() {
            self.available.push(resource);
            self.borrowed -= 1;
        }
    }

    /// Get the number of available resources.
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Get the number of borrowed resources.
    pub fn borrowed_count(&self) -> usize {
        self.borrowed
    }
}

/// A resource borrowed from a pool.
///
/// Must be returned to the pool when done.
#[derive(Debug)]
pub struct PooledResource<T> {
    /// The borrowed resource
    resource: Option<T>,
    /// Pool identifier (for validation)
    pool_id: usize,
}

impl<T> PooledResource<T> {
    /// Get a reference to the resource.
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get a mutable reference to the resource.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }
}

// ============================================================================
// RAII Guard for Resources
// ============================================================================

/// A RAII guard that ensures a resource is finalized when dropped.
///
/// This is useful for resources that should be affine (can be dropped)
/// but have cleanup requirements.
pub struct ResourceGuard<T, F: FnOnce(T)> {
    /// The resource
    resource: Option<T>,
    /// The finalizer
    finalizer: Option<F>,
}

impl<T, F: FnOnce(T)> ResourceGuard<T, F> {
    /// Create a new guard.
    pub fn new(resource: T, finalizer: F) -> Self {
        Self {
            resource: Some(resource),
            finalizer: Some(finalizer),
        }
    }

    /// Get a reference to the resource.
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get a mutable reference to the resource.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Consume the resource without running the finalizer.
    pub fn into_inner(mut self) -> Option<T> {
        self.finalizer = None;
        self.resource.take()
    }
}

impl<T, F: FnOnce(T)> Drop for ResourceGuard<T, F> {
    fn drop(&mut self) {
        if let (Some(resource), Some(finalizer)) = (self.resource.take(), self.finalizer.take()) {
            finalizer(resource);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_implies() {
        assert!(Capability::Full.implies(&Capability::Read));
        assert!(Capability::Full.implies(&Capability::Write));
        assert!(Capability::Read.implies(&Capability::Read));
        assert!(!Capability::Read.implies(&Capability::Write));
        assert!(!Capability::None.implies(&Capability::Read));
    }

    #[test]
    fn test_capability_and() {
        let rw = Capability::Read.and(Capability::Write);
        assert!(rw.implies(&Capability::Read));
        assert!(rw.implies(&Capability::Write));
    }

    #[test]
    fn test_resource_type_file() {
        let file = ResourceType::file();
        assert_eq!(file.kind, ResourceKind::File);
        assert!(file.has_capability(&Capability::Read));
        assert!(file.has_capability(&Capability::Write));
        assert!(file.has_capability(&Capability::Close));
        assert!(!file.has_capability(&Capability::Execute));
    }

    #[test]
    fn test_resource_type_database() {
        let db = ResourceType::database();
        assert_eq!(db.kind, ResourceKind::Database);
        assert!(db.has_capability(&Capability::Read));
        assert!(db.has_capability(&Capability::Execute));
    }

    #[test]
    fn test_resource_handle() {
        let handle = ResourceHandle::new(42, ResourceType::memory());

        assert!(!handle.is_finalized());
        assert_eq!(handle.get(), Some(&42));

        let value = handle.consume();
        assert_eq!(value, Some(42));
    }

    #[test]
    fn test_linear_resource() {
        let resource = LinearResource::file(42);
        assert_eq!(resource.get(), Some(&42));

        let mut closed = false;
        resource.close(|_| closed = true);
        assert!(closed);
    }

    #[test]
    fn test_resource_pool() {
        let mut pool: ResourcePool<i32> = ResourcePool::new(ResourceType::memory());

        pool.add(1);
        pool.add(2);
        pool.add(3);

        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.borrowed_count(), 0);

        let borrowed = pool.borrow().unwrap();
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.borrowed_count(), 1);

        pool.return_resource(borrowed);
        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.borrowed_count(), 0);
    }

    #[test]
    fn test_resource_guard() {
        use std::cell::Cell;

        let finalized = Cell::new(false);

        {
            let _guard = ResourceGuard::new(42, |_| finalized.set(true));
            assert!(!finalized.get());
        }

        assert!(finalized.get());
    }

    #[test]
    fn test_resource_guard_into_inner() {
        use std::cell::Cell;

        let finalized = Cell::new(false);

        let guard = ResourceGuard::new(42, |_| finalized.set(true));
        let value = guard.into_inner();

        assert_eq!(value, Some(42));
        assert!(!finalized.get()); // Finalizer not called
    }
}
