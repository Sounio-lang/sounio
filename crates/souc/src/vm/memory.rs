//! Memory management for the VM

use crate::vm::VmError;
use std::collections::HashMap;

/// Manages heap memory allocations
pub struct Heap {
    allocations: HashMap<usize, Vec<u8>>,
    next_addr: usize,
}

impl Heap {
    /// Creates a new heap
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            next_addr: 0x1000, // Start from 0x1000 to avoid null pointer issues
        }
    }

    /// Allocates memory and returns the address
    pub fn alloc(&mut self, size: usize) -> usize {
        let addr = self.next_addr;
        self.next_addr = self.next_addr.wrapping_add(size).wrapping_add(64); // Add padding
        self.allocations.insert(addr, vec![0u8; size]);
        tracing::trace!("Heap: alloc({}) -> 0x{:x}", size, addr);
        addr
    }

    /// Deallocates memory at the given address
    pub fn dealloc(&mut self, addr: usize) -> Result<(), VmError> {
        self.allocations.remove(&addr);
        tracing::trace!("Heap: dealloc(0x{:x})", addr);
        Ok(())
    }

    /// Reads bytes from a heap address
    pub fn read(&self, addr: usize, size: usize) -> Result<Vec<u8>, VmError> {
        let data = self
            .allocations
            .get(&addr)
            .ok_or(VmError::MemoryError(format!(
                "Invalid pointer: 0x{:x}",
                addr
            )))?;

        if size > data.len() {
            return Err(VmError::MemoryError(format!(
                "Read overflow at 0x{:x}: requested {}, available {}",
                addr,
                size,
                data.len()
            )));
        }

        Ok(data[..size].to_vec())
    }

    /// Writes bytes to a heap address
    pub fn write(&mut self, addr: usize, data: &[u8]) -> Result<(), VmError> {
        let heap_data = self
            .allocations
            .get_mut(&addr)
            .ok_or(VmError::MemoryError(format!(
                "Invalid pointer: 0x{:x}",
                addr
            )))?;

        if data.len() > heap_data.len() {
            return Err(VmError::MemoryError(format!(
                "Write overflow at 0x{:x}: requested {}, available {}",
                addr,
                data.len(),
                heap_data.len()
            )));
        }

        heap_data[..data.len()].copy_from_slice(data);
        tracing::trace!("Heap: write(0x{:x}, {} bytes)", addr, data.len());
        Ok(())
    }

    /// Gets the size of an allocation
    pub fn size(&self, addr: usize) -> Result<usize, VmError> {
        self.allocations
            .get(&addr)
            .map(|data| data.len())
            .ok_or(VmError::MemoryError(format!(
                "Invalid pointer: 0x{:x}",
                addr
            )))
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}
