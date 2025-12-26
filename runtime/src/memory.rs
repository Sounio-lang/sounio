//! Memory management for Sounio runtime

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Memory allocator interface
pub trait Allocator {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>>;
    fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout);
    fn reallocate(&mut self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>>;
}

/// Default system allocator
pub struct SystemAllocator;

impl Allocator for SystemAllocator {
    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return NonNull::new(layout.align() as *mut u8);
        }

        unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr)
        }
    }

    fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }
    }

    fn reallocate(&mut self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Option<NonNull<u8>> {
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }

        if new_layout.size() == 0 {
            self.deallocate(ptr, old_layout);
            return NonNull::new(new_layout.align() as *mut u8);
        }

        unsafe {
            let new_ptr = std::alloc::realloc(ptr.as_ptr(), old_layout, new_layout.size());
            NonNull::new(new_ptr)
        }
    }
}

/// Arena allocator for temporary allocations
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: usize,
    offset: usize,
    chunk_size: usize,
}

impl Arena {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: vec![vec![0u8; chunk_size]],
            current: 0,
            offset: 0,
            chunk_size,
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8 {
        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size <= self.chunk_size {
            let ptr = self.chunks[self.current].as_mut_ptr().wrapping_add(aligned_offset);
            self.offset = aligned_offset + size;
            return ptr;
        }

        // Need new chunk
        let new_chunk_size = std::cmp::max(self.chunk_size, size);
        self.chunks.push(vec![0u8; new_chunk_size]);
        self.current += 1;
        self.offset = size;

        self.chunks[self.current].as_mut_ptr()
    }

    pub fn reset(&mut self) {
        self.current = 0;
        self.offset = 0;
    }

    pub fn bytes_allocated(&self) -> usize {
        self.chunks.iter().take(self.current).map(|c| c.len()).sum::<usize>() + self.offset
    }
}

/// Reference counted pointer (for shared ownership)
#[repr(C)]
pub struct RcBox<T> {
    ref_count: std::sync::atomic::AtomicUsize,
    value: T,
}

impl<T> RcBox<T> {
    pub fn new(value: T) -> NonNull<Self> {
        let boxed = Box::new(Self {
            ref_count: std::sync::atomic::AtomicUsize::new(1),
            value,
        });
        NonNull::from(Box::leak(boxed))
    }

    pub unsafe fn increment(ptr: NonNull<Self>) {
        (*ptr.as_ptr()).ref_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub unsafe fn decrement(ptr: NonNull<Self>) -> bool {
        let prev = (*ptr.as_ptr()).ref_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
        if prev == 1 {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            true // Caller should deallocate
        } else {
            false
        }
    }

    pub unsafe fn get_ref(ptr: NonNull<Self>) -> &'static T {
        &(*ptr.as_ptr()).value
    }

    pub unsafe fn get_mut(ptr: NonNull<Self>) -> &'static mut T {
        &mut (*ptr.as_ptr()).value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = Arena::new(1024);

        let ptr1 = arena.allocate(64, 8);
        let ptr2 = arena.allocate(128, 16);

        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);

        arena.reset();
    }

    #[test]
    fn test_arena_large_allocation() {
        let mut arena = Arena::new(64);

        // Allocate more than chunk size
        let ptr = arena.allocate(128, 8);
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_rc_box() {
        let ptr = RcBox::new(42i32);

        unsafe {
            RcBox::increment(ptr);
            assert!(!RcBox::decrement(ptr)); // Still has ref
            assert!(RcBox::decrement(ptr));  // Last ref

            // Deallocate
            let _ = Box::from_raw(ptr.as_ptr());
        }
    }

    #[test]
    fn test_system_allocator() {
        let mut alloc = SystemAllocator;

        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr = alloc.allocate(layout).unwrap();

        // Write to memory
        unsafe {
            std::ptr::write(ptr.as_ptr(), 42u8);
            assert_eq!(std::ptr::read(ptr.as_ptr()), 42u8);
        }

        alloc.deallocate(ptr, layout);
    }
}
