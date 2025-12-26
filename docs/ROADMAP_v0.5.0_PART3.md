# SOUNIO v0.5.0 — IMPLEMENTATION ROADMAP (Part 3)

## Semanas 9-12: Runtime, Integration, Polish

---

# FASE 5: RUNTIME (Semana 9)

## Semana 9: Core Runtime

### Dia 29-30: Memory Management

**Arquivo:** `runtime/src/memory.rs`

```rust
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
}
```

---

### Dia 31: FFI / Intrinsics

**Arquivo:** `runtime/src/intrinsics.rs`

```rust
//! Runtime intrinsics for Sounio

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_double};

/// Print a string to stdout
#[no_mangle]
pub extern "C" fn sounio_print_str(s: *const c_char) {
    if s.is_null() {
        return;
    }

    unsafe {
        let cstr = CStr::from_ptr(s);
        if let Ok(s) = cstr.to_str() {
            print!("{}", s);
        }
    }
}

/// Print an integer
#[no_mangle]
pub extern "C" fn sounio_print_i64(value: i64) {
    print!("{}", value);
}

/// Print a float
#[no_mangle]
pub extern "C" fn sounio_print_f64(value: f64) {
    print!("{}", value);
}

/// Print a boolean
#[no_mangle]
pub extern "C" fn sounio_print_bool(value: bool) {
    print!("{}", value);
}

/// Print newline
#[no_mangle]
pub extern "C" fn sounio_println() {
    println!();
}

/// Allocate memory
#[no_mangle]
pub extern "C" fn sounio_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::alloc(layout)
    }
}

/// Deallocate memory
#[no_mangle]
pub extern "C" fn sounio_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::dealloc(ptr, layout);
    }
}

/// Panic with message
#[no_mangle]
pub extern "C" fn sounio_panic(msg: *const c_char) -> ! {
    let message = if msg.is_null() {
        "panic".to_string()
    } else {
        unsafe {
            CStr::from_ptr(msg)
                .to_str()
                .unwrap_or("panic")
                .to_string()
        }
    };

    panic!("{}", message);
}

// Math intrinsics

#[no_mangle]
pub extern "C" fn sounio_sqrt_f64(x: f64) -> f64 { x.sqrt() }

#[no_mangle]
pub extern "C" fn sounio_sin_f64(x: f64) -> f64 { x.sin() }

#[no_mangle]
pub extern "C" fn sounio_cos_f64(x: f64) -> f64 { x.cos() }

#[no_mangle]
pub extern "C" fn sounio_exp_f64(x: f64) -> f64 { x.exp() }

#[no_mangle]
pub extern "C" fn sounio_ln_f64(x: f64) -> f64 { x.ln() }

#[no_mangle]
pub extern "C" fn sounio_pow_f64(base: f64, exp: f64) -> f64 { base.powf(exp) }

// Knowledge/Uncertainty intrinsics

/// Create a Knowledge value
#[repr(C)]
pub struct Knowledge {
    pub value: f64,
    pub uncertainty: f64,
    pub confidence: f64,
}

#[no_mangle]
pub extern "C" fn sounio_knowledge_new(value: f64, uncertainty: f64, confidence: f64) -> Knowledge {
    Knowledge { value, uncertainty, confidence }
}

/// Add two Knowledge values (uncertainty propagation)
#[no_mangle]
pub extern "C" fn sounio_knowledge_add(a: Knowledge, b: Knowledge) -> Knowledge {
    Knowledge {
        value: a.value + b.value,
        uncertainty: (a.uncertainty.powi(2) + b.uncertainty.powi(2)).sqrt(),
        confidence: a.confidence.min(b.confidence),
    }
}

/// Multiply two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_mul(a: Knowledge, b: Knowledge) -> Knowledge {
    let value = a.value * b.value;
    let rel_a = if a.value != 0.0 { a.uncertainty / a.value.abs() } else { 0.0 };
    let rel_b = if b.value != 0.0 { b.uncertainty / b.value.abs() } else { 0.0 };
    let rel_result = (rel_a.powi(2) + rel_b.powi(2)).sqrt();

    Knowledge {
        value,
        uncertainty: value.abs() * rel_result,
        confidence: a.confidence.min(b.confidence),
    }
}
```

---

## CRONOGRAMA RESUMIDO

| Semana | Foco | Deliverables |
|--------|------|--------------|
| 1-2 | Type System | types/, symbol table |
| 3-4 | Semantic Analysis | typeck, inference |
| 5-6 | IR Generation | HIR, MIR |
| 7-8 | Code Generation | Cranelift backend |
| 9 | Runtime | memory, intrinsics, stdlib |
| 10 | Integration | stdlib conectada |
| 11 | Examples | 5+ exemplos funcionais |
| 12 | Polish | docs, release |

---

## MÉTRICAS DE SUCESSO

| Métrica | Target |
|---------|--------|
| Hello World funciona | ✅ |
| 5+ exemplos executam | ✅ |
| Testes passam | >90% |
| Docs completas | Getting Started + Reference |
| Binários disponíveis | Linux, macOS, Windows |
| Website online | souniolang.org |

---

🏛️ SOUNIO v0.5.0 — Implementation Complete

**Estimated Total: 12 weeks (~300 hours of focused work)**
