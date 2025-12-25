/// Callbacks and Function Pointers for FFI
///
/// This module provides types and utilities for working with function pointers
/// across FFI boundaries, including panic safety and closure wrapping.

module ffi::callback;

use ffi::ctypes::*;

// =============================================================================
// Function Pointer Types
// =============================================================================

/// A C-compatible function pointer type.
///
/// This represents a function pointer that can be passed to and from C code.
/// It uses the C calling convention by default.
///
/// # Example
///
/// ```d
/// // Define a callback type
/// type Comparator = extern "C" fn(*const c_void, *const c_void) -> c_int;
///
/// // Use with qsort
/// extern "C" {
///     fn qsort(
///         base: *mut c_void,
///         nmemb: c_size_t,
///         size: c_size_t,
///         compar: Comparator,
///     );
/// }
/// ```
pub type FnPtr<Args, Ret> = extern "C" fn(Args) -> Ret;

/// Option wrapper for nullable function pointers.
///
/// In C, function pointers can be NULL. This type provides a safe wrapper.
pub enum OptionFnPtr<F> {
    Some(F),
    None,
}

impl<F> OptionFnPtr<F> {
    /// Check if the function pointer is present.
    pub fn is_some(&self) -> bool {
        match self {
            OptionFnPtr::Some(_) => true,
            OptionFnPtr::None => false,
        }
    }

    /// Check if the function pointer is null.
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    /// Get the function pointer, or panic if null.
    pub fn unwrap(self) -> F with Panic {
        match self {
            OptionFnPtr::Some(f) => f,
            OptionFnPtr::None => panic("called unwrap on null function pointer"),
        }
    }

    /// Get the function pointer, or return a default.
    pub fn unwrap_or(self, default: F) -> F {
        match self {
            OptionFnPtr::Some(f) => f,
            OptionFnPtr::None => default,
        }
    }
}

// =============================================================================
// Panic Safety
// =============================================================================

/// Result of a panic-safe FFI call.
pub enum FfiResult<T> {
    /// The call succeeded with a value
    Ok(T),
    /// A panic occurred during the call
    Panicked(PanicInfo),
    /// An exception was caught (platform-specific)
    Exception(string),
}

impl<T> FfiResult<T> {
    /// Check if the call succeeded.
    pub fn is_ok(&self) -> bool {
        match self {
            FfiResult::Ok(_) => true,
            _ => false,
        }
    }

    /// Get the value, or panic if the call failed.
    pub fn unwrap(self) -> T with Panic {
        match self {
            FfiResult::Ok(value) => value,
            FfiResult::Panicked(info) => {
                panic(format!("FFI call panicked: {}", info.message))
            }
            FfiResult::Exception(msg) => {
                panic(format!("FFI call threw exception: {}", msg))
            }
        }
    }

    /// Convert to a Result type.
    pub fn into_result(self) -> Result<T, string> {
        match self {
            FfiResult::Ok(value) => Ok(value),
            FfiResult::Panicked(info) => Err(info.message),
            FfiResult::Exception(msg) => Err(msg),
        }
    }
}

/// Information about a panic that occurred during an FFI call.
pub struct PanicInfo {
    /// The panic message
    pub message: string,
    /// The file where the panic occurred
    pub file: Option<string>,
    /// The line number
    pub line: Option<u32>,
}

impl PanicInfo {
    /// Create a new panic info.
    pub fn new(message: string) -> Self {
        PanicInfo {
            message: message,
            file: None,
            line: None,
        }
    }

    /// Create panic info with location.
    pub fn with_location(message: string, file: string, line: u32) -> Self {
        PanicInfo {
            message: message,
            file: Some(file),
            line: Some(line),
        }
    }
}

/// Catch panics at FFI boundaries.
///
/// This function wraps a D closure in a panic handler, converting any
/// panics to a controlled error return instead of unwinding across
/// the FFI boundary (which is undefined behavior).
///
/// # Example
///
/// ```d
/// extern "C" fn my_callback(data: *mut c_void) -> c_int {
///     catch_panic(|| {
///         let value = unsafe { *(data as *const i32) };
///         process(value); // Might panic
///         0
///     }, -1) // Return -1 on panic
/// }
/// ```
pub fn catch_panic<T>(f: fn() -> T, on_panic: T) -> T {
    // Use catch_unwind internally
    match catch_unwind(f) {
        FfiResult::Ok(value) => value,
        _ => on_panic,
    }
}

/// Catch panics and return detailed information.
pub fn catch_unwind<T>(f: fn() -> T) -> FfiResult<T> {
    // Implementation uses compiler intrinsic
    __builtin_catch_unwind(f)
}

/// Abort the process if a panic occurs.
///
/// Use this when you need to guarantee that a panic won't unwind.
pub fn abort_on_panic<T>(f: fn() -> T) -> T {
    match catch_unwind(f) {
        FfiResult::Ok(value) => value,
        FfiResult::Panicked(info) => {
            eprintln!("fatal: panic in no-unwind context: {}", info.message);
            std::process::abort();
        }
        FfiResult::Exception(msg) => {
            eprintln!("fatal: exception in no-unwind context: {}", msg);
            std::process::abort();
        }
    }
}

// =============================================================================
// Closure to Function Pointer Conversion
// =============================================================================

/// Context for a wrapped closure.
///
/// This struct is used to pass closure data through FFI boundaries.
#[repr(C)]
pub struct ClosureContext<F> {
    /// The closure itself
    closure: F,
    /// Drop flag
    dropped: bool,
}

/// A wrapper that allows passing D closures to C as function pointers.
///
/// Since C function pointers cannot capture environment, this wrapper
/// uses a context pointer pattern: a separate pointer carries the
/// closure data.
///
/// # Example
///
/// ```d
/// // C function that takes a callback with user data
/// extern "C" {
///     fn set_callback(
///         callback: extern "C" fn(*mut c_void, c_int) -> c_int,
///         user_data: *mut c_void,
///     );
/// }
///
/// // Create a closure wrapper
/// let multiplier = 2;
/// let wrapper = ClosureWrapper::new(move |x: c_int| -> c_int {
///     x * multiplier
/// });
///
/// unsafe {
///     set_callback(wrapper.as_fn_ptr(), wrapper.as_user_data());
/// }
/// ```
pub struct ClosureWrapper<F, Args, Ret> {
    /// Boxed context
    context: *mut ClosureContext<F>,
    /// The trampoline function
    trampoline: extern "C" fn(*mut c_void, Args) -> Ret,
}

impl<F, Args, Ret> ClosureWrapper<F, Args, Ret>
where
    F: Fn(Args) -> Ret,
{
    /// Create a new closure wrapper.
    pub fn new(closure: F) -> Self with Alloc {
        let context = Box::new(ClosureContext {
            closure: closure,
            dropped: false,
        });

        let context_ptr = Box::into_raw(context);

        // Create trampoline function
        extern "C" fn trampoline<F, Args, Ret>(
            user_data: *mut c_void,
            args: Args,
        ) -> Ret
        where
            F: Fn(Args) -> Ret,
        {
            let context = unsafe { &*(user_data as *const ClosureContext<F>) };
            (context.closure)(args)
        }

        ClosureWrapper {
            context: context_ptr,
            trampoline: trampoline::<F, Args, Ret>,
        }
    }

    /// Get the function pointer to pass to C.
    pub fn as_fn_ptr(&self) -> extern "C" fn(*mut c_void, Args) -> Ret {
        self.trampoline
    }

    /// Get the user data pointer to pass to C.
    pub fn as_user_data(&self) -> *mut c_void {
        self.context as *mut c_void
    }

    /// Consume the wrapper and return the raw pointers.
    ///
    /// The caller is responsible for calling `from_raw` to clean up.
    pub fn into_raw(self) -> (*mut c_void, extern "C" fn(*mut c_void, Args) -> Ret) {
        let context = self.context;
        let trampoline = self.trampoline;
        std::mem::forget(self);
        (context as *mut c_void, trampoline)
    }

    /// Reconstruct a wrapper from raw pointers.
    ///
    /// # Safety
    ///
    /// The pointers must have been created by `into_raw` on a compatible wrapper.
    pub unsafe fn from_raw(
        user_data: *mut c_void,
        trampoline: extern "C" fn(*mut c_void, Args) -> Ret,
    ) -> Self {
        ClosureWrapper {
            context: user_data as *mut ClosureContext<F>,
            trampoline: trampoline,
        }
    }
}

impl<F, Args, Ret> Drop for ClosureWrapper<F, Args, Ret> {
    fn drop(&mut self) with Alloc {
        if !is_null(self.context) {
            unsafe {
                let _ = Box::from_raw(self.context);
            }
        }
    }
}

// =============================================================================
// Thread Safety for Callbacks
// =============================================================================

/// Marker trait for callbacks that are safe to call from any thread.
pub trait ThreadSafe {}

/// Marker trait for callbacks that can only be called from the main thread.
pub trait MainThreadOnly {}

/// A thread-safe callback wrapper.
///
/// This ensures the wrapped callback is only called on the appropriate thread.
pub struct ThreadSafeCallback<F>
where
    F: ThreadSafe,
{
    callback: F,
}

impl<F> ThreadSafeCallback<F>
where
    F: ThreadSafe,
{
    /// Create a new thread-safe callback.
    pub fn new(callback: F) -> Self {
        ThreadSafeCallback { callback: callback }
    }

    /// Get a reference to the callback.
    pub fn get(&self) -> &F {
        &self.callback
    }
}

/// Ensure a callback is called on the main thread.
pub struct MainThreadCallback<F> {
    callback: F,
    /// Thread ID where the callback was created
    thread_id: ThreadId,
}

impl<F> MainThreadCallback<F> {
    /// Create a new main-thread callback.
    pub fn new(callback: F) -> Self {
        MainThreadCallback {
            callback: callback,
            thread_id: current_thread_id(),
        }
    }

    /// Call the callback, panicking if not on the correct thread.
    pub fn call<Args, Ret>(&self, args: Args) -> Ret
    where
        F: Fn(Args) -> Ret,
    {
        if current_thread_id() != self.thread_id {
            panic("MainThreadCallback called from wrong thread");
        }
        (self.callback)(args)
    }
}

// =============================================================================
// Callback Registration
// =============================================================================

/// A registered callback that can be unregistered later.
///
/// This pattern is common in C APIs that take callbacks with a way to
/// remove them later.
pub struct RegisteredCallback<F> {
    /// The callback
    callback: Option<F>,
    /// Unregister function
    unregister: Option<fn()>,
}

impl<F> RegisteredCallback<F> {
    /// Create a new registered callback.
    pub fn new(callback: F, unregister: fn()) -> Self {
        RegisteredCallback {
            callback: Some(callback),
            unregister: Some(unregister),
        }
    }

    /// Unregister the callback manually.
    pub fn unregister(&mut self) {
        if let Some(unregister) = self.unregister.take() {
            unregister();
        }
        self.callback = None;
    }

    /// Check if the callback is still registered.
    pub fn is_registered(&self) -> bool {
        self.callback.is_some()
    }
}

impl<F> Drop for RegisteredCallback<F> {
    fn drop(&mut self) {
        self.unregister();
    }
}

// =============================================================================
// Common Callback Type Aliases
// =============================================================================

/// A simple callback with no arguments.
pub type SimpleCallback = extern "C" fn();

/// A callback with a user data pointer.
pub type UserDataCallback = extern "C" fn(*mut c_void);

/// A callback that returns an int status.
pub type StatusCallback = extern "C" fn(*mut c_void) -> c_int;

/// A progress callback.
pub type ProgressCallback = extern "C" fn(current: c_size_t, total: c_size_t, user_data: *mut c_void) -> c_int;

/// An error callback.
pub type ErrorCallback = extern "C" fn(error_code: c_int, message: *const c_char, user_data: *mut c_void);

/// A comparison callback (for qsort, etc.).
pub type CompareCallback = extern "C" fn(a: *const c_void, b: *const c_void) -> c_int;

/// A destructor callback.
pub type DestructorCallback = extern "C" fn(*mut c_void);

// =============================================================================
// Helper Functions
// =============================================================================

/// Get the current thread ID.
fn current_thread_id() -> ThreadId {
    // Implementation provided by runtime
    __builtin_current_thread_id()
}

/// Thread identifier type.
pub type ThreadId = usize;

/// Create a comparison callback from a D comparison function.
///
/// # Example
///
/// ```d
/// let compare = make_compare_callback::<i32>(|a, b| a.cmp(b));
///
/// extern "C" {
///     fn qsort(base: *mut c_void, n: c_size_t, size: c_size_t, cmp: CompareCallback);
/// }
///
/// let mut arr = [3, 1, 4, 1, 5, 9, 2, 6];
/// unsafe {
///     qsort(
///         arr.as_mut_ptr() as *mut c_void,
///         arr.len(),
///         size_of::<i32>(),
///         compare,
///     );
/// }
/// ```
pub fn make_compare_callback<T>(cmp: fn(&T, &T) -> Ordering) -> CompareCallback
where
    T: 'static,
{
    extern "C" fn compare_wrapper<T>(a: *const c_void, b: *const c_void) -> c_int
    where
        T: 'static,
    {
        // Note: This requires storing the comparison function somewhere accessible
        // In practice, this would use thread-local storage or a static
        todo!("requires runtime support for storing comparison function")
    }

    compare_wrapper::<T>
}

/// Ordering enum for comparisons
pub enum Ordering {
    Less,
    Equal,
    Greater,
}

impl Ordering {
    /// Convert to a C-style comparison result.
    pub fn to_c_int(&self) -> c_int {
        match self {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}
