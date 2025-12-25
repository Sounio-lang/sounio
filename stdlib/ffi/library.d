/// Dynamic Library Loading for FFI
///
/// This module provides functionality for loading dynamic libraries at runtime,
/// resolving symbols, and creating plugin systems.

module ffi::library;

use ffi::ctypes::*;
use ffi::cstring::*;

// =============================================================================
// Library Loading Error Types
// =============================================================================

/// Error type for library loading operations
pub enum LibraryError {
    /// Library file not found
    NotFound(string),
    /// Failed to load the library
    LoadFailed(string),
    /// Symbol not found in library
    SymbolNotFound(string),
    /// Invalid library handle
    InvalidHandle,
    /// Platform-specific error
    OsError(c_int, string),
}

impl LibraryError {
    /// Get a human-readable error message
    pub fn message(&self) -> string {
        match self {
            LibraryError::NotFound(path) =>
                format!("library not found: {}", path),
            LibraryError::LoadFailed(msg) =>
                format!("failed to load library: {}", msg),
            LibraryError::SymbolNotFound(name) =>
                format!("symbol not found: {}", name),
            LibraryError::InvalidHandle =>
                "invalid library handle".to_string(),
            LibraryError::OsError(code, msg) =>
                format!("OS error {}: {}", code, msg),
        }
    }
}

// =============================================================================
// Library Handle
// =============================================================================

/// A handle to a dynamically loaded library.
///
/// This type manages the lifecycle of a loaded dynamic library. When dropped,
/// the library is automatically unloaded.
///
/// # Platform Support
///
/// - **Unix/Linux/macOS**: Uses `dlopen`, `dlsym`, `dlclose`
/// - **Windows**: Uses `LoadLibrary`, `GetProcAddress`, `FreeLibrary`
///
/// # Example
///
/// ```d
/// let lib = Library::open("libm.so")?;
/// let sin: fn(f64) -> f64 = unsafe { lib.get_fn("sin")? };
/// let result = sin(3.14159 / 2.0);
/// assert!(result > 0.99);
/// ```
pub linear struct Library {
    /// Platform-specific handle
    handle: *mut c_void,
    /// Library path (for error messages)
    path: string,
}

impl Library {
    /// Load a dynamic library from the given path.
    ///
    /// # Platform Behavior
    ///
    /// - On Unix, if the path doesn't contain a `/`, the library is searched
    ///   for in the standard library paths (LD_LIBRARY_PATH, etc.)
    /// - On Windows, if the path doesn't contain a `\`, the library is searched
    ///   for in the DLL search path.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded.
    pub fn open(path: &str) -> Result<Self, LibraryError> with IO, Alloc {
        let cpath = CString::new(path.to_string())
            .ok_or(LibraryError::LoadFailed("invalid path".to_string()))?;

        #[cfg(target_os = "windows")]
        {
            let handle = unsafe { LoadLibraryA(cpath.as_ptr()) };
            if is_null(handle) {
                let error = unsafe { GetLastError() };
                return Err(LibraryError::OsError(error as c_int, get_win32_error(error)));
            }
            Ok(Library { handle: handle, path: path.to_string() })
        }

        #[cfg(not(target_os = "windows"))]
        {
            let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if is_null(handle) {
                let error = unsafe { dlerror() };
                let msg = if is_null(error) {
                    "unknown error".to_string()
                } else {
                    unsafe { CStr::from_ptr(error).to_string_lossy() }
                };
                return Err(LibraryError::LoadFailed(msg));
            }
            Ok(Library { handle: handle, path: path.to_string() })
        }
    }

    /// Load a library with custom flags (Unix only).
    #[cfg(not(target_os = "windows"))]
    pub fn open_with_flags(path: &str, flags: c_int) -> Result<Self, LibraryError> with IO, Alloc {
        let cpath = CString::new(path.to_string())
            .ok_or(LibraryError::LoadFailed("invalid path".to_string()))?;

        let handle = unsafe { dlopen(cpath.as_ptr(), flags) };
        if is_null(handle) {
            let error = unsafe { dlerror() };
            let msg = if is_null(error) {
                "unknown error".to_string()
            } else {
                unsafe { CStr::from_ptr(error).to_string_lossy() }
            };
            return Err(LibraryError::LoadFailed(msg));
        }
        Ok(Library { handle: handle, path: path.to_string() })
    }

    /// Get the main program as a library handle.
    ///
    /// This allows looking up symbols in the main executable and all
    /// loaded libraries.
    #[cfg(not(target_os = "windows"))]
    pub fn this() -> Result<Self, LibraryError> with IO {
        let handle = unsafe { dlopen(NULL as *const c_char, RTLD_NOW | RTLD_LOCAL) };
        if is_null(handle) {
            return Err(LibraryError::LoadFailed("failed to get main handle".to_string()));
        }
        Ok(Library { handle: handle, path: "<main>".to_string() })
    }

    /// Get a raw symbol pointer from the library.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid as long as the library is loaded.
    /// The caller must ensure the pointer is used with the correct type.
    pub unsafe fn get_raw(&self, name: &str) -> Result<*mut c_void, LibraryError> with IO, Alloc {
        let cname = CString::new(name.to_string())
            .ok_or(LibraryError::SymbolNotFound(name.to_string()))?;

        #[cfg(target_os = "windows")]
        {
            let ptr = GetProcAddress(self.handle, cname.as_ptr());
            if is_null(ptr) {
                return Err(LibraryError::SymbolNotFound(name.to_string()));
            }
            Ok(ptr)
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Clear any existing error
            dlerror();

            let ptr = dlsym(self.handle, cname.as_ptr());
            let error = dlerror();

            if is_not_null(error) {
                let msg = CStr::from_ptr(error).to_string_lossy();
                return Err(LibraryError::SymbolNotFound(format!("{}: {}", name, msg)));
            }
            Ok(ptr)
        }
    }

    /// Get a function pointer from the library.
    ///
    /// # Safety
    ///
    /// The caller must ensure the function signature matches the actual
    /// symbol in the library.
    ///
    /// # Example
    ///
    /// ```d
    /// let lib = Library::open("libmath.so")?;
    /// let add: fn(i32, i32) -> i32 = unsafe { lib.get_fn("add")? };
    /// assert_eq!(add(2, 3), 5);
    /// ```
    pub unsafe fn get_fn<F>(&self, name: &str) -> Result<F, LibraryError> with IO, Alloc {
        let ptr = self.get_raw(name)?;
        Ok(transmute::<*mut c_void, F>(ptr))
    }

    /// Get a pointer to a static variable in the library.
    ///
    /// # Safety
    ///
    /// The caller must ensure the type matches the actual symbol.
    pub unsafe fn get_static<T>(&self, name: &str) -> Result<*mut T, LibraryError> with IO, Alloc {
        let ptr = self.get_raw(name)?;
        Ok(ptr as *mut T)
    }

    /// Check if a symbol exists in the library.
    pub fn contains(&self, name: &str) -> bool with IO, Alloc {
        unsafe { self.get_raw(name).is_ok() }
    }

    /// Get the library path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the raw handle (platform-specific).
    pub fn raw_handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for Library {
    fn drop(&mut self) with IO {
        if is_not_null(self.handle) {
            #[cfg(target_os = "windows")]
            unsafe { FreeLibrary(self.handle); }

            #[cfg(not(target_os = "windows"))]
            unsafe { dlclose(self.handle); }
        }
    }
}

// =============================================================================
// Symbol Resolution
// =============================================================================

/// A resolved symbol from a dynamic library.
///
/// This type provides a safe wrapper around a symbol pointer, ensuring
/// the library remains loaded while the symbol is in use.
pub struct Symbol<'lib, T> {
    /// The symbol pointer
    ptr: T,
    /// Phantom to tie lifetime to library
    _marker: PhantomData<&'lib Library>,
}

impl<'lib, T> Symbol<'lib, T> {
    /// Create a new symbol wrapper.
    fn new(ptr: T) -> Self {
        Symbol { ptr: ptr, _marker: PhantomData }
    }

    /// Get the symbol value.
    pub fn get(&self) -> &T {
        &self.ptr
    }
}

// =============================================================================
// Plugin System
// =============================================================================

/// A plugin interface trait.
///
/// Plugins implement this trait to define their initialization and
/// cleanup behavior.
pub trait Plugin {
    /// Plugin name for identification.
    fn name(&self) -> &str;

    /// Plugin version.
    fn version(&self) -> (u32, u32, u32);

    /// Initialize the plugin.
    fn init(&mut self) -> Result<(), string> with IO;

    /// Shutdown the plugin.
    fn shutdown(&mut self) with IO;
}

/// Plugin metadata returned by the plugin's registration function.
#[repr(C)]
pub struct PluginInfo {
    /// Plugin API version (for compatibility checking)
    pub api_version: u32,
    /// Plugin name
    pub name: *const c_char,
    /// Plugin version (major, minor, patch)
    pub version: (u32, u32, u32),
    /// Plugin description
    pub description: *const c_char,
    /// Plugin author
    pub author: *const c_char,
}

/// Function signature for plugin registration.
pub type PluginRegisterFn = extern "C" fn() -> *const PluginInfo;

/// Function signature for plugin factory.
pub type PluginFactoryFn = extern "C" fn() -> *mut c_void;

/// Current plugin API version.
pub const PLUGIN_API_VERSION: u32 = 1;

/// A loaded plugin.
pub struct LoadedPlugin {
    /// The underlying library
    library: Library,
    /// Plugin info
    info: PluginInfo,
    /// Plugin instance (if created)
    instance: Option<*mut c_void>,
}

impl LoadedPlugin {
    /// Load a plugin from a library path.
    ///
    /// The library must export a `plugin_register` function that returns
    /// a `PluginInfo` pointer.
    pub fn load(path: &str) -> Result<Self, LibraryError> with IO, Alloc {
        let library = Library::open(path)?;

        // Get the registration function
        let register: PluginRegisterFn = unsafe {
            library.get_fn("plugin_register")?
        };

        // Get plugin info
        let info_ptr = register();
        if is_null(info_ptr) {
            return Err(LibraryError::LoadFailed("plugin_register returned null".to_string()));
        }

        let info = unsafe { read(info_ptr) };

        // Check API version compatibility
        if info.api_version != PLUGIN_API_VERSION {
            return Err(LibraryError::LoadFailed(
                format!("incompatible plugin API version: expected {}, got {}",
                        PLUGIN_API_VERSION, info.api_version)
            ));
        }

        Ok(LoadedPlugin {
            library: library,
            info: info,
            instance: None,
        })
    }

    /// Get the plugin name.
    pub fn name(&self) -> string {
        if is_null(self.info.name) {
            "<unknown>".to_string()
        } else {
            unsafe { CStr::from_ptr(self.info.name).to_string_lossy() }
        }
    }

    /// Get the plugin version.
    pub fn version(&self) -> (u32, u32, u32) {
        self.info.version
    }

    /// Get the plugin description.
    pub fn description(&self) -> string {
        if is_null(self.info.description) {
            "".to_string()
        } else {
            unsafe { CStr::from_ptr(self.info.description).to_string_lossy() }
        }
    }

    /// Create a plugin instance.
    ///
    /// The library must export a `plugin_create` function.
    pub fn create_instance(&mut self) -> Result<*mut c_void, LibraryError> with IO, Alloc {
        let factory: PluginFactoryFn = unsafe {
            self.library.get_fn("plugin_create")?
        };

        let instance = factory();
        if is_null(instance) {
            return Err(LibraryError::LoadFailed("plugin_create returned null".to_string()));
        }

        self.instance = Some(instance);
        Ok(instance)
    }

    /// Destroy the plugin instance.
    pub fn destroy_instance(&mut self) -> Result<(), LibraryError> with IO, Alloc {
        if let Some(instance) = self.instance.take() {
            // Try to get a destroy function
            if let Ok(destroy) = unsafe { self.library.get_fn::<extern "C" fn(*mut c_void)>("plugin_destroy") } {
                destroy(instance);
            }
        }
        Ok(())
    }
}

impl Drop for LoadedPlugin {
    fn drop(&mut self) with IO, Alloc {
        let _ = self.destroy_instance();
    }
}

/// A plugin manager for loading and managing multiple plugins.
pub struct PluginManager {
    /// Loaded plugins
    plugins: Vec<LoadedPlugin>,
    /// Plugin search paths
    search_paths: Vec<string>,
}

impl PluginManager {
    /// Create a new plugin manager.
    pub fn new() -> Self {
        PluginManager {
            plugins: Vec::new(),
            search_paths: Vec::new(),
        }
    }

    /// Add a search path for plugins.
    pub fn add_search_path(&mut self, path: string) {
        self.search_paths.push(path);
    }

    /// Load a plugin by name or path.
    pub fn load(&mut self, name: &str) -> Result<usize, LibraryError> with IO, Alloc {
        // Try to load directly first
        if let Ok(plugin) = LoadedPlugin::load(name) {
            let index = self.plugins.len();
            self.plugins.push(plugin);
            return Ok(index);
        }

        // Search in search paths
        for search_path in self.search_paths.iter() {
            let full_path = format!("{}/{}", search_path, name);
            if let Ok(plugin) = LoadedPlugin::load(&full_path) {
                let index = self.plugins.len();
                self.plugins.push(plugin);
                return Ok(index);
            }

            // Try with platform-specific extension
            #[cfg(target_os = "windows")]
            let full_path_ext = format!("{}/{}.dll", search_path, name);

            #[cfg(target_os = "macos")]
            let full_path_ext = format!("{}/lib{}.dylib", search_path, name);

            #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
            let full_path_ext = format!("{}/lib{}.so", search_path, name);

            if let Ok(plugin) = LoadedPlugin::load(&full_path_ext) {
                let index = self.plugins.len();
                self.plugins.push(plugin);
                return Ok(index);
            }
        }

        Err(LibraryError::NotFound(name.to_string()))
    }

    /// Get a loaded plugin by index.
    pub fn get(&self, index: usize) -> Option<&LoadedPlugin> {
        self.plugins.get(index)
    }

    /// Get a mutable reference to a loaded plugin.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut LoadedPlugin> {
        self.plugins.get_mut(index)
    }

    /// Unload a plugin by index.
    pub fn unload(&mut self, index: usize) -> Option<LoadedPlugin> {
        if index < self.plugins.len() {
            Some(self.plugins.remove(index))
        } else {
            None
        }
    }

    /// Get the number of loaded plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if any plugins are loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Iterate over loaded plugins.
    pub fn iter(&self) -> impl Iterator<Item = &LoadedPlugin> {
        self.plugins.iter()
    }
}

// =============================================================================
// Platform-Specific Functions
// =============================================================================

#[cfg(not(target_os = "windows"))]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

/// dlopen flags
#[cfg(not(target_os = "windows"))]
pub const RTLD_LAZY: c_int = 0x0001;
#[cfg(not(target_os = "windows"))]
pub const RTLD_NOW: c_int = 0x0002;
#[cfg(not(target_os = "windows"))]
pub const RTLD_LOCAL: c_int = 0x0000;
#[cfg(not(target_os = "windows"))]
pub const RTLD_GLOBAL: c_int = 0x0100;

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hModule: *mut c_void) -> c_int;
    fn GetLastError() -> DWORD;
    fn FormatMessageA(
        dwFlags: DWORD,
        lpSource: *const c_void,
        dwMessageId: DWORD,
        dwLanguageId: DWORD,
        lpBuffer: *mut c_char,
        nSize: DWORD,
        Arguments: *mut c_void,
    ) -> DWORD;
}

#[cfg(target_os = "windows")]
const FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
#[cfg(target_os = "windows")]
const FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

#[cfg(target_os = "windows")]
fn get_win32_error(code: DWORD) -> string with Alloc {
    let mut buffer: [c_char; 256] = [0; 256];
    let len = unsafe {
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            code,
            0,
            buffer.as_mut_ptr(),
            256,
            NULL as *mut c_void,
        )
    };

    if len == 0 {
        format!("error code {}", code)
    } else {
        unsafe {
            CStr::from_ptr(buffer.as_ptr()).to_string_lossy()
        }
    }
}

// =============================================================================
// Marker Types
// =============================================================================

/// Zero-sized marker type for lifetime annotations
pub struct PhantomData<T>;
