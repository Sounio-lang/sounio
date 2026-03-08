use std::ffi::{c_char, c_int, c_void, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

#[repr(C)]
struct RawPoseidonModule {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawPoseidonRunConfig {
    max_steps: i64,
    print_int_cb: Option<extern "C" fn(i64, *mut c_void)>,
    user_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawPoseidonRunResult {
    status: c_int,
    exit_code: i64,
    steps_executed: i64,
}

unsafe extern "C" {
    fn poseidon_default_config() -> RawPoseidonRunConfig;
    fn poseidon_module_load_file(path: *const c_char, out_module: *mut *mut RawPoseidonModule) -> c_int;
    fn poseidon_module_load_bytes(buf: *const u8, size: usize, out_module: *mut *mut RawPoseidonModule) -> c_int;
    fn poseidon_module_free(module: *mut RawPoseidonModule);
    fn poseidon_run_module(module: *const RawPoseidonModule, config: *const RawPoseidonRunConfig) -> RawPoseidonRunResult;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoseidonError {
    InvalidArgument,
    IoError,
    FileTooLarge,
    InvalidBytecode,
    UnsupportedVersion,
    OutOfMemory,
    NoMainFunction,
    ExecutionError,
    ExecutionTimeout,
}

impl PoseidonError {
    fn from_status(code: c_int) -> Option<Self> {
        match code {
            0 => None,
            1 => Some(Self::InvalidArgument),
            2 => Some(Self::IoError),
            3 => Some(Self::FileTooLarge),
            4 => Some(Self::InvalidBytecode),
            5 => Some(Self::UnsupportedVersion),
            6 => Some(Self::OutOfMemory),
            7 => Some(Self::NoMainFunction),
            8 => Some(Self::ExecutionError),
            9 => Some(Self::ExecutionTimeout),
            _ => Some(Self::ExecutionError),
        }
    }

    pub fn status_code(self) -> i32 {
        match self {
            Self::InvalidArgument => 1,
            Self::IoError => 2,
            Self::FileTooLarge => 3,
            Self::InvalidBytecode => 4,
            Self::UnsupportedVersion => 5,
            Self::OutOfMemory => 6,
            Self::NoMainFunction => 7,
            Self::ExecutionError => 8,
            Self::ExecutionTimeout => 9,
        }
    }

    pub fn status_name(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid argument",
            Self::IoError => "i/o error",
            Self::FileTooLarge => "file too large",
            Self::InvalidBytecode => "invalid bytecode",
            Self::UnsupportedVersion => "unsupported version",
            Self::OutOfMemory => "out of memory",
            Self::NoMainFunction => "no main function",
            Self::ExecutionError => "execution error",
            Self::ExecutionTimeout => "execution timeout",
        }
    }
}

impl fmt::Display for PoseidonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.status_name())
    }
}

impl std::error::Error for PoseidonError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmConfig {
    pub max_steps: i64,
}

impl Default for VmConfig {
    fn default() -> Self {
        let raw = unsafe { poseidon_default_config() };
        Self {
            max_steps: raw.max_steps,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunResult {
    pub exit_code: i64,
    pub steps_executed: i64,
}

#[derive(Debug)]
pub struct Module {
    raw: NonNull<RawPoseidonModule>,
}

impl Module {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PoseidonError> {
        let path_string = path.as_ref().to_string_lossy();
        let c_path = CString::new(path_string.as_bytes()).map_err(|_| PoseidonError::InvalidArgument)?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe { poseidon_module_load_file(c_path.as_ptr(), &mut raw) };
        match PoseidonError::from_status(status) {
            None => Ok(Self {
                raw: NonNull::new(raw).ok_or(PoseidonError::ExecutionError)?,
            }),
            Some(err) => Err(err),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PoseidonError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { poseidon_module_load_bytes(bytes.as_ptr(), bytes.len(), &mut raw) };
        match PoseidonError::from_status(status) {
            None => Ok(Self {
                raw: NonNull::new(raw).ok_or(PoseidonError::ExecutionError)?,
            }),
            Some(err) => Err(err),
        }
    }

    pub fn run(&self, config: &VmConfig) -> Result<RunResult, PoseidonError> {
        let mut raw_config = unsafe { poseidon_default_config() };
        raw_config.max_steps = config.max_steps;

        let result = unsafe { poseidon_run_module(self.raw.as_ptr(), &raw_config) };
        match PoseidonError::from_status(result.status) {
            None => Ok(RunResult {
                exit_code: result.exit_code,
                steps_executed: result.steps_executed,
            }),
            Some(err) => Err(err),
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            poseidon_module_free(self.raw.as_ptr());
        }
    }
}
