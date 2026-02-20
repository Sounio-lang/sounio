//! HDF5 FFI Bindings for Sounio
//!
//! Provides C-callable functions for reading and writing HDF5 files from
//! Sounio programs. Feature-gated behind `hdf5` to avoid requiring libhdf5
//! at build time.
//!
//! These bindings use the HDF5 C API directly via extern "C" declarations.
//! At link time, the system's libhdf5 must be available.
//!
//! # Usage from Sounio
//!
//! ```sio
//! let file = hdf5_open("data.h5", "r")
//! let dataset = hdf5_dataset_open(file, "/measurements/temperature")
//! let dims = hdf5_dataset_dims(dataset)
//! let data = hdf5_read_f64(dataset)
//! hdf5_close(file)
//! ```
//!
//! # References
//!
//! - HDF5 C API: <https://docs.hdfgroup.org/hdf5/v1_14/group___h5.html>

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

// =============================================================================
// HDF5 C API type aliases
// =============================================================================

/// HDF5 identifier type (file, dataset, dataspace, etc.)
pub type Hid = i64;

/// HDF5 size type
pub type Hsize = u64;

/// HDF5 error type
pub type Herr = c_int;

// Well-known HDF5 datatype constants (from H5Tpublic.h)
// These are runtime values obtained from H5T_NATIVE_DOUBLE etc.
// We'll use H5T_NATIVE_DOUBLE_g which is the global variable in libhdf5.

// =============================================================================
// HDF5 C API extern declarations
// =============================================================================

#[link(name = "hdf5")]
extern "C" {
    // File operations
    fn H5Fopen(filename: *const c_char, flags: c_int, fapl: Hid) -> Hid;
    fn H5Fcreate(filename: *const c_char, flags: c_int, fcpl: Hid, fapl: Hid) -> Hid;
    fn H5Fclose(file_id: Hid) -> Herr;

    // Dataset operations
    fn H5Dopen2(loc: Hid, name: *const c_char, dapl: Hid) -> Hid;
    fn H5Dcreate2(
        loc: Hid,
        name: *const c_char,
        type_id: Hid,
        space_id: Hid,
        lcpl: Hid,
        dcpl: Hid,
        dapl: Hid,
    ) -> Hid;
    fn H5Dread(
        dset: Hid,
        mem_type: Hid,
        mem_space: Hid,
        file_space: Hid,
        xfer: Hid,
        buf: *mut c_void,
    ) -> Herr;
    fn H5Dwrite(
        dset: Hid,
        mem_type: Hid,
        mem_space: Hid,
        file_space: Hid,
        xfer: Hid,
        buf: *const c_void,
    ) -> Herr;
    fn H5Dget_space(dset: Hid) -> Hid;
    fn H5Dclose(dset: Hid) -> Herr;

    // Dataspace operations
    fn H5Sget_simple_extent_ndims(space: Hid) -> c_int;
    fn H5Sget_simple_extent_dims(space: Hid, dims: *mut Hsize, maxdims: *mut Hsize) -> c_int;
    fn H5Screate_simple(rank: c_int, dims: *const Hsize, maxdims: *const Hsize) -> Hid;
    fn H5Sclose(space: Hid) -> Herr;

    // Predefined datatypes (globals in libhdf5)
    static H5T_NATIVE_DOUBLE_g: Hid;
    static H5T_NATIVE_FLOAT_g: Hid;
    static H5T_NATIVE_INT_g: Hid;
    static H5T_NATIVE_LONG_g: Hid;
}

// HDF5 constants
const H5F_ACC_RDONLY: c_int = 0x0000;
const H5F_ACC_TRUNC: c_int = 0x0002;
const H5P_DEFAULT: Hid = 0;
const H5S_ALL: Hid = 0;

// =============================================================================
// Sounio FFI: HDF5 file operations
// =============================================================================

/// Open an HDF5 file for reading.
///
/// Returns an HDF5 file identifier, or -1 on failure.
///
/// # Safety
///
/// `path_ptr` must point to `path_len` valid UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_open_read(
    path_ptr: *const u8,
    path_len: usize,
) -> Hid {
    if path_ptr.is_null() {
        return -1;
    }

    let path_slice = std::slice::from_raw_parts(path_ptr, path_len);
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let c_path = match CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let fid = H5Fopen(c_path.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if fid < 0 { -1 } else { fid }
}

/// Create a new HDF5 file (truncates if exists).
///
/// Returns an HDF5 file identifier, or -1 on failure.
///
/// # Safety
///
/// `path_ptr` must point to `path_len` valid UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_create(
    path_ptr: *const u8,
    path_len: usize,
) -> Hid {
    if path_ptr.is_null() {
        return -1;
    }

    let path_slice = std::slice::from_raw_parts(path_ptr, path_len);
    let path_str = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let c_path = match CString::new(path_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let fid = H5Fcreate(c_path.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if fid < 0 { -1 } else { fid }
}

/// Close an HDF5 file.
///
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_close(file_id: Hid) -> i32 {
    if H5Fclose(file_id) < 0 { -1 } else { 0 }
}

// =============================================================================
// Sounio FFI: Dataset operations
// =============================================================================

/// Open a dataset within an HDF5 file.
///
/// Returns a dataset identifier, or -1 on failure.
///
/// # Safety
///
/// `name_ptr` must point to `name_len` valid UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_dataset_open(
    file_id: Hid,
    name_ptr: *const u8,
    name_len: usize,
) -> Hid {
    if name_ptr.is_null() {
        return -1;
    }

    let name_slice = std::slice::from_raw_parts(name_ptr, name_len);
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let c_name = match CString::new(name_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let did = H5Dopen2(file_id, c_name.as_ptr(), H5P_DEFAULT);
    if did < 0 { -1 } else { did }
}

/// Get the number of dimensions of a dataset.
///
/// Returns ndims on success, -1 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_dataset_ndims(dataset_id: Hid) -> i32 {
    let space = H5Dget_space(dataset_id);
    if space < 0 {
        return -1;
    }

    let ndims = H5Sget_simple_extent_ndims(space);
    H5Sclose(space);
    ndims
}

/// Get the dimensions of a dataset.
///
/// Writes dimensions into `out_dims`. Returns ndims on success, -1 on failure.
///
/// # Safety
///
/// `out_dims` must point to space for at least `max_dims` u64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_dataset_dims(
    dataset_id: Hid,
    out_dims: *mut u64,
    max_dims: usize,
) -> i32 {
    if out_dims.is_null() {
        return -1;
    }

    let space = H5Dget_space(dataset_id);
    if space < 0 {
        return -1;
    }

    let ndims = H5Sget_simple_extent_ndims(space);
    if ndims < 0 || ndims as usize > max_dims {
        H5Sclose(space);
        return -1;
    }

    let result = H5Sget_simple_extent_dims(space, out_dims as *mut Hsize, std::ptr::null_mut());
    H5Sclose(space);

    if result < 0 { -1 } else { result }
}

/// Read an entire dataset as f64 values.
///
/// Returns a heap-allocated buffer (caller must free with `__sounio_hdf5_free`).
/// Writes the total number of elements to `out_len`.
///
/// # Safety
///
/// `out_len` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_read_f64(
    dataset_id: Hid,
    out_len: *mut usize,
) -> *mut f64 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }

    // Get dataspace and compute total elements
    let space = H5Dget_space(dataset_id);
    if space < 0 {
        return std::ptr::null_mut();
    }

    let ndims = H5Sget_simple_extent_ndims(space);
    if ndims < 0 || ndims > 16 {
        H5Sclose(space);
        return std::ptr::null_mut();
    }

    let mut dims = [0u64; 16];
    if H5Sget_simple_extent_dims(space, dims.as_mut_ptr(), std::ptr::null_mut()) < 0 {
        H5Sclose(space);
        return std::ptr::null_mut();
    }

    let total: usize = dims[..ndims as usize].iter().product::<u64>() as usize;
    if total == 0 {
        H5Sclose(space);
        *out_len = 0;
        return std::ptr::null_mut();
    }

    // Allocate buffer and read
    let mut buffer = vec![0.0f64; total];
    let status = H5Dread(
        dataset_id,
        H5T_NATIVE_DOUBLE_g,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buffer.as_mut_ptr() as *mut c_void,
    );

    H5Sclose(space);

    if status < 0 {
        return std::ptr::null_mut();
    }

    *out_len = total;
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Read an entire dataset as f32 values.
///
/// # Safety
///
/// `out_len` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_read_f32(
    dataset_id: Hid,
    out_len: *mut usize,
) -> *mut f32 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }

    let space = H5Dget_space(dataset_id);
    if space < 0 {
        return std::ptr::null_mut();
    }

    let ndims = H5Sget_simple_extent_ndims(space);
    if ndims < 0 || ndims > 16 {
        H5Sclose(space);
        return std::ptr::null_mut();
    }

    let mut dims = [0u64; 16];
    if H5Sget_simple_extent_dims(space, dims.as_mut_ptr(), std::ptr::null_mut()) < 0 {
        H5Sclose(space);
        return std::ptr::null_mut();
    }

    let total: usize = dims[..ndims as usize].iter().product::<u64>() as usize;
    if total == 0 {
        H5Sclose(space);
        *out_len = 0;
        return std::ptr::null_mut();
    }

    let mut buffer = vec![0.0f32; total];
    let status = H5Dread(
        dataset_id,
        H5T_NATIVE_FLOAT_g,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buffer.as_mut_ptr() as *mut c_void,
    );

    H5Sclose(space);

    if status < 0 {
        return std::ptr::null_mut();
    }

    *out_len = total;
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Write f64 data to a new dataset.
///
/// Creates a dataset with the given name and dimensions, then writes data.
/// Returns 0 on success, -1 on failure.
///
/// # Safety
///
/// - `name_ptr` must point to `name_len` valid UTF-8 bytes.
/// - `dims_ptr` must point to `ndims` u64 values.
/// - `data_ptr` must point to `product(dims)` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_write_f64(
    file_id: Hid,
    name_ptr: *const u8,
    name_len: usize,
    data_ptr: *const f64,
    dims_ptr: *const u64,
    ndims: i32,
) -> i32 {
    if name_ptr.is_null() || data_ptr.is_null() || dims_ptr.is_null() || ndims <= 0 {
        return -1;
    }

    let name_slice = std::slice::from_raw_parts(name_ptr, name_len);
    let name_str = match std::str::from_utf8(name_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let c_name = match CString::new(name_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let dims = std::slice::from_raw_parts(dims_ptr as *const Hsize, ndims as usize);

    // Create dataspace
    let space = H5Screate_simple(ndims, dims.as_ptr(), std::ptr::null());
    if space < 0 {
        return -1;
    }

    // Create dataset
    let dset = H5Dcreate2(
        file_id,
        c_name.as_ptr(),
        H5T_NATIVE_DOUBLE_g,
        space,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT,
    );
    if dset < 0 {
        H5Sclose(space);
        return -1;
    }

    // Write data
    let status = H5Dwrite(
        dset,
        H5T_NATIVE_DOUBLE_g,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        data_ptr as *const c_void,
    );

    H5Dclose(dset);
    H5Sclose(space);

    if status < 0 { -1 } else { 0 }
}

/// Close a dataset.
///
/// Returns 0 on success, -1 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_dataset_close(dataset_id: Hid) -> i32 {
    if H5Dclose(dataset_id) < 0 { -1 } else { 0 }
}

/// Free a buffer allocated by `__sounio_hdf5_read_f64` or `__sounio_hdf5_read_f32`.
///
/// # Safety
///
/// `ptr` must have been allocated by one of the HDF5 read functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __sounio_hdf5_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_defined() {
        assert_eq!(H5F_ACC_RDONLY, 0);
        assert_eq!(H5F_ACC_TRUNC, 2);
        assert_eq!(H5P_DEFAULT, 0);
        assert_eq!(H5S_ALL, 0);
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            assert_eq!(__sounio_hdf5_open_read(std::ptr::null(), 0), -1);
            assert_eq!(__sounio_hdf5_create(std::ptr::null(), 0), -1);
            assert_eq!(__sounio_hdf5_dataset_open(-1, std::ptr::null(), 0), -1);
            assert_eq!(__sounio_hdf5_dataset_dims(-1, std::ptr::null_mut(), 0), -1);
            assert!(
                __sounio_hdf5_read_f64(-1, std::ptr::null_mut()).is_null()
            );
        }
    }
}
