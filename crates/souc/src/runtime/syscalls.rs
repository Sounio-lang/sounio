//! Minimal raw syscall wrappers.
//!
//! These are thin wrappers intended for bootstrap / self-hosting work.
//! We avoid libc function calls and invoke the kernel directly via inline asm.
//!
//! Notes:
//! - Return values are raw syscall return values (negative on Linux, -1 on macOS).
//! - The `sys_open` flags in this module follow Linux values; on macOS we translate
//!   `O_CREAT` and `O_TRUNC` to the platform equivalents.
//!
//! Safety:
//! - All `unsafe` is contained inside small `syscall*` helpers.

use core::mem::MaybeUninit;
use core::ptr;

// The self-hosted toolchain currently assumes Linux-style open(2) flags.
pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_RDWR: i32 = 2;
pub const O_CREAT: i32 = 0x40;
pub const O_TRUNC: i32 = 0x200;

#[cfg(target_os = "linux")]
fn normalize_open_flags(flags: i32) -> i32 {
    flags
}

#[cfg(target_os = "macos")]
fn normalize_open_flags(flags: i32) -> i32 {
    // Translate the Linux-ish bits we expose into Darwin values.
    let mut out = flags;
    if (flags & O_CREAT) != 0 {
        out = (out & !O_CREAT) | libc::O_CREAT;
    }
    if (flags & O_TRUNC) != 0 {
        out = (out & !O_TRUNC) | libc::O_TRUNC;
    }
    out
}

#[cfg(target_os = "linux")]
fn syscall_nr(n: libc::c_long) -> u64 {
    n as u64
}

#[cfg(target_os = "macos")]
fn syscall_nr(n: libc::c_long) -> u64 {
    // Darwin x86_64/arm64 uses a 0x2000000 prefix for UNIX syscalls.
    0x2000000_u64 + (n as u64)
}

#[cfg(all(target_os = "linux"))]
fn map_anon_flag() -> i32 {
    libc::MAP_ANONYMOUS
}

#[cfg(all(target_os = "macos"))]
fn map_anon_flag() -> i32 {
    libc::MAP_ANON
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use core::arch::asm;

    #[inline(always)]
    pub(super) unsafe fn syscall1(nr: u64, a1: u64) -> u64 {
        let ret: u64;
        unsafe {
            asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a1,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }

    #[inline(always)]
    pub(super) unsafe fn syscall2(nr: u64, a1: u64, a2: u64) -> u64 {
        let ret: u64;
        unsafe {
            asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a1,
                in("rsi") a2,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }

    #[inline(always)]
    pub(super) unsafe fn syscall3(nr: u64, a1: u64, a2: u64, a3: u64) -> u64 {
        let ret: u64;
        unsafe {
            asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }

    #[inline(always)]
    pub(super) unsafe fn syscall4(nr: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> u64 {
        let ret: u64;
        unsafe {
            asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }

    #[inline(always)]
    pub(super) unsafe fn syscall6(
        nr: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        a6: u64,
    ) -> u64 {
        let ret: u64;
        unsafe {
            asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                in("r8") a5,
                in("r9") a6,
                lateout("rax") ret,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod arch {
    use core::arch::asm;

    #[inline(always)]
    pub(super) unsafe fn syscall1(nr: u64, a1: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0",
                in("x8") nr,
                inout("x0") x0,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall2(nr: u64, a1: u64, a2: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0",
                in("x8") nr,
                inout("x0") x0,
                in("x1") a2,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall3(nr: u64, a1: u64, a2: u64, a3: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0",
                in("x8") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall4(nr: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0",
                in("x8") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall6(
        nr: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        a6: u64,
    ) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0",
                in("x8") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                in("x4") a5,
                in("x5") a6,
                options(nostack),
            );
        }
        x0
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod arch {
    use core::arch::asm;

    #[inline(always)]
    pub(super) unsafe fn syscall1(nr: u64, a1: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0x80",
                in("x16") nr,
                inout("x0") x0,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall2(nr: u64, a1: u64, a2: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0x80",
                in("x16") nr,
                inout("x0") x0,
                in("x1") a2,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall3(nr: u64, a1: u64, a2: u64, a3: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0x80",
                in("x16") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall4(nr: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0x80",
                in("x16") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                options(nostack),
            );
        }
        x0
    }

    #[inline(always)]
    pub(super) unsafe fn syscall6(
        nr: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        a6: u64,
    ) -> u64 {
        let mut x0 = a1;
        unsafe {
            asm!(
                "svc 0x80",
                in("x16") nr,
                inout("x0") x0,
                in("x1") a2,
                in("x2") a3,
                in("x3") a4,
                in("x4") a5,
                in("x5") a6,
                options(nostack),
            );
        }
        x0
    }
}

pub fn sys_write(fd: i32, buf: *const u8, len: usize) -> isize {
    // Linux: SYS_write = 1 (x86_64), 64 (aarch64)
    // macOS: SYS_write = 4 (prefix 0x2000000)
    let nr = syscall_nr(libc::SYS_write);
    unsafe { arch::syscall3(nr, fd as u64, buf as u64, len as u64) as isize }
}

pub fn sys_read(fd: i32, buf: *mut u8, len: usize) -> isize {
    let nr = syscall_nr(libc::SYS_read);
    unsafe { arch::syscall3(nr, fd as u64, buf as u64, len as u64) as isize }
}

#[cfg(target_os = "linux")]
pub fn sys_open(path: *const u8, flags: i32, mode: i32) -> isize {
    let nr = syscall_nr(libc::SYS_openat);
    let flags = normalize_open_flags(flags);
    let dirfd = libc::AT_FDCWD as i64 as u64;
    unsafe { arch::syscall4(nr, dirfd, path as u64, flags as u64, mode as u64) as isize }
}

#[cfg(target_os = "macos")]
pub fn sys_open(path: *const u8, flags: i32, mode: i32) -> isize {
    let nr = syscall_nr(libc::SYS_open);
    let flags = normalize_open_flags(flags);
    unsafe { arch::syscall3(nr, path as u64, flags as u64, mode as u64) as isize }
}

pub fn sys_close(fd: i32) -> isize {
    let nr = syscall_nr(libc::SYS_close);
    unsafe { arch::syscall1(nr, fd as u64) as isize }
}

pub fn sys_exit(code: i32) -> ! {
    let nr = syscall_nr(libc::SYS_exit);
    unsafe {
        let _ = arch::syscall1(nr, code as u64);
    }
    // If the syscall returns, something is very wrong; avoid UB.
    loop {
        core::hint::spin_loop();
    }
}

pub fn sys_fstat_size(fd: i32) -> isize {
    let nr = syscall_nr(libc::SYS_fstat);
    let mut st = MaybeUninit::<libc::stat>::uninit();
    let ret = unsafe { arch::syscall2(nr, fd as u64, st.as_mut_ptr() as u64) as isize };
    if ret != 0 {
        return ret;
    }
    // SAFETY: kernel wrote a full `libc::stat` on success.
    let st = unsafe { st.assume_init() };
    st.st_size as isize
}

pub fn sys_mmap(len: usize) -> *mut u8 {
    let nr = syscall_nr(libc::SYS_mmap);
    let prot = (libc::PROT_READ | libc::PROT_WRITE) as u64;
    let flags = (libc::MAP_PRIVATE | map_anon_flag()) as u64;
    let fd = (-1_i64) as u64;
    let ret = unsafe { arch::syscall6(nr, 0, len as u64, prot, flags, fd, 0) };
    if (ret as isize) < 0 {
        return ptr::null_mut();
    }
    ret as *mut u8
}

pub fn sys_munmap(ptr: *mut u8, len: usize) -> isize {
    let nr = syscall_nr(libc::SYS_munmap);
    unsafe { arch::syscall2(nr, ptr as u64, len as u64) as isize }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path() -> Vec<u8> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let s = format!("/tmp/sounio_syscall_test_{}_{}\0", std::process::id(), n);
        s.into_bytes()
    }

    #[test]
    fn test_sys_write() {
        let msg = b"hello from syscall\n";
        let ret = sys_write(1, msg.as_ptr(), msg.len());
        assert_eq!(ret, msg.len() as isize);
    }

    #[test]
    fn test_sys_open_read_close_and_fstat() {
        let path = temp_path();
        let data = b"test data";

        let fd = sys_open(path.as_ptr(), O_WRONLY | O_CREAT | O_TRUNC, 0o644);
        assert!(fd >= 0, "open(write) failed: {fd}");
        let wrote = sys_write(fd as i32, data.as_ptr(), data.len());
        assert_eq!(wrote, data.len() as isize);
        assert_eq!(sys_close(fd as i32), 0);

        let fd = sys_open(path.as_ptr(), O_RDONLY, 0);
        assert!(fd >= 0, "open(read) failed: {fd}");
        let size = sys_fstat_size(fd as i32);
        assert_eq!(size, data.len() as isize);

        let mut buf = [0u8; 64];
        let n = sys_read(fd as i32, buf.as_mut_ptr(), buf.len());
        assert_eq!(n, data.len() as isize);
        assert_eq!(&buf[..data.len()], data);
        assert_eq!(sys_close(fd as i32), 0);

        let path_str = std::str::from_utf8(&path[..path.len() - 1]).unwrap();
        let _ = std::fs::remove_file(path_str);
    }
}
