//! Installed memory, to decide how much processing state the app can afford to keep.

/// Physical RAM in bytes, if the OS reports it.
pub fn installed() -> Option<u64> {
    imp::installed()
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn installed() -> Option<u64> {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib = info
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kib * 1024)
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    pub fn installed() -> Option<u64> {
        let mut bytes = 0u64;
        let mut len = size_of::<u64>();
        // SAFETY: hw.memsize is a 64-bit integer and `len` is the size of `bytes`.
        let status = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0).then_some(bytes)
    }
}

#[cfg(windows)]
mod imp {
    /// `MEMORYSTATUSEX`.
    #[repr(C)]
    struct MemoryStatus {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatus) -> i32;
    }

    pub fn installed() -> Option<u64> {
        let mut status = MemoryStatus {
            length: size_of::<MemoryStatus>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a MEMORYSTATUSEX with its length set, as the API requires.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
        ok.then_some(status.total_phys)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod imp {
    pub fn installed() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn reports_installed_memory() {
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            let ram = super::installed().expect("installed memory is reported");
            assert!(ram >= 1 << 30, "implausible installed memory: {ram} bytes");
        }
    }
}
