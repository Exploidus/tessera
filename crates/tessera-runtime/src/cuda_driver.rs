#![allow(non_snake_case, dead_code, clippy::all)]

use std::ffi::{c_void, CString};

pub type CUresult = i32;
pub const CUDA_SUCCESS: CUresult = 0;

pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub const CU_STREAM_DEFAULT: u32 = 0;
pub const CU_STREAM_NON_BLOCKING: u32 = 1;

pub const CU_EVENT_DEFAULT: u32 = 0;
pub const CU_EVENT_BLOCKING_SYNC: u32 = 1;
pub const CU_EVENT_DISABLE_TIMING: u32 = 2;

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
    fn FreeLibrary(hModule: *mut c_void) -> i32;
}

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

pub struct CudaDriver {
    _handle: *mut c_void,
    pub cu_init: unsafe extern "system" fn(flags: u32) -> CUresult,
    pub cu_driver_get_version: unsafe extern "system" fn(version: *mut i32) -> CUresult,
    pub cu_device_get_count: unsafe extern "system" fn(count: *mut i32) -> CUresult,
    pub cu_device_get: unsafe extern "system" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cu_device_get_name:
        unsafe extern "system" fn(name: *mut u8, len: i32, dev: CUdevice) -> CUresult,
    pub cu_ctx_create:
        unsafe extern "system" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    pub cu_ctx_destroy: unsafe extern "system" fn(ctx: CUcontext) -> CUresult,
    pub cu_ctx_sync: unsafe extern "system" fn() -> CUresult,
    pub cu_module_load_data:
        unsafe extern "system" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    pub cu_module_load_data_ex: unsafe extern "system" fn(
        module: *mut CUmodule,
        image: *const c_void,
        numOptions: u32,
        options: *mut u32,
        optionValues: *mut *mut c_void,
    ) -> CUresult,
    pub cu_module_get_function: unsafe extern "system" fn(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const u8,
    ) -> CUresult,
    pub cu_module_unload: unsafe extern "system" fn(hmod: CUmodule) -> CUresult,
    pub cu_mem_alloc:
        unsafe extern "system" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cu_mem_free: unsafe extern "system" fn(dptr: CUdeviceptr) -> CUresult,
    pub cu_memcpy_htod:
        unsafe extern "system" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult,
    pub cu_memcpy_dtoh:
        unsafe extern "system" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult,
    pub cu_memcpy_htod_async: unsafe extern "system" fn(
        dst: CUdeviceptr,
        src: *const c_void,
        bytes: usize,
        hStream: CUstream,
    ) -> CUresult,
    pub cu_memcpy_dtoh_async: unsafe extern "system" fn(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytes: usize,
        hStream: CUstream,
    ) -> CUresult,
    pub cu_launch_kernel: unsafe extern "system" fn(
        f: CUfunction,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    pub cu_stream_create:
        unsafe extern "system" fn(phStream: *mut CUstream, flags: u32) -> CUresult,
    pub cu_stream_destroy: unsafe extern "system" fn(hStream: CUstream) -> CUresult,
    pub cu_stream_synchronize: unsafe extern "system" fn(hStream: CUstream) -> CUresult,
    pub cu_stream_wait_event:
        unsafe extern "system" fn(hStream: CUstream, hEvent: CUevent, flags: u32) -> CUresult,
    pub cu_event_create: unsafe extern "system" fn(phEvent: *mut CUevent, flags: u32) -> CUresult,
    pub cu_event_destroy: unsafe extern "system" fn(hEvent: CUevent) -> CUresult,
    pub cu_event_record: unsafe extern "system" fn(hEvent: CUevent, hStream: CUstream) -> CUresult,
    pub cu_event_synchronize: unsafe extern "system" fn(hEvent: CUevent) -> CUresult,
    pub cu_event_elapsed_time: unsafe extern "system" fn(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult,
    pub cu_device_can_access_peer: unsafe extern "system" fn(
        canAccessPeer: *mut i32,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult,
    pub cu_ctx_enable_peer_access:
        unsafe extern "system" fn(peerCtx: CUcontext, flags: u32) -> CUresult,
    pub cu_ctx_disable_peer_access: unsafe extern "system" fn(peerCtx: CUcontext) -> CUresult,
    pub cu_memcpy_peer: unsafe extern "system" fn(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
    ) -> CUresult,
    pub cu_memcpy_peer_async: unsafe extern "system" fn(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult,
}

impl CudaDriver {
    pub fn load() -> Result<Self, String> {
        #[cfg(windows)]
        let handle = unsafe {
            let lib = b"nvcuda.dll\0";
            LoadLibraryA(lib.as_ptr())
        };

        #[cfg(unix)]
        let handle = unsafe {
            let lib = b"libcuda.so.1\0";
            dlopen(lib.as_ptr(), 1)
        };

        if handle.is_null() {
            return Err("NVIDIA Driver (nvcuda.dll / libcuda.so) not found on system".into());
        }

        unsafe {
            macro_rules! sym {
                ($name:expr) => {{
                    let symbol = CString::new($name).unwrap();
                    #[cfg(windows)]
                    let ptr = GetProcAddress(handle, symbol.as_ptr() as *const u8);
                    #[cfg(unix)]
                    let ptr = dlsym(handle, symbol.as_ptr() as *const u8);
                    if ptr.is_null() {
                        return Err(format!("Failed to locate CUDA driver symbol: {}", $name));
                    }
                    std::mem::transmute(ptr)
                }};
            }

            Ok(Self {
                _handle: handle,
                cu_init: sym!("cuInit"),
                cu_driver_get_version: sym!("cuDriverGetVersion"),
                cu_device_get_count: sym!("cuDeviceGetCount"),
                cu_device_get: sym!("cuDeviceGet"),
                cu_device_get_name: sym!("cuDeviceGetName"),
                cu_ctx_create: sym!("cuCtxCreate_v2"),
                cu_ctx_destroy: sym!("cuCtxDestroy_v2"),
                cu_ctx_sync: sym!("cuCtxSynchronize"),
                cu_module_load_data: sym!("cuModuleLoadData"),
                cu_module_load_data_ex: sym!("cuModuleLoadDataEx"),
                cu_module_get_function: sym!("cuModuleGetFunction"),
                cu_module_unload: sym!("cuModuleUnload"),
                cu_mem_alloc: sym!("cuMemAlloc_v2"),
                cu_mem_free: sym!("cuMemFree_v2"),
                cu_memcpy_htod: sym!("cuMemcpyHtoD_v2"),
                cu_memcpy_dtoh: sym!("cuMemcpyDtoH_v2"),
                cu_memcpy_htod_async: sym!("cuMemcpyHtoDAsync_v2"),
                cu_memcpy_dtoh_async: sym!("cuMemcpyDtoHAsync_v2"),
                cu_launch_kernel: sym!("cuLaunchKernel"),
                cu_stream_create: sym!("cuStreamCreate"),
                cu_stream_destroy: sym!("cuStreamDestroy_v2"),
                cu_stream_synchronize: sym!("cuStreamSynchronize"),
                cu_stream_wait_event: sym!("cuStreamWaitEvent"),
                cu_event_create: sym!("cuEventCreate"),
                cu_event_destroy: sym!("cuEventDestroy_v2"),
                cu_event_record: sym!("cuEventRecord"),
                cu_event_synchronize: sym!("cuEventSynchronize"),
                cu_event_elapsed_time: sym!("cuEventElapsedTime"),
                cu_device_can_access_peer: sym!("cuDeviceCanAccessPeer"),
                cu_ctx_enable_peer_access: sym!("cuCtxEnablePeerAccess"),
                cu_ctx_disable_peer_access: sym!("cuCtxDisablePeerAccess"),
                cu_memcpy_peer: sym!("cuMemcpyPeer"),
                cu_memcpy_peer_async: sym!("cuMemcpyPeerAsync"),
            })
        }
    }
}
