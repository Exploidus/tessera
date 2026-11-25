#![allow(clippy::all)]

pub mod autotune;
pub mod cuda_driver;

use std::ffi::{c_void, CStr, CString};
use std::sync::Arc;
use tessera_backend_ptx::PtxGenerator;
use tessera_ir::KernelModule;

use cuda_driver::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUmodule, CUstream, CudaDriver,
    CUDA_SUCCESS, CU_EVENT_DEFAULT, CU_STREAM_NON_BLOCKING,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBackend {
    NvidiaPtx,
    VulkanSpirv,
    AppleMetal,
    HostCpu,
}

pub struct JITCompiler {
    backend: DeviceBackend,
    target_arch: u32,
}

impl JITCompiler {
    pub fn new(backend: DeviceBackend, target_arch: u32) -> Self {
        Self {
            backend,
            target_arch,
        }
    }

    pub fn compile_and_assemble(&self, module: &KernelModule) -> String {
        match self.backend {
            DeviceBackend::NvidiaPtx => {
                let gen = PtxGenerator::new(self.target_arch);
                gen.compile(module)
            }
            DeviceBackend::VulkanSpirv => {
                let gen = tessera_backend_spirv::SpirvGenerator::new();
                let config = module.gemm_config.clone().unwrap_or_default();
                gen.emit_vulkan_compute_shader(&module.name, &config)
            }
            DeviceBackend::AppleMetal => {
                let gen = tessera_backend_metal::MetalGenerator::new();
                let config = module.gemm_config.clone().unwrap_or_default();
                gen.emit_metal_gemm(&module.name, &config)
            }
            DeviceBackend::HostCpu => {
                todo!("Host CPU fallback backend in development")
            }
        }
    }
}

pub fn pad_matrix_2d<T: Copy + Default>(
    data: &[T],
    rows: usize,
    cols: usize,
    tile_dim: usize,
) -> (Vec<T>, usize, usize) {
    let pad_rows = ((rows + tile_dim - 1) / tile_dim) * tile_dim;
    let pad_cols = ((cols + tile_dim - 1) / tile_dim) * tile_dim;

    if pad_rows == rows && pad_cols == cols {
        return (data.to_vec(), pad_rows, pad_cols);
    }

    let mut padded = vec![T::default(); pad_rows * pad_cols];
    for r in 0..rows {
        for c in 0..cols {
            padded[r * pad_cols + c] = data[r * cols + c];
        }
    }
    (padded, pad_rows, pad_cols)
}

pub fn unpad_matrix_2d<T: Copy>(
    data: &[T],
    pad_cols: usize,
    orig_rows: usize,
    orig_cols: usize,
) -> Vec<T> {
    if orig_cols == pad_cols {
        return data[..orig_rows * orig_cols].to_vec();
    }
    let mut unpadded = Vec::with_capacity(orig_rows * orig_cols);
    for r in 0..orig_rows {
        for c in 0..orig_cols {
            unpadded.push(data[r * pad_cols + c]);
        }
    }
    unpadded
}

pub struct GpuBuffer {
    driver: Arc<CudaDriver>,
    pub dptr: CUdeviceptr,
    pub bytes: usize,
}

impl GpuBuffer {
    pub fn upload<T>(&self, host_data: &[T]) -> Result<(), String> {
        let data_bytes = host_data.len() * std::mem::size_of::<T>();
        if data_bytes > self.bytes {
            return Err("Host data exceeds device buffer allocation".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_htod)(
                self.dptr,
                host_data.as_ptr() as *const c_void,
                data_bytes,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD failed with code {}", res));
            }
        }
        Ok(())
    }

    pub fn download<T>(&self, host_data: &mut [T]) -> Result<(), String> {
        let data_bytes = host_data.len() * std::mem::size_of::<T>();
        if data_bytes > self.bytes {
            return Err("Host slice exceeds device buffer size".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_dtoh)(
                host_data.as_mut_ptr() as *mut c_void,
                self.dptr,
                data_bytes,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH failed with code {}", res));
            }
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_mem_free)(self.dptr);
        }
    }
}

pub struct GpuStream {
    driver: Arc<CudaDriver>,
    pub stream: CUstream,
}

impl GpuStream {
    pub fn new(driver: Arc<CudaDriver>, non_blocking: bool) -> Result<Self, String> {
        let flags = if non_blocking {
            CU_STREAM_NON_BLOCKING
        } else {
            0
        };
        let mut stream: CUstream = std::ptr::null_mut();
        unsafe {
            let res = (driver.cu_stream_create)(&mut stream, flags);
            if res != CUDA_SUCCESS {
                return Err(format!("cuStreamCreate failed with code: {}", res));
            }
        }
        Ok(Self { driver, stream })
    }

    pub fn sync(&self) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_stream_synchronize)(self.stream);
            if res != CUDA_SUCCESS {
                return Err(format!("cuStreamSynchronize failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn wait_event(&self, event: &GpuEvent) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_stream_wait_event)(self.stream, event.event, 0);
            if res != CUDA_SUCCESS {
                return Err(format!("cuStreamWaitEvent failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn upload_async<T>(&self, buffer: &GpuBuffer, host_data: &[T]) -> Result<(), String> {
        let data_bytes = host_data.len() * std::mem::size_of::<T>();
        if data_bytes > buffer.bytes {
            return Err("Host data exceeds device buffer allocation".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_htod_async)(
                buffer.dptr,
                host_data.as_ptr() as *const c_void,
                data_bytes,
                self.stream,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoDAsync failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn download_async<T>(&self, buffer: &GpuBuffer, host_data: &mut [T]) -> Result<(), String> {
        let data_bytes = host_data.len() * std::mem::size_of::<T>();
        if data_bytes > buffer.bytes {
            return Err("Host slice exceeds device buffer size".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_dtoh_async)(
                host_data.as_mut_ptr() as *mut c_void,
                buffer.dptr,
                data_bytes,
                self.stream,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoHAsync failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub unsafe fn launch_kernel(
        &self,
        kernel: &GpuKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let res = (self.driver.cu_launch_kernel)(
            kernel.func,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem_bytes,
            self.stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if res != CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel async failed with code: {}", res));
        }
        Ok(())
    }
}

impl Drop for GpuStream {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                let _ = (self.driver.cu_stream_destroy)(self.stream);
            }
        }
    }
}

pub struct GpuEvent {
    driver: Arc<CudaDriver>,
    pub event: CUevent,
}

impl GpuEvent {
    pub fn new(driver: Arc<CudaDriver>) -> Result<Self, String> {
        let mut event: CUevent = std::ptr::null_mut();
        unsafe {
            let res = (driver.cu_event_create)(&mut event, CU_EVENT_DEFAULT);
            if res != CUDA_SUCCESS {
                return Err(format!("cuEventCreate failed with code: {}", res));
            }
        }
        Ok(Self { driver, event })
    }

    pub fn record(&self, stream: &GpuStream) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_event_record)(self.event, stream.stream);
            if res != CUDA_SUCCESS {
                return Err(format!("cuEventRecord failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_event_synchronize)(self.event);
            if res != CUDA_SUCCESS {
                return Err(format!("cuEventSynchronize failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn elapsed_time(&self, end: &GpuEvent) -> Result<f32, String> {
        let mut ms = 0.0f32;
        unsafe {
            let res = (self.driver.cu_event_elapsed_time)(&mut ms, self.event, end.event);
            if res != CUDA_SUCCESS {
                return Err(format!("cuEventElapsedTime failed with code: {}", res));
            }
        }
        Ok(ms)
    }
}

impl Drop for GpuEvent {
    fn drop(&mut self) {
        unsafe {
            if !self.event.is_null() {
                let _ = (self.driver.cu_event_destroy)(self.event);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub name: String,
}

pub struct MultiGpuTopology {
    pub devices: Vec<DeviceInfo>,
}

impl MultiGpuTopology {
    pub fn discover(driver: &Arc<CudaDriver>) -> Result<Self, String> {
        let mut count = 0i32;
        unsafe {
            let res = (driver.cu_device_get_count)(&mut count);
            if res != CUDA_SUCCESS {
                return Err(format!("cuDeviceGetCount failed with code: {}", res));
            }
        }

        let mut devices = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut dev: CUdevice = 0;
            unsafe {
                let res = (driver.cu_device_get)(&mut dev, i);
                if res != CUDA_SUCCESS {
                    continue;
                }
                let mut name_buf = [0u8; 256];
                let _ = (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev);
                let name = CStr::from_ptr(name_buf.as_ptr() as *const i8)
                    .to_string_lossy()
                    .to_string();
                devices.push(DeviceInfo { ordinal: i, name });
            }
        }
        Ok(Self { devices })
    }
}

pub struct CollectiveComm;

impl CollectiveComm {
    pub fn all_reduce_sum(buffers: &mut [&mut [f32]]) {
        if buffers.is_empty() {
            return;
        }
        let len = buffers[0].len();
        for i in 0..len {
            let mut sum = 0.0f32;
            for buf in buffers.iter() {
                sum += buf[i];
            }
            for buf in buffers.iter_mut() {
                buf[i] = sum;
            }
        }
    }

    pub fn broadcast(src: &[f32], dsts: &mut [&mut [f32]]) {
        for dst in dsts.iter_mut() {
            dst.copy_from_slice(src);
        }
    }
}

pub struct GpuContext {
    pub driver: Arc<CudaDriver>,
    _device: CUdevice,
    pub ctx: CUcontext,
    pub device_name: String,
    pub driver_version: i32,
}

impl GpuContext {
    pub fn init() -> Result<Self, String> {
        Self::init_device(0)
    }

    pub fn init_device(ordinal: i32) -> Result<Self, String> {
        let driver = Arc::new(CudaDriver::load()?);
        unsafe {
            let res = (driver.cu_init)(0);
            if res != CUDA_SUCCESS {
                return Err(format!("cuInit failed with error code: {}", res));
            }

            let mut version: i32 = 0;
            let _ = (driver.cu_driver_get_version)(&mut version);

            let mut device: CUdevice = 0;
            let res = (driver.cu_device_get)(&mut device, ordinal);
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "cuDeviceGet failed for ordinal {}: code {}",
                    ordinal, res
                ));
            }

            let mut name_buf = [0u8; 256];
            let _ = (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, device);
            let device_name = CStr::from_ptr(name_buf.as_ptr() as *const i8)
                .to_string_lossy()
                .to_string();

            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = (driver.cu_ctx_create)(&mut ctx, 0, device);
            if res != CUDA_SUCCESS {
                return Err(format!("cuCtxCreate failed with error code: {}", res));
            }

            Ok(Self {
                driver,
                _device: device,
                ctx,
                device_name,
                driver_version: version,
            })
        }
    }

    pub fn alloc(&self, bytes: usize) -> Result<GpuBuffer, String> {
        let mut dptr: CUdeviceptr = 0;
        unsafe {
            let res = (self.driver.cu_mem_alloc)(&mut dptr, bytes);
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemAlloc failed with code: {}", res));
            }
        }
        Ok(GpuBuffer {
            driver: Arc::clone(&self.driver),
            dptr,
            bytes,
        })
    }

    pub fn create_stream(&self, non_blocking: bool) -> Result<GpuStream, String> {
        GpuStream::new(Arc::clone(&self.driver), non_blocking)
    }

    pub fn create_event(&self) -> Result<GpuEvent, String> {
        GpuEvent::new(Arc::clone(&self.driver))
    }

    pub fn get_topology(&self) -> Result<MultiGpuTopology, String> {
        MultiGpuTopology::discover(&self.driver)
    }

    pub fn load_ptx(&self, ptx_code: &str) -> Result<GpuModule, String> {
        let c_ptx = CString::new(ptx_code).map_err(|e| e.to_string())?;
        let mut module: CUmodule = std::ptr::null_mut();
        let mut error_log = vec![0u8; 4096];
        let mut options: [u32; 2] = [5, 6];
        let mut option_values: [*mut c_void; 2] = [
            error_log.as_mut_ptr() as *mut c_void,
            4096usize as *mut c_void,
        ];
        unsafe {
            let res = (self.driver.cu_module_load_data_ex)(
                &mut module,
                c_ptx.as_ptr() as *const c_void,
                2,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            );
            if res != CUDA_SUCCESS {
                let err_str = CStr::from_ptr(error_log.as_ptr() as *const i8).to_string_lossy();
                return Err(format!(
                    "cuModuleLoadDataEx failed (code: {}): {}",
                    res, err_str
                ));
            }
        }
        Ok(GpuModule {
            driver: Arc::clone(&self.driver),
            module,
        })
    }

    pub fn sync(&self) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_ctx_sync)();
            if res != CUDA_SUCCESS {
                return Err(format!("cuCtxSynchronize failed with code: {}", res));
            }
        }
        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                let _ = (self.driver.cu_ctx_destroy)(self.ctx);
            }
        }
    }
}

pub struct GpuModule {
    driver: Arc<CudaDriver>,
    module: CUmodule,
}

impl GpuModule {
    pub fn get_kernel(&self, name: &str) -> Result<GpuKernel, String> {
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let res = (self.driver.cu_module_get_function)(
                &mut func,
                self.module,
                c_name.as_ptr() as *const u8,
            );
            if res != CUDA_SUCCESS {
                return Err(format!(
                    "Kernel function '{}' not found in module (code: {})",
                    name, res
                ));
            }
        }
        Ok(GpuKernel {
            driver: Arc::clone(&self.driver),
            func,
        })
    }
}

impl Drop for GpuModule {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_module_unload)(self.module);
        }
    }
}

pub struct GpuKernel {
    driver: Arc<CudaDriver>,
    pub func: CUfunction,
}

impl GpuKernel {
    pub unsafe fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let res = (self.driver.cu_launch_kernel)(
            self.func,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem_bytes,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if res != CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel failed with code: {}", res));
        }
        Ok(())
    }
}
