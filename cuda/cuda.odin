package cuda

import "core:strings"
import "core:log"

foreign import cuda "system:cuda"

Device :: distinct i32
Context :: distinct rawptr
Stream :: distinct rawptr
Module :: distinct rawptr
Function :: distinct rawptr
MemoryPool :: distinct rawptr

UserContext :: struct {
	ctx: Context,
	mod: Module,
	handles: map[string]Handle,
	cache: map[string]Handle,
}

Handle :: struct {
	ptr: rawptr,
	destroy: proc(string, rawptr),
}

@(default_calling_convention="c", private)
foreign cuda {
	cuInit :: proc(uint) -> Result ---
	cuGetErrorName :: proc(error: Result, pstr: ^cstring) --- 
	cuDriverGetVersion :: proc(version: ^i32) -> Result ---
	cuDeviceGetCount :: proc(count: ^i32) -> Result ---
	cuDeviceGetAttribute :: proc(pi: ^i32, attrib: Device_Attribute, dev: Device) -> Result ---
	cuDeviceGetName :: proc(name: cstring, len: i32, dev: Device) -> Result ---
	cuDeviceTotalMem_v2 :: proc(bytes: ^uint, dev: Device) -> Result ---
	cuDeviceGetMemPool :: proc(pool: ^MemoryPool, dev: Device) -> Result ---
	cuMemPoolGetAttribute :: proc(pool: MemoryPool, attr: MemPool_Attribute, value: rawptr) -> Result ---
	cuMemPoolSetAttribute :: proc(pool: MemoryPool, attr: MemPool_Attribute, value: rawptr) -> Result ---
	cuCtxCreate_v2 :: proc(ctx: ^Context, flags: u32, dev: Device) -> Result ---
	cuCtxDestroy :: proc(ctx: Context) -> Result ---
	cuCtxSynchronize :: proc() -> Result ---
	cuCtxGetDevice :: proc(dev: ^Device) -> Result --- 
	cuMemAllocAsync :: proc(dptr: ^rawptr, bytes: uint, stream: Stream) -> Result ---
	cuMemFreeAsync :: proc(dptr: rawptr, stream: Stream) -> Result ---
	cuMemcpyDtoH_v2 :: proc(dst, src: rawptr, bytes: uint) -> Result ---
	cuMemcpyHtoD_v2 :: proc(dst, src: rawptr, bytes: uint) -> Result ---
	cuMemcpyDtoD_v2 :: proc(dst, src: rawptr, bytes: uint) -> Result ---
	cuMemsetD8_v2 :: proc(dst: rawptr, val: u8, n: uint) -> Result ---
	cuMemsetD16_v2 :: proc(dst: rawptr, val: u16, n: uint) -> Result ---
	cuMemsetD32_v2 :: proc(dst: rawptr, val: u32, n: uint) -> Result ---
 	cuModuleLoadData :: proc(mod: ^Module, image: rawptr) -> Result ---
 	cuModuleGetFunction :: proc(func: ^Function, mod: Module, name: cstring) -> Result ---
 	cuLaunchKernel :: proc(func: Function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes: u32,
 						   stream: Stream, kernelParams, extra: rawptr) -> Result ---
}

@(init)
init_cuda :: proc() {
	must(cuInit(0))
}

// Initialise cuda context and load ptx module - will panic on error.
create_context :: proc(device := 0, ptx := "") -> ^UserContext {
	if device >= num_devices() {
		panic("cuda device not available")
	}
	log.debug("create cuda context on device", device)
	c := new(UserContext)
	must(cuCtxCreate_v2(&c.ctx, 0, Device(device)))
	err: Result
	if ptx != "" {
		if c.mod, err = load_module(ptx); err != .SUCCESS {
			panic(error(err))
		}
	}
	c.handles = make(map[string]Handle)
	c.cache = make(map[string]Handle)
	return c
}

// Register a new handle in user context - e.g. for cublas or cudnn
register_handle :: proc(name: string, ptr: rawptr, destroy: proc(string, rawptr), ctx: ^UserContext = nil,
						loc := #caller_location) {
	ctx := ctx
	if ctx == nil {
		assert (context.user_ptr != nil, "cuda user context not set", loc)
		ctx = cast(^UserContext)context.user_ptr
	}
	ctx.handles[name] = {ptr = ptr, destroy = destroy}
}

// Get handle from user context
get_handle :: proc(name: string, ctx: ^UserContext = nil, loc := #caller_location) -> rawptr {
	ctx := ctx
	if ctx == nil {
		assert (context.user_ptr != nil, "cuda user context not set", loc)
		ctx = cast(^UserContext)context.user_ptr
	}
	h := ctx.handles[name] or_else panic("handle not registered", loc)
	return h.ptr
}


// Destroy current context and free resources
destroy_context :: proc(c: ^UserContext) {
	for key, h in c.handles {
		h.destroy(key, h.ptr)
	}
	for key, h in c.cache {
		h.destroy(key, h.ptr)
	}
	must(cuCtxDestroy(c.ctx))
	delete(c.handles)
	delete(c.cache)
	free(c)
}

driver_error :: proc(rc: Result) -> string {
	err: cstring
	cuGetErrorName(rc, &err)
	return string(err)
}

error :: proc{driver_error, nvrtc_error}

driver_must :: proc(rc: Result, loc := #caller_location) {
	if rc != .SUCCESS {
		panic(driver_error(rc), loc)
	}
}

must :: proc{driver_must, nvrtc_must}

driver_version :: proc() -> (major, minor: int) {
	version: i32
	must(cuDriverGetVersion(&version))
	return int(version/1000), int((version%1000)/10)
}

num_devices :: proc() -> int {
	count: i32
	must(cuDeviceGetCount(&count))
	return int(count)
}

device_attribute :: proc(dev: Device, attr: Device_Attribute) -> int {
	info: i32
	must(cuDeviceGetAttribute(&info, attr, dev))
	return int(info)
}

get_mempool_attribute :: proc(dev: Device, attr: MemPool_Attribute) -> int {
	pool: MemoryPool
	must(cuDeviceGetMemPool(&pool, dev))
	value: u64
	must(cuMemPoolGetAttribute(pool, attr, &value))
	return int(value)
}

set_mempool_attribute :: proc(dev: Device, attr: MemPool_Attribute, value: int) {
	pool: MemoryPool
	must(cuDeviceGetMemPool(&pool, dev))
	value := u64(value)
	must(cuMemPoolSetAttribute(pool, attr, &value))
}

device_name :: proc(dev: Device) -> string {
	MAX_BUF :: 128
	cstr := make_cstring(MAX_BUF)
	must(cuDeviceGetName(cstr, MAX_BUF, dev))
	return string(cstr)
}

device_total_mem :: proc(dev: Device) -> int {
	bytes: uint
	must(cuDeviceTotalMem_v2(&bytes, dev))
	return int(bytes)
}

get_device :: proc() -> Device {
	dev: Device
	must(cuCtxGetDevice(&dev))
	return dev
}

synchronize :: proc() {
	must(cuCtxSynchronize())
}

memalloc :: proc(bytes: int) -> rawptr {
	dptr: rawptr
	must(cuMemAllocAsync(&dptr, uint(bytes), nil))
	return dptr
}

memfree :: proc(dptr: rawptr) {
	must(cuMemFreeAsync(dptr, nil))
}

memcpyDtoH :: proc(dst, src: rawptr, bytes: int) {
	must(cuMemcpyDtoH_v2(dst, src, uint(bytes)))
}

memcpyHtoD :: proc(dst, src: rawptr, bytes: int) {
	must(cuMemcpyHtoD_v2(dst, src, uint(bytes)))
}

memcpyDtoD :: proc(dst, src: rawptr, bytes: int) {
	must(cuMemcpyDtoD_v2(dst, src, uint(bytes)))
}

memsetD8 :: proc(dptr: rawptr, val: u8, n: int) {
	must(cuMemsetD8_v2(dptr, val, uint(n)))
}

memsetD16 :: proc(dptr: rawptr, val: u16, n: int) {
	must(cuMemsetD16_v2(dptr, val, uint(n)))
}

memsetD32 :: proc(dptr: rawptr, val: u32, n: int) {
	must(cuMemsetD32_v2(dptr, val, uint(n)))
}

load_module :: proc(ptx: string) -> (Module, Result) {
	cptx := strings.clone_to_cstring(ptx)
	defer delete(cptx)
	mod: Module
	rc := cuModuleLoadData(&mod, cast([^]u8)cptx)
	return mod, rc
}

get_function :: proc(name: cstring, loc := #caller_location) -> Function {
	assert (context.user_ptr != nil, "cuda user context not set", loc)
	ctx := cast(^UserContext)context.user_ptr
	func: Function
	must(cuModuleGetFunction(&func, ctx.mod, name), loc)
	return func
}

launch_kernel :: proc(func: Function, gridX := 1, gridY := 1, gridZ := 1, blockX := 1, blockY := 1, blockZ := 1, 
						sharedMemBytes := 0, params: []rawptr = nil, loc := #caller_location) {
	must(cuLaunchKernel(
		func, u32(gridX), u32(gridY), u32(gridZ), u32(blockX), u32(blockY), u32(blockZ), 
		u32(sharedMemBytes), nil, raw_data(params), nil), loc)
}

make_cstring :: proc(size: uint) -> cstring {
	buf := make([]u8, size)
	return cstring(raw_data(buf))
}
