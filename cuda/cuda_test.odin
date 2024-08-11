package cuda

import "core:testing"
import "core:log"
import "core:fmt"
import "core:mem"
import "core:slice"
import "../util"

@(test)
device_test :: proc(t: ^testing.T) {
	major, minor := driver_version()
	log.debugf("Cuda version: %d.%d", major, minor)
	devices := num_devices()
	log.debugf("Got %d devices", devices)
	testing.expect(t, devices > 0)
	dev := Device(0)
	name := device_name(dev)
	defer delete(name)
	bytes := device_total_mem(dev)
	log.debugf("%s %.0f MB", name, f64(bytes) / mem.Megabyte)
	comp_major := device_attribute(dev, .COMPUTE_CAPABILITY_MAJOR)
	comp_minor := device_attribute(dev, .COMPUTE_CAPABILITY_MINOR)
	log.debugf("compute capability = %d.%d", comp_major, comp_minor)
	procs := device_attribute(dev, .MULTIPROCESSOR_COUNT)
	threads := device_attribute(dev, .MAX_THREADS_PER_MULTIPROCESSOR)
	log.debugf("max threads = %d x %d", procs, threads)
}

@(test)
memset_test :: proc(t: ^testing.T) {
	ctx := create_context()
	defer destroy_context(ctx)
	SIZE :: 100
	BSIZE :: SIZE * size_of(f32)
	dptr := memalloc(BSIZE)
	defer memfree(dptr)
	val := transmute(u32)(f32(42))
	memsetD32(dptr, val, SIZE)
	out := make([]f32, SIZE)
	defer delete(out)
	memcpyDtoH(raw_data(out), dptr, BSIZE)
	log.debug(out)
	testing.expect(t, slice.equal(out, []f32{0 ..< SIZE = 42}))
}

compile_test_func :: proc(t: ^testing.T, verbose := false) -> string {
	src := `extern "C" __global__
void axpy(float alpha, float *x, float *y, size_t n) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] += alpha * x[i];
	}
}`
	ptx, err1 := compile_to_ptx(src, "axpy.cu", "--gpu-architecture=compute_80")
	if err1 != .SUCCESS {
		fmt.print(ptx)
		testing.fail_now(t, nvrtc_error(err1))
	}
	if verbose {
		log.debug(ptx)
	}
	return ptx
}

@(test)
compile_test :: proc(t: ^testing.T) {
	ptx := compile_test_func(t, verbose=true)
	defer delete(ptx)
}

@(test)
launch_test :: proc(t: ^testing.T) {
	ptx := compile_test_func(t)
	defer delete(ptx)

	ctx := create_context(ptx=ptx)
	defer destroy_context(ctx)
	context.user_ptr = ctx

	fn := get_function("axpy")
	BLOCK :: 128
	size := 1000
	bsize := size * size_of(f32)

	x := memalloc(bsize)
	defer memfree(x)
	y := memalloc(bsize)
	defer memfree(y)

	xa := make([]f32, size)
	defer delete(xa)
	for i in 0 ..< size {
		xa[i] = f32(i+1)
	}
	memcpyHtoD(x, raw_data(xa), bsize)

	y_val := f32(4)
	memsetD32(y, transmute(u32)(y_val), size)

	alpha := f32(0.01)
	launch_kernel(fn, blockX=util.ceil_div(size, BLOCK), gridX=BLOCK, params={&alpha, &x, &y, &size})
	synchronize()

	res := make([]f32, size)
	defer delete(res)
	memcpyDtoH(raw_data(res), y, bsize)
	log.debugf("out: %.2f .. %.2f", res[:5], res[size-5:])

	for v, i in res {
		exp := alpha*f32(i+1) + y_val
		testing.expect(t, abs(v-exp) < 1e-6)
	}
}
