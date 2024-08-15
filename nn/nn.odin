package nn

import "base:runtime"
import "core:fmt"
import "core:io"
import "core:log"
import "core:math/rand"
import "core:text/table"

import "../array"
import "../cublas"
import "../cuda"
import "../cudnn"
import "../util"

cuda_ptx: string

Array :: array.Array
Shape :: array.Shape
BF16 :: array.BF16
CPU :: array.CPU
Cuda :: array.Cuda

// Compile cuda kernels to ptx - will panic on compile error.
@(init)
compile_kernels :: proc() {
	src := #load("kernels.cu", string)
	ptx, err := cuda.compile_to_ptx(src, "kernels.cu", "-arch=compute_80", "-use_fast_math", "-I/usr/local/cuda/include")
	if err != nil {
		fmt.println(ptx)
		panic(cuda.error(err))
	}
	cuda_ptx = ptx
}

// Initialize cuda context
init_cuda :: proc(device := 0, verbose := false) -> ^cuda.UserContext {
	ctx := cuda.create_context(device, ptx = cuda_ptx)
	cublas.register_handle(ctx)
	cudnn.register_handle(ctx)
	if verbose {
		major, minor := cuda.driver_version()
		dev := cuda.Device(device)
		device_name := cuda.device_name(dev)
		defer delete(device_name)
		total_mem := cuda.device_total_mem(dev)
		log.infof("Cuda %d.%d %s %.0f MB", major, minor, device_name, f64(total_mem) / (1024 * 1024))
	}
	return ctx
}

// Destroy context and free resources.
end_cuda :: proc() {
	ctx := cast(^cuda.UserContext)context.user_ptr
	cuda.destroy_context(ctx)
}

// uniform weight initialization
uniform_init :: proc(a: Array($D, $T), low: f32 = 0, high: f32 = 1) {
	tmp := make([]f32, a.size)
	defer delete(tmp)
	for i in 0 ..< a.size {
		tmp[i] = rand.float32_range(low, high)
	}
	array.copy(a, tmp)
}

// normal weight initialization
normal_init :: proc(a: Array($D, $T), mean: f32 = 0, stddev: f32 = 1) {
	tmp := make([]f32, a.size)
	defer delete(tmp)
	for i in 0 ..< a.size {
		tmp[i] = rand.float32_normal(mean, stddev)
	}
	array.copy(a, tmp)
}

// convert from bfloat16 to float32 on GPU
to_float32 :: proc(inp: Array(Cuda, BF16), out: Array(Cuda, f32), loc := #caller_location) {
	assert(inp.size == out.size, "size mismatch", loc)
	assert(inp.size % 2 == 0, "size must be aligned to 2", loc)
	fn := cuda.get_function("bf16_to_f32")
	BLOCK :: 512
	xp, yp, size := array.ptr(inp), array.ptr(out), inp.size
	cuda.launch_kernel(fn, gridX = util.ceil_div(size / 2, BLOCK), blockX = BLOCK, params = {&yp, &xp, &size})
}

// Calculate out = x + y
add :: proc {
	add_cpu,
	add_cuda,
}

add_cpu :: proc(x, y, out: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	assert(x.size == out.size && y.size == out.size, "size mismatch", loc)
	xp, yp, outp := array.ptr(x), array.ptr(y), array.ptr(out)
	for i in 0 ..< y.size {
		outp[i] = xp[i] + yp[i]
	}
}

add_cuda :: proc(x, y, out: Array(Cuda, $T), loc := #caller_location) where T == f32 || T == BF16 {
	assert(x.size == out.size && y.size == out.size, "size mismatch", loc)
	when T == f32 {
		fn := cuda.get_function("add_f32")
		grid := util.ceil_div(x.size, BLOCK)
	} else {
		assert(x.size % 2 == 0, "size must be aligned to 2", loc)
		fn := cuda.get_function("add_bf16")
		grid := util.ceil_div(x.size / 2, BLOCK)
	}
	BLOCK :: 512
	xp, yp, outp, size := array.ptr(x), array.ptr(y), array.ptr(out), y.size
	cuda.launch_kernel(fn, gridX = grid, blockX = BLOCK, params = {&outp, &xp, &yp, &size})
}

// Write a nicely formatted summary of the model. If expanded is set then shows all layers, else just the top level
write_summary :: proc(w: io.Writer, model: ^Layer($D, $T), expanded := false) {

	align_right :: proc(t: ^table.Table, column: int) {
		table.set_cell_alignment(t, table.last_row(t), column, .Right)
	}

	format_row :: proc(t: ^table.Table, l: ^Layer($D, $T), indent := "") {
		num := util.comma_format(l.num_params)
		defer delete(num)
		table.row(t, table.format(t, "%s%s (%s)", indent, l.info, l.type), table.format(t, "%v", l.out_shape), table.format(t, "%s", num))
		align_right(t, 2)
	}

	t := table.init(&table.Table{})
	defer table.destroy(t)

	table.padding(t, 2, 2)
	device, dtype: ^runtime.Type_Info
	device, dtype = type_info_of(D), type_info_of(T)
	table.caption(t, table.format(t, "== %s - device: %s, data type: %s ==", model.name, device, dtype))
	table.header(t, "Layer", "Output shape", "Parameters")

	for &layer in model.layers {
		format_row(t, layer)
		if expanded {
			for &l in layer.layers {
				format_row(t, l, "  ")
			}
		}
	}
	table.row(t, "", "", "──────")
	align_right(t, 2)
	num := util.comma_format(model.num_params)
	defer delete(num)
	table.row(t, "", "", num)
	align_right(t, 2)

	decorations := table.Decorations{"┌", "┬", "┐", "├", "┼", "┤", "└", "┴", "┘", "│", "─"}
	table.write_decorated_table(w, t, decorations)
}
