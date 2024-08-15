package nn

import "core:fmt"
import "core:log"
import "core:math"
import "core:strings"

import "../array"
import "../cublas"
import "../cuda"
import "../cudnn"
import "../openblas"
import "../util"

USE_CUBLAS_DBIAS :: true

// Parameter has array and optional associated gradient. If type is BF16 then also store copy in f32 format.
Parameter :: struct($D, $T: typeid) where T == f32 || T == BF16 {
	using arr: Array(D, T),
	grad:      Array(D, T),
}

delete_parameter :: proc(p: Parameter($D, $T)) {
	array.delete(p.arr)
	array.delete(p.grad)
}

// Layer state info
Layer :: struct($D, $T: typeid) {
	name:       string,
	type:       string,
	info:       string,
	params:     []Parameter(D, T),
	num_params: int,
	out_shape:  array.Shape,
	layers:     []^Layer(D, T),
}

// Set batch size and sequence length
build :: proc(layer: ^Layer($D, $T), dims: ..int) {
	copy(layer.out_shape.dims[:], dims)
	for &l in layer.layers {
		build(l, ..dims)
	}
}

// Allocate gradient arrays if nil
alloc_grads :: proc(layer: ^Layer($D, $T), loc := #caller_location) {
	for &p in layer.params {
		if array.is_nil(p.grad) {
			p.grad = array.zeros_like(p.arr)
		}
	}
}

delete_layer_base :: proc(l: Layer($D, $T)) {
	if l.params != nil {
		for p in l.params {
			delete_parameter(p)
		}
		delete(l.params)
	}
	if l.layers != nil {
		delete(l.layers)
	}
	delete(l.info)
	delete(l.name)
}

delete_layer :: proc {
	delete_linear,
	delete_encoder,
	delete_layernorm,
	delete_attention,
}

// Zero parameter gradients. Allocate them if not already done.
zero_gradients :: proc(layer: ^Layer($D, $T)) {
	for &p in layer.params {
		if array.is_nil(p.grad) {
			p.grad = array.zeros_like(p.arr)
		} else {
			array.zero(p.grad)
		}
	}
	for l in layer.layers {
		zero_gradients(l)
	}
}

// Linear matrix multiply layer with optional bias
Linear :: struct($D, $T: typeid) {
	using layer: Layer(D, T),
	weight:      ^Parameter(D, T), // -> .params[0]
	bias:        ^Parameter(D, T), // -> .params[1]  (may be nil)
}

make_linear :: proc($D, $T: typeid, name: string, in_channels, out_channels: int, bias := true, no_init := false) -> (l: Linear(D, T)) {
	if !no_init {
		l.params = make([]Parameter(D, T), bias ? 2 : 1)
		l.params[0].arr = array.zeros(D, T, {out_channels, in_channels})
		l.weight = &l.params[0]
		l.num_params = in_channels * out_channels
		if bias {
			l.params[1].arr = array.zeros(D, T, {out_channels})
			l.bias = &l.params[1]
			l.num_params += out_channels
		}
	}
	l.type = "Linear"
	l.name = strings.clone(name)
	l.info = fmt.aprintf("%s{{ nin:%d nout:%d%s }}", name, in_channels, out_channels, l.bias != nil ? " bias" : "")
	l.out_shape = array.make_shape({0, 0, out_channels})
	return l
}

delete_linear :: proc(l: Linear($D, $T)) {
	delete_layer_base(l.layer)
}


linear_forward :: proc {
	linear_forward_cpu,
	linear_forward_cuda,
}

linear_forward_cpu :: proc(l: ^Linear(CPU, f32), inp, out: Array(CPU, f32), loc := #caller_location) {
	in_shape, out_shape := flatten(inp.shape), flatten(out.shape)
	linear_fwd_checkargs(l, in_shape, out_shape, loc = loc)
	// out = matmul(in, weight.T)  [BT, C] x [C, OC] => [BT, OC]
	m, n, k := i32(out_shape.dims[0]), i32(out_shape.dims[1]), i32(in_shape.dims[1])
	openblas.sgemm(.RowMajor, .NoTrans, .Trans, m, n, k, 1, array.ptr(inp), k, array.ptr(l.weight.arr), k, 0, array.ptr(out), n)
	if l.bias != nil {
		// add bias
		biasp := array.ptr(l.bias.arr)
		for i in 0 ..< int(m) {
			openblas.saxpy(n, 1, biasp, 1, array.ptr(out, offset = i * int(n)), 1)
		}
	}
}

linear_forward_cuda :: proc(l: ^Linear(Cuda, $T), inp, out: Array(Cuda, T), loc := #caller_location) where T == f32 || T == BF16 {
	in_shape, out_shape := flatten(inp.shape), flatten(out.shape)
	linear_fwd_checkargs(l, in_shape, out_shape, loc = loc)
	dtype: cublas.DataType = .R_16BF when T == BF16 else .R_32F
	// optional bias
	bias: [^]T
	if l.bias != nil {
		bias = array.ptr(l.bias.arr)
	}
	// out = matmul(in, weight.T)  [BT, C] x [C, OC] => [BT, OC]
	m, n, k := out_shape.dims[0], out_shape.dims[1], in_shape.dims[1]
	err := cublas.gemm(dtype, .OP_N, .OP_T, array.ptr(inp), array.ptr(l.weight.arr), array.ptr(out), m, n, k, bias = bias)
	if err != nil {
		log.panic("no cublas matmul algoritm found:", err, location = loc)
	}
}

linear_fwd_checkargs :: proc(l: ^Linear($D, $T), inp, out: Shape, loc := #caller_location) {
	assert(
		inp.dims[1] == l.weight.dims[1] && out.dims[1] == l.weight.dims[0] && inp.dims[0] == out.dims[0] && (l.bias == nil || out.dims[1] == l.bias.size),
		"invalid input shapes",
		loc,
	)
}

linear_backward :: proc {
	linear_backward_cpu,
	linear_backward_cuda,
}

linear_backward_cpu :: proc(l: ^Linear(CPU, f32), inp, dout, din: Array(CPU, f32), loc := #caller_location) {
	in_shape, dout_shape, din_shape := flatten(inp.shape), flatten(dout.shape), flatten(din.shape)
	linear_bwd_checkargs(l, in_shape, dout_shape, din_shape, loc = loc)
	BT, C, OC := i32(in_shape.dims[0]), i32(l.weight.dims[1]), i32(l.weight.dims[0])
	alloc_grads(&l.layer)
	// din = matmul(dout, weight)    [BT, OC] x [OC, C] => [BT, C]
	openblas.sgemm(.RowMajor, .NoTrans, .NoTrans, BT, C, OC, 1, array.ptr(dout), OC, array.ptr(l.weight.arr), C, 0, array.ptr(din), C)
	// dweight += matmul(dout.T, in)  [OC, BT] x [BT, C] => [OC, C]
	openblas.sgemm(.RowMajor, .Trans, .NoTrans, OC, C, BT, 1, array.ptr(dout), OC, array.ptr(inp), C, 1, array.ptr(l.weight.grad), C)
	// dbias += sum(dout, axis=0)
	if l.bias != nil {
		ones := make([]f32, OC)
		defer delete(ones)
		for i in 0 ..< OC {
			ones[i] = 1
		}
		openblas.sgemv(.RowMajor, .Trans, BT, OC, 1, array.ptr(dout), OC, raw_data(ones), 1, 1, array.ptr(l.bias.grad), 1)
	}
}

linear_backward_cuda :: proc(l: ^Linear(Cuda, $T), inp, dout, din: Array(Cuda, T), loc := #caller_location) where T == f32 || T == BF16 {
	in_shape, dout_shape, din_shape := flatten(inp.shape), flatten(dout.shape), flatten(din.shape)
	linear_bwd_checkargs(l, in_shape, dout_shape, din_shape, loc = loc)
	dtype: cublas.DataType = .R_16BF when T == BF16 else .R_32F
	BT, C, OC := in_shape.dims[0], l.weight.dims[1], l.weight.dims[0]
	alloc_grads(&l.layer)
	// din = matmul(dout, weight)    [BT, OC] x [OC, C] => [BT, C]
	err := cublas.gemm(dtype, .OP_N, .OP_N, array.ptr(dout), array.ptr(l.weight.arr), array.ptr(din), BT, C, OC)
	if err != nil {
		log.panic("no cublas matmul algoritm found:", err, location = loc)
	}
	when USE_CUBLAS_DBIAS {
		// optinal bias gradient
		dbias: Array(Cuda, T)
		have_bias := false
		if l.bias != nil {
			// fuse bias reduction with weight grad update
			have_bias = true
			dbias = array.zeros_like(l.bias.grad)
		}
		// dweight += matmul(dout.T, in)  [OC, BT] x [BT, C] => [OC, C]
		err = cublas.gemm(
			dtype,
			.OP_T,
			.OP_N,
			array.ptr(dout),
			array.ptr(inp),
			array.ptr(l.weight.grad),
			OC,
			C,
			BT,
			beta = 1,
			bias = array.ptr(dbias),
			dbias = have_bias,
		)
		if err != nil {
			log.panic("no cublas matmul algoritm found:", err, location = loc)
		}
		if have_bias {
			add(l.bias.grad, dbias, l.bias.grad)
			defer array.delete(dbias)
		}
	} else {
		// dweight += matmul(dout.T, in)  [OC, BT] x [BT, C] => [OC, C]
		err = cublas.gemm(dtype, .OP_T, .OP_N, array.ptr(dout), array.ptr(inp), array.ptr(l.weight.grad), OC, C, BT, beta = 1)
		if err != nil {
			log.panic("no cublas matmul algoritm found:", err, location = loc)
		}
		if l.bias != nil {
			// separate bias reduction
			assert(OC % 32 == 0, "output channels must be divisible by 32 for backward bias kernel", loc)
			when T == BF16 {
				fn := cuda.get_function("matmul_backward_bias_bf16")
			} else {
				fn := cuda.get_function("matmul_backward_bias_f32")
			}
			dbias, doutput := array.ptr(l.bias.grad), array.ptr(dout)
			bt, oc := i32(BT), i32(OC)
			BLOCK := 512
			cuda.launch_kernel(fn, gridX = OC / 32, blockX = BLOCK, sharedMemBytes = BLOCK * 4, params = {&dbias, &doutput, &bt, &oc})
		}
	}
}

linear_bwd_checkargs :: proc(l: ^Linear($D, $T), inp, dout, din: Shape, loc := #caller_location) {
	assert(
		inp == din &&
		din.dims[1] == l.weight.dims[1] &&
		dout.dims[1] == l.weight.dims[0] &&
		din.dims[0] == dout.dims[0] &&
		(l.bias == nil || dout.dims[1] == l.bias.size),
		"invalid input shapes",
		loc,
	)
}

flatten :: proc(a: Shape, loc := #caller_location) -> (s: Shape) {
	a := a
	assert(a.ndims >= 2, "invalid matrix shape", loc = loc)
	s.ndims = 2
	s.dims[0] = math.prod(a.dims[:a.ndims - 1])
	s.dims[1] = a.dims[a.ndims - 1]
	s.size = s.dims[0] * s.dims[1]
	return s
}

// Token and position embedding layer
Encoder :: struct($D, $T: typeid) {
	using layer: Layer(D, T),
	wte:         ^Parameter(D, T), // -> .params[0]
	wpe:         ^Parameter(D, T), // -> .params[1]
}

make_encoder :: proc($D, $T: typeid, name: string, vocab_size, max_seq_len, channels: int) -> (l: Encoder(D, T)) {
	l.params = make([]Parameter(D, T), 2)
	l.params[0].arr = array.zeros(D, T, {vocab_size, channels})
	l.params[1].arr = array.zeros(D, T, {max_seq_len, channels})
	l.wte = &l.params[0]
	l.wpe = &l.params[1]
	l.type = "Encoder"
	l.name = strings.clone(name)
	l.info = fmt.aprintf("%s{{ wte:%v wpe:%v }}", name, array.shape(&l.wte.shape), array.shape(&l.wpe.shape))
	l.out_shape = array.make_shape({0, 0, channels})
	l.num_params = vocab_size * channels + max_seq_len * channels
	return l
}

delete_encoder :: proc(l: Encoder($D, $T)) {
	delete_layer_base(l.layer)
}

encoder_forward :: proc {
	encoder_forward_cpu,
	encoder_forward_cuda,
}

encoder_forward_cpu :: proc(l: ^Encoder(CPU, f32), inp: Array(CPU, i32), out: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	encoder_checkargs(l.wpe.shape, inp.shape, out.shape, loc = loc)
	B, T, C := out.dims[0], out.dims[1], out.dims[2]
	input, output := array.ptr(inp), array.ptr(out)
	wte, wpe := array.ptr(l.wte.arr), array.ptr(l.wpe.arr)
	for bt in 0 ..< B * T {
		outV := output[bt * C:]
		ix := int(input[bt])
		t := bt % T
		for c in 0 ..< C {
			outV[c] = wte[ix * C + c] + wpe[t * C + c]
		}
	}
}

encoder_forward_cuda :: proc(l: ^Encoder(Cuda, BF16), inp: Array(Cuda, i32), out: Array(Cuda, BF16), loc := #caller_location) {
	encoder_checkargs(l.wpe.shape, inp.shape, out.shape, loc = loc)
	B, T, C := i32(out.dims[0]), i32(out.dims[1]), i32(out.dims[2])
	assert(C % 2 == 0, "num channels must be aligned to 2")
	fn := cuda.get_function("encoder_forward_bf16")
	BLOCK :: 512
	input, output := array.ptr(inp), array.ptr(out)
	wte, wpe := array.ptr(l.wte.arr), array.ptr(l.wpe.arr)
	cuda.launch_kernel(fn, gridX = util.ceil_div(out.size / 2, BLOCK), blockX = BLOCK, params = {&output, &input, &wte, &wpe, &B, &T, &C})
}

// Backprop weights - don't generate input gradient as this is first layer
encoder_backward :: proc {
	encoder_backward_cpu,
	encoder_backward_cuda,
}

encoder_backward_cpu :: proc(l: ^Encoder(CPU, f32), inp: Array(CPU, i32), dout: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	encoder_checkargs(l.wpe.shape, inp.shape, dout.shape, loc = loc)
	alloc_grads(&l.layer)
	B, T, C := dout.dims[0], dout.dims[1], dout.dims[2]
	input, doutput := array.ptr(inp), array.ptr(dout)
	dwte, dwpe := array.ptr(l.wte.grad), array.ptr(l.wpe.grad)
	for bt in 0 ..< B * T {
		ix_t := int(input[bt]) * C
		ix_p := (bt % T) * C
		for c in 0 ..< C {
			d := doutput[bt * C + c]
			dwte[ix_t + c] += d
			dwpe[ix_p + c] += d
		}
	}
}

encoder_backward_cuda :: proc(l: ^Encoder(Cuda, BF16), inp: Array(Cuda, i32), dout: Array(Cuda, BF16), loc := #caller_location) {
	encoder_checkargs(l.wpe.shape, inp.shape, dout.shape, loc = loc)
	alloc_grads(&l.layer)
	B, T, C := i32(dout.dims[0]), i32(dout.dims[1]), i32(dout.dims[2])
	fn := cuda.get_function("encoder_backward_bf16")
	BLOCK :: 256
	input, doutput := array.ptr(inp), array.ptr(dout)
	dwte, dwpe := array.ptr(l.wte.grad), array.ptr(l.wpe.grad)
	cuda.launch_kernel(fn, gridX = util.ceil_div(dout.size, BLOCK), blockX = BLOCK, params = {&dwte, &dwpe, &doutput, &input, &B, &T, &C})
}

encoder_checkargs :: proc(wpe, inp, out: Shape, loc := #caller_location) {
	assert(
		inp.ndims == 2 && out.ndims == 3 && out.dims[0] == inp.dims[0] && out.dims[1] <= wpe.dims[0] && out.dims[2] == wpe.dims[1],
		"invalid input shapes",
		loc,
	)
}

// Layer normalization by channel
Layernorm :: struct($D, $T: typeid) {
	using layer: Layer(D, T),
	scale:       ^Parameter(D, T), // -> .params[0]
	bias:        ^Parameter(D, T), // -> .params[1]
	mean:        Array(D, T),
	rstd:        Array(D, T),
	eps:         f32,
}

make_layernorm :: proc($D, $T: typeid, name: string, channels: int, epsilon: f32 = 1e-5) -> (l: Layernorm(D, T)) {
	l.params = make([]Parameter(D, T), 2)
	l.params[0].arr = array.zeros(D, T, {channels})
	l.params[1].arr = array.zeros(D, T, {channels})
	l.scale = &l.params[0]
	l.bias = &l.params[1]
	l.eps = epsilon
	l.type = "Layernorm"
	l.name = strings.clone(name)
	l.info = fmt.aprintf("%s{{ channels:%d }}", name, channels)
	l.out_shape = array.make_shape({0, 0, channels})
	l.num_params = 2 * channels
	return l
}

delete_layernorm :: proc(l: Layernorm($D, $T)) {
	delete_layer_base(l.layer)
	array.delete(l.mean)
	array.delete(l.rstd)
}

layernorm_forward :: proc {
	layernorm_forward_cpu,
	layernorm_forward_cuda,
}

layernorm_forward_cpu :: proc(l: ^Layernorm(CPU, f32), inp, out: Array(CPU, f32), train := true, loc := #caller_location) #no_bounds_check {
	layernorm_fwd_checkargs(l, inp.shape, out.shape, loc = loc)
	C := out.dims[out.ndims - 1]
	N := out.size / C
	if train && l.mean.size != N {
		layernorm_init(l, N)
	}
	input, output := array.ptr(inp), array.ptr(out)
	scale, bias := array.ptr(l.scale.arr), array.ptr(l.bias.arr)
	means, rstds := array.ptr(l.mean), array.ptr(l.rstd)
	for n in 0 ..< N {
		x := input[n * C:]
		mean: f32
		for i in 0 ..< C {
			mean += x[i]
		}
		mean /= f32(C)
		variance: f32
		for i in 0 ..< C {
			variance += (x[i] - mean) * (x[i] - mean)
		}
		variance /= f32(C)
		rstd := 1.0 / math.sqrt(variance + l.eps)
		for i in 0 ..< C {
			output[n * C + i] = rstd * (x[i] - mean) * scale[i] + bias[i]
		}
		if train {
			means[n] = mean
			rstds[n] = rstd
		}
	}
}

layernorm_init :: proc(l: ^Layernorm($D, $T), N: int) {
	array.delete(l.mean)
	array.delete(l.rstd)
	l.mean = array.zeros(D, T, {N})
	l.rstd = array.zeros(D, T, {N})
}

layernorm_forward_cuda :: proc(l: ^Layernorm(Cuda, BF16), inp, out: Array(Cuda, BF16), train := true, loc := #caller_location) {
	layernorm_fwd_checkargs(l, inp.shape, out.shape, loc = loc)
	C := out.dims[out.ndims - 1]
	N := out.size / C
	if l.mean.size != N {
		layernorm_init(l, N)
	}
	id := fmt.aprintf("norm_fwd_%d_%d", N, C)
	defer delete(id)
	plan, ok := cudnn.plan_from_cache(id)
	if !ok {
		plan = layernorm_fwd_init_cuda(N, C, loc = loc)
		cudnn.cache_plan(id, plan)
	}
	x, y := array.ptr(inp), array.ptr(out)
	scale, bias := array.ptr(l.scale.arr), array.ptr(l.bias.arr)
	mean, rstd := array.ptr(l.mean), array.ptr(l.rstd)
	cudnn.execute(plan, {'x', 'y', 's', 'b', 'm', 'v', 'e'}, {x, y, scale, bias, mean, rstd, &l.eps})
}

layernorm_fwd_init_cuda :: proc(N, C: int, loc := #caller_location) -> cudnn.BackendDescriptor {
	log.debugf("init layernorm fwd: N=%d C=%d", N, C, location = loc)
	x := cudnn.tensor_descriptor('x', {N, C, 1, 1}, {C, 1, 1, 1}, dtype = .BFLOAT16)
	y := cudnn.tensor_descriptor('y', {N, C, 1, 1}, {C, 1, 1, 1}, dtype = .BFLOAT16)
	scale := cudnn.tensor_descriptor('s', {1, C, 1, 1}, dtype = .BFLOAT16)
	bias := cudnn.tensor_descriptor('b', {1, C, 1, 1}, dtype = .BFLOAT16)
	mean := cudnn.tensor_descriptor('m', {N, 1, 1, 1}, dtype = .BFLOAT16)
	rvar := cudnn.tensor_descriptor('v', {N, 1, 1, 1}, dtype = .BFLOAT16)
	eps := cudnn.tensor_descriptor('e', {1, 1, 1, 1}, by_value = true)
	defer cudnn.destroy(x, y, scale, bias, mean, rvar, eps)

	mode := cudnn.BackendNormMode.LAYER_NORM
	phase := cudnn.BackendNormFwdPhase.TRAINING
	d := cudnn.make_descriptor(.OPERATION_NORM_FORWARD_DESCRIPTOR)
	defer cudnn.destroy(d)
	cudnn.set(d, .OPERATION_NORM_FWD_MODE, .NORM_MODE, &mode)
	cudnn.set(d, .OPERATION_NORM_FWD_PHASE, .NORM_FWD_PHASE, &phase)
	cudnn.set(d, .OPERATION_NORM_FWD_XDESC, .BACKEND_DESCRIPTOR, &x)
	cudnn.set(d, .OPERATION_NORM_FWD_YDESC, .BACKEND_DESCRIPTOR, &y)
	cudnn.set(d, .OPERATION_NORM_FWD_SCALE_DESC, .BACKEND_DESCRIPTOR, &scale)
	cudnn.set(d, .OPERATION_NORM_FWD_BIAS_DESC, .BACKEND_DESCRIPTOR, &bias)
	cudnn.set(d, .OPERATION_NORM_FWD_EPSILON_DESC, .BACKEND_DESCRIPTOR, &eps)
	cudnn.set(d, .OPERATION_NORM_FWD_MEAN_DESC, .BACKEND_DESCRIPTOR, &mean)
	cudnn.set(d, .OPERATION_NORM_FWD_INV_VARIANCE_DESC, .BACKEND_DESCRIPTOR, &rvar)
	cudnn.finalize(d)
	return must_build(d)
}

layernorm_fwd_checkargs :: proc(l: ^Layernorm($D, $T), inp, out: Shape, loc := #caller_location) {
	assert(inp.ndims >= 2 && inp.dims == out.dims && out.dims[out.ndims - 1] == l.scale.dims[0], "invalid input shapes", loc)
}

// backprop layernorm: accumulates gradient in din
layernorm_backward :: proc {
	layernorm_backward_cpu,
	layernorm_backward_cuda,
}

layernorm_backward_cpu :: proc(l: ^Layernorm(CPU, f32), inp, dout, din: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	layernorm_bwd_checkargs(l, inp.shape, dout.shape, din.shape, loc = loc)
	alloc_grads(&l.layer)

	C := dout.dims[dout.ndims - 1]
	N := dout.size / C
	input, doutput, dinput := array.ptr(inp), array.ptr(dout), array.ptr(din)
	scale, dscale, dbias := array.ptr(l.scale.arr), array.ptr(l.scale.grad), array.ptr(l.bias.grad)
	means, rstds := array.ptr(l.mean), array.ptr(l.rstd)

	for n in 0 ..< N {
		inputs := input[n * C:]
		doutputs := doutput[n * C:]
		dinputs := dinput[n * C:]
		mean := means[n]
		rstd := rstds[n]
		dnorm_mean, dnorm2_mean: f32
		for i in 0 ..< C {
			dnorm := scale[i] * doutputs[i]
			dnorm_mean += dnorm
			dnorm2_mean += dnorm * (inputs[i] - mean) * rstd
		}
		dnorm_mean /= f32(C)
		dnorm2_mean /= f32(C)
		for i in 0 ..< C {
			d := doutputs[i]
			norm := (inputs[i] - mean) * rstd
			dbias[i] += d
			dscale[i] += norm * d
			dinputs[i] += (scale[i] * d - dnorm_mean - norm * dnorm2_mean) * rstd
		}
	}
}

layernorm_backward_cuda :: proc(l: ^Layernorm(Cuda, BF16), inp, dout, din: Array(Cuda, BF16), loc := #caller_location) {
	layernorm_bwd_checkargs(l, inp.shape, dout.shape, din.shape, loc = loc)
	alloc_grads(&l.layer)
	C := din.dims[din.ndims - 1]
	N := din.size / C
	id := fmt.aprintf("norm_bwd_%d_%d", N, C)
	defer delete(id)
	plan, ok := cudnn.plan_from_cache(id)
	if !ok {
		plan = layernorm_bwd_init_cuda(N, C, loc = loc)
		cudnn.cache_plan(id, plan)
	}
	// inputs
	x, dy := array.ptr(inp), array.ptr(dout)
	scale, mean, rstd := array.ptr(l.scale.arr), array.ptr(l.mean), array.ptr(l.rstd)
	// outputs
	dx := array.zeros_like(din)
	dscale := array.zeros_like(l.scale.grad)
	dbias := array.zeros_like(l.bias.grad)
	// exec and accumulate
	cudnn.execute(plan, {'i', 'y', 'x', 'w', 's', 'b', 'm', 'v'}, {x, dy, array.ptr(dx), scale, array.ptr(dscale), array.ptr(dbias), mean, rstd})
	add(din, dx, din)
	add(l.scale.grad, dscale, l.scale.grad)
	add(l.bias.grad, dbias, l.bias.grad)
	array.delete(dx)
	array.delete(dscale)
	array.delete(dbias)
}

layernorm_bwd_init_cuda :: proc(N, C: int, loc := #caller_location) -> cudnn.BackendDescriptor {
	log.debugf("init layernorm bwd: N=%d C=%d", N, C, location = loc)
	x := cudnn.tensor_descriptor('i', {N, C, 1, 1}, {C, 1, 1, 1}, dtype = .BFLOAT16)
	dy := cudnn.tensor_descriptor('y', {N, C, 1, 1}, {C, 1, 1, 1}, dtype = .BFLOAT16)
	dx := cudnn.tensor_descriptor('x', {N, C, 1, 1}, {C, 1, 1, 1}, dtype = .BFLOAT16)
	scale := cudnn.tensor_descriptor('w', {1, C, 1, 1}, dtype = .BFLOAT16)
	dscale := cudnn.tensor_descriptor('s', {1, C, 1, 1}, dtype = .BFLOAT16)
	dbias := cudnn.tensor_descriptor('b', {1, C, 1, 1}, dtype = .BFLOAT16)
	mean := cudnn.tensor_descriptor('m', {N, 1, 1, 1}, dtype = .BFLOAT16)
	rvar := cudnn.tensor_descriptor('v', {N, 1, 1, 1}, dtype = .BFLOAT16)
	defer cudnn.destroy(x, dy, dx, dscale, dbias, scale, mean, rvar)

	mode := cudnn.BackendNormMode.LAYER_NORM
	d := cudnn.make_descriptor(.OPERATION_NORM_BACKWARD_DESCRIPTOR)
	defer cudnn.destroy(d)
	cudnn.set(d, .OPERATION_NORM_BWD_MODE, .NORM_MODE, &mode)
	cudnn.set(d, .OPERATION_NORM_BWD_XDESC, .BACKEND_DESCRIPTOR, &x)
	cudnn.set(d, .OPERATION_NORM_BWD_DYDESC, .BACKEND_DESCRIPTOR, &dy)
	cudnn.set(d, .OPERATION_NORM_BWD_DXDESC, .BACKEND_DESCRIPTOR, &dx)
	cudnn.set(d, .OPERATION_NORM_BWD_SCALE_DESC, .BACKEND_DESCRIPTOR, &scale)
	cudnn.set(d, .OPERATION_NORM_BWD_DSCALE_DESC, .BACKEND_DESCRIPTOR, &dscale)
	cudnn.set(d, .OPERATION_NORM_BWD_DBIAS_DESC, .BACKEND_DESCRIPTOR, &dbias)
	cudnn.set(d, .OPERATION_NORM_BWD_MEAN_DESC, .BACKEND_DESCRIPTOR, &mean)
	cudnn.set(d, .OPERATION_NORM_BWD_INV_VARIANCE_DESC, .BACKEND_DESCRIPTOR, &rvar)
	cudnn.finalize(d)
	return must_build(d)
}

layernorm_bwd_checkargs :: proc(l: ^Layernorm($D, $T), inp, dout, din: Shape, loc := #caller_location) {
	assert(inp.ndims >= 2 && inp.dims == dout.dims && inp.dims == din.dims && inp.dims[inp.ndims - 1] == l.scale.size, "invalid input shapes", loc)
	assert(!array.is_nil(l.mean), "mean and rstd not populated - run layernorm_forward in train mode", loc)
}


// Multi-head attention layer
Attention :: struct($D, $T: typeid) {
	using layer: Layer(D, T),
	num_heads:   int,
	channels:    int,
	att:         []f32,
	stats:       Array(D, f32),
	out_ptr:     [^]T,
}


make_attention :: proc($D, $T: typeid, name: string, num_heads, channels: int) -> (l: Attention(D, T)) {
	l.num_heads = num_heads
	l.channels = channels
	l.name = strings.clone(name)
	l.type = "Attention"
	l.info = fmt.aprintf("%s{{ num_heads:%d channels:%d }}", name, num_heads, channels)
	l.out_shape = array.make_shape({0, 0, channels})
	return l
}

delete_attention :: proc(l: Attention($D, $T)) {
	delete_layer_base(l.layer)
	if l.att != nil {
		delete(l.att)
	}
	array.delete(l.stats)
}

attention_forward :: proc {
	attention_forward_cpu,
	attention_forward_cuda,
}

attention_forward_cpu :: proc(l: ^Attention(CPU, f32), inp, out: Array(CPU, f32), train := true, loc := #caller_location) #no_bounds_check {
	attention_fwd_checkargs(l, inp, out)
	B, T, C, NH := inp.dims[0], inp.dims[1], l.channels, l.num_heads
	preatt, att := attention_init_cpu(l, B * NH * T * T, train)
	input, output := array.ptr(inp), array.ptr(out)

	stride, offset := 3 * C, C
	headSize := C / NH
	scale := 1 / math.sqrt(f32(headSize))

	for b in 0 ..< B {
		for t in 0 ..< T {
			for h in 0 ..< NH {
				query := input[b * T * stride + t * stride + h * headSize:]
				base := b * NH * T * T + h * T * T + t * T
				preatt_bt := preatt[base:base + T]
				att_bt := att[base:base + T]
				// pass 1: calculate query dot key and maxval
				maxval := f32(-10000.0)
				for t2 in 0 ..= t {
					key := input[b * T * stride + t2 * stride + h * headSize + offset:]
					val: f32
					for i in 0 ..< headSize {
						val += query[i] * key[i]
					}
					val *= scale
					maxval = max(maxval, val)
					preatt_bt[t2] = val
				}
				// pass 2: calculate the exp and keep track of sum
				expsum: f32
				for t2 in 0 ..= t {
					expv := math.exp(preatt_bt[t2] - maxval)
					expsum += expv
					att_bt[t2] = expv
				}
				expsum_inv := expsum != 0 ? 1 / expsum : 0
				// pass 3: normalize to get the softmax
				for t2 in 0 ..< T {
					if t2 <= t {
						att_bt[t2] *= expsum_inv
					} else {
						att_bt[t2] = 0
					}
				}
				// pass 4: accumulate weighted values into the output of attention
				outv := output[b * T * C + t * C + h * headSize:]
				for i in 0 ..< headSize {
					outv[i] = 0
				}
				for t2 in 0 ..= t {
					value := input[b * T * stride + t2 * stride + h * headSize + 2 * offset:]
					for i in 0 ..< headSize {
						outv[i] += att_bt[t2] * value[i]
					}
				}
			}
		}
	}
	if !train {
		delete(att)
	}
	delete(preatt)
}

attention_init_cpu :: proc(l: ^Attention(CPU, f32), size: int, train: bool) -> (preatt, att: []f32) {
	preatt = make([]f32, size)
	if train && len(l.att) == size {
		return preatt, l.att
	}
	att = make([]f32, size)
	if train {
		l.att = att
	}
	return preatt, att
}


attention_forward_cuda :: proc(l: ^Attention(Cuda, BF16), inp, out: Array(Cuda, BF16), train := true, loc := #caller_location) {
	attention_fwd_checkargs(l, inp, out)
	B, T, C, NH := inp.dims[0], inp.dims[1], l.channels, l.num_heads
	id := fmt.aprintf("attn_fwd_%d_%d_%d_%d%s", B, T, C, NH, train ? "_train" : "")
	defer delete(id)
	plan, ok := cudnn.plan_from_cache(id)
	if !ok {
		plan = attention_fwd_init_cuda(B, T, C, NH, train, loc = loc)
		cudnn.cache_plan(id, plan)
	}
	if train && l.stats.size != B * NH * T {
		array.delete(l.stats)
		l.stats = array.zeros(Cuda, f32, {B, NH, T, 1})
	}
	q, k, v := array.ptr(inp), array.ptr(inp, offset = C), array.ptr(inp, offset = 2 * C)
	output := array.ptr(out)
	scale := 1 / math.sqrt(f32(C / NH))
	neg_inf := f32(-math.F32_MAX)
	if train {
		stats := array.ptr(l.stats)
		cudnn.execute(plan, {'q', 'k', 'v', 'c', 'n', 'o', 's'}, {q, k, v, &scale, &neg_inf, output, stats})
	} else {
		cudnn.execute(plan, {'q', 'k', 'v', 'c', 'n', 'o'}, {q, k, v, &scale, &neg_inf, output})
	}
	l.out_ptr = output
}

attention_fwd_init_cuda :: proc(B, T, C, NH: int, train: bool, loc := #caller_location) -> cudnn.BackendDescriptor {
	log.debugf("init attention fwd: B=%d T=%d C=%d NH=%d train=%v", B, T, C, NH, train, location = loc)
	assert(C % NH == 0, "number of channels must be multiple of number of heads", loc)
	stride := 3 * C
	HS := C / NH
	// inputs
	q := cudnn.tensor_descriptor('q', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	k := cudnn.tensor_descriptor('k', {B, NH, HS, T}, {stride * T, HS, 1, stride}, dtype = .BFLOAT16)
	v := cudnn.tensor_descriptor('v', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	scale := cudnn.tensor_descriptor('c', {1, 1, 1, 1}, by_value = true)
	neg_inf := cudnn.tensor_descriptor('n', {1, 1, 1, 1}, by_value = true)
	defer cudnn.destroy(q, k, v, scale, neg_inf)
	// outputs
	out := cudnn.tensor_descriptor('o', {B, NH, T, HS}, {C * T, HS, C, 1}, dtype = .BFLOAT16)
	stats := cudnn.tensor_descriptor('s', {B, NH, T, 1}, virtual = !train)
	defer cudnn.destroy(out, stats)
	// virtuals
	qk := cudnn.tensor_descriptor(1, {B, NH, T, T}, virtual = true)
	qks := cudnn.tensor_descriptor(2, {B, NH, T, T}, virtual = true)
	row := cudnn.tensor_descriptor(3, {B, NH, T, T}, virtual = true)
	col := cudnn.tensor_descriptor(4, {B, NH, T, T}, virtual = true)
	cmp := cudnn.tensor_descriptor(5, {B, NH, T, T}, virtual = true, dtype = .BOOLEAN)
	preatt := cudnn.tensor_descriptor(6, {B, NH, T, T}, virtual = true)
	att := cudnn.tensor_descriptor(7, {B, NH, T, T}, virtual = true, dtype = .BFLOAT16)
	maxval := cudnn.tensor_descriptor(8, {B, NH, T, 1}, virtual = true)
	delta := cudnn.tensor_descriptor(9, {B, NH, T, T}, virtual = true)
	p := cudnn.tensor_descriptor(10, {B, NH, T, T}, virtual = true)
	total := cudnn.tensor_descriptor(11, {B, NH, T, 1}, virtual = true)
	logt := cudnn.tensor_descriptor(12, {B, NH, T, 1}, virtual = true)
	defer cudnn.destroy(qk, qks, row, col, cmp, preatt, att, maxval, delta, p, total, logt)

	ops := []cudnn.BackendDescriptor {
		// input scaled dot product
		cudnn.matmul_op(q, k, qk),
		cudnn.binary_op(.MUL, qk, scale, qks),
		// causal mask
		cudnn.unary_op(.GEN_INDEX, qks, col, axis = 3),
		cudnn.unary_op(.GEN_INDEX, qks, row, axis = 2),
		cudnn.binary_op(.CMP_GE, row, col, cmp, type = .BOOLEAN),
		cudnn.binary_op(.BINARY_SELECT, qks, neg_inf, preatt, mask = cmp),
		// softmax with stats for backward step
		cudnn.reduce_op(.MAX, preatt, maxval),
		cudnn.binary_op(.SUB, preatt, maxval, delta),
		cudnn.unary_op(.EXP, delta, p),
		cudnn.reduce_op(.ADD, p, total),
		cudnn.unary_op(.LOG, total, logt),
		cudnn.binary_op(.DIV, p, total, att),
		cudnn.binary_op(.ADD, maxval, logt, stats),
		// output dot product
		cudnn.matmul_op(att, v, out),
	}
	defer cudnn.destroy(..ops)
	return must_build(..ops)
}

attention_fwd_checkargs :: proc(l: ^Attention($D, $T), inp, out: Shape, loc := #caller_location) {
	assert(
		inp.ndims == 3 &&
		out.ndims == 3 &&
		out.dims[2] == l.channels &&
		inp.dims[0] == out.dims[0] &&
		inp.dims[1] == out.dims[1] &&
		inp.dims[2] == l.channels * 3,
		"invalid input shapes",
		loc,
	)
}

// backprop attention: overwrites gradient values in din
attention_backward :: proc {
	attention_backward_cpu,
	attention_backward_cuda,
}

attention_backward_cpu :: proc(l: ^Attention(CPU, f32), inp, dout, din: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	attention_bwd_checkargs(l, inp.shape, dout.shape, din.shape, loc = loc)
	array.zero(din)
	B, T, C, NH := inp.dims[0], inp.dims[1], l.channels, l.num_heads
	input, doutput, dinput := array.ptr(inp), array.ptr(dout), array.ptr(din)
	preattD := make([]f32, B * NH * T * T)
	defer delete(preattD)
	attD := make([]f32, B * NH * T * T)
	defer delete(attD)
	stride, offset := 3 * C, C
	headSize := C / NH
	scale := 1 / math.sqrt(f32(headSize))
	for b in 0 ..< B {
		for t in 0 ..< T {
			for h in 0 ..< NH {
				base := b * NH * T * T + h * T * T + t * T
				att := l.att[base:]
				datt := attD[base:]
				dpreatt := preattD[base:]
				// Backward pass 4: value accumulation
				dout := doutput[b * T * C + t * C + h * headSize:]
				for t2 in 0 ..= t {
					value := b * T * stride + t2 * stride + h * headSize + 2 * offset
					for i in 0 ..< headSize {
						// Compute gradients for attention and value accumulation
						datt[t2] += input[value + i] * dout[i]
						dinput[value + i] += att[t2] * dout[i]
					}
				}
				// Backward pass 2 & 3: softmax backward
				for t2 in 0 ..= t {
					for t3 in 0 ..= t {
						indicator: f32 = t2 == t3 ? 1 : 0
						localDerivative := att[t2] * (indicator - att[t3])
						dpreatt[t3] += localDerivative * datt[t2]
					}
				}
				// Backward pass 1: query @ key matmul
				query := b * T * stride + t * stride + h * headSize
				for t2 in 0 ..= t {
					key := b * T * stride + t2 * stride + h * headSize + offset
					for i in 0 ..< headSize {
						// Compute gradients for query and key
						dinput[query + i] += input[key + i] * dpreatt[t2] * scale
						dinput[key + i] += input[query + i] * dpreatt[t2] * scale
					}
				}
			}
		}
	}
}

attention_backward_cuda :: proc(l: ^Attention(Cuda, BF16), inp, dout, din: Array(Cuda, BF16), loc := #caller_location) {
	attention_bwd_checkargs(l, inp.shape, dout.shape, din.shape, loc = loc)
	B, T, C, NH := inp.dims[0], inp.dims[1], l.channels, l.num_heads
	id := fmt.aprintf("attn_bwd_%d_%d_%d_%d", B, T, C, NH)
	defer delete(id)
	plan, ok := cudnn.plan_from_cache(id)
	if !ok {
		plan = attention_bwd_init_cuda(B, T, C, NH, loc = loc)
		cudnn.cache_plan(id, plan)
	}
	q, k, v := array.ptr(inp), array.ptr(inp, offset = C), array.ptr(inp, offset = 2 * C)
	dq, dk, dv := array.ptr(din), array.ptr(din, offset = C), array.ptr(din, offset = 2 * C)
	doutput, stats := array.ptr(dout), array.ptr(l.stats)
	scale := 1 / math.sqrt(f32(C / NH))
	one := f32(1)
	neg_inf := f32(-math.F32_MAX)
	cudnn.execute(plan, {'q', 'k', 'v', 'x', 'y', 'z', 'o', 'd', 's', 'c', 'n', '1'}, {q, k, v, dq, dk, dv, l.out_ptr, doutput, stats, &scale, &neg_inf, &one})
}

attention_bwd_init_cuda :: proc(B, T, C, NH: int, loc := #caller_location) -> cudnn.BackendDescriptor {
	log.debugf("init attention bwd: B=%d T=%d C=%d NH=%d", B, T, C, NH, location = loc)
	assert(C % NH == 0, "number of channels must be multiple of number of heads", loc)
	stride := 3 * C
	HS := C / NH
	// inputs
	q := cudnn.tensor_descriptor('q', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	k := cudnn.tensor_descriptor('k', {B, NH, HS, T}, {stride * T, HS, 1, stride}, dtype = .BFLOAT16)
	v := cudnn.tensor_descriptor('v', {B, NH, HS, T}, {stride * T, HS, 1, stride}, dtype = .BFLOAT16)
	out := cudnn.tensor_descriptor('o', {B, NH, T, HS}, {C * T, HS, C, 1}, dtype = .BFLOAT16)
	dout := cudnn.tensor_descriptor('d', {B, NH, T, HS}, {C * T, HS, C, 1}, dtype = .BFLOAT16)
	stats := cudnn.tensor_descriptor('s', {B, NH, T, 1})
	scale := cudnn.tensor_descriptor('c', {1, 1, 1, 1}, by_value = true)
	one := cudnn.tensor_descriptor('1', {1, 1, 1, 1}, by_value = true)
	neg_inf := cudnn.tensor_descriptor('n', {1, 1, 1, 1}, by_value = true)
	defer cudnn.destroy(q, k, v, out, dout, stats, scale, one, neg_inf)
	// outputs
	dq := cudnn.tensor_descriptor('x', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	dk := cudnn.tensor_descriptor('y', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	dv := cudnn.tensor_descriptor('z', {B, NH, T, HS}, {stride * T, HS, stride, 1}, dtype = .BFLOAT16)
	defer cudnn.destroy(dq, dk, dv)
	// virtuals
	o := cudnn.tensor_descriptor(1, {B, NH, T, HS}, {C * T, T * HS, C, 1}, virtual = true)
	osum := cudnn.tensor_descriptor(2, {B, NH, T, 1}, virtual = true)
	osums := cudnn.tensor_descriptor(3, {B, NH, T, 1}, virtual = true)
	qk := cudnn.tensor_descriptor(4, {B, NH, T, T}, virtual = true)
	qks := cudnn.tensor_descriptor(5, {B, NH, T, T}, virtual = true)
	row := cudnn.tensor_descriptor(6, {B, NH, T, T}, virtual = true, dtype = .INT32)
	col := cudnn.tensor_descriptor(7, {B, NH, T, T}, virtual = true, dtype = .INT32)
	cmp := cudnn.tensor_descriptor(8, {B, NH, T, T}, {NH * T * T, 1, NH * T, NH}, virtual = true, dtype = .BOOLEAN)
	mask := cudnn.tensor_descriptor(9, {B, NH, T, T}, {NH * T * T, 1, NH * T, NH}, virtual = true)
	val := cudnn.tensor_descriptor(10, {B, NH, T, T}, {NH * T * T, 1, NH * T, NH}, virtual = true)
	p := cudnn.tensor_descriptor(11, {B, NH, T, T}, {NH * T * T, 1, NH * T, NH}, virtual = true)
	p_t := cudnn.tensor_descriptor(12, {B, NH, T, T}, {NH * T * T, T * T, 1, T}, virtual = true, dtype = .BFLOAT16)
	dout_v := cudnn.tensor_descriptor(13, {B, NH, T, T}, virtual = true)
	dout_vs := cudnn.tensor_descriptor(14, {B, NH, T, T}, virtual = true)
	delta := cudnn.tensor_descriptor(15, {B, NH, T, T}, virtual = true)
	prod := cudnn.tensor_descriptor(16, {B, NH, T, T}, virtual = true)
	prods := cudnn.tensor_descriptor(17, {B, NH, T, T}, virtual = true)
	prods_t := cudnn.tensor_descriptor(18, {B, NH, T, T}, {NH * T * T, T * T, 1, T}, virtual = true, dtype = .BFLOAT16)
	k_t := cudnn.tensor_descriptor(19, {B, NH, T, HS}, {stride * T, HS, stride, 1}, virtual = true)
	defer cudnn.destroy(o, osum, osums, qk, qks, row, col, cmp, mask, val, p, p_t, dout_v, dout_vs, delta, prod, prods, prods_t, k_t)

	ops := []cudnn.BackendDescriptor {
		// dOut x V input
		cudnn.matmul_op(dout, v, dout_v),
		cudnn.binary_op(.MUL, dout_v, one, dout_vs),
		// Q x K input
		cudnn.matmul_op(q, k, qk),
		cudnn.binary_op(.MUL, qk, scale, qks),
		// causal mask
		cudnn.unary_op(.GEN_INDEX, qks, col, axis = 3, type = .INT32),
		cudnn.unary_op(.GEN_INDEX, qks, row, axis = 2, type = .INT32),
		cudnn.binary_op(.CMP_GE, row, col, cmp, type = .BOOLEAN),
		cudnn.binary_op(.BINARY_SELECT, qks, neg_inf, mask, mask = cmp),
		// dOut * out sum
		cudnn.binary_op(.MUL, dout, out, o),
		cudnn.reduce_op(.ADD, o, osum),
		cudnn.binary_op(.MUL, osum, one, osums),
		// reverse softmax
		cudnn.binary_op(.SUB, mask, stats, val),
		cudnn.unary_op(.EXP, val, p),
		cudnn.binary_op(.SUB, dout_vs, osums, delta),
		cudnn.binary_op(.MUL, delta, p, prod),
		cudnn.binary_op(.MUL, prod, scale, prods),
		// -> dQ output
		cudnn.reshape_op(k, k_t),
		cudnn.matmul_op(prods, k_t, dq),
		// -> dK output
		cudnn.reshape_op(prods, prods_t),
		cudnn.matmul_op(prods_t, q, dk),
		// -> dV output
		cudnn.reshape_op(p, p_t),
		cudnn.matmul_op(p_t, dout, dv),
	}
	defer cudnn.destroy(..ops)
	return must_build(..ops)
}

attention_bwd_checkargs :: proc(l: ^Attention($D, $T), inp, dout, din: Shape, loc := #caller_location) {
	assert(
		inp.ndims == 3 &&
		dout.ndims == 3 &&
		inp.dims == din.dims &&
		inp.dims[0] == dout.dims[0] &&
		inp.dims[1] == dout.dims[1] &&
		inp.dims[2] == l.channels * 3,
		"invalid input shapes",
		loc,
	)
	when D == CPU {
		assert(l.att != nil, "intermediate activation not populated - run attention_forward in train mode", loc)
	} else {
		assert(!array.is_nil(l.stats) && l.out_ptr != nil, "prior output and stats not populated - run attention_forward in train mode", loc)
	}
}

// Calculate gelu activation - supports inplace update
gelu_scale := math.sqrt(f32(2) / f32(math.PI))

gelu_forward :: proc {
	gelu_forward_cpu,
	gelu_forward_cuda,
}

gelu_forward_cpu :: proc(inp, out: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	assert(inp.size == out.size, "size mismatch", loc)
	xp, yp := array.ptr(inp), array.ptr(out)
	for i in 0 ..< inp.size {
		x := xp[i]
		cube := 0.044715 * x * x * x
		yp[i] = 0.5 * x * (1 + math.tanh(gelu_scale * (x + cube)))
	}
}

gelu_forward_cuda :: proc(inp, out: Array(Cuda, BF16), loc := #caller_location) {
	assert(inp.size == out.size, "size mismatch", loc)
	fn := cuda.get_function("gelu_bf16")
	BLOCK :: 512
	xp, yp, size := array.ptr(inp), array.ptr(out), inp.size
	cuda.launch_kernel(fn, gridX = util.ceil_div(inp.size, BLOCK), blockX = BLOCK, params = {&xp, &yp, &size})
}

gelu_backward :: proc {
	gelu_backward_cpu,
	gelu_backward_cuda,
}

gelu_backward_cpu :: proc(inp, dout, din: Array(CPU, f32), loc := #caller_location) #no_bounds_check {
	assert(inp.size == dout.size && dout.size == din.size, "size mismatch", loc)
	xp, dx, dy := array.ptr(inp), array.ptr(din), array.ptr(dout)
	for i in 0 ..< inp.size {
		x := xp[i]
		cube := 0.044715 * x * x * x
		tanh_arg := gelu_scale * (x + cube)
		tanh_out := math.tanh(tanh_arg)
		cosh_out := math.cosh(tanh_arg)
		sech_out := 1 / (cosh_out * cosh_out)
		local_grad := 0.5 * (1 + tanh_out) + x * 0.5 * sech_out * gelu_scale * (1 + 3 * 0.044715 * x * x)
		dx[i] = local_grad * dy[i]
	}
}

gelu_backward_cuda :: proc(inp, dout, din: Array(Cuda, $T), loc := #caller_location) where T == f32 || T == BF16 {
	assert(inp.size == dout.size && dout.size == din.size, "size mismatch", loc)
	fn := cuda.get_function("gelu_bwd_bf16")
	BLOCK :: 512
	xp, dx, dy, size := array.ptr(inp), array.ptr(din), array.ptr(dout), din.size
	cuda.launch_kernel(fn, gridX = util.ceil_div(din.size, BLOCK), blockX = BLOCK, params = {&xp, &dy, &dx, &size})
}

// Calculate cross entropy loss
// if train flag is set then overwrites the logits buffer with the logits gradient assuming an output gradient 
// of 1 / (B*T*grad_accum_steps)
cross_entropy_loss :: proc {
	cross_entropy_loss_cpu,
	cross_entropy_loss_cuda,
}

cross_entropy_loss_cpu :: proc(
	logits: Array(CPU, f32),
	targets: Array(CPU, i32),
	losses: Array(CPU, f32),
	vocab_size: int,
	train := true,
	grad_accum_steps := 1,
	loc := #caller_location,
) #no_bounds_check {

	cross_entropy_checkargs(logits.shape, targets.shape, losses.shape, vocab_size, loc = loc)
	B, T, V := logits.dims[0], logits.dims[1], logits.dims[2]
	logit, target, loss := array.ptr(logits), array.ptr(targets), array.ptr(losses)
	probs := make([]f32, vocab_size)
	defer delete(probs)
	dloss := 1 / f32(B * T * grad_accum_steps)

	for i in 0 ..< B * T {
		// calc softmax probability
		maxval := logit[i * V]
		for j in 1 ..< vocab_size {
			maxval = max(maxval, logit[i * V + j])
		}
		sum: f32
		for j in 0 ..< vocab_size {
			probs[j] = math.exp(logit[i * V + j] - maxval)
			sum += probs[j]
		}
		for j in 0 ..< vocab_size {
			probs[j] /= sum
		}
		// calc loss
		ix := int(target[i])
		loss[i] = -math.ln(probs[ix])
		// back propagate
		if train {
			for j in 0 ..< vocab_size {
				logit[i * V + j] = j == ix ? (probs[j] - 1) * dloss : probs[j] * dloss
			}
		}
	}
}

cross_entropy_loss_cuda :: proc(
	logits: Array(Cuda, BF16),
	targets: Array(Cuda, i32),
	losses: Array(Cuda, f32),
	vocab_size: int,
	train := true,
	grad_accum_steps := 1,
	loc := #caller_location,
) {

	cross_entropy_checkargs(logits.shape, targets.shape, losses.shape, vocab_size, loc = loc)
	fn := cuda.get_function("crossentropy_loss_bf16")
	logitp, lossp, targetp := array.ptr(logits), array.ptr(losses), array.ptr(targets)
	BT, V, Vp := i32(logits.dims[0] * logits.dims[1]), i32(vocab_size), i32(logits.dims[2])
	tflag: i32 = train ? 1 : 0
	dloss := 1 / f32(int(BT) * grad_accum_steps)
	BLOCK :: 512
	rows_per_block := BLOCK / 32
	grid := int(BT) / rows_per_block
	cuda.launch_kernel(fn, gridX = grid, blockX = BLOCK, params = {&logitp, &lossp, &logitp, &targetp, &dloss, &BT, &V, &Vp, &tflag})
}

cross_entropy_checkargs :: proc(logits, targets, losses: Shape, vocab_size: int, loc := #caller_location) {
	assert(
		logits.ndims == 3 &&
		targets.ndims == 2 &&
		targets.dims == losses.dims &&
		logits.dims[0] == targets.dims[0] &&
		logits.dims[1] == targets.dims[1] &&
		logits.dims[2] >= vocab_size,
		"input shapes invalid",
		loc = loc,
	)
}

// build graph and get best plan from heuristics
must_build :: proc(ops: ..cudnn.BackendDescriptor, loc := #caller_location) -> cudnn.BackendDescriptor {
	graph := cudnn.make_graph(..ops, loc = loc)
	defer cudnn.destroy(graph)
	plan, err := cudnn.get_plan(graph, loc = loc)
	if err != nil {
		log.panic(err, location = loc)
	}
	return plan
}

// Create unique ID from string. Value is case insensitive and anything outside [0-9A-Za-z] range is ignored
tid :: proc(str: string) -> (value: u64) {
	digit_value :: proc(r: rune) -> u64 {
		ri := u64(r)
		switch r {
		case '0' ..= '9':
			return ri - '0'
		case 'a' ..= 'z':
			return ri - 'a' + 10
		case 'A' ..= 'Z':
			return ri - 'A' + 10
		}
		panic("invalid digit")
	}
	for r in str {
		value *= 36
		value += digit_value(r)
	}
	return
}
