package nn

import "core:testing"
import "core:log"
import "core:math/rand"

import "../array"
import "../util"

uniform_init_func :: proc(ctx: rawptr) -> f32 {
	return rand.float32()
}

integer_init_func :: proc(ctx: rawptr) -> i32 {
	return i32(rand.int_max(1024))
}

grad_init :: proc(ctx: rawptr) -> f32 {
	return rand.float32_normal(0, 0.1)
}


expect_slice :: proc(t: ^testing.T, name: string, out: Array($D, $T), expected: []f32, max_diff: f32 = 1e-6, loc := #caller_location) {
	res := make([]f32, out.size)
	defer delete(res)
	array.copy(res, out)
	diff := util.max_difference(res, expected)
	log.debugf("%s delta = %g\n% 10.4g", name, diff, out)
	testing.expect(t, diff <= max_diff, loc=loc)
}

@(test)
add_test_cpu :: proc(t: ^testing.T) {
	add_test_on(t, CPU, f32)
}

@(test)
add_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()
	add_test_on(t, Cuda, f32)
	add_test_on(t, Cuda, BF16)
}

add_test_on :: proc(t: ^testing.T, $Device, $T: typeid) {
	x := array.new(Device, T, {3, 4}, util.seq(f32, 1, 13), move=true)
	defer array.delete(x)
	y := array.zeros_like(x)
	array.fill(y, 10)
	defer array.delete(y)
	out := array.zeros_like(x)
	defer array.delete(out)
	add(x, y, out)
	expect_slice(t, "out", out, {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
}

@(test)
linear_test_cpu :: proc(t: ^testing.T) {
	linear_test_on(t, CPU, f32)
}

@(test)
linear_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()	
	linear_test_on(t, Cuda, f32)
	linear_test_on(t, Cuda, BF16, max_diff=0.02)
}

@(test)
linear_bwd_test_cpu :: proc(t: ^testing.T) {
	linear_bwd_test_on(t, CPU, f32)
}

@(test)
linear_bwd_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()
	linear_bwd_test_on(t, Cuda, f32)
	linear_bwd_test_on(t, Cuda, BF16, max_diff=0.01)
}


linear_test_on :: proc(t: ^testing.T, $Device, $Type: typeid, max_diff: f32 = 1e-6) {
	B, T, C, OC := 1, 4, 2, 3
	l := make_linear_layer(Device, Type, C, OC)
	defer delete_layer(l)
	
	input := array.new(Device, Type, {B, T, C}, []f32{1, 2, 0.5, 1, -2, -1, 1.1, 2})
	defer array.delete(input)
	output := array.zeros(Device, Type, {B, T, OC})
	defer array.delete(output)
	
	linear_forward(&l, input, output)
	expect_slice(t, "output", output, {6, 13, 20, 3.5, 7.5, 11.5, -3, -8, -13, 6.1, 13.3, 20.5}, max_diff)
}

linear_bwd_test_on :: proc(t: ^testing.T, $Device, $Type: typeid, max_diff: f32 = 1e-6) {
	B, T, C, OC := 1, 1, 2, 4
	l := make_linear_layer(Device, Type, C, OC)
	defer delete_layer(l)

	input := array.new(Device, Type, {B, T, C}, []f32{1, 2})
	defer array.delete(input)
	output := array.zeros(Device, Type, {B, T, OC})
	defer array.delete(output)

	linear_forward(&l, input, output)
	expect_slice(t, "output", output, {6, 13, 20, 27})

	dout := array.new(Device, Type, {B, T, OC}, []f32{0.1, 0.2, -0.25, 0.5})
	defer array.delete(dout)
	din := array.zeros_like(input)
	defer array.delete(din)

	linear_backward(&l, input, dout, din)
	expect_slice(t, "din", din, {2.95, 3.5}, max_diff)
	expect_slice(t, "dweight", l.weight.grad, {0.1, 0.2, 0.2, 0.4, -0.25, -0.5, 0.5, 1}, max_diff)
	expect_slice(t, "dbias", l.bias.grad, {0.1, 0.2, -0.25, 0.5}, max_diff)
}

make_linear_layer :: proc($Device, $Type: typeid, C, OC: int) -> Linear(Device, Type) {
	l := make_linear(Device, Type, "linear", C, OC, bias=true)
	log.debug(l.info)
	s1 := util.seq(f32, 1, C*OC+1)
	defer delete(s1)
	array.copy(l.weight.arr, s1)
	s2 := util.seq(f32, 1, OC+1)
	defer delete(s2)
	array.copy(l.bias.arr, s2)
	return l	
}

@(test)
encoder_test_cpu :: proc(t: ^testing.T) {
	encoder_test_on(t, CPU, f32)
}

@(test)
encoder_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()
	encoder_test_on(t, Cuda, BF16)
}

@(test)
encoder_bwd_test_cpu :: proc(t: ^testing.T) {
	encoder_bwd_test_on(t, CPU, f32)
}

@(test)
encoder_bwd_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()
	encoder_bwd_test_on(t, Cuda, BF16)
}

encoder_test_on :: proc(t: ^testing.T, $Device, $Type: typeid) {
	B, T, C, V := 1, 2, 2, 2
	l := make_encoder(Device, Type, "encoder", V, T, C)
	defer delete_layer(l)
	log.debug(l.info)
	array.copy(l.wte.arr, []f32{0, 1, 2, 3})
	array.copy(l.wpe.arr, []f32{4, 5, 6, 7})

	input := array.zeros(Device, i32, {B, T})
	defer array.delete(input)
	array.copy(input, []f32{0, 1})
	output := array.zeros(Device, Type, {B, T, C})
	defer array.delete(output)

	encoder_forward(&l, input, output)
	expect_slice(t, "output", output, {4, 6, 8, 10})
}

encoder_bwd_test_on :: proc(t: ^testing.T, $Device, $Type: typeid) {
	B, T, max_T, C, V := 1, 1, 2, 2, 2
	l := make_encoder(Device, Type, "encoder", V, max_T, C)
	defer delete_layer(l)
	log.debug(l.info)
	zero_gradients(&l.layer)

	input := array.new(Device, i32, {B, T}, []f32{1})
	defer array.delete(input)
	doutput := array.new(Device, Type, {B, T, C}, []f32{1, 2})
	defer array.delete(doutput)
	log.debug(doutput)

	encoder_backward(&l, input, doutput)
	expect_slice(t, "dwte", l.wte.grad, {0, 0, 1, 2})
	expect_slice(t, "dwpe", l.wpe.grad, {1, 2, 0, 0})
}


@(test)
layernorm_test_cpu :: proc(t: ^testing.T) {
	layernorm_test_on(t, CPU, f32)
}

@(test)
layernorm_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()
	layernorm_test_on(t, Cuda, BF16, max_diff=0.1)
}

@(test)
layernorm_bwd_test_cpu :: proc(t: ^testing.T) {
	layernorm_test_on(t, CPU, f32, backward=true)
}

layernorm_test_on :: proc(t: ^testing.T, $Device, $Type: typeid, backward := false, max_diff: f32 = 1e-6) {
	B, T, C := 2, 1, 3
	l := make_layernorm(Device, Type, "layernorm", C)
	defer delete_layer(l)
	log.debug(l.info)
	array.fill(l.scale.arr, 1)

	input := array.new(Device, Type, {B, T, C}, []f32{0.2, 0.1, 0.3, 0.5, 0.1, 0.1})
	defer array.delete(input)
	output := array.zeros(Device, Type, {B, T, C})
	defer array.delete(output)

	layernorm_forward(&l, input, output)
	if !backward {
		expect_slice(t, "output", output, {0, -1.2238272, 1.2238274, 1.4140146, -0.70700747, -0.70700747}, max_diff)
		expect_slice(t, "mean", l.mean, {0.2, 0.23333335}, max_diff)
		expect_slice(t, "rstd", l.rstd, {12.238273, 5.302555}, max_diff)
		return
	}
	dout := array.new(Device, Type, {B, T, C}, []f32{0.1, 0.2, 0.3, -0.4, -0.6, -0.1})
	defer array.delete(dout)
	din := array.zeros(Device, Type, {B, T, C})
	defer array.delete(din)

	layernorm_backward(&l, input, dout, din)
	expect_slice(t, "din", din, {-1.2238272, 0.61099726, 0.61283004, -5.01e-05, -1.3256139, 1.3256639}, max_diff)	
	expect_slice(t, "dscale", l.scale.grad, {-0.5656058, 0.17943905, 0.437849}, max_diff)	
	expect_slice(t, "dbias", l.bias.grad, {-0.3, -0.4, 0.2}, max_diff)	
}


@(test)
attention_test_cpu :: proc(t: ^testing.T) {
	B, T, C, NH := 1, 2, 3, 1
	l := make_attention(CPU, f32, "attention", NH, C)
	defer delete_layer(l)
	log.debug(l.info)

	input := array.new(CPU, f32, {B, T, 3*C}, util.seq(f32, 1, 19), move=true)
	defer array.delete(input)
	output := array.zeros(CPU, f32, {B, T, C})
	defer array.delete(output)

	attention_forward(&l, input, output)
	diff := util.max_difference(l.att, {1, 0, 0, 1})
	log.debugf("att delta = %g  %v", diff, l.att)
	testing.expect(t, diff <= 1e-6)
	expect_slice(t, "output", output, {7, 8, 9, 16, 17, 18})
}

@(test)
attention_bwd_test_cpu :: proc(t: ^testing.T) {
	B, T, C, NH := 1, 1, 2, 1
	l := make_attention(CPU, f32, "attention", NH, C)
	defer delete_layer(l)
	log.debug(l.info)

	input := array.new(CPU, f32, {B, T, 3*C}, util.seq(f32, 1, 7), move=true)
	defer array.delete(input)
	output := array.zeros(CPU, f32, {B, T, C})
	defer array.delete(output)

	attention_forward(&l, input, output)
	diff := util.max_difference(l.att, {1})
	log.debugf("att delta = %g  %v", diff, l.att)
	testing.expect(t, diff <= 1e-6)
	expect_slice(t, "output", output, {5, 6})

	dout := array.new(CPU, f32, {B, T, C}, []f32{0.1, -0.2})
	defer array.delete(dout)
	din := array.zeros(CPU, f32, {B, T, 3*C})
	defer array.delete(din)

	attention_backward(&l, input, dout, din)
	expect_slice(t, "din", din, {0, 0, 0, 0, 0.1, -0.2})
}


@(test)
attention_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()

	out_cpu := attention_test_on(CPU, f32)
	defer array.delete(out_cpu)
	log.debugf("CPU output % 8.3f", out_cpu)

	out_cuda := attention_test_on(Cuda, BF16)
	defer array.delete(out_cuda)
	log.debugf("Cuda output % 8.3f", out_cuda)

	testing.expect(t, array.compare("out", out_cuda, out_cpu, epsilon=0.01))
}

@(test)
attention_bwd_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()

	din_cpu := attention_test_on(CPU, f32, backward=true)
	defer array.delete(din_cpu)
	log.debugf("CPU din % 8.3f", din_cpu)

	din_cuda := attention_test_on(Cuda, BF16, backward=true)
	defer array.delete(din_cuda)
	log.debugf("Cuda din % 8.3f", din_cuda)

	testing.expect(t, array.compare("din", din_cuda, din_cpu, epsilon=0.1, threshold=0.01))
}

attention_test_on :: proc($Device, $Type: typeid, backward := false) -> Array(Device, Type) {
	rand.reset(42)
	B, T, C, NH := 1, 256, 384, 6
	l := make_attention(Device, Type, "attention", NH, C)
	defer delete_layer(l)
	log.debug(l.info)

	input := array.zeros(Device, Type, {B, T, 3*C})
	defer array.delete(input)
	array.initialize(input, uniform_init_func)

	output := array.zeros(Device, Type, {B, T, C})
	attention_forward(&l, input, output, train=backward)
	if !backward {
		return output
	}

	defer array.delete(output)

	dout := array.zeros(Device, Type, {B, T, C})
	defer array.delete(dout)
	array.initialize(dout, uniform_init_func)

	din := array.zeros(Device, Type, {B, T, 3*C})
	attention_backward(&l, input, dout, din)
	return din
}

@(test)
gelu_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()

	out_cpu := gelu_test_on(CPU, f32)
	defer array.delete(out_cpu)
	log.debugf("CPU output %.2g", out_cpu)

	out_cuda := gelu_test_on(Cuda, BF16)
	defer array.delete(out_cuda)
	log.debugf("Cuda output %.2g", out_cuda)

	testing.expect(t, array.compare("out", out_cuda, out_cpu, epsilon=0.05))
}

gelu_test_on :: proc($Device, $Type: typeid) -> Array(Device, Type) {
	rand.reset(42)
	input := array.zeros(Device, Type, {1, 256, 768})
	defer array.delete(input)
	array.initialize(input, uniform_init_func)
	output := array.zeros_like(input)
	gelu_forward(input, output)
	return output
}

@(test)
cross_entropy_test_cuda :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()

	loss_cpu, din_cpu := cross_entropy_test_on(CPU, f32)
	defer array.delete(loss_cpu)
	defer array.delete(din_cpu)

	loss_cuda, din_cuda := cross_entropy_test_on(Cuda, BF16)
	defer array.delete(loss_cuda)
	defer array.delete(din_cuda)

	log.debug("CPU loss", loss_cpu)
	log.debug("Cuda loss", loss_cuda)
	testing.expect(t, array.compare("loss", loss_cuda, loss_cpu, epsilon=0.01))

	log.debug("CPU dlogits", din_cpu)
	log.debug("Cuda dlogits", din_cuda)
	testing.expect(t, array.compare("dlogits", din_cuda, din_cpu, epsilon=0.01))
}

cross_entropy_test_on :: proc($Device, $Type: typeid) -> (Array(Device, f32), Array(Device, Type)) {
	B, T, V := 4, 256, 1024
	rand.reset(42)
	logits := array.zeros(Device, Type, {B, T, V})
	array.initialize(logits, uniform_init_func)

	targets := array.zeros(Device, i32, {B, T})
	defer array.delete(targets)
	array.initialize(targets, integer_init_func)

	losses := array.zeros(Device, f32, {B, T})
	cross_entropy_loss(logits, targets, losses, V, train=true)
	return losses, logits
}


@(test)
adamw_test :: proc(t: ^testing.T) {
	context.user_ptr = init_cuda()
	defer end_cuda()

	cpu_model := make_test_model(CPU, f32)
	defer delete_test_model(cpu_model)

	cuda_model := make_test_model(Cuda, BF16)
	defer delete_test_model(cuda_model)

	cpu_opt := new_optimizer(&cpu_model, learning_rate=0.1)
	defer delete_optimizer(cpu_opt)
	cuda_opt := new_optimizer(&cuda_model, learning_rate=0.1)
	defer delete_optimizer(cuda_opt)

	epsilon, threshold: f32 = 0.1, 0.005
	for step in 1 ..= 10 {
		for i in 0 ..< len(cpu_model.params) {
			array.initialize(cpu_model.params[i].grad, grad_init)
			grad := array.ptr(cpu_model.params[i].grad)[:cpu_model.params[i].size]
			array.copy(cuda_model.params[i].grad, grad)
		}
		cpu_norm := optimizer_step(cpu_opt, &cpu_model)
		cuda_norm := optimizer_step(cuda_opt, &cuda_model)
		log.debugf("step %d CPU norm=%.4g, Cuda norm=%.4g", step, cpu_norm, cuda_norm)
		testing.expect(t, util.nearly_equal(cuda_norm, cpu_norm, epsilon, threshold))
		for i in 0 ..< len(cpu_model.params) {
			testing.expect(t, array.compare("m", cuda_opt.param_m[i], cpu_opt.param_m[i], epsilon, threshold))
			testing.expect(t, array.compare("v", cuda_opt.param_v[i], cpu_opt.param_v[i], epsilon, threshold))
			testing.expect(t, array.compare("weight", cuda_opt.weights[i], cpu_model.params[i].arr, epsilon, threshold))
		}
	}
}

make_test_model :: proc($D, $T: typeid) -> (m: Layer(D, T)) {
	rand.reset(123)
	m.name = "Test"
	m.params = make([]Parameter(D, T), 2)
	for &p in m.params {
		p.arr = array.zeros(D, T, {4, 8})
		p.grad = array.zeros(D, T, {4, 8})
		array.initialize(p.arr, uniform_init_func)	
	}
	return m
}

delete_test_model :: proc(m: Layer($D, $T)) {
	for p in m.params {
		array.delete(p.arr)
		array.delete(p.grad)
	}
	delete(m.params)
}

