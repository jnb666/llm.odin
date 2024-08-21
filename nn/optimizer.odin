package nn

import "core:log"
import "core:math"

import "../array"
import "../cuda"
import "../util"


// Cosine learning rate decay
Cosine_Decay :: struct {
	initial_rate:  f32,
	final_rate:    f32,
	decay_steps:   int,
	warmup_target: f32,
	warmup_steps:  int,
}

decay_learning_rate :: proc(c: ^Cosine_Decay, step: int, learning_rate: ^f32) {
	if step < c.warmup_steps {
		frac := f32(step) / f32(c.warmup_steps)
		learning_rate^ = c.initial_rate * (1 - frac) + c.warmup_target * frac
	} else {
		step := min(step - c.warmup_steps, c.decay_steps)
		decay := f32(0.5 * (1 + math.cos(math.PI * f32(step) / f32(c.decay_steps))))
		rmax := c.warmup_steps > 0 ? c.warmup_target : c.initial_rate
		learning_rate^ = c.final_rate + (rmax - c.final_rate) * decay
	}
}

// AdamW optimizer config settings
AdamW_Config :: struct {
	learning_rate: f32,
	weight_decay:  f32,
	gradient_clip: f32,
	beta1:         f32,
	beta2:         f32,
	epsilon:       f32,
}

// AdamW optimizer state
AdamW :: struct($D, $T: typeid) {
	using config: AdamW_Config,
	steps:        int,
	params:       [dynamic]^Parameter(D, T),
	weights:      [dynamic]Array(D, f32),
	param_m:      [dynamic]Array(D, f32),
	param_v:      [dynamic]Array(D, f32),
}

// Create new optimizer
new_optimizer :: proc(
	model: ^Layer($D, $T),
	learning_rate: f32 = 3e-4,
	weight_decay: f32 = 0,
	gradient_clip: f32 = 0,
	beta1: f32 = 0.9,
	beta2: f32 = 0.95,
	epsilon: f32 = 1e-8,
) -> ^AdamW(D, T) {
	opt := new(AdamW(D, T))
	opt.config = {learning_rate, weight_decay, gradient_clip, beta1, beta2, epsilon}
	init_optimizer(opt, model)
	log.debugf("AdamW optimizer - model = %s, %d parameters", model.name, len(opt.params))
	return opt
}

// Initialize optimizer state arrays for each parameter
init_optimizer :: proc(opt: ^AdamW($D, $T), layer: ^Layer(D, T)) {
	for i in 0 ..< len(layer.params) {
		append(&opt.params, &layer.params[i])
		s := layer.params[i].shape
		dims := array.shape(&s)
		when T != f32 {
			w := array.zeros(D, f32, dims)
			to_float32(layer.params[i].arr, w)
			append(&opt.weights, w)
		}
		append(&opt.param_m, array.zeros(D, f32, dims))
		append(&opt.param_v, array.zeros(D, f32, dims))
	}
	for l in layer.layers {
		init_optimizer(opt, l)
	}
}

// Free allocated optimizer state
delete_optimizer :: proc(opt: ^AdamW($D, $T)) {
	for p in opt.weights {
		array.delete(p)
	}
	for p in opt.param_m {
		array.delete(p)
	}
	for p in opt.param_v {
		array.delete(p)
	}
	delete(opt.weights)
	delete(opt.param_m)
	delete(opt.param_v)
	delete(opt.params)
	free(opt)
}

// Apply one batch step - updates model weights from the gradients + optimizer parameters - returns gradient norm
optimizer_step :: proc {
	optimizer_step_cpu,
	optimizer_step_cuda,
}

grad_scale :: proc(norm, clip: f32) -> f32 {
	scale: f32 = 1
	if clip > 0 && norm > clip {
		log.debugf("norm = %.4g - clipping gradients to %g", norm, clip)
		scale = clip / norm
	}
	return scale
}

optimizer_step_cpu :: proc(opt: ^AdamW(CPU, f32), model: ^Layer(CPU, f32)) -> f32 #no_bounds_check {
	// calc gradient norm
	norm_sq: f32
	for p in opt.params {
		grads := array.ptr(p.grad)
		for v in grads[:p.size] {
			norm_sq += v * v
		}
	}
	norm := math.sqrt(norm_sq)
	scale := grad_scale(norm, opt.gradient_clip)
	// update weights
	opt.steps += 1
	beta1c := f32(1.0 - math.pow(f64(opt.beta1), f64(opt.steps)))
	beta2c := f32(1.0 - math.pow(f64(opt.beta2), f64(opt.steps)))
	for p, i in opt.params {
		weights, grads := array.ptr(p.arr), array.ptr(p.grad)
		m, v := array.ptr(opt.param_m[i]), array.ptr(opt.param_v[i])
		wdecay: f32 = p.ndims > 1 ? opt.weight_decay : 1
		for j in 0 ..< p.size {
			grad := grads[j] * scale
			// momentum update
			m[j] = opt.beta1 * m[j] + (1 - opt.beta1) * grad
			// RMSprop update
			v[j] = opt.beta2 * v[j] + (1 - opt.beta2) * grad * grad
			// bias correction
			bias_c := (m[j] / beta1c) / (math.sqrt(v[j] / beta2c) + opt.epsilon)
			// weight update
			weights[j] -= opt.learning_rate * (bias_c + wdecay * weights[j])
		}
	}
	return norm
}

optimizer_step_cuda :: proc(opt: ^AdamW(Cuda, BF16), model: ^Layer(Cuda, BF16)) -> f32 {
	BLOCK :: 512
	dev := cuda.get_device()
	max_threads := cuda.device_attribute(dev, .MULTIPROCESSOR_COUNT) * cuda.device_attribute(dev, .MAX_THREADS_PER_MULTIPROCESSOR)
	assert(max_threads % BLOCK == 0, "threads must be multiple of block size")
	grid := max_threads / BLOCK

	// calc gradient norm
	fn := cuda.get_function("norm_squared_bf16")
	sum := array.zeros(Cuda, f32, {})
	defer array.delete(sum)
	sump := array.ptr(sum)
	for p in opt.params {
		grads, size := array.ptr(p.grad), p.size
		cuda.launch_kernel(fn, gridX = grid, blockX = BLOCK, params = {&sump, &grads, &size})
	}
	norm_sq: [1]f32
	array.copy(norm_sq[:], sum)
	norm := math.sqrt(norm_sq[0])
	scale := grad_scale(norm, opt.gradient_clip)

	// update weights
	fn = cuda.get_function("adamw_step_bf16")
	opt.steps += 1
	beta1c := f32(1.0 - math.pow(f64(opt.beta1), f64(opt.steps)))
	beta2c := f32(1.0 - math.pow(f64(opt.beta2), f64(opt.steps)))
	for p, i in opt.params {
		assert(p.arr.shape == opt.weights[i].shape, "size of float32 weights must match")
		weights, grads, size := array.ptr(p.arr), array.ptr(p.grad), p.size
		w32, m, v := array.ptr(opt.weights[i]), array.ptr(opt.param_m[i]), array.ptr(opt.param_v[i])
		wdecay: f32 = p.ndims > 1 ? opt.weight_decay : 1

		cuda.launch_kernel(
			fn,
			gridX = util.ceil_div(size, BLOCK),
			blockX = BLOCK,
			params = {&w32, &weights, &grads, &m, &v, &size, &opt.learning_rate, &wdecay, &opt.beta1, &opt.beta2, &beta1c, &beta2c, &opt.epsilon, &scale},
		)
	}
	return norm
}
