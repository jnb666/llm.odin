package main

import "core:fmt"
import "core:log"
import "core:os"
import "core:time"

import "array"
import "gpt2"
import "nn"
import "util"

Test_Options :: struct {
	debug:       bool `usage:"enable debug logging"`,
	track:       bool `usage:"use tracking allocator to find memory leaks"`,
	cuda:        bool `usage:"use Cuda acceleration"`,
	model:       string `usage:"model checkpoint file"`,
	debug_state: string `usage:"debug state checkpoint file"`,
	loss_file:   string `usage:"json file with saved mean losses"`,
	verbose:     bool `usage:"show verbose output"`,
	quiet:       bool `usage:"suppress differences if ok"`,
	steps:       int `usage:"number of test steps"`,
	recompute:   bool `usage:"recompute Gelu activations to save memory"`,
}

Debug_State :: struct {
	batch_size: int,
	seq_len:    int,
	x, y:       Array(CPU, i32),
	logits:     Array(CPU, f32),
	loss:       f32,
	grads:      ^gpt2.Model(CPU, f32),
	losses:     []f32,
}

// run test training session to compare model outputs with saved pytorch reference
test_main :: proc(args: []string) {
	opt := Test_Options {
		model       = "gpt2_124M.bin",
		debug_state = "gpt2_124M_debug_state.bin",
		loss_file   = "gpt2_124M_losses.json",
		steps       = 10,
	}
	parse_args(&opt, "llm test", args)
	run(test_run, &opt)
}

test_run :: proc(opt_ptr: rawptr) {
	opt := cast(^Test_Options)opt_ptr
	log.debugf("\n%v", opt)
	if opt.cuda {
		test_start(Cuda, BF16, opt)
	} else {
		test_start(CPU, f32, opt)
	}
}

test_start :: proc($Device, $Type: typeid, opt: ^Test_Options) {
	model_file := data_file(opt.model, "snapshots")
	defer delete(model_file)
	model, err := gpt2.load_checkpoint(Device, Type, model_file)
	if err != nil {
		fatal_error("Error loading %s: %v", model_file, err)
	}
	defer gpt2.delete_model(model)
	model.recompute_gelu = opt.recompute
	log.info(model.config)

	state: ^Debug_State
	state, err = load_debug_state(opt.debug_state, opt.loss_file, model.config)
	if err != nil {
		fatal_error("Error loading debug state: %v", err)
	}
	defer delete_debug_state(state)

	gpt2.build(model, state.x.dims[0], state.x.dims[1])
	if opt.verbose {
		nn.write_summary(stdout, &model.layer)
	}

	adamw := nn.new_optimizer(&model.layer, learning_rate = 1e-4, weight_decay = 0.01, gradient_clip = 1)
	log.infof("%.4v", adamw.config)
	defer nn.delete_optimizer(adamw)

	when Device == Cuda {
		input := array.to_device(state.x)
		output := array.to_device(state.y)
		defer array.delete(input)
		defer array.delete(output)
	} else {
		input, output := state.x, state.y
	}

	when Device == CPU {
		epsilon, threshold: f32 = 0.12, 0.005
		check_loss_steps := len(state.losses)
	} else {
		epsilon, threshold: f32 = 0.2, 0.05
		check_loss_steps := min(2, len(state.losses))
	}
	log.infof("compare epsilon=%.4g threshold=%.4g check_loss_steps=%d", epsilon, threshold, check_loss_steps)

	start_run := time.now()
	all_ok := true
	for step in 1 ..= opt.steps {
		start := time.now()
		gpt2.forward(model, input)
		if step == 1 {
			if !array.compare("logits", model.act.logits, state.logits, epsilon = epsilon, threshold = threshold, verbose = opt.verbose, quiet = opt.quiet) {
				all_ok = false
			}
		}
		loss := gpt2.calc_loss(model, output)
		if step == 1 {
			log.debugf("mean loss = %.4f, expect %.4f - losses %v", loss, state.loss, model.act.losses)
		}
		nn.zero_gradients(&model.layer)
		gpt2.backward(model, input)
		elapsed_step := round_ms(time.since(start))
		if step == 1 {
			compare_params(&model.layer, &state.grads.layer, epsilon, threshold, &all_ok, verbose = opt.verbose, quiet = opt.quiet)
		}
		grad_norm := nn.optimizer_step(adamw, model)
		elapsed := round_ms(time.since(start))
		exp_loss := step <= len(state.losses) ? state.losses[step - 1] : -1
		log.infof("step % 2d : norm = % 6.2f  loss = %.4f - expect %.4f  elapsed %v / %v", step, grad_norm, loss, exp_loss, elapsed_step, elapsed)
		if step <= check_loss_steps && !util.nearly_equal(loss, exp_loss, epsilon, threshold) {
			log.errorf("loss not matching expected")
			all_ok = false
		}
	}
	run_elapsed := round_ms(time.since(start_run))
	log.infof("total elapsed time = %v", run_elapsed)
	if !all_ok {
		log.error("data mismatch within given tolerances")
	}
}

compare_params :: proc(m1: ^nn.Layer($D1, $T1), m2: ^nn.Layer($D2, $T2), epsilon, threshold: f32, ok: ^bool, verbose, quiet: bool) {
	buf: [64]u8
	assert(m1.name == m2.name && len(m1.params) == len(m2.params))
	for i in 0 ..< len(m1.params) {
		name := fmt.bprintf(buf[:], "%s.p[%d]", m1.name, i)
		if !array.compare(name, m1.params[i].grad, m2.params[i].arr, epsilon = epsilon, threshold = threshold, verbose = verbose, quiet = quiet) {
			ok^ = false
		}
	}
	for i := len(m1.layers) - 1; i >= 0; i -= 1 {
		compare_params(m1.layers[i], m2.layers[i], epsilon, threshold, ok, verbose, quiet)
	}
}

load_debug_state :: proc(file, loss_file: string, cfg: gpt2.Config) -> (s: ^Debug_State, err: os.Error) {
	state_file := data_file(file, "snapshots")
	defer delete(state_file)
	log.info("load debug state from", state_file)
	file := os.open(state_file) or_return
	defer os.close(file)

	r := os.stream_from_handle(file)
	header := make([]i32, 256)
	defer delete(header)
	util.read_slice(r, header) or_return
	if header[0] != 20240327 || header[1] != 2 {
		return nil, .Invalid_File
	}
	s = new(Debug_State)
	s.batch_size = int(header[2])
	s.seq_len = int(header[3])

	s.x = array.zeros(CPU, i32, {s.batch_size, s.seq_len})
	s.y = array.zeros(CPU, i32, {s.batch_size, s.seq_len})
	array.read(i32, r, s.x) or_return
	array.read(i32, r, s.y) or_return

	s.logits = array.zeros(CPU, f32, {s.batch_size, s.seq_len, cfg.vocab_padded})
	logits := make([]f32, cfg.vocab_size)
	defer delete(logits)
	for i in 0 ..< s.batch_size * s.seq_len {
		util.read_slice(r, logits) or_return
		array.copy(array.view(s.logits, {cfg.vocab_size}, offset = i * cfg.vocab_padded), logits)
	}
	loss: [1]f32
	util.read_slice(r, loss[:]) or_return
	s.loss = loss[0]

	s.grads = gpt2.new_model(CPU, f32, cfg)
	gpt2.load_parameters(f32, r, s.grads) or_return

	if loss_file != "" {
		log.info("loading mean losses from", loss_file)
		err = util.unmarshal_json_file(loss_file, &s.losses)
	}
	return s, err
}

delete_debug_state :: proc(s: ^Debug_State) {
	array.delete(s.x)
	array.delete(s.y)
	array.delete(s.logits)
	gpt2.delete_model(s.grads)
	delete(s.losses)
	free(s)
}
