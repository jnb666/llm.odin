package main

import "core:fmt"
import "core:log"
import "core:os"
import "core:strings"
import "core:time"

import "array"
import "cuda"
import "gpt2"
import "nn"
import "plot"
import "util"

Plot_Width :: 1000
Plot_Height :: 1000

Train_Options :: struct {
	debug:            bool `usage:"enable debug logging"`,
	track:            bool `usage:"use tracking allocator to find memory leaks"`,
	cuda:             bool `usage:"use Cuda acceleration - default true"`,
	config:           string `usage:"initialize new model with config from json file"`,
	model:            string `usage:"model checkpoint file - default gpt2_124M.bin"`,
	dataset:          string `usage:"dataset file name"`,
	verbose:          bool `usage:"show verbose output"`,
	steps:            int `usage:"number of training steps - default 50"`,
	val_steps:        int `usage:"limit on number of validation batches- default 20"`,
	val_every:        int `usage:"check validation loss every n steps - default 20"`,
	sample_every:     int `usage:"sample output every n steps - default 100"`,
	save_every:       int `usage:"save checkpoint every n steps - default 100"`,
	tokenizer:        string `usage:"tokenizer name (gpt2, byte) - default gpt2"`,
	sample_len:       int `usage:"length of sample text - default 256"`,
	temperature:      f32 `usage:"sampler temperature parameter - default 1.0"`,
	top_p:            f32 `usage:"sampler top_p parameter - default 0.9"`,
	top_k:            int `usage:"sampler top_k parameter"`,
	batch:            int `usage:"number of samples in each training batch - default 4"`,
	seq_len:          int `usage:"sequence length of each training sample - default 1024"`,
	learn_rate:       f32 `usage:"AdamW learning rate parameter - default 3e-4"`,
	cosine_decay:     bool `usage:"Enable cosine decay of learning rate"`,
	warmup_steps:     int `usage:"Number of warmup steps if cosine decay enabled"`,
	warmup_target:    f32 `usage:"Learning rate after warmup if cosine decay warmup enabled"`,
	final_learn_rate: f32 `usage:"Final rate after cosine decay"`,
	weight_decay:     f32 `usage:"AdamW weight decay parameter"`,
	grad_clip:        f32 `usage:"AdamW gradient clip parameter"`,
	beta1:            f32 `usage:AdamW beta1 parameter - default 0.9"`,
	beta2:            f32 `usage:AdamW beta2 parameter - default 0.95"`,
	recompute:        bool `usage:"recompute Gelu activations to save memory - default true"`,
	nonstop:          bool `usage:"don't stop generating text when get the end token"`,
	plot:             bool `usage:"plot loss values to a webview window"`,
}

// run test training session to compare model outputs with saved pytorch reference
train_main :: proc(args: []string) {
	opt := Train_Options {
		model        = "gpt2_124M.bin",
		tokenizer    = "gpt2",
		steps        = 50,
		val_every    = 20,
		val_steps    = 20,
		save_every   = 100,
		batch        = 4,
		seq_len      = 1024,
		learn_rate   = 3e-4,
		beta1        = 0.9,
		beta2        = 0.95,
		sample_every = 100,
		sample_len   = 256,
		temperature  = 1.0,
		top_p        = 0.9,
		cuda         = true,
		recompute    = true,
	}
	parse_args(&opt, "llm train", args)
	run(train_run, &opt)
}

train_run :: proc(opt_ptr: rawptr) {
	opt := cast(^Train_Options)opt_ptr
	log.debugf("\n%v", opt)
	if opt.cuda {
		train_start(Cuda, BF16, opt)
	} else {
		train_start(CPU, f32, opt)
	}
}


train_start :: proc($Device, $Type: typeid, opt: ^Train_Options) {
	// load and initialise model
	model, tokenizer := load_model(Device, Type, opt)
	defer gpt2.delete_model(model)
	defer nn.delete_tokenizer(&tokenizer)
	gpt2.build(model, opt.batch, opt.seq_len)
	if opt.verbose {
		nn.write_summary(stdout, &model.layer)
	}

	// load training and validation data
	train_data := load_dataset(Device, opt, train = true)
	defer nn.delete_dataset(train_data)
	test_data := load_dataset(Device, opt, train = false)
	defer nn.delete_dataset(test_data)

	// initialize optimizer
	adamw := nn.new_optimizer(
		&model.layer,
		learning_rate = opt.learn_rate,
		weight_decay = opt.weight_decay,
		gradient_clip = opt.grad_clip,
		beta1 = opt.beta1,
		beta2 = opt.beta2,
	)
	log.infof("%.4v", adamw.config)
	defer nn.delete_optimizer(adamw)
	decay: nn.Cosine_Decay
	if opt.cosine_decay {
		decay = {
			initial_rate  = opt.learn_rate,
			final_rate    = opt.final_learn_rate,
			decay_steps   = opt.steps - opt.warmup_steps,
			warmup_target = opt.warmup_target,
			warmup_steps  = opt.warmup_steps,
		}
		log.infof("%.4v", decay)
	}

	// GUI - plotting
	stats: plot.Stats
	defer plot.delete_stats(&stats)
	view := opt.plot ? plot.start(Plot_Width, Plot_Height, xrange = {opt.steps + 1}) : nil

	// main training loop
	start_run := time.now()
	start_check := time.now()
	mean_norm, mean_loss: util.Running_Mean
	checkpoint_file := checkpoint_filename(opt.model, opt.dataset)
	defer delete(checkpoint_file)
	prompt := [1]u16{opt.tokenizer == "gpt2" ? gpt2.End_Token_ID : '\n'}

	for step in 1 ..= opt.steps {
		start_step := time.now()
		loss, norm := train_step(model, adamw, train_data)
		util.add_mean(&mean_loss, loss)
		util.add_mean(&mean_norm, norm)

		if step == opt.steps || step % opt.val_every == 0 {
			val_loss := calc_validation_loss(model, test_data, opt.val_steps)
			elapsed := time.since(start_check) / time.Duration(opt.val_every)
			log_stats(Device, &stats, step, opt.steps, elapsed, adamw.learning_rate, loss, mean_norm.value, mean_loss.value, val_loss)
			mean_norm, mean_loss = {}, {}
			start_check = time.now()
		} else {
			log_stats(Device, &stats, step, opt.steps, time.since(start_step), adamw.learning_rate, loss, mean_norm.value, mean_loss.value)
		}
		if opt.plot {
			if err := plot.update_plot(view, &stats); err != nil {
				log.error("Error updating plot:", err)
			}
		}
		if step == opt.steps || step % opt.sample_every == 0 {
			sample_text(model, &tokenizer, prompt[:], opt, step, &stats)
			if opt.plot {
				if err := plot.update_table(view, &stats); err != nil {
					log.error("Error updating table:", err)
				}
			}
		}
		if step == opt.steps || step % opt.save_every == 0 {
			if err := gpt2.save_checkpoint(model, checkpoint_file); err != nil {
				log.error(err)
			}
		}
		if opt.cosine_decay {
			nn.decay_learning_rate(&decay, step, &adamw.learning_rate)
		}
	}

	run_elapsed := round_sec(time.since(start_run))
	log.infof("total elapsed time = %v", run_elapsed)
	if opt.plot {
		plot.wait(view)
	}
}

load_model :: proc($Device, $Type: typeid, opt: ^Train_Options) -> (model: ^gpt2.Model(Device, Type), tok: nn.Tokenizer(u16)) {
	err: os.Error
	tok = new_tokenizer(opt.tokenizer)
	if opt.config != "" {
		cfg: gpt2.Config
		err = unmarshal_json_file(opt.config, &cfg)
		if err != nil {
			fatal_error("Error loading config %s: %v", opt.config, err)
		}
		cfg.max_seq = opt.seq_len
		cfg.vocab_size = tok.vocab_size
		cfg.recompute_gelu = opt.recompute
		gpt2.pad_vocab(&cfg)
		log.info(cfg)
		model = gpt2.new_model(Device, Type, cfg)
		gpt2.init_weights(model)
	} else {
		model_file := data_file(opt.model, "snapshots")
		defer delete(model_file)
		model, err = gpt2.load_checkpoint(Device, Type, model_file)
		if err != nil {
			fatal_error("Error loading model %s: %v", model_file, err)
		}
		if opt.seq_len > model.max_seq {
			fatal_error("requested seqlen=%d is greater than model max_seq=%d", opt.seq_len, model.max_seq)
		}
		model.recompute_gelu = opt.recompute
		log.info(model.config)
	}
	return model, tok
}

log_stats :: proc($Device: typeid, s: ^plot.Stats, step, steps: int, elapsed: time.Duration, learn_rate, batch_loss, norm, loss: f32, val_loss: f32 = -1) {
	fmt.printf("\rstep % 3d/% 3d: lr % 7.2e  norm % 6.2f  train loss % 6.3f  ", step, steps, learn_rate, norm, loss)
	buf: [64]u8
	gpu_mem := max_device_memory_used(Device, buf[:])
	plot.add(s, "training loss", step, batch_loss, width = 1, opacity = 0.5)
	if val_loss >= 0 {
		fmt.printf("val loss % 6.3f  %s  %v  \n", val_loss, gpu_mem, round_ms(elapsed))
		plot.add(s, "avg training loss", step, loss)
		plot.add(s, "avg validation loss", step, val_loss)
	} else {
		fmt.printf("%17s%s  %v  ", "", gpu_mem, round_ms(elapsed))
	}
}

Sample_Context :: struct {
	tok: ^nn.Tokenizer(u16),
	buf: strings.Builder,
}

sample_text :: proc(model: ^gpt2.Model($D, $T), tokenizer: ^nn.Tokenizer(u16), prompt: []u16, opt: ^Train_Options, step: int, s: ^plot.Stats) {
	log.debugf("sample text: step=%d", step)
	sampler := nn.Sampler {
		temperature = opt.temperature,
		top_k       = opt.top_k,
		top_p       = opt.top_p,
	}
	stop := -1
	if end_tok, ok := tokenizer.end_token.?; ok && !opt.nonstop {
		stop = int(end_tok)
	}
	ctx := Sample_Context {
		tok = tokenizer,
	}
	gpt2.generate(model, sampler, &ctx, sample_callback, prompt, max_length = opt.sample_len, stop_token = stop)
	text := strings.to_string(ctx.buf)
	plot.add_sample(s, step, strings.trim_space(text))
	if !opt.plot {
		fmt.print("\n", text, "\n\n", sep = "")
	}
	strings.builder_destroy(&ctx.buf)
}

sample_callback :: proc(p: rawptr, token: u16, done: bool) {
	ctx := cast(^Sample_Context)p
	if end, ok := ctx.tok.end_token.?; ok && token == end {
		fmt.sbprint(&ctx.buf, "\n\n")
	} else {
		text := nn.decode(ctx.tok, token)
		fmt.sbprint(&ctx.buf, text)
		delete(text)
	}
}

train_step :: proc(model: ^gpt2.Model($D, $T), adamw: ^nn.AdamW(D, T), data: ^nn.Dataset(D)) -> (loss, norm: f32) {
	nn.next_batch(data)
	gpt2.forward(model, data.inputs, train = true)
	loss = gpt2.calc_loss(model, data.targets)
	nn.zero_gradients(&model.layer)
	gpt2.backward(model, data.inputs)
	norm = nn.optimizer_step(adamw, model)
	return loss, norm
}

calc_validation_loss :: proc(model: ^gpt2.Model($D, $T), data: ^nn.Dataset(D), max_steps: int) -> f32 {
	mean_loss: util.Running_Mean
	data.offset = 0
	step := 1
	for {
		if !nn.next_batch(data) {
			break
		}
		gpt2.forward(model, data.inputs, train = false)
		loss := gpt2.calc_loss(model, data.targets)
		util.add_mean(&mean_loss, loss)
		step += 1
		if max_steps > 0 && step >= max_steps {
			break
		}
	}
	return mean_loss.value
}

load_dataset :: proc($Device: typeid, opt: ^Train_Options, train: bool) -> ^nn.Dataset(Device) {
	name := strings.concatenate({opt.dataset, train ? "_train.bin" : "_val.bin"})
	defer delete(name)
	file_name := data_file(name, "datasets")
	defer delete(file_name)
	ds, err := nn.read_dataset(Device, file_name, opt.batch, opt.seq_len, shuffle = train)
	if err != nil {
		fatal_error("Error loading %s: %v", file_name, err)
	}
	return ds
}

checkpoint_filename :: proc(model, dataset: string) -> string {
	model := model
	if strings.has_suffix(model, ".bin") {
		model = model[:len(model) - 4]
	}
	file_name := strings.concatenate({model, "_", dataset, ".bin"})
	defer delete(file_name)
	return data_file(file_name, "snapshots")
}

max_device_memory_used :: proc($Device: typeid, buf: []u8) -> string {
	when Device == Cuda {
		dev := cuda.get_device()
		used := cuda.get_mempool_attribute(dev, .USED_MEM_HIGH)
		cuda.set_mempool_attribute(dev, .USED_MEM_HIGH, 0)
		return fmt.bprintf(buf, "%d MB", used / (1024 * 1024))
	} else {
		return ""
	}
}
