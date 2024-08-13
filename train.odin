package main

import "core:fmt"
import "core:log"
import "core:os"
import "core:time"
import "core:strings"

import "gpt2"
import "nn"
import "array"
import "util"
import "cuda"


Train_Options :: struct{
	debug: bool `usage:"enable debug logging"`,
	track: bool `usage:"use tracking allocator to find memory leaks"`,
	cuda: bool `usage:"use Cuda acceleration"`,
	model: string `usage:"model checkpoint file"`,
	dataset: string `usage:"dataset file name"`,
	verbose: bool `usage:"show verbose output"`,
	steps: int `usage:"number of training steps"`,
	val_steps: int `usage:"limit on number of validation batches"`,
	val_every: int `usage:"check validation loss every n steps"`,
	sample_every: int `usage:"sample output every n steps"`,
	save_every: int `usage:"save checkpoint every n steps"`,
	sample_len: int `usage:"length of sample text"`,
	temperature: f32 `usage:"sampler temperature parameter"`,
	top_p: f32 `usage:"sampler top_p parameter"`,
	top_k: int `usage:"sampler top_k parameter"`,
	batch: int `usage:"number of samples in each training batch"`,
	seq_len: int `usage:"sequence length of each training sample"`,
	learn_rate: f32 `usage:"AdamW learning rate parameter"`,
	weight_decay: f32 `usage:"AdamW weight decay parameter"`,
	grad_clip: f32 `usage:"AdamW gradient clip parameter"`,
}

// run test training session to compare model outputs with saved pytorch reference
train_main :: proc(args: []string) {
	opt := Train_Options{
		model 	    = "gpt2_124M.bin",
		steps       = 50,
		val_every   = 20,
		val_steps   = 20,
		save_every  = 100,
		batch       = 4,
		seq_len     = 1024,
		learn_rate  = 3e-4,
		sample_every= 100,
		sample_len  = 256,
		temperature = 1.0,
		top_p       = 0.9,
		cuda        = true,
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
	model_file := data_file(opt.model, "snapshots")
	model, err := gpt2.load_checkpoint(Device, Type, model_file)
	if err != nil {
		fatal_error("Error loading %s: %v", model_file, err)
	}
	delete(model_file)
	defer gpt2.delete_model(model)
	if opt.seq_len > model.max_seq {
		log.fatalf("requested seqlen=%d is greater than model max_seq=%d", opt.seq_len, model.max_seq)
		os.exit(1)
	}
	gpt2.build(model, opt.batch, opt.seq_len)
	if opt.verbose {
		nn.write_summary(stdout, &model.layer)
	}
	tokenizer := gpt2.new_tokenizer()
	defer gpt2.delete_tokenizer(tokenizer)

	// load training and validation data
	if opt.dataset == "" {
		log.fatal("dataset parameter is required")
		os.exit(1)
	}
	train_data := load_dataset(Device, opt, train=true)
	defer nn.delete_dataset(train_data)

	test_data := load_dataset(Device, opt, train=false)
	defer nn.delete_dataset(test_data)

	// initialize optimizer
	adamw := nn.new_optimizer(&model.layer, learning_rate=opt.learn_rate, weight_decay=opt.weight_decay, gradient_clip=opt.grad_clip)
	log.infof("%.4v", adamw.config)
	defer nn.delete_optimizer(adamw)

	// main training loop
	start_run := time.now()
	start_check := time.now()
	mean_norm, mean_loss: util.Running_Mean
	checkpoint_file := checkpoint_filename(opt.model, opt.dataset)
	defer delete(checkpoint_file)
	buf: [64]u8

	for step in 1 ..= opt.steps {
		start_step := time.now()
		loss, norm := train_step(model, adamw, train_data)
		util.add_mean(&mean_loss, loss)
		util.add_mean(&mean_norm, norm)

		fmt.printf("\rstep % 3d/% 3d: norm % 6.2f  train loss % 6.3f  ", step, opt.steps, mean_norm.value, mean_loss.value)
		gpu_mem := max_device_memory_used(Device, buf[:])
		if step == opt.steps || step % opt.val_every == 0 {
			val_loss := calc_validation_loss(model, test_data, opt.val_steps)
			elapsed := time.since(start_check) / time.Duration(opt.val_every)
			fmt.printf("val loss % 6.3f  %s  %v  \n", val_loss, gpu_mem, round_ms(elapsed))
			mean_norm, mean_loss = {}, {}
			start_check = time.now() 
		} else {
			fmt.printf("%17s%s  %v  ", "", gpu_mem, round_ms(time.since(start_step)))
		}
		if step == opt.steps || step % opt.sample_every == 0 {
			sample_text(model, tokenizer, opt)
		}
		if step == opt.steps || step % opt.save_every == 0 {
			if err := gpt2.save_checkpoint(model, checkpoint_file); err != nil {
				log.error(err)
			}
		}
	}

	run_elapsed := round_sec(time.since(start_run))
	log.infof("total elapsed time = %v", run_elapsed)
}

sample_text :: proc(model: ^gpt2.Model($D, $T), tokenizer: ^gpt2.Tokenizer, opt: ^Train_Options) {
	fmt.println()
	tokenizer := tokenizer
	sampler := nn.Sampler{ temperature=opt.temperature, top_k=opt.top_k, top_p=opt.top_p }
	prompt := [1]u16{gpt2.End_Token_ID}
	gpt2.generate(model, sampler, tokenizer, sample_callback, prompt[:], max_length=opt.sample_len)
	fmt.println("\n")
}

sample_callback :: proc(p: rawptr, token: u16, done: bool) {
	tok := cast(^gpt2.Tokenizer)p
	if token == gpt2.End_Token_ID {
		fmt.printf("\n\n")
	} else {
		text := gpt2.decode(tok, token)
		fmt.print(text)
		delete(text)
	}
}

train_step :: proc(model: ^gpt2.Model($D, $T), adamw: ^nn.AdamW(D, T), data: ^nn.Dataset(D)) -> (loss, norm: f32) {
	nn.next_batch(data)
	gpt2.forward(model, data.inputs, train=true)
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
		gpt2.forward(model, data.inputs, train=false)
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
	name := strings.concatenate( {opt.dataset, train ? "_train.bin" : "_val.bin"} )
	defer delete(name)
	file_name := data_file(name, "datasets")
	defer delete(file_name)
	ds, err := nn.read_dataset(Device, file_name, opt.batch, opt.seq_len, shuffle=train)
	if err != nil {
		fatal_error("Error loading %s: %v", file_name, err)
	}
	return ds
}

checkpoint_filename :: proc(model, dataset: string) -> string {
	model := model
	if strings.has_suffix(model, ".bin") {
		model = model[:len(model)-4]
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
		return fmt.bprintf(buf, "%d MB", used/(1024*1024))
	} else {
		return ""
	}
}