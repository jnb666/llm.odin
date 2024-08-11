package main

import "core:flags"
import "core:log"
import "core:fmt"
import "core:time"
import "core:strings"
import "core:os"

import "../gpt2"
import "../nn"
import "../util"


Generate_Options :: struct {
	debug: bool `usage:"enable debug logging"`,
	track: bool `usage:"use tracking allocator to find memory leaks"`,
	cuda: bool `usage:"use Cuda acceleration"`,
	model: string `usage:"model checkpoint file"`,
	prompt: string `usage:"input prompt string"`,
	verbose: bool `usage:"show verbose output"`,
	maxlen: int `usage:"max number of tokens generated"`,
	sampler: string `usage:"sampler type (greedy, random, top_k or top_p)"`,
	temp: f32 `usage:"sampler temerature"`,
	topk: int `usage:"top k sampler cutoff"`,
	topp: f32 `usage:"top p sampler cutoff"`,
}

Context :: struct {
	using opt: ^Generate_Options,
	tok: ^gpt2.Tokenizer,
	ntokens: int,
}

// run session to generate sample text from the model
generate_main :: proc(args: []string) {
	opt := Generate_Options{
		model 		= "gpt2_124M.bin",
		maxlen		= 256,
		sampler     = "top_p",
		temp        = 1.0,
		topk        = 10,
		topp        = 0.9,
	}
	flags.parse_or_exit(&opt, args, .Unix)
	run(generate_run, &opt)
}

generate_run :: proc(opt_ptr: rawptr) {
	opt := cast(^Generate_Options)opt_ptr
	log.debugf("\n%v", opt)
	if opt.cuda {
		generate_start(Cuda, BF16, opt)
	} else {
		generate_start(CPU, f32, opt)
	}
}

generate_start :: proc($Device, $Type: typeid, opt: ^Generate_Options) {
	model_file := util.must_get_cache_file_path(opt.model)
	defer delete(model_file)
	model, err := gpt2.load_checkpoint(Device, Type, model_file)
	if err != nil {
		fatal_file_error("Error loading model", model_file, err)
	}
	defer gpt2.delete_model(model)
	if opt.verbose {
		nn.write_summary(stdout, &model.layer)
	}

	sampler := init_sampler(opt)
	log.infof("%.4v", sampler)

	tokenizer := gpt2.new_tokenizer()
	defer gpt2.delete_tokenizer(tokenizer)
	stop_token := int(tokenizer.encoder[gpt2.End_Token])

	prompt := strings.concatenate({gpt2.End_Token, opt.prompt})
	defer delete(prompt)
	tokens := gpt2.encode(tokenizer, prompt)
	defer delete(tokens)

	ctx := Context{opt=opt, tok=tokenizer}
	start := time.now()
	log.debugf("calling generate with prompt=%v max_length=%d stop_token=%d", prompt, opt.maxlen, stop_token)
	if opt.debug {
		fmt.printf("%v => %s\n", tokens, prompt)
	} else {
		fmt.printf(opt.prompt)
	}
	gpt2.generate(model, sampler, &ctx, generate_callback, tokens, max_length=opt.maxlen, stop_token=stop_token)

	elapsed := time.duration_seconds(time.since(start))
	fmt.println()
	log.infof("Generated %d tokens in % .2fs - % .0fms/token", ctx.ntokens, elapsed, 1000*elapsed/f64(ctx.ntokens))
}

generate_callback :: proc(p: rawptr, token: u16, done: bool) {
	ctx := cast(^Context)p
	text := gpt2.decode(ctx.tok, token)
	defer delete(text)
	if ctx.debug {
		fmt.printf("[%5d] => %q\n", token, text)
	} else {
		fmt.print(text)
	}
	ctx.ntokens += 1
}

init_sampler :: proc(opt: ^Generate_Options) -> (s: nn.Sampler) {
	switch opt.sampler {
	case "greedy":
	case "random":
		s.temperature = opt.temp
	case "top_k":
		s.temperature = opt.temp
		s.top_k = opt.topk
	case "top_p":
		s.temperature = opt.temp
		s.top_p = opt.topp
	case:
		log.fatal("invalid sampler type", opt.sampler)
		os.exit(1)
	}
	return s
}






