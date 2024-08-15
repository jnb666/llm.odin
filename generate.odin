package main

import "core:fmt"
import "core:log"
import "core:os"
import "core:strings"
import "core:time"

import "gpt2"
import "nn"
import "util"


Generate_Options :: struct {
	debug:   bool `usage:"enable debug logging"`,
	track:   bool `usage:"use tracking allocator to find memory leaks"`,
	cuda:    bool `usage:"use Cuda acceleration - default true"`,
	model:   string `usage:"model checkpoint file - default gpt2_124M.bin"`,
	prompt:  string `usage:"input prompt string"`,
	verbose: bool `usage:"show verbose output"`,
	maxlen:  int `usage:"max number of tokens generated- defalt 256"`,
	sampler: string `usage:"sampler type (greedy, random, top_k or top_p) - default top_p"`,
	temp:    f32 `usage:"sampler temerature - default 1.0"`,
	topk:    int `usage:"top k sampler cutoff - default 10"`,
	topp:    f32 `usage:"top p sampler cutoff - default 0.9"`,
	nonstop: bool `usage:"don't stop generating text when get the end token"`,
}

Context :: struct {
	using opt: ^Generate_Options,
	tok:       ^gpt2.Tokenizer,
	ntokens:   int,
}

// run session to generate sample text from the model
generate_main :: proc(args: []string) {
	opt := Generate_Options {
		model   = "gpt2_124M.bin",
		maxlen  = 256,
		sampler = "top_p",
		temp    = 1.0,
		topk    = 10,
		topp    = 0.9,
		cuda    = true,
	}
	parse_args(&opt, "llm generate", args)
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
	model_file := data_file(opt.model, "snapshots")
	defer delete(model_file)
	model, err := gpt2.load_checkpoint(Device, Type, model_file)
	if err != nil {
		fatal_error("Error loading %s: %v", model_file, err)
	}
	defer gpt2.delete_model(model)
	if opt.verbose {
		nn.write_summary(stdout, &model.layer)
	}

	sampler := init_sampler(opt)
	log.infof("%.4v", sampler)

	tokenizer := gpt2.new_tokenizer()
	defer gpt2.delete_tokenizer(tokenizer)

	ctx := Context {
		opt = opt,
		tok = tokenizer,
	}
	start := time.now()
	prompt := get_prompt(opt.prompt, opt.debug)
	defer delete(prompt)
	tokens := gpt2.encode(tokenizer, prompt)
	defer delete(tokens)
	if opt.debug {
		fmt.printf("%v => %q\n", tokens, prompt)
	}
	stop := !opt.nonstop ? gpt2.End_Token_ID : -1
	gpt2.generate(model, sampler, &ctx, generate_callback, tokens, max_length = opt.maxlen, stop_token = stop)

	elapsed := time.duration_seconds(time.since(start))
	fmt.println()
	log.infof("Generated %d tokens in % .2fs - % .0fms/token", ctx.ntokens, elapsed, 1000 * elapsed / f64(ctx.ntokens))
}

get_prompt :: proc(s: string, debug: bool) -> string {
	txt, is_alloc := strings.replace_all(s, "\\n", "\n")
	if !debug {
		fmt.print(txt)
	}
	prompt := strings.concatenate({gpt2.End_Token, txt})
	if is_alloc {
		delete(txt)
	}
	return prompt
}

generate_callback :: proc(p: rawptr, token: u16, done: bool) {
	ctx := cast(^Context)p
	if token == gpt2.End_Token_ID {
		if ctx.debug {
			fmt.printf("[%5d] => %s\n", token, gpt2.End_Token)
		} else if !done {
			fmt.print("\n\n")
		}
	} else {
		text := gpt2.decode(ctx.tok, token)
		defer delete(text)
		if ctx.debug {
			fmt.printf("[%5d] => %q\n", token, text)
		} else {
			fmt.print(text)
		}
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
