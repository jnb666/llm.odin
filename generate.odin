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
	debug:     bool `usage:"enable debug logging"`,
	track:     bool `usage:"use tracking allocator to find memory leaks"`,
	cuda:      bool `usage:"use Cuda acceleration - default true"`,
	model:     string `usage:"model checkpoint file - default gpt2_124M.bin"`,
	prompt:    string `usage:"input prompt string"`,
	verbose:   bool `usage:"show verbose output"`,
	maxlen:    int `usage:"max number of tokens generated- defalt 256"`,
	sampler:   string `usage:"sampler type (greedy, random, top_k or top_p) - default top_p"`,
	tokenizer: string `usage:"tokenizer name (gpt2, byte) - default gpt2"`,
	temp:      f32 `usage:"sampler temerature - default 1.0"`,
	topk:      int `usage:"top k sampler cutoff - default 10"`,
	topp:      f32 `usage:"top p sampler cutoff - default 0.9"`,
	nonstop:   bool `usage:"don't stop generating text when get the end token"`,
}

Context :: struct {
	using opt: ^Generate_Options,
	tok:       nn.Tokenizer(u16),
	ntokens:   int,
}

// run session to generate sample text from the model
generate_main :: proc(args: []string) {
	opt := Generate_Options {
		model     = "gpt2_124M.bin",
		maxlen    = 256,
		tokenizer = "gpt2",
		sampler   = "top_p",
		temp      = 1.0,
		topk      = 10,
		topp      = 0.9,
		cuda      = true,
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

	ctx := Context {
		opt = opt,
		tok = new_tokenizer(opt.tokenizer),
	}
	defer nn.delete_tokenizer(&ctx.tok)

	start := time.now()
	end := gpt2.End_Token if opt.tokenizer == "gpt2" else "\n\n"
	prompt := get_prompt(opt.prompt, end, opt.debug)
	defer delete(prompt)
	tokens := nn.encode(&ctx.tok, prompt)
	defer delete(tokens)
	if opt.debug {
		fmt.printf("%v => %q\n", tokens, prompt)
	}
	stop := -1
	if end_tok, ok := ctx.tok.end_token.?; ok && !opt.nonstop {
		stop = int(end_tok)
	}
	gpt2.generate(model, sampler, &ctx, generate_callback, tokens, max_length = opt.maxlen, stop_token = stop)

	elapsed := time.duration_seconds(time.since(start))
	fmt.println()
	log.infof("Generated %d tokens in % .2fs - % .0fms/token", ctx.ntokens, elapsed, 1000 * elapsed / f64(ctx.ntokens))
}

get_prompt :: proc(s, end: string, debug: bool) -> string {
	txt, is_alloc := strings.replace_all(s, "\\n", "\n")
	if !debug {
		fmt.print(txt)
	}
	prompt := strings.concatenate({end, txt})
	if is_alloc {
		delete(txt)
	}
	return prompt
}

generate_callback :: proc(p: rawptr, token: u16, done: bool) {
	ctx := cast(^Context)p
	if end, ok := ctx.tok.end_token.?; ok && token == end {
		if ctx.debug {
			fmt.printf("[%5d] => %s\n", token, gpt2.End_Token)
		} else if !done {
			fmt.print("\n\n")
		}
	} else {
		text := nn.decode(&ctx.tok, token)
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
