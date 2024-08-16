package main

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:mem"
import "core:os"
import "core:strings"

import "nn"
import "util"


Prepare_Options :: struct {
	debug:   bool `usage:"enable debug logging"`,
	track:   bool `usage:"use tracking allocator to find memory leaks"`,
	cuda:    bool `usage:"use Cuda acceleration"`,
	config:  string `usage:"config json file with dataset definitions"`,
	dataset: string `args:"required" usage:"name of dataset to encode"`,
}

Dataset_Config :: struct {
	train_source:     string,
	train_file:       string,
	val_source:       string,
	val_file:         string,
	val_tokens:       int,
	record_separator: string,
	trim_space:       bool,
	format:           string,
	template:         string,
	tokenizer:        string,
}

// run session to encode and save tokens to dataset file
prepare_main :: proc(args: []string) {
	opt := Prepare_Options {
		config = "datasets.json",
	}
	parse_args(&opt, "llm prepare", args)
	run(prepare_run, &opt)
}

prepare_run :: proc(opt_ptr: rawptr) {
	opt := cast(^Prepare_Options)opt_ptr
	log.debugf("\n%v", opt)
	config := make(map[string]Dataset_Config)
	defer delete(config)
	err := unmarshal_json_file(opt.config, &config)
	if err != nil {
		fatal_error("error reading file: %s", opt.config)
	}
	cfg, ok := config[opt.dataset]
	if !ok {
		fatal_error("dataset %s not defined in config", opt.dataset)
	}
	if cfg.tokenizer == "" {
		cfg.tokenizer = "gpt2"
	}
	prepare_dataset(opt.dataset, cfg)
}

prepare_dataset :: proc(dataset_name: string, cfg: Dataset_Config) {
	log.debug(cfg)
	tok := new_tokenizer(cfg.tokenizer)
	defer nn.delete_tokenizer(&tok)

	train_file := data_file(cfg.train_file, "datasets")
	train_data := download_file(train_file, cfg.train_source)
	log.info("tokenize", train_file)
	delete(train_file)
	train_tokens := tokenize(&tok, train_data, cfg)
	delete(train_data)

	if cfg.val_file != "" {
		write_tokens(dataset_name, train_tokens, train = true)
		delete(train_tokens)
		val_file := data_file(cfg.val_file, "datasets")
		val_data := download_file(val_file, cfg.val_source)
		log.info("tokenize", val_file)
		delete(val_file)
		val_tokens := tokenize(&tok, val_data, cfg)
		delete(val_data)
		write_tokens(dataset_name, val_tokens, train = false)
		delete(val_tokens)

	} else if cfg.val_tokens > 0 {
		write_tokens(dataset_name, train_tokens[:cfg.val_tokens], train = false)
		write_tokens(dataset_name, train_tokens[cfg.val_tokens:], train = true)
		delete(train_tokens)

	} else {
		write_tokens(dataset_name, train_tokens, train = true)
		delete(train_tokens)
	}
}

write_tokens :: proc(dataset_name: string, tokens: []u16, train: bool) {
	file_name := strings.concatenate({dataset_name, train ? "_train.bin" : "_val.bin"})
	defer delete(file_name)
	out_file := data_file(file_name, "datasets")
	defer delete(out_file)
	err := nn.write_dataset(out_file, tokens)
	if err != nil {
		fatal_error("error writing %s: %v", out_file, err)
	}
}

tokenize :: proc(tok: ^nn.Tokenizer(u16), data: string, cfg: Dataset_Config) -> []u16 {
	sep: [dynamic]u16
	defer delete(sep)
	if end, ok := tok.end_token.?; ok {
		append(&sep, end)
	} else {
		nn.encode_to(tok, cfg.record_separator, &sep)
	}
	log.info("output record separator:", sep)

	switch cfg.format {
	case "jsonl":
		if cfg.template == "" {
			fatal_error("template option is required for jsonl data")
		}
		data, err := tokenize_jsonl(tok, data, cfg, sep[:])
		if err != nil {
			fatal_error("json parse error: %v", err)
		}
		return data
	case "text", "":
		return tokenize_text(tok, data, cfg, sep[:])
	case:
		fatal_error("invalid dataset format: %s", cfg.format)
	}
}

tokenize_text :: proc(tok: ^nn.Tokenizer(u16), text: string, cfg: Dataset_Config, sep: []u16) -> []u16 {
	text := text
	tokens: [dynamic]u16
	append(&tokens, ..sep)
	n := 1
	done := 0
	length := len(text)
	for record in strings.split_iterator(&text, cfg.record_separator) {
		r := record
		if cfg.trim_space {
			r = strings.trim_space(r)
		}
		nn.encode_to(tok, r, &tokens)
		append(&tokens, ..sep)
		if n % 1000 == 0 {
			fmt.printf("\rtokenize record %d - %.0f%% done", n, 100 * f64(done) / f64(length))
		}
		n += 1
		done += len(record) + len(cfg.record_separator)
	}
	fmt.printf("\rtokenized %d records%20s\n", n, "")
	return tokens[:]
}

tokenize_jsonl :: proc(tok: ^nn.Tokenizer(u16), data: string, cfg: Dataset_Config, sep: []u16) -> ([]u16, json.Error) {
	text := data
	tokens: [dynamic]u16
	append(&tokens, ..sep)
	n := 1
	done := 0
	length := len(text)
	for record in strings.split_lines_iterator(&text) {
		json_data, err := json.parse(transmute([]u8)record, allocator = context.temp_allocator)
		if err != nil {
			return nil, err
		}
		root, ok := json_data.(json.Object)
		if !ok {
			return nil, .Unexpected_Token
		}
		r := make(map[string]string, allocator = context.temp_allocator)
		for key, val in root {
			r[key] = fmt.aprint(val, allocator = context.temp_allocator)
		}
		text := util.parse_template(cfg.template, r, allocator = context.temp_allocator)
		if cfg.trim_space {
			text = strings.trim_space(text)
		}
		nn.encode_to(tok, text, &tokens)
		append(&tokens, ..sep)
		if n % 1000 == 0 {
			fmt.printf("\rtokenize record %d - %.0f%% done", n, 100 * f64(done) / f64(length))
		}
		n += 1
		done += len(record) + 1
		mem.free_all(context.temp_allocator)
	}
	fmt.printf("\rtokenized %d records%20s\n", n, "")
	return tokens[:], nil
}
