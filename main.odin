package main

import "core:encoding/json"
import "core:flags"
import "core:fmt"
import "core:log"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:sys/linux"
import "core:time"

import "../back"
import "array"
import "cuda"
import "gpt2"
import "nn"
import "util"


Array :: array.Array
BF16 :: array.BF16
CPU :: array.CPU
Cuda :: array.Cuda

stdout := os.stream_from_handle(os.stdout)
stderr := os.stream_from_handle(os.stderr)

Base_Options :: struct {
	debug: bool,
	track: bool,
	cuda:  bool,
}

Command :: struct {
	func: proc(_: []string),
	name: string,
	desc: string,
}

commands := []Command {
	{prepare_main, "prepare", "download sample datasets and encode tokens"},
	{train_main, "train", "general training handler: train the model on a training data and evaluate the output"},
	{generate_main, "generate", "generate text by sampling the model output logits"},
	{test_main, "test", "run training session to compare model output with saved pytorch state"},
	{help_main, "help", "info on available commands"},
}

main :: proc() {
	when ODIN_DEBUG {
		context.assertion_failure_proc = back.assertion_failure_proc
		back.register_segfault_handler()
	}
	context.logger = log.create_console_logger(.Info, {.Level, .Terminal_Color, .Time})
	defer log.destroy_console_logger(context.logger)
	if len(os.args) < 2 {
		help_main({})
		return
	}
	for cmd in commands {
		if len(os.args) >= 2 && os.args[1] == cmd.name {
			cmd.func(os.args[2:])
			return
		}
	}
	help_main({})
}

// run fn with common context settings
run :: proc(fn: proc(_: rawptr), opt_ptr: rawptr) {
	c := context
	opt := cast(^Base_Options)opt_ptr
	if opt.debug {
		c.logger.lowest_level = .Debug
	}
	track: mem.Tracking_Allocator
	if opt.track {
		log.info("Using tracking allocator to check for leaks")
		mem.tracking_allocator_init(&track, c.allocator)
		c.allocator = mem.tracking_allocator(&track)
	}
	if opt.cuda {
		c.user_ptr = nn.init_cuda(verbose = true)
	}
	context = c
	fn(opt)
	if opt.cuda {
		nn.end_cuda()
	}
	if opt.track {
		for _, leak in track.allocation_map {
			fmt.printf("[ERROR] %v leaked %m\n", leak.location, leak.size)
		}
		for bad_free in track.bad_free_array {
			fmt.printf("[ERROR] %v allocation %p was freed badly\n", bad_free.location, bad_free.memory)
		}
		mem.tracking_allocator_destroy(&track)
	}
}

help_main :: proc(args: []string) {
	fmt.eprintln("Usage: llm <cmd> [args...]\n\nAvailable commands:")
	for cmd in commands {
		fmt.eprintln("  ", cmd.name, "\t", cmd.desc)
	}
}

parse_args :: proc(model: ^$T, program: string, args: []string) {
	err := flags.parse(model, args, .Unix)
	if err == nil {
		return
	}
	if _, ok := err.(flags.Help_Request); ok {
		flags.write_usage(stderr, T, program, .Unix)
		os.exit(0)
	} else {
		flags.print_errors(T, err, program, .Unix)
		flags.write_usage(stderr, T, program, .Unix)
		os.exit(1)
	}
}

fatal_error :: proc(format: string, args: ..any) -> ! {
	log.fatalf(format, ..args)
	os.exit(1)
}

// Load model config and weights from preset or snapshot
load_model :: proc($Device, $Type: typeid, preset, snapshot: string, seq_len: int) -> (model: ^gpt2.Model(Device, Type)) {
	err: os.Error
	start := time.now()
	if preset != "" {
		model, err = gpt2.load_huggingface(Device, Type, preset)
		if err != nil {
			fatal_error("Error loading preset %s: %v", preset, err)
		}
	} else {
		model_file := data_file(snapshot, "snapshots")
		defer delete(model_file)
		model, err = gpt2.load_checkpoint(Device, Type, model_file)
		if err != nil {
			fatal_error("Error loading %s: %v", model_file, err)
		}
	}
	if seq_len > model.max_seq {
		fatal_error("requested seq_len=%d is greater than model max_seq=%d", seq_len, model.max_seq)
	}
	log.infof("time to load model = %.2fs ", time.duration_seconds(time.since(start)))
	return model
}

// get named tokenizer
new_tokenizer :: proc(name: string) -> nn.Tokenizer(u16) {
	switch name {
	case "gpt2":
		return gpt2.tokenizer()
	case "byte":
		return nn.byte_tokenizer()
	case:
		fatal_error("unknown tokenizer: %s", name)
	}
}

round_ms :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Millisecond)
}

round_sec :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Second)
}

// Get file path as data/<subdir>/<name> creating dirs if needed
data_file :: proc(name, subdir: string) -> string {
	util.make_dir_if_not_exist("data")
	dir := filepath.join({"data", subdir})
	defer delete(dir)
	util.make_dir_if_not_exist(dir)
	return filepath.join({dir, name})
}

// get file via http if not already cached
download_file :: proc(file, url: string) -> string {
	if !os.exists(file) {
		log.infof("downloading %s from %s", file, url)
		status, err := http_get(url, file)
		if err != nil {
			fatal_error("http_get: error launching curl process: %v", err)
		}
		if status != 0 {
			fatal_error("http_get: curl error %d", status)
		}
	}
	data, err := os.read_entire_file_or_err(file)
	if err != nil {
		fatal_error("error reading %s: %v", file, err)
	}
	return string(data)
}

// fork curl process to download a file - will only work on Linux
// if no system error then returns the status code from the curl child process
http_get :: proc(url, file: string) -> (status: i32, err: os.Error) {
	status = util.system({"curl", "-#", "-L", "-f", "-o", file, url}) or_return
	return status / 256, nil
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
