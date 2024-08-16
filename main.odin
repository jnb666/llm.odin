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
import "gpt2"
import "nn"


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

// read data from file unmarshal into value pointed to by ptr
unmarshal_json_file :: proc(file_name: string, ptr: ^$T, allocator := context.allocator) -> os.Error {
	data := os.read_entire_file_or_err(file_name) or_return
	defer delete(data)
	if err := json.unmarshal(data, ptr, allocator = allocator); err != nil {
		log.panic(err)
	}
	return nil
}

round_ms :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Millisecond)
}

round_sec :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Second)
}

make_dir_if_not_exist :: proc(name: string) {
	if os.is_dir(name) {
		return
	}
	if os.exists(name) {
		log.panicf("%s is not a directory!", name)
	}
	if err := os.make_directory(name, 0o755); err != os.ERROR_NONE {
		log.panicf("error creating %s dir: %v", name, err)
	}
}

// Get file path as data/<subdir>/<name> creating dirs if needed
data_file :: proc(name, subdir: string) -> string {
	make_dir_if_not_exist("data")
	dir := filepath.join({"data", subdir})
	defer delete(dir)
	make_dir_if_not_exist(dir)
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
http_get :: proc(url, file: string) -> (status: u32, err: os.Error) {
	pid := os.fork() or_return
	if pid == 0 {
		os.execvp("curl", {"-#", "-L", "-f", "-o", file, url}) or_return
	} else {
		got_pid, child_status := wait_pid(pid) or_return
		if got_pid < 0 || got_pid != pid {
			return 0, .Unsupported
		}
		status = child_status / 256
	}
	return
}

wait_pid :: proc(pid: os.Pid) -> (os.Pid, u32, os.Error) {
	rusage: linux.RUsage
	opt: linux.Wait_Options
	id := linux.Pid(pid)
	status: u32
	pid2, err := linux.wait4(id, &status, opt, &rusage)
	return os.Pid(pid2), status, err
}
