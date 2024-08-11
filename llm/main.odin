package main

import "core:fmt"
import "core:os"
import "core:log"
import "core:mem"
import "core:strings"
import "core:time"

import "../array"
import "../nn"
import "../util"

Array :: array.Array
BF16 :: array.BF16
CPU :: array.CPU
Cuda :: array.Cuda

stdout := os.stream_from_handle(os.stdout)

Base_Options :: struct {
	debug: bool,
	track: bool,
	cuda: bool,
}

Command :: struct{
	func: proc([]string),
	name: string,
	desc: string,
}

commands := []Command {
	{train_main, 	"train", 	"general training handler: train the model on a training data and evaluate the output"},
	{generate_main, "generate", "generate text by sampling the model output logits"},
	{test_main, 	"test", 	"run training session to compare model output with saved pytorch state"},
	{help_main, 	"help", 	"info on available commands"},
}

main :: proc() {
	context.logger = log.create_console_logger(.Info, {.Level, .Terminal_Color, .Time})
	defer log.destroy_console_logger(context.logger)
	if len(os.args) < 2 {
		help_main({})
		return
	}
	args := munge_args()
	for cmd in commands {
		if len(os.args) >= 2 && os.args[1] == cmd.name {
			cmd.func(args)
			return
		}
	}	
	help_main({})
}

// run fn with common context settings
run :: proc(fn: proc(rawptr), opt_ptr: rawptr) {
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
		c.user_ptr = nn.init_cuda(verbose=true)
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

// args => args[0]+args[1], args[2]...
munge_args :: proc() -> []string {
	args := make([]string, len(os.args)-1)
	args[0] = strings.join({os.args[0], os.args[1]}, " ")
	copy(args[1:], os.args[2:])
	return args
}

fatal_file_error :: proc(msg, file: string, err: os.Error) {
	log.fatalf("%s %s: %s", msg, file, os.error_string(err))
	os.exit(1)
}

round_ms :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Millisecond)
}

round_sec :: proc(d: time.Duration) -> time.Duration {
	return time.duration_round(d, time.Second)
}
