package cuda

import "core:strings"

@(extra_linker_flags="-L/usr/local/cuda/lib64")
foreign import nvrtc "system:nvrtc"

Program :: distinct rawptr

NvrtcResult :: enum i32 {
	SUCCESS = 0,
	OUT_OF_MEMORY = 1,
	PROGRAM_CREATION_FAILURE = 2,
	INVALID_INPUT = 3,
	INVALID_PROGRAM = 4,
	INVALID_OPTION = 5,
	COMPILATION = 6,
	BUILTIN_OPERATION_FAILURE = 7,
	NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
	NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
	NAME_EXPRESSION_NOT_VALID = 10,
	INTERNAL_ERROR = 11,
	TIME_FILE_WRITE_FAILED = 12
}


@(default_calling_convention="c", private)
foreign nvrtc {
	nvrtcGetErrorString :: proc(rc: NvrtcResult) -> cstring ---
	nvrtcCreateProgram :: proc(prog: ^Program, src, name: cstring, numHeaders: i32, headers, includeNames: cstring)  -> NvrtcResult ---
	nvrtcDestroyProgram :: proc(prog: ^Program) -> NvrtcResult ---
	nvrtcCompileProgram :: proc(prog: Program, numOptions: i32, options: ^cstring) -> NvrtcResult ---
	nvrtcGetProgramLogSize :: proc(prog: Program, logSize: ^uint) -> NvrtcResult ---
	nvrtcGetProgramLog :: proc(prog: Program, log: cstring) -> NvrtcResult ---
	nvrtcGetPTXSize :: proc(prog: Program, ptxSize: ^uint) -> NvrtcResult ---
	nvrtcGetPTX :: proc(prog: Program, ptx: cstring) -> NvrtcResult ---
}

nvrtc_error :: proc(rc: NvrtcResult) -> string {
	return string(nvrtcGetErrorString(rc))
}

nvrtc_must :: proc(rc: NvrtcResult, loc := #caller_location) {
	if rc != .SUCCESS {
		panic(nvrtc_error(rc), loc)
	}
}

// Compile cuda source to ptx assembly. Returned string will have error log if err != .NVRTC_SUCCESS
compile_to_ptx :: proc(src: string, name: cstring, opts: ..string) -> (ptx: string, err: NvrtcResult) {
	prog: Program
	{
		csrc := strings.clone_to_cstring(src)
		defer delete(csrc)
		must(nvrtcCreateProgram(&prog, csrc, name, 0, nil, nil))
	}
	defer nvrtcDestroyProgram(&prog)
	if len(opts) > 0 {
		buf := make([]cstring, len(opts))
		defer delete(buf)
		for opt, i in opts {
			buf[i] = strings.clone_to_cstring(opt)
		}
		err = nvrtcCompileProgram(prog, i32(len(opts)), &buf[0])
		for i in 0 ..< len(opts) {
			delete(buf[i])
		}
	} else {
		err = nvrtcCompileProgram(prog, 0, nil)
	}
	size: uint
	if err != .SUCCESS {
		must(nvrtcGetProgramLogSize(prog, &size))
		logmsg := make_cstring(size)
		must(nvrtcGetProgramLog(prog, logmsg))
		return string(logmsg), err
	}
	must(nvrtcGetPTXSize(prog, &size))
	cptx := make_cstring(size)
	must(nvrtcGetPTX(prog, cptx))
	return string(cptx), .SUCCESS
}

