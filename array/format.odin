package array

import "base:runtime"
import "core:fmt"
import "core:io"
import "core:slice"

// Parameters for array format summarization
Edgeitems: int = 5
Threshold: int = 100
Default_Width: int = 10
Default_Precision: int = 3

@(init)
init_formatters :: proc() {
	fmt.register_user_formatter(BF16, fmt_bfloat16)
	fmt.register_user_formatter(Shape, fmt_shape)
	fmt.register_user_formatter(Array(CPU, i32), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cpu_array_fmt(i32, fi, arg, verb)})
	fmt.register_user_formatter(Array(CPU, f32), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cpu_array_fmt(f32, fi, arg, verb)})
	fmt.register_user_formatter(Array(CPU, BF16), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cpu_array_fmt(BF16, fi, arg, verb)})
	fmt.register_user_formatter(Array(Cuda, i32), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cuda_array_fmt(i32, fi, arg, verb)})
	fmt.register_user_formatter(Array(Cuda, f32), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cuda_array_fmt(f32, fi, arg, verb)})
	fmt.register_user_formatter(Array(Cuda, BF16), proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {return cuda_array_fmt(BF16, fi, arg, verb)})
}

fmt_bfloat16 :: proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {
	v := bf16_to_f32(arg.(BF16))
	fmt.fmt_float(fi, f64(v), 32, verb)
	return true
}

fmt_shape :: proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {
	v := arg.(Shape)
	_putc(fi, '[')
	for dim, i in shape(&v) {
		if i > 0 do _putc(fi, ' ')
		fmt.fmt_int(fi, u64(dim), false, size_of(u32), 'd')
	}
	_putc(fi, ']')
	return true
}

Array_Fmt_State :: struct {
	fi:        ^fmt.Info,
	verb:      rune,
	indent:    int,
	newline:   int,
	summarize: bool,
}

cuda_array_fmt :: proc($T: typeid, fi: ^fmt.Info, arg: any, verb: rune) -> bool {
	a := to_host(arg.(Array(Cuda, T)))
	defer delete(a)
	_puts(fi, "Cuda")
	return array_fmt(a, fi, verb)
}

cpu_array_fmt :: proc($T: typeid, fi: ^fmt.Info, arg: any, verb: rune) -> bool {
	a := arg.(Array(CPU, T))
	_puts(fi, "CPU")
	return array_fmt(a, fi, verb)
}

array_fmt :: proc(a: Array(CPU, $T), fi: ^fmt.Info, verb: rune) -> bool {
	fi2 := fi^
	fi2.state = {}
	_puts(fi, "Array<")
	info: ^runtime.Type_Info
	info = type_info_of(T)
	fmt.fmt_value(&fi2, info, 's')
	_putc(fi, '>')
	if a.dptr == nil {
		_puts(fi, "(nil)")
	} else {
		fmt.fmt_value(&fi2, a.shape, 's')
		_putc(fi, '\n')
		fi.space = true
		if !fi.width_set {
			fi.width, fi.width_set = Default_Width, true
		}
		when T == f32 || T == BF16 {
			if !fi.prec_set {
				fi.prec, fi.prec_set = Default_Precision, true
			}
		}
		s := Array_Fmt_State {
			fi        = fi,
			verb      = verb,
			summarize = Threshold > 0 && a.size > Threshold,
		}
		fmt_array_data(&s, a)
		_putc(fi, '\n')
	}
	return true
}

fmt_array_data :: proc(s: ^Array_Fmt_State, a: Array(CPU, $T), pos: []int = nil, last := -1, dummy := false) {
	pos := slice.to_dynamic(pos)
	defer delete(pos)
	if last >= 0 {
		append(&pos, last)
	}
	ix := len(pos)
	if ix == a.ndims {
		if pos[ix - 1] > 0 do _putc(s.fi, ' ')
		if dummy {
			fmt.fmt_string(s.fi, "...", 's')
		} else {
			fmt.fmt_value(s.fi, a.dptr[index(a.shape, ..pos[:])], s.verb)
		}
		return
	}
	_putc(s.fi, '[')
	if ix > 0 {
		s.indent += 1
	}
	if s.summarize && a.dims[ix] > 2 * Edgeitems {
		for i in 0 ..< Edgeitems {
			fmt_array_data(s, a, pos[:], i, dummy)
		}
		fmt_array_data(s, a, pos[:], Edgeitems, true)
		for i in a.dims[ix] - Edgeitems ..< a.dims[ix] {
			fmt_array_data(s, a, pos[:], i, dummy)
		}
	} else {
		for i in 0 ..< a.dims[ix] {
			fmt_array_data(s, a, pos[:], i, dummy)
		}
	}
	row, rows := 0, 1
	if ix > 0 {
		row, rows = pos[ix - 1], a.dims[ix - 1]
	}
	_putc(s.fi, ']')
	s.newline += 1
	if row < rows - 1 {
		_putc(s.fi, '\n', s.newline)
		_putc(s.fi, ' ', s.indent)
		s.newline = 0
	}
	s.indent -= 1
}

_puts :: proc(fi: ^fmt.Info, s: string) {
	io.write_string(fi.writer, s, &fi.n)
}

_putc :: proc(fi: ^fmt.Info, r: rune, n := 1) {
	for _ in 0 ..< n {
		io.write_rune(fi.writer, r, &fi.n)
	}
}
