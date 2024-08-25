package safetensors

import "core:encoding/json"
import "core:io"
import "core:math"
import "core:mem/virtual"
import "core:os"
import "core:slice"
import "core:sys/linux"

import "../array"

Error :: union #shared_nil {
	os.Error,
	io.Error,
	linux.Errno,
	json.Unmarshal_Error,
	virtual.Allocator_Error,
	Tensor_Error,
}

Tensor_Error :: enum {
	None,
	Tensor_Not_Found,
	Invalid_Shape,
	Tensor_Shape_Mismatch,
	Data_Type_Not_Supported,
	Invalid_File_Format,
}

Data_Type :: enum {
	F64,
	F32,
	F16,
	BF16,
	I64,
	I32,
	I16,
	I8,
	U8,
	BOOL,
}

// Parsed safetensors file header
File :: struct {
	data:  [^]u8,
	size:  i64,
	base:  i64,
	desc:  map[string]Tensor_Desc,
	arena: ^virtual.Arena,
}

Tensor_Desc :: struct {
	dtype:        Data_Type,
	shape:        []int,
	data_offsets: [2]i64,
}

// Open and mmap file and read and parse header data
open :: proc(file_name: string) -> (f: File, err: Error) {
	fd := os.open(file_name) or_return
	defer os.close(fd)
	f.size = os.file_size(fd) or_return
	f.data = cast([^]u8)linux.mmap(0, uint(f.size), {.READ, .WRITE}, {.PRIVATE}, linux.Fd(fd), 0) or_return
	header_size: i64 = (cast(^i64)f.data)^
	f.base = size_of(u64) + header_size
	f.arena = new(virtual.Arena)
	virtual.arena_init_growing(f.arena) or_return
	json.unmarshal(f.data[size_of(u64):f.base], &f.desc, allocator = virtual.arena_allocator(f.arena)) or_return
	return
}

// Read named tensor from file - only supports f32 format currently
read :: proc(f: File, name: string, a: array.Array($D, $T), transposed := false) -> Error {
	d, ok := f.desc[name]
	if !ok {
		return .Tensor_Not_Found
	}
	if d.dtype != .F32 {
		return .Data_Type_Not_Supported
	}
	size := math.prod(d.shape)
	if i64(size) * size_of(f32) != d.data_offsets[1] - d.data_offsets[0] {
		return .Invalid_File_Format
	}
	shape := d.shape
	if transposed {
		if len(shape) != 2 {
			return .Invalid_Shape
		}
		shape[0], shape[1] = shape[1], shape[0]
	}
	s := a.shape
	if !slice.equal(shape, array.shape(&s)) {
		return .Tensor_Shape_Mismatch
	}
	buf := slice.reinterpret([]f32, f.data[f.base + d.data_offsets[0]:f.base + d.data_offsets[1]])
	if transposed {
		buf2, alloc := transpose(shape[1], shape[0], buf)
		array.copy(a, buf2)
		if alloc {
			delete(buf2)
		}
	} else {
		array.copy(a, buf)
	}
	return nil
}

// Close mmap region and free allocated memory
close :: proc(f: File) {
	linux.munmap(f.data, uint(f.size))
	free_all(virtual.arena_allocator(f.arena))
	free(f.arena)
}

@(private)
transpose :: proc(rows, cols: int, inp: []f32) -> ([]f32, bool) #no_bounds_check {
	if rows == cols {
		for row in 0 ..< rows {
			for col in 0 ..< row {
				inp[col * rows + row], inp[row * cols + col] = inp[row * cols + col], inp[col * rows + row]
			}
		}
		return inp, false
	} else {
		out := make([]f32, rows * cols)
		for row in 0 ..< rows {
			for col in 0 ..< cols {
				out[col * rows + row] = inp[row * cols + col]
			}
		}
		return out, true
	}
}
