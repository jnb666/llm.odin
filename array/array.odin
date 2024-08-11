package array

import "base:intrinsics"
import "core:math"
import "core:slice"
import "core:io"
import "core:fmt"
import "core:log"

import "../util"

MAX_DIMS :: 4
ALIGN_DATA :: 64

// Bfloat16 data type
BF16 :: distinct u16

// Device indicates if the array is resident on CPU or the Cuda GPU device.
CPU :: struct {}

Cuda :: struct {}

// Array is a dynamically allocated multi-dimensional regular array. 
Array :: struct($D, $T: typeid) where (D == CPU || D == Cuda) && (T == i32 || T == f32 || T == BF16) {
	dptr: [^]T,
	using shape: Shape,
}

zeros :: proc{zeros_cpu, zeros_cuda}

delete :: proc{delete_cpu, delete_cuda,
			   delete_string, delete_cstring, delete_dynamic_array, delete_slice, delete_map}

zero :: proc{zero_cpu, zero_cuda}

fill :: proc{fill_cpu, fill_cuda}

copy :: proc{copy_array_cpu, copy_array_cuda,
			 copy_array_to_slice_cpu, copy_slice_to_array_cpu,
			 copy_array_to_slice_cuda, copy_slice_to_array_cuda,
			 copy_slice, copy_from_string}

read :: proc{read_cpu, read_cuda}

mean :: proc{mean_cpu, mean_cuda}

ptr :: proc "contextless" (a: Array($D, $T), offset := 0) -> [^]T {
	if a.dptr == nil {
		return nil
	}
	return &a.dptr[a.offset+offset]
}

is_nil :: proc "contextless" (a: Array($D, $T)) -> bool {
	return a.dptr == nil
}

// Create a new array using a copy of the provided data. If move is set then calls delete(data)
new :: proc($D, $T: typeid, dims: []int, data: []$V, move := false, loc := #caller_location) -> Array(D, T) {
	a := zeros(D, T, dims, loc=loc)
	copy(a, data)
	if move {
		delete(data)
	}
	return a
}

// Allocate a new array with given dimensions and fill with zeros
zeros_cpu :: proc($D, $T: typeid, dims: []int, loc := #caller_location) -> (a: Array(D, T)) where D == CPU {
	a.shape = make_shape(dims, loc=loc)
	a.dptr = raw_data(make([]T, a.size))
	return a
}

// Allocate new zeroed array with same device, type and shape as a.
zeros_like :: proc(a: Array($D, $T)) -> Array(D, T) {
	s := a.shape
	return zeros(D, T, shape(&s))
}

// Return a view on the same data with new shape and optional relative offset
view :: proc(a: Array($D, $T), dims: []int, offset := 0, loc := #caller_location) -> Array(D, T) {
	assert(offset >= 0 && offset + math.prod(dims) <= a.size, "invalid size", loc) 
	return {dptr = a.dptr, shape = make_shape(dims, a.offset+offset, loc=loc)}
}

// Deallocate memory
delete_cpu :: proc(a: Array(CPU, $T)) {
	if a.dptr != nil {
		free(a.dptr)
	}
}

// Clear all elements to zero
zero_cpu :: proc(a: Array(CPU, $T)) {
	slice.zero(a.dptr[a.offset : a.offset+a.size])
}

// Fill all elements with value
fill_cpu :: proc(a: Array(CPU, $T), value: $V) where intrinsics.type_is_numeric(V) {
	val := convert(T, value)
	slice.fill(a.dptr[a.offset : a.offset+a.size], val)
}

// Initialize values from function
initialize :: proc(a: Array($D, $T), fn: proc(ctx: rawptr) -> $T2, ctx: rawptr = nil) {
	temp := make([]T2, a.size)
	defer delete(temp)
	for i in 0 ..< a.size {
		temp[i] = fn(ctx)
	}
	copy(a, temp)
}

// Copy array of same type
copy_array_cpu :: proc(dst: Array(CPU, $T), src: Array(CPU, T)) {
	copy(dst.dptr[dst.offset : dst.offset+dst.size], src.dptr[src.offset : src.offset+src.size])
}

// Copy data from array to slice
copy_array_to_slice_cpu :: proc(dst: []$V, src: Array(CPU, $T)) {
	when T == V {
		copy(dst, src.dptr[src.offset : src.offset+src.size])
	} else {
		for i in 0 ..< min(len(dst), src.size) {
			dst[i] = convert(V, src.dptr[src.offset+i])
		}
	}
}

// Copy data from slice to array
copy_slice_to_array_cpu :: proc(dst: Array(CPU, $T), src: []$V) {
	when T == V {
		copy(dst.dptr[dst.offset: dst.offset+dst.size], src)
	} else {
		for i in 0 ..< min(len(src), dst.size) {
			dst.dptr[dst.offset+i] = convert(T, src[i])
		}
	}
}

// Read data from stream
read_cpu :: proc($Data_Type: typeid, r: io.Stream, dst: Array(CPU, $T), loc := #caller_location) -> io.Error {
	assert(dst.size > 0, "zero size array", loc)
	when Data_Type == T {
		data := ptr(dst)
		return util.read_slice(r, data[:dst.size])
	} else {
		buf := make([]Data_Type, dst.size)
		defer delete(buf)
		util.read_slice(r, buf[:dst.size]) or_return
		copy(dst, buf)
		return nil
	}
}

// Get mean value of array elements
mean_cpu :: proc(a: Array(CPU, $T)) -> f32 {
	ap := ptr(a)
	sum: f32
	for i in 0 ..< a.size {
		sum += ap[i]
	}
	return sum / f32(a.size)
}

// Shape contains the array dimensions. As a struct so can be stored on stack.
Shape :: struct {
	dims: [MAX_DIMS]int,
	ndims: int,
	offset: int,
	size: int,
}

make_shape :: proc(dims: []int, offset := 0, loc := #caller_location) -> (s: Shape) {
	assert(len(dims) <= MAX_DIMS, "too many dimensions", loc)
	copy(s.dims[:], dims)
	s.ndims = len(dims)
	s.size = 1
	for i in 0 ..< s.ndims {
		assert(dims[i] >= 0, "invalid dimension", loc)
		s.size *= dims[i]
	}
	s.offset = offset
	return s
}

// Get shape as a slice - don't change the returned data!
shape :: proc(s: ^Shape) -> []int {
	return s.dims[:s.ndims]
}

// Convert from index in each dimension to relative offset in no. of elements from base pointer.
index :: proc(s: Shape, ix: ..int, loc := #caller_location) -> int {
	assert(len(ix) <= s.ndims, "too many indices", loc)
	pos := s.offset
	stride := 1
	for i := len(ix)-1; i >= 0; i -= 1 {
		assert(ix[i] >= 0 && ix[i] < s.dims[i], "index out of range", loc)
		pos += ix[i] * stride
		stride *= int(s.dims[i])
	}
	return pos
}

// Convert from bf16 to f32
bf16_to_f32 :: proc(v: BF16) -> f32 {
	bits := u32(v) << 16
	return transmute(f32)bits
}

// Convert from f32 to bf16 - round to nearest
f32_to_bf16 :: proc(v: f32) -> BF16 {
	bits := transmute(u32)(v * 1.001957)
	return BF16(bits >> 16)
}

// Generic conversions
convert :: proc($T: typeid, v: $V) -> T where intrinsics.type_is_numeric(T) && intrinsics.type_is_numeric(V) {
	when T == BF16 {
		return f32_to_bf16(f32(v))
	} else when V == BF16 {
		return T(bf16_to_f32(v))
	} else {
		return T(v)
	}
}

convert_slice :: proc(dst: []$T, src: []$V) where intrinsics.type_is_numeric(T) && intrinsics.type_is_numeric(V) {
	n := min(len(src), len(dst))
	for i in 0 ..< n {
		dst[i] = convert(T, src[i])
	}
}

// Check if arrays a and b have all elements with matching within relative + absolute error threshold
compare :: proc(name: string, arr_a: Array($D1, $T1), arr_b: Array($D2, $T2), epsilon: f32 = 1e-3, 
				threshold: f32 = 1e-6, max_print := 10, verbose := false, loc := #caller_location) -> bool {
	b1, b2: [32]u8

	fmt_num :: proc(buf: []u8, x: f32) -> string {
		return fmt.bprintf(buf, abs(x) >= 1e-4 ? "% 10.3g" : "% 10.2e", x)
	}

	if arr_a.dims != arr_b.dims {
		log.errorf("%-20s: shape mismatch: a=%v b=%v", name, arr_a.shape, arr_b.shape, location=loc)
		return false
	}
	a := make([]f32, arr_a.size)
	defer delete(a)
	copy(a, arr_a)
	b := make([]f32, arr_b.size)
	defer delete(b)
	copy(b, arr_b)
	n_err, n_print := 0, 0
	max_diff: f32
	for i in 0 ..< len(a) {
		diff := abs(a[i] - b[i])
		max_diff = max(max_diff, diff)
		warned := false
		if !util.nearly_equal(a[i], b[i], epsilon, threshold) {
			if n_err < max_print {
				log.warnf("%-20s: % 6d a=%s b=%s", name, i, fmt_num(b1[:], a[i]), fmt_num(b2[:], b[i]), location=loc)
				warned = true
			}
			n_err += 1
		}
		if verbose && n_print + n_err < max_print && !warned {
			log.infof("%-20s: % 6d a=%s b=%s", name, i, fmt_num(b1[:], a[i]), fmt_num(b2[:], b[i]), location=loc)
			n_print += 1
		}
	}
	if n_err > 0 {
		log.errorf("%-20s: %d / %d values outside threshold  max_diff=%.2g", name, n_err, len(a), max_diff, location=loc)
		return false
	}
	if !verbose {
		log.debugf("%-20s: all ok  max_diff=%.2g", name, max_diff, location=loc)
	} else {
		log.infof("%-20s: all ok  max_diff=%.2g", name, max_diff, location=loc)
	}
	return true
}
