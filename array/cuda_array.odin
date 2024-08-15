package array

import "../cuda"
import "../util"
import "base:intrinsics"
import "core:io"

// Allocate a new device array with given dimensions and fill with zeros
zeros_cuda :: proc($D, $T: typeid, dims: []int, loc := #caller_location) -> (a: Array(D, T)) where D == Cuda {
	a.shape = make_shape(dims, loc = loc)
	a.dptr = cast([^]T)cuda.memalloc(a.size * size_of(T))
	zero_cuda(a)
	return a
}

// Either to_host if Device == CPU or to_device if Device == Cuda
to :: proc($Device: typeid, a: Array($D, $T)) -> Array(Device, T) where Device != D {
	when Device == CPU {
		return to_host(a)
	} else {
		return to_device(a)
	}
}

// Copy Cuda array to new CPU array
to_host :: proc(a: Array(Cuda, $T)) -> (dst: Array(CPU, T)) {
	if a.dptr != nil {
		s := a.shape
		dst = zeros(CPU, T, shape(&s))
		cuda.memcpyDtoH(dst.dptr, ptr(a), a.size * size_of(T))
	}
	return dst
}

// Copy CPU array to new Cuda array
to_device :: proc(a: Array(CPU, $T)) -> (dst: Array(Cuda, T)) {
	if a.dptr != nil {
		s := a.shape
		dst = zeros(Cuda, T, shape(&s))
		cuda.memcpyHtoD(dst.dptr, ptr(a), a.size * size_of(T))
	}
	return dst
}

// Deallocate device memory
delete_cuda :: proc(a: Array(Cuda, $T)) {
	if a.dptr != nil {
		cuda.memfree(a.dptr)
	}
}

// Clear all elements to zero
zero_cuda :: proc(a: Array(Cuda, $T)) {
	when T == BF16 {
		cuda.memsetD16(ptr(a), 0, a.size)
	} else {
		cuda.memsetD32(ptr(a), 0, a.size)
	}
}

// Fill all elements with value
fill_cuda :: proc(a: Array(Cuda, $T), value: $V) where intrinsics.type_is_numeric(V) {
	when T == BF16 {
		val := u16(f32_to_bf16(f32(value)))
		cuda.memsetD16(ptr(a), val, a.size)
	} else {
		val := transmute(u32)(T(value))
		cuda.memsetD32(ptr(a), val, a.size)
	}
}

// Copy array of same type
copy_array_cuda :: proc(dst: Array(Cuda, $T), src: Array(Cuda, T)) {
	n := min(dst.size, src.size)
	if n > 0 {
		cuda.memcpyDtoD(ptr(dst), ptr(src), n * size_of(T))
	}
}

// Copy data from array to slice
copy_array_to_slice_cuda :: proc(dst: []$V, src: Array(Cuda, $T)) {
	n := min(len(dst), src.size)
	if n == 0 {
		return
	}
	when T == V {
		dst_a := dst
	} else {
		dst_a := make([]T, n)
		defer delete(dst_a)
	}
	cuda.memcpyDtoH(raw_data(dst_a), ptr(src), n * size_of(T))
	when T != V {
		convert_slice(dst, dst_a)
	}
}

// Copy data from slice to array
copy_slice_to_array_cuda :: proc(dst: Array(Cuda, $T), src: []$V) {
	n := min(len(src), dst.size)
	if n == 0 {
		return
	}
	when T == V {
		src_a := src
	} else {
		src_a := make([]T, n)
		defer delete(src_a)
		convert_slice(src_a, src)
	}
	cuda.memcpyHtoD(ptr(dst), raw_data(src_a), n * size_of(T))
}

// Read data from stream into array
read_cuda :: proc($Data_Type: typeid, r: io.Stream, dst: Array(Cuda, $T), loc := #caller_location) -> io.Error {
	assert(dst.size > 0, "zero size array", loc)
	buf := make([]Data_Type, dst.size)
	defer delete(buf)
	util.read_slice(r, buf) or_return
	copy(dst, buf)
	return nil
}

// Write data to stream
write_cuda :: proc($Data_Type: typeid, w: io.Stream, src: Array(Cuda, $T), loc := #caller_location) -> io.Error {
	assert(src.size > 0, "zero size array", loc)
	buf := make([]Data_Type, src.size)
	defer delete(buf)
	copy(buf, src)
	return util.write_slice(w, buf)
}

// Get mean value of array elements - calculated on CPU so best for small arrays
mean_cuda :: proc(a: Array(Cuda, $T)) -> f32 {
	vals := make([]f32, a.size)
	defer delete(vals)
	copy(vals, a)
	sum: f32
	for x in vals {
		sum += x
	}
	return sum / f32(a.size)
}
