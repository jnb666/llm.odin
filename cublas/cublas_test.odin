package cublas

import "core:testing"
import "core:log"
import "../cuda"
import "../array"

Array :: array.Array
Cuda :: array.Cuda
CPU :: array.CPU

create_matrices :: proc() -> (a, b, c: Array(Cuda, f32)) {
	a = array.new(Cuda, f32, {2, 3}, []f32{1, 2, 1, -3, 4, -1})
	b = array.new(Cuda, f32, {2, 3}, []f32{1, 2, 1, -3, 4, -1})
	c = array.zeros(Cuda, f32, {3, 3})	
	return
}

@(test)
matmul_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	context.user_ptr = ctx
	register_handle()

	a, b, c := create_matrices()
	defer array.delete(a)
	defer array.delete(b)
	defer array.delete(c)
	array.fill(c, 1)

	rc := gemm(.R_32F, .OP_T, .OP_N, array.ptr(a), array.ptr(b), array.ptr(c), 3, 3, 2, beta=1)
	testing.expect_value(t, rc, Status.SUCCESS)
	
	log.debug("trans(A) x B = ", c)
	expect := array.new(CPU, f32, {3, 3}, []f32{11, -9, 5, -9, 21, -1, 5, -1, 3})
	defer array.delete(expect)
	testing.expect(t, array.compare("matmul", c, expect))
}

@(test)
matmul_bias_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	context.user_ptr = ctx
	register_handle()

	a, b, c := create_matrices()
	defer array.delete(a)
	defer array.delete(b)
	defer array.delete(c)
	bias := array.new(Cuda, f32, {3}, []f32{0.1, 0.2, 0.3})
	defer array.delete(bias)

	rc := gemm(.R_32F, .OP_T, .OP_N, array.ptr(a), array.ptr(b), array.ptr(c), 3, 3, 2, bias=array.ptr(bias))
	testing.expect_value(t, rc, Status.SUCCESS)
	
	log.debug("trans(A) x B + bias = ", c)
	expect := array.new(CPU, f32, {3, 3}, []f32{10.1, -9.8, 4.3, -9.9, 20.2, -1.7, 4.1, -1.8, 2.3})
	defer array.delete(expect)
	testing.expect(t, array.compare("matmul + bias", c, expect))
}

@(test)
matmul_dbias_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	context.user_ptr = ctx
	register_handle()

	a, b, c := create_matrices()
	defer array.delete(a)
	defer array.delete(b)
	defer array.delete(c)
	dbias := array.zeros(Cuda, f32, {3})
	defer array.delete(dbias)

	rc := gemm(.R_32F, .OP_T, .OP_N, array.ptr(a), array.ptr(b), array.ptr(c), 3, 3, 2, bias=array.ptr(dbias), dbias=true)
	testing.expect_value(t, rc, Status.SUCCESS)
	
	log.debug("sum(A) => ", dbias)
	expect := array.new(CPU, f32, {3}, []f32{-2, 6, 0})
	defer array.delete(expect)
	testing.expect(t, array.compare("dbias", dbias, expect))
}





