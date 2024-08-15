package cudnn

import "../cuda"
import "core:log"
import "core:slice"
import "core:testing"

alloc_array :: proc(size: int, value: f32) -> rawptr {
	mem := cuda.memalloc(size * 4)
	val := transmute(u32)(value)
	cuda.memsetD32(mem, val, size)
	return mem
}

@(test)
add_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	context.user_ptr = ctx
	register_handle()

	N, C, H, W := 1, 5, 2, 2
	dims := []int{N, C, H, W}
	strides := []int{C * H * W, 1, C * W, C}
	a_desc := tensor_descriptor('a', dims, strides)
	defer destroy(a_desc)
	b_desc := tensor_descriptor('b', dims, strides)
	defer destroy(b_desc)
	c_desc := tensor_descriptor('c', dims, strides)
	defer destroy(c_desc)

	add := binary_op(.ADD, a_desc, b_desc, c_desc)
	defer destroy(add)
	graph := make_graph(add)
	defer destroy(graph)
	plan, err := get_plan(graph)
	testing.expect_value(t, err, Status.SUCCESS)
	defer destroy(plan)

	wsize := workspace_size(plan)
	log.debug("workspace size =", wsize)
	testing.expect_value(t, wsize, 0)

	size := N * C * H * W
	a := alloc_array(size, 2.5)
	defer cuda.memfree(a)
	b := alloc_array(size, 39.5)
	defer cuda.memfree(b)
	c := alloc_array(size, 0)
	defer cuda.memfree(c)

	execute(plan, {'a', 'b', 'c'}, {a, b, c})
	res := make([]f32, size)
	defer delete(res)
	cuda.memcpyDtoH(raw_data(res), c, size * 4)
	log.debug(res)
	testing.expect(t, slice.equal(res, []f32{0 ..< 20 = 42}))
}
