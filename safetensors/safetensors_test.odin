package safetensors

import "core:log"
import "core:slice"
import "core:testing"

import "../array"
import "../util"

@(test)
read_test :: proc(t: ^testing.T) {
	file_name, err := util.huggingface_cache_file("gpt2", "model.safetensors")
	defer delete(file_name)
	log.debug(file_name)
	testing.expect_value(t, err, nil)

	f, err2 := open(file_name)
	testing.expect_value(t, err2, nil)
	defer close(f)

	d := f.desc["wte.weight"]
	log.debug(d)
	testing.expect_value(t, d.dtype, Data_Type.F32)
	testing.expect(t, slice.equal(d.shape, []int{50257, 768}))

	a := array.zeros(array.CPU, f32, {50257, 768})
	defer array.delete(a)
	err2 = read(f, "wte.weight", a)
	testing.expect_value(t, err2, nil)
	log.debugf("\n%v", a)
}

@(test)
transpose_test :: proc(t: ^testing.T) {
	file_name, err := util.huggingface_cache_file("gpt2", "model.safetensors")
	defer delete(file_name)
	log.debug(file_name)
	testing.expect_value(t, err, nil)

	f, err2 := open(file_name)
	testing.expect_value(t, err2, nil)
	defer close(f)

	d := f.desc["h.0.mlp.c_fc.weight"]
	log.debug(d)

	a := array.zeros(array.CPU, f32, {3072, 768})
	defer array.delete(a)
	err2 = read(f, "h.0.mlp.c_fc.weight", a, transposed = true)
	testing.expect_value(t, err2, nil)
	log.debugf("\n%v", a)
}
