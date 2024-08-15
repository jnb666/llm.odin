package array

import "../cuda"
import "../util"
import "core:fmt"
import "core:log"
import "core:slice"
import "core:testing"

@(test)
basic_test :: proc(t: ^testing.T) {
	basic_test_on(t, CPU)
}

@(test)
basic_cuda_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	basic_test_on(t, Cuda)
}

basic_test_on :: proc(t: ^testing.T, $Device: typeid) {
	a := zeros(Device, f32, {2, 3})
	defer delete(a)
	log.debug(a.shape)
	testing.expect(t, slice.equal(shape(&a.shape), []int{2, 3}))
	testing.expect_value(t, a.size, 6)
	fill(a, 42)
	arr: [6]f32
	copy(arr[:], a)
	log.debug(arr)
	testing.expect(t, slice.equal(arr[:], []f32{0 ..< 6 = 42}))
	zero(a)
	copy(arr[:], a)
	testing.expect(t, slice.equal(arr[:], []f32{0 ..< 6 = 0}))
}

@(test)
bfloat16_test :: proc(t: ^testing.T) {
	bfloat16_test_on(t, CPU)
}

@(test)
bfloat16_cuda_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	bfloat16_test_on(t, Cuda)
}

bfloat16_test_on :: proc(t: ^testing.T, $Device: typeid) {
	a := new(Device, BF16, {4}, []f32{0.111, 0.222, 0.333, 0.444})
	defer delete(a)
	arr: [4]f32
	copy(arr[:], a)
	delta := util.max_difference(arr[:], []f32{0.1108, 0.2217, 0.3320, 0.4434})
	log.debugf("a = %.5f  delta = %g", arr[:], delta)
	testing.expect(t, delta < 1e-4)
}

@(test)
view_test :: proc(t: ^testing.T) {
	a := new(CPU, i32, {2, 20, 40}, util.seq(i32, 1600), move = true)
	defer delete(a)
	v := view(a, {5, 10}, offset = 800)
	log.debugf("\n% 6d", v)
	v.dptr[index(v.shape, 4, 4)] = 42
	testing.expect_value(t, a.dptr[index(a.shape, 1, 1, 4)], 42)
}

@(test)
format_test1 :: proc(t: ^testing.T) {
	a := new(CPU, f32, {2, 3}, []f32{1.1, -2.2, 3.3, 40, 50, 60})
	defer delete(a)
	a_str := fmt.aprintf("% 6.1f", a)
	defer delete(a_str)
	log.debugf("\n%s", a_str)
	testing.expect_value(t, a_str, `CPUArray<f32>[2 3]
[[   1.1   -2.2    3.3]
 [  40.0   50.0   60.0]]
`)
}

@(test)
format_cuda_test :: proc(t: ^testing.T) {
	ctx := cuda.create_context()
	defer cuda.destroy_context(ctx)
	a := new(Cuda, f32, {2, 3}, []f32{1.1, -2.2, 3.3, 40, 50, 60})
	defer delete(a)
	a_str := fmt.aprintf("% 6.1f", a)
	defer delete(a_str)
	log.debugf("\n%s", a_str)
	testing.expect_value(t, a_str, `CudaArray<f32>[2 3]
[[   1.1   -2.2    3.3]
 [  40.0   50.0   60.0]]
`)
}

@(test)
format_nil_test :: proc(t: ^testing.T) {
	a: Array(CPU, f32)
	defer delete(a)
	a_str := fmt.aprintf("% 6.1f", a)
	defer delete(a_str)
	log.debug(a_str)
	testing.expect_value(t, a_str, `CPUArray<f32>(nil)`)

	b: Array(Cuda, f32)
	defer delete(b)
	b_str := fmt.aprintf("% 6.1f", b)
	defer delete(b_str)
	log.debug(b_str)
	testing.expect_value(t, b_str, `CudaArray<f32>(nil)`)
}

@(test)
format_test2 :: proc(t: ^testing.T) {
	a := new(CPU, i32, {2, 20, 40}, util.seq(i32, 1600), move = true)
	defer delete(a)
	a_str := fmt.aprintf("% 6d", a)
	defer delete(a_str)
	log.debugf("\n%s", a_str)
	testing.expect_value(
		t,
		a_str,
		`CPUArray<i32>[2 20 40]
[[[     0      1      2      3      4    ...     35     36     37     38     39]
  [    40     41     42     43     44    ...     75     76     77     78     79]
  [    80     81     82     83     84    ...    115    116    117    118    119]
  [   120    121    122    123    124    ...    155    156    157    158    159]
  [   160    161    162    163    164    ...    195    196    197    198    199]
  [   ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...]
  [   600    601    602    603    604    ...    635    636    637    638    639]
  [   640    641    642    643    644    ...    675    676    677    678    679]
  [   680    681    682    683    684    ...    715    716    717    718    719]
  [   720    721    722    723    724    ...    755    756    757    758    759]
  [   760    761    762    763    764    ...    795    796    797    798    799]]

 [[   800    801    802    803    804    ...    835    836    837    838    839]
  [   840    841    842    843    844    ...    875    876    877    878    879]
  [   880    881    882    883    884    ...    915    916    917    918    919]
  [   920    921    922    923    924    ...    955    956    957    958    959]
  [   960    961    962    963    964    ...    995    996    997    998    999]
  [   ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...]
  [  1400   1401   1402   1403   1404    ...   1435   1436   1437   1438   1439]
  [  1440   1441   1442   1443   1444    ...   1475   1476   1477   1478   1479]
  [  1480   1481   1482   1483   1484    ...   1515   1516   1517   1518   1519]
  [  1520   1521   1522   1523   1524    ...   1555   1556   1557   1558   1559]
  [  1560   1561   1562   1563   1564    ...   1595   1596   1597   1598   1599]]]
`,
	)
}
