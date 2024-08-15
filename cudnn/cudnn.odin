package cudnn

import "../cuda"
import "core:log"
import "core:strings"

foreign import "system:cudnn"

Handle :: distinct rawptr
BackendDescriptor :: distinct rawptr
RuntimeTag :: distinct rawptr
Fraction :: FractionStruct

@(default_calling_convention = "c")
foreign cudnn {

	@(link_name = "cudnnGetVersion")
	GetVersion :: proc() -> uint ---

	@(link_name = "cudnnGetErrorString")
	GetErrorString :: proc(status: Status) -> cstring ---

	@(link_name = "cudnnGetLastErrorString")
	GetLastErrorString :: proc(message: cstring, max_size: uint) ---

	@(link_name = "cudnnQueryRuntimeError")
	QueryRuntimeError :: proc(handle: Handle, rstatus: ^Status, mode: ErrQueryMode, tag: ^RuntimeTag) -> Status ---

	@(link_name = "cudnnCreate")
	Create :: proc(handle: ^Handle) -> Status ---

	@(link_name = "cudnnDestroy")
	Destroy :: proc(handle: Handle) -> Status ---

	@(link_name = "cudnnSetStream")
	SetStream :: proc(handle: Handle, streamId: cuda.Stream) -> Status ---

	@(link_name = "cudnnGetStream")
	GetStream :: proc(handle: Handle, streamId: ^cuda.Stream) -> Status ---

	@(link_name = "cudnnBackendCreateDescriptor")
	BackendCreateDescriptor :: proc(descriptorType: BackendDescriptorType, descriptor: ^BackendDescriptor) -> Status ---

	@(link_name = "cudnnBackendDestroyDescriptor")
	BackendDestroyDescriptor :: proc(descriptor: BackendDescriptor) -> Status ---

	@(link_name = "cudnnBackendFinalize")
	BackendFinalize :: proc(descriptor: BackendDescriptor) -> Status ---

	@(link_name = "cudnnBackendSetAttribute")
	BackendSetAttribute :: proc(descriptor: BackendDescriptor, attributeName: BackendAttributeName, attributeType: BackendAttributeType, elementCount: i64, arrayOfElements: rawptr) -> Status ---

	@(link_name = "cudnnBackendGetAttribute")
	BackendGetAttribute :: proc(descriptor: BackendDescriptor, attributeName: BackendAttributeName, attributeType: BackendAttributeType, requestedElementCount: i64, elementCount: ^i64, arrayOfElements: rawptr) -> Status ---

	@(link_name = "cudnnBackendExecute")
	BackendExecute :: proc(handle: Handle, executionPlan: BackendDescriptor, variantPack: BackendDescriptor) -> Status ---
}

// Register cudnn handle in cuda user context
register_handle :: proc(ctx: ^cuda.UserContext = nil) {
	destroy_handle :: proc(key: string, h: rawptr) {
		Destroy(cast(Handle)h)
	}
	h: Handle
	must(Create(&h))
	cuda.register_handle("cudnn", h, destroy_handle, ctx)
}

error :: proc(rc: Status) -> string {
	return string(GetErrorString(rc))
}

must :: proc(rc: Status, loc := #caller_location) {
	if rc != .SUCCESS {
		panic(error(rc), loc)
	}
}

query_runtime_error :: proc(loc := #caller_location) -> Status {
	handle := cast(Handle)cuda.get_handle("cudnn", loc = loc)
	rc: Status
	must(QueryRuntimeError(handle, &rc, .BLOCKING, nil), loc)
	return rc
}

set :: proc(d: BackendDescriptor, name: BackendAttributeName, type: BackendAttributeType, ptr: rawptr, count: int = 1, loc := #caller_location) {
	must(BackendSetAttribute(d, name, type, i64(count), ptr), loc)
}

get :: proc(d: BackendDescriptor, name: BackendAttributeName, type: BackendAttributeType, req_count: int, elements: rawptr) -> int {
	count: i64
	must(BackendGetAttribute(d, name, type, i64(req_count), &count, elements))
	return int(count)
}

make_descriptor :: proc(type: BackendDescriptorType) -> BackendDescriptor {
	d: BackendDescriptor
	must(BackendCreateDescriptor(type, &d))
	return d
}

destroy :: proc(desc: ..BackendDescriptor) {
	for d in desc {
		if d != nil {
			BackendDestroyDescriptor(d)
		}
	}
}

// generate tensor descriptor. if strides is nil then allocate using packed layout
tensor_descriptor :: proc(
	id: u64,
	dims: []int,
	strides: []int = nil,
	dtype: DataType = .FLOAT,
	virtual := false,
	by_value := false,
	loc := #caller_location,
) -> BackendDescriptor {
	id, dtype := id, dtype
	alignment := 16
	d := make_descriptor(.TENSOR_DESCRIPTOR)
	set(d, .TENSOR_UNIQUE_ID, .INT64, &id)
	set(d, .TENSOR_DATA_TYPE, .DATA_TYPE, &dtype)
	set(d, .TENSOR_BYTE_ALIGNMENT, .INT64, &alignment)
	if virtual {
		virt: i32 = 1
		set(d, .TENSOR_IS_VIRTUAL, .BOOLEAN, &virt)
	}
	if by_value {
		byval: i32 = 1
		set(d, .TENSOR_IS_BY_VALUE, .BOOLEAN, &byval)
	}
	set(d, .TENSOR_DIMENSIONS, .INT64, raw_data(dims), len(dims))
	if strides != nil {
		set(d, .TENSOR_STRIDES, .INT64, raw_data(strides), len(strides))
	} else {
		stride := make([]int, len(dims))
		defer delete(stride)
		n := 1
		for i := len(dims) - 1; i >= 0; i -= 1 {
			stride[i] = n
			n *= dims[i]
		}
		set(d, .TENSOR_STRIDES, .INT64, raw_data(stride), len(stride))
	}
	return finalize(d, loc)
}

finalize :: proc(d: BackendDescriptor, loc := #caller_location) -> BackendDescriptor {
	must(BackendFinalize(d), loc)
	return d
}

make_graph :: proc(ops: ..BackendDescriptor, loc := #caller_location) -> BackendDescriptor {
	handle := cast(Handle)cuda.get_handle("cudnn", loc = loc)
	d := make_descriptor(.OPERATIONGRAPH_DESCRIPTOR)
	set(d, .OPERATIONGRAPH_HANDLE, .HANDLE, &handle)
	set(d, .OPERATIONGRAPH_OPS, .BACKEND_DESCRIPTOR, raw_data(ops), len(ops))
	return finalize(d, loc)
}

get_configs_from_heuristic :: proc(graph: BackendDescriptor, mode: BackendHeurMode, loc := #caller_location) -> []BackendDescriptor {
	graph, mode := graph, mode
	d := make_descriptor(.ENGINEHEUR_DESCRIPTOR)
	defer destroy(d)
	set(d, .ENGINEHEUR_OPERATION_GRAPH, .BACKEND_DESCRIPTOR, &graph)
	set(d, .ENGINEHEUR_MODE, .HEUR_MODE, &mode)
	finalize(d, loc)
	max_count := get(d, .ENGINEHEUR_RESULTS, .BACKEND_DESCRIPTOR, 0, nil)
	if max_count == 0 {
		return nil
	}
	configs := make([]BackendDescriptor, max_count)
	for i in 0 ..< max_count {
		configs[i] = make_descriptor(.ENGINECFG_DESCRIPTOR)
	}
	count := get(d, .ENGINEHEUR_RESULTS, .BACKEND_DESCRIPTOR, max_count, raw_data(configs))
	log.debugf("found %d / %d candidate configs using %s", count, max_count, mode, location = loc)
	if count == 0 {
		delete(configs)
		return nil
	}
	return configs[:count]
}

// get first supported execution plan from heuristics
get_plan :: proc(graph: BackendDescriptor, mode: BackendHeurMode = .HEUR_MODE_A, loc := #caller_location) -> (plan: BackendDescriptor, err: Status) {
	handle := cast(Handle)cuda.get_handle("cudnn", loc = loc)
	configs := get_configs_from_heuristic(graph, mode, loc = loc)
	if len(configs) == 0 {
		return nil, .NOT_SUPPORTED
	}
	defer {
		destroy(..configs)
		delete(configs)
	}
	for i in 0 ..< len(configs) {
		plan = make_descriptor(.EXECUTION_PLAN_DESCRIPTOR)
		set(plan, .EXECUTION_PLAN_HANDLE, .HANDLE, &handle)
		set(plan, .EXECUTION_PLAN_ENGINE_CONFIG, .BACKEND_DESCRIPTOR, &configs[i])
		err = BackendFinalize(plan)
		log.debugf("plan %d - %s", i, err, location = loc)
		if err == .SUCCESS {
			return plan, .SUCCESS
		}
		destroy(plan)
	}
	return nil, err
}

workspace_size :: proc(plan: BackendDescriptor) -> int {
	size: i64
	get(plan, .EXECUTION_PLAN_WORKSPACE_SIZE, .INT64, 1, &size)
	return int(size)
}

// execute plan with given tensor inputs
execute :: proc(plan: BackendDescriptor, ids: []u64, arrays: []rawptr, loc := #caller_location) {
	assert(len(ids) == len(arrays), "invalid input params", loc)
	handle := cast(Handle)cuda.get_handle("cudnn", loc = loc)

	vars := make_descriptor(.VARIANT_PACK_DESCRIPTOR)
	defer destroy(vars)
	set(vars, .VARIANT_PACK_UNIQUE_IDS, .INT64, raw_data(ids), len(ids))
	set(vars, .VARIANT_PACK_DATA_POINTERS, .VOID_PTR, raw_data(arrays), len(arrays))
	wspace: rawptr
	if workspace := workspace_size(plan); workspace > 0 {
		wspace = cuda.memalloc(workspace)
		set(vars, .VARIANT_PACK_WORKSPACE, .VOID_PTR, &wspace)
	}
	finalize(vars, loc)

	must(BackendExecute(handle, plan, vars))

	if wspace != nil {
		cuda.memfree(wspace)
	}
}

// save plan to cache
cache_plan :: proc(id: string, plan: BackendDescriptor, loc := #caller_location) {
	destroy_entry :: proc(key: string, p: rawptr) {
		BackendDestroyDescriptor(cast(BackendDescriptor)p)
		delete(key)
	}
	assert(context.user_ptr != nil, "cuda user context not set", loc)
	ctx := cast(^cuda.UserContext)context.user_ptr
	ctx.cache[strings.clone(id)] = cuda.Handle {
		ptr     = plan,
		destroy = destroy_entry,
	}
}

// lookup previously saved plan
plan_from_cache :: proc(id: string, loc := #caller_location) -> (BackendDescriptor, bool) {
	assert(context.user_ptr != nil, "cuda user context not set", loc)
	ctx := cast(^cuda.UserContext)context.user_ptr
	if h, ok := ctx.cache[id]; ok {
		return cast(BackendDescriptor)h.ptr, true
	}
	return nil, false
}

pointwise_descriptor :: proc(mode: PointwiseMode, prec: DataType, axis := -1) -> BackendDescriptor {
	mode, prec := mode, prec
	d := make_descriptor(.POINTWISE_DESCRIPTOR)
	set(d, .POINTWISE_MODE, .POINTWISE_MODE, &mode)
	set(d, .POINTWISE_MATH_PREC, .DATA_TYPE, &prec)
	if axis >= 0 {
		axis := i64(axis)
		set(d, .POINTWISE_AXIS, .INT64, &axis)
	}
	return finalize(d)
}

// y = op(x)
unary_op :: proc(mode: PointwiseMode, x, y: BackendDescriptor, axis := -1, type: DataType = .FLOAT) -> BackendDescriptor {
	x, y := x, y
	typ := pointwise_descriptor(mode, type, axis)
	defer destroy(typ)
	d := make_descriptor(.OPERATION_POINTWISE_DESCRIPTOR)
	set(d, .OPERATION_POINTWISE_PW_DESCRIPTOR, .BACKEND_DESCRIPTOR, &typ)
	set(d, .OPERATION_POINTWISE_XDESC, .BACKEND_DESCRIPTOR, &x)
	set(d, .OPERATION_POINTWISE_YDESC, .BACKEND_DESCRIPTOR, &y)
	return finalize(d)
}

// y = x op b
binary_op :: proc(mode: PointwiseMode, x, b, y: BackendDescriptor, mask: BackendDescriptor = nil, type: DataType = .FLOAT) -> BackendDescriptor {
	x, b, y := x, b, y
	typ := pointwise_descriptor(mode, type)
	defer destroy(typ)
	d := make_descriptor(.OPERATION_POINTWISE_DESCRIPTOR)
	set(d, .OPERATION_POINTWISE_PW_DESCRIPTOR, .BACKEND_DESCRIPTOR, &typ)
	set(d, .OPERATION_POINTWISE_XDESC, .BACKEND_DESCRIPTOR, &x)
	set(d, .OPERATION_POINTWISE_BDESC, .BACKEND_DESCRIPTOR, &b)
	set(d, .OPERATION_POINTWISE_YDESC, .BACKEND_DESCRIPTOR, &y)
	if mode == .BINARY_SELECT {
		assert(mask != nil, "mask is needed for binary select")
		mask := mask
		set(d, .OPERATION_POINTWISE_TDESC, .BACKEND_DESCRIPTOR, &mask)
	}
	return finalize(d)
}

// y = reduce_op(x)
reduce_op :: proc(mode: ReduceTensorOp, x, y: BackendDescriptor, type: DataType = .FLOAT) -> BackendDescriptor {
	mode, type := mode, type
	typ := make_descriptor(.REDUCTION_DESCRIPTOR)
	set(typ, .REDUCTION_OPERATOR, .REDUCTION_OPERATOR_TYPE, &mode)
	set(typ, .REDUCTION_COMP_TYPE, .DATA_TYPE, &type)
	finalize(typ)
	defer destroy(typ)
	x, y := x, y
	d := make_descriptor(.OPERATION_REDUCTION_DESCRIPTOR)
	set(d, .OPERATION_REDUCTION_DESC, .BACKEND_DESCRIPTOR, &typ)
	set(d, .OPERATION_REDUCTION_XDESC, .BACKEND_DESCRIPTOR, &x)
	set(d, .OPERATION_REDUCTION_YDESC, .BACKEND_DESCRIPTOR, &y)
	return finalize(d)
}

// c = matmul(a, b)
matmul_op :: proc(a, b, c: BackendDescriptor, type: DataType = .FLOAT) -> BackendDescriptor {
	type := type
	typ := make_descriptor(.MATMUL_DESCRIPTOR)
	set(typ, .MATMUL_COMP_TYPE, .DATA_TYPE, &type)
	finalize(typ)
	defer destroy(typ)
	a, b, c := a, b, c
	d := make_descriptor(.OPERATION_MATMUL_DESCRIPTOR)
	set(d, .OPERATION_MATMUL_DESC, .BACKEND_DESCRIPTOR, &typ)
	set(d, .OPERATION_MATMUL_ADESC, .BACKEND_DESCRIPTOR, &a)
	set(d, .OPERATION_MATMUL_BDESC, .BACKEND_DESCRIPTOR, &b)
	set(d, .OPERATION_MATMUL_CDESC, .BACKEND_DESCRIPTOR, &c)
	return finalize(d)
}

// for attention graph
reshape_op :: proc(x, y: BackendDescriptor) -> BackendDescriptor {
	x, y := x, y
	d := make_descriptor(.OPERATION_RESHAPE_DESCRIPTOR)
	set(d, .OPERATION_RESHAPE_XDESC, .BACKEND_DESCRIPTOR, &x)
	set(d, .OPERATION_RESHAPE_YDESC, .BACKEND_DESCRIPTOR, &y)
	return finalize(d)
}
