package cublas

import "../cuda"

@(extra_linker_flags="-L/usr/local/cuda/lib64")
foreign import cublaslt "system:cublasLt"

DEFAULT_MAX_WORKSPACE :: 32*1024*1024

Handle :: distinct rawptr
MatrixLayout :: distinct rawptr
MatmulDesc :: distinct rawptr
MatrixTransformDesc :: distinct rawptr
MatmulPreference :: distinct rawptr
NumericalImplFlags :: u64

MatmulAlgo :: struct {
	data : [8]u64,
}

MatmulHeuristicResult :: struct {
	algo : MatmulAlgo,
	workspaceSize : uint,
	state : Status,
	wavesCount : f32,
	reserved : [4]i32,
}

@(default_calling_convention="c")
foreign cublaslt {

	@(link_name="cublasLtCreate")
	create :: proc(lightHandle : ^Handle) -> Status ---

	@(link_name="cublasLtDestroy")
	destroy :: proc(lightHandle : Handle) -> Status ---

	@(link_name="cublasLtGetStatusName")
	getStatusName :: proc(status : Status) -> cstring ---

	@(link_name="cublasLtGetStatusString")
	getStatusString :: proc(status : Status) -> cstring ---

	@(link_name="cublasLtGetVersion")
	getVersion :: proc() -> uint ---

	@(link_name="cublasLtGetCudartVersion")
	getCudartVersion :: proc() -> uint ---

	@(link_name="cublasLtGetProperty")
	getProperty :: proc(type : LibraryPropertyType, value : ^i32) -> Status ---

	@(link_name="cublasLtHeuristicsCacheGetCapacity")
	heuristicsCacheGetCapacity :: proc(capacity : ^uint) -> Status ---

	@(link_name="cublasLtHeuristicsCacheSetCapacity")
	heuristicsCacheSetCapacity :: proc(capacity : uint) -> Status ---

	@(link_name="cublasLtDisableCpuInstructionsSetMask")
	disableCpuInstructionsSetMask :: proc(mask : u32) -> u32 ---

	@(link_name="cublasLtMatmul")
	matmul :: proc(lightHandle : Handle, computeDesc : MatmulDesc, alpha : rawptr, A : rawptr, Adesc : MatrixLayout, B : rawptr, Bdesc : MatrixLayout, beta : rawptr, C : rawptr, Cdesc : MatrixLayout, D : rawptr, Ddesc : MatrixLayout, algo : ^MatmulAlgo, workspace : rawptr, workspaceSizeInBytes : uint, stream : cuda.Stream) -> Status ---

	@(link_name="cublasLtMatrixTransform")
	matrixTransform :: proc(lightHandle : Handle, transformDesc : MatrixTransformDesc, alpha : rawptr, A : rawptr, Adesc : MatrixLayout, beta : rawptr, B : rawptr, Bdesc : MatrixLayout, C : rawptr, Cdesc : MatrixLayout, stream : cuda.Stream) -> Status ---

	@(link_name="cublasLtMatrixLayoutInit")
	matrixLayoutInit :: proc(matLayout : MatrixLayout, type : DataType, rows : u64, cols : u64, ld : i64) -> Status ---

	@(link_name="cublasLtMatrixLayoutCreate")
	matrixLayoutCreate :: proc(matLayout : ^MatrixLayout, type : DataType, rows : u64, cols : u64, ld : i64) -> Status ---

	@(link_name="cublasLtMatrixLayoutDestroy")
	matrixLayoutDestroy :: proc(matLayout : MatrixLayout) -> Status ---

	@(link_name="cublasLtMatrixLayoutSetAttribute")
	matrixLayoutSetAttribute :: proc(matLayout : MatrixLayout, attr : MatrixLayoutAttribute, buf : rawptr, sizeInBytes : uint) -> Status ---

	@(link_name="cublasLtMatrixLayoutGetAttribute")
	matrixLayoutGetAttribute :: proc(matLayout : MatrixLayout, attr : MatrixLayoutAttribute, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---

	@(link_name="cublasLtMatmulDescInit")
	matmulDescInit :: proc(matmulDesc : MatmulDesc, computeType : ComputeType, scaleType : DataType) -> Status ---

	@(link_name="cublasLtMatmulDescCreate")
	matmulDescCreate :: proc(matmulDesc : ^MatmulDesc, computeType : ComputeType, scaleType : DataType) -> Status ---

	@(link_name="cublasLtMatmulDescDestroy")
	matmulDescDestroy :: proc(matmulDesc : MatmulDesc) -> Status ---

	@(link_name="cublasLtMatmulDescSetAttribute")
	matmulDescSetAttribute :: proc(matmulDesc : MatmulDesc, attr : MatmulDescAttributes, buf : rawptr, sizeInBytes : uint) -> Status ---

	@(link_name="cublasLtMatmulDescGetAttribute")
	matmulDescGetAttribute :: proc(matmulDesc : MatmulDesc, attr : MatmulDescAttributes, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---

	@(link_name="cublasLtMatrixTransformDescInit")
	matrixTransformDescInit :: proc(transformDesc : MatrixTransformDesc, scaleType : DataType) -> Status ---

	@(link_name="cublasLtMatrixTransformDescCreate")
	matrixTransformDescCreate :: proc(transformDesc : ^MatrixTransformDesc, scaleType : DataType) -> Status ---

	@(link_name="cublasLtMatrixTransformDescDestroy")
	matrixTransformDescDestroy :: proc(transformDesc : MatrixTransformDesc) -> Status ---

	@(link_name="cublasLtMatrixTransformDescSetAttribute")
	matrixTransformDescSetAttribute :: proc(transformDesc : MatrixTransformDesc, attr : MatrixTransformDescAttributes, buf : rawptr, sizeInBytes : uint) -> Status ---

	@(link_name="cublasLtMatrixTransformDescGetAttribute")
	matrixTransformDescGetAttribute :: proc(transformDesc : MatrixTransformDesc, attr : MatrixTransformDescAttributes, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---

	@(link_name="cublasLtMatmulPreferenceInit")
	matmulPreferenceInit :: proc(pref : MatmulPreference) -> Status ---

	@(link_name="cublasLtMatmulPreferenceCreate")
	matmulPreferenceCreate :: proc(pref : ^MatmulPreference) -> Status ---

	@(link_name="cublasLtMatmulPreferenceDestroy")
	matmulPreferenceDestroy :: proc(pref : MatmulPreference) -> Status ---

	@(link_name="cublasLtMatmulPreferenceSetAttribute")
	matmulPreferenceSetAttribute :: proc(pref : MatmulPreference, attr : MatmulPreferenceAttributes, buf : rawptr, sizeInBytes : uint) -> Status ---

	@(link_name="cublasLtMatmulPreferenceGetAttribute")
	matmulPreferenceGetAttribute :: proc(pref : MatmulPreference, attr : MatmulPreferenceAttributes, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---

	@(link_name="cublasLtMatmulAlgoGetHeuristic")
	matmulAlgoGetHeuristic :: proc(lightHandle : Handle, operationDesc : MatmulDesc, Adesc : MatrixLayout, Bdesc : MatrixLayout, Cdesc : MatrixLayout, Ddesc : MatrixLayout, preference : MatmulPreference, requestedAlgoCount : i32, heuristicResultsArray : ^MatmulHeuristicResult, returnAlgoCount : ^i32) -> Status ---

	@(link_name="cublasLtMatmulAlgoGetIds")
	matmulAlgoGetIds :: proc(lightHandle : Handle, computeType : ComputeType, scaleType : DataType, Atype : DataType, Btype : DataType, Ctype : DataType, Dtype : DataType, requestedAlgoCount : i32, algoIdsArray : ^i32, returnAlgoCount : ^i32) -> Status ---

	@(link_name="cublasLtMatmulAlgoInit")
	matmulAlgoInit :: proc(lightHandle : Handle, computeType : ComputeType, scaleType : DataType, Atype : DataType, Btype : DataType, Ctype : DataType, Dtype : DataType, algoId : i32, algo : ^MatmulAlgo) -> Status ---

	@(link_name="cublasLtMatmulAlgoCheck")
	matmulAlgoCheck :: proc(lightHandle : Handle, operationDesc : MatmulDesc, Adesc : MatrixLayout, Bdesc : MatrixLayout, Cdesc : MatrixLayout, Ddesc : MatrixLayout, algo : ^MatmulAlgo, result : ^MatmulHeuristicResult) -> Status ---

	@(link_name="cublasLtMatmulAlgoCapGetAttribute")
	matmulAlgoCapGetAttribute :: proc(algo : ^MatmulAlgo, attr : MatmulAlgoCapAttributes, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---

	@(link_name="cublasLtMatmulAlgoConfigSetAttribute")
	matmulAlgoConfigSetAttribute :: proc(algo : ^MatmulAlgo, attr : MatmulAlgoConfigAttributes, buf : rawptr, sizeInBytes : uint) -> Status ---

	@(link_name="cublasLtMatmulAlgoConfigGetAttribute")
	matmulAlgoConfigGetAttribute :: proc(algo : ^MatmulAlgo, attr : MatmulAlgoConfigAttributes, buf : rawptr, sizeInBytes : uint, sizeWritten : ^uint) -> Status ---
}

// Register cublasLt handle in cuda user context
register_handle :: proc(ctx: ^cuda.UserContext = nil) {
	destroy_handle :: proc(key: string, h: rawptr) {
		destroy(cast(Handle)h)
	}
	h: Handle
	must(create(&h))
	cuda.register_handle("cublas", h, destroy_handle, ctx)
}

// Convenience wrapper for matrix multiply with optional bias
//  C = alpha * op(A) @ op(B) + beta * C
// where op is optional transpose
//    op(A) = Array[m, k]
//    op(B) = Array[k, n]
//    C     = Array[m, n]
//
// Array layout is in fully packed row major format. A, B, C and bias must all have the same type.
// If bias is set then bias pointer is the input bias which is added to C after the calculation
// If dbias is set then bias pointer is the output bias gradient from sum over rows of A 
//
// Internally the transpose and argument order is reversed to use column major as the cublas API prefers this.
//
// Note: must call register_handle at startup to store the cublas handle in the cuda user context.
//
gemm :: proc (type: DataType, a_trans, b_trans: Operation, A, B, C: rawptr, m, n, k: int, alpha: f32 = 1, beta: f32 = 0, bias: rawptr = nil, 
				dbias := false, stream: cuda.Stream = nil, max_workspace: u64 = DEFAULT_MAX_WORKSPACE, loc := #caller_location) -> Status {
	type, a_trans, b_trans, alpha, beta := type, a_trans, b_trans, alpha, beta

	handle := cast(Handle)cuda.get_handle("cublas", loc=loc)
	assert(uintptr(A) % 16 == 0 && uintptr(B) % 16 == 0 && uintptr(C) % 16 == 0, "matrix pointers not aligned", loc)

	// all swapped over as input is row major but matmul call is col major
	adesc, bdesc, cdesc: MatrixLayout
	adims: [2]int = a_trans != .OP_N ? { m, k } : { k, m }
	bdims: [2]int = b_trans != .OP_N ? { k, n } : { n, k }
	matrixLayoutCreate(&adesc, type, u64(adims[0]), u64(adims[1]), i64(adims[0]))
	defer matrixLayoutDestroy(adesc)
	matrixLayoutCreate(&bdesc, type, u64(bdims[0]), u64(bdims[1]), i64(bdims[0]))
	defer matrixLayoutDestroy(bdesc)
	matrixLayoutCreate(&cdesc, type, u64(n), u64(m), i64(n))
	defer matrixLayoutDestroy(cdesc)

	// computation and alpha, beta scale factors are always float32
	desc : MatmulDesc
	must(matmulDescCreate(&desc, .COMPUTE_32F, .R_32F))
	defer matmulDescDestroy(desc)
	must(matmulDescSetAttribute(desc, .TRANSA, &b_trans, size_of(b_trans)))
	must(matmulDescSetAttribute(desc, .TRANSB, &a_trans, size_of(a_trans)))

	if bias != nil {
		bias_ptr := bias
		assert(uintptr(bias_ptr) % 16 == 0, "bias pointer not aligned", loc)
		// dbias is for a matrix - b & a are swapped
		epilogue: Epilogue = dbias ? .EPILOGUE_BGRADB : .EPILOGUE_BIAS 
		must(matmulDescSetAttribute(desc, .EPILOGUE, &epilogue, size_of(epilogue)))
		must(matmulDescSetAttribute(desc, .BIAS_DATA_TYPE, &type, size_of(type)))
		must(matmulDescSetAttribute(desc, .BIAS_POINTER, &bias_ptr, size_of(bias_ptr)))
	}

	// get best algorithm from heuristics - will use cache if previously queried
	pref: MatmulPreference
	must(matmulPreferenceCreate(&pref))
	defer matmulPreferenceDestroy(pref)
	max_workspace := max_workspace
	must(matmulPreferenceSetAttribute(pref, .MAX_WORKSPACE_BYTES, &max_workspace, size_of(max_workspace)))
	if type == .R_16BF {
		// accumulate in float32 when using bfloat16 inputs
		mask := ReductionScheme.COMPUTE_TYPE
		must(matmulPreferenceSetAttribute(pref, .REDUCTION_SCHEME_MASK, &mask, size_of(mask)))		
	}
	heur: MatmulHeuristicResult
	count: i32
	must(matmulAlgoGetHeuristic(handle, desc, bdesc, adesc, cdesc, cdesc, pref, 1, &heur, &count))
	if count == 0 {
		return .NOT_SUPPORTED
	}

	// launch kernel - with workspace if needed
	ws: rawptr
	if heur.workspaceSize > 0 {
		ws = cuda.memalloc(int(heur.workspaceSize))
	}
	must(matmul(handle, desc, &alpha, B, bdesc, A, adesc, &beta, C, cdesc, C, cdesc, &heur.algo, ws, heur.workspaceSize, stream))
	if ws != nil {
		cuda.memfree(ws)
	}
	return .SUCCESS
}

// Error checking
error :: proc(rc: Status) -> string {
	return string(getStatusName(rc))
}

must :: proc(rc: Status, loc := #caller_location) {
	if rc != .SUCCESS {
		panic(error(rc), loc)
	}
}
