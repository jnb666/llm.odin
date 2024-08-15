package cublas

Status :: enum i32 {
	SUCCESS          = 0,
	NOT_INITIALIZED  = 1,
	ALLOC_FAILED     = 3,
	INVALID_VALUE    = 7,
	ARCH_MISMATCH    = 8,
	MAPPING_ERROR    = 11,
	EXECUTION_FAILED = 13,
	INTERNAL_ERROR   = 14,
	NOT_SUPPORTED    = 15,
	LICENSE_ERROR    = 16,
}

LibraryPropertyType :: enum i32 {
	MAJOR_VERSION,
	MINOR_VERSION,
	PATCH_LEVEL,
}

FillMode :: enum i32 {
	FILL_MODE_LOWER = 0,
	FILL_MODE_UPPER = 1,
	FILL_MODE_FULL  = 2,
}

DiagType :: enum i32 {
	DIAG_NON_UNIT = 0,
	DIAG_UNIT     = 1,
}

SideMode :: enum i32 {
	SIDE_LEFT  = 0,
	SIDE_RIGHT = 1,
}

Operation :: enum i32 {
	OP_N        = 0,
	OP_T        = 1,
	OP_C        = 2,
	OP_HERMITAN = 2,
	OP_CONJG    = 3,
}

AtomicsMode :: enum i32 {
	ATOMICS_NOT_ALLOWED = 0,
	ATOMICS_ALLOWED     = 1,
}

GemmAlgo :: enum i32 {
	DEFAULT           = -1,
	ALGO0             = 0,
	ALGO1             = 1,
	ALGO2             = 2,
	ALGO3             = 3,
	ALGO4             = 4,
	ALGO5             = 5,
	ALGO6             = 6,
	ALGO7             = 7,
	ALGO8             = 8,
	ALGO9             = 9,
	ALGO10            = 10,
	ALGO11            = 11,
	ALGO12            = 12,
	ALGO13            = 13,
	ALGO14            = 14,
	ALGO15            = 15,
	ALGO16            = 16,
	ALGO17            = 17,
	ALGO18            = 18,
	ALGO19            = 19,
	ALGO20            = 20,
	ALGO21            = 21,
	ALGO22            = 22,
	ALGO23            = 23,
	DEFAULT_TENSOR_OP = 99,
	ALGO0_TENSOR_OP   = 100,
	ALGO1_TENSOR_OP   = 101,
	ALGO2_TENSOR_OP   = 102,
	ALGO3_TENSOR_OP   = 103,
	ALGO4_TENSOR_OP   = 104,
	ALGO5_TENSOR_OP   = 105,
	ALGO6_TENSOR_OP   = 106,
	ALGO7_TENSOR_OP   = 107,
	ALGO8_TENSOR_OP   = 108,
	ALGO9_TENSOR_OP   = 109,
	ALGO10_TENSOR_OP  = 110,
	ALGO11_TENSOR_OP  = 111,
	ALGO12_TENSOR_OP  = 112,
	ALGO13_TENSOR_OP  = 113,
	ALGO14_TENSOR_OP  = 114,
	ALGO15_TENSOR_OP  = 115,
}

Math :: enum i32 {
	DEFAULT_MATH                              = 0,
	TENSOR_OP_MATH                            = 1,
	PEDANTIC_MATH                             = 2,
	TF32_TENSOR_OP_MATH                       = 3,
	MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
}

DataType :: enum i32 {
	R_16F     = 2, /* real as a half */
	C_16F     = 6, /* complex as a pair of half numbers */
	R_16BF    = 14, /* real as a nv_bfloat16 */
	C_16BF    = 15, /* complex as a pair of nv_bfloat16 numbers */
	R_32F     = 0, /* real as a float */
	C_32F     = 4, /* complex as a pair of float numbers */
	R_64F     = 1, /* real as a double */
	C_64F     = 5, /* complex as a pair of double numbers */
	R_4I      = 16, /* real as a signed 4-bit int */
	C_4I      = 17, /* complex as a pair of signed 4-bit int numbers */
	R_4U      = 18, /* real as a unsigned 4-bit int */
	C_4U      = 19, /* complex as a pair of unsigned 4-bit int numbers */
	R_8I      = 3, /* real as a signed 8-bit int */
	C_8I      = 7, /* complex as a pair of signed 8-bit int numbers */
	R_8U      = 8, /* real as a unsigned 8-bit int */
	C_8U      = 9, /* complex as a pair of unsigned 8-bit int numbers */
	R_16I     = 20, /* real as a signed 16-bit int */
	C_16I     = 21, /* complex as a pair of signed 16-bit int numbers */
	R_16U     = 22, /* real as a unsigned 16-bit int */
	C_16U     = 23, /* complex as a pair of unsigned 16-bit int numbers */
	R_32I     = 10, /* real as a signed 32-bit int */
	C_32I     = 11, /* complex as a pair of signed 32-bit int numbers */
	R_32U     = 12, /* real as a unsigned 32-bit int */
	C_32U     = 13, /* complex as a pair of unsigned 32-bit int numbers */
	R_64I     = 24, /* real as a signed 64-bit int */
	C_64I     = 25, /* complex as a pair of signed 64-bit int numbers */
	R_64U     = 26, /* real as a unsigned 64-bit int */
	C_64U     = 27, /* complex as a pair of unsigned 64-bit int numbers */
	R_8F_E4M3 = 28, /* real as a nv_fp8_e4m3 */
	R_8F_E5M2 = 29, /* real as a nv_fp8_e5m2 */
}

ComputeType :: enum i32 {
	COMPUTE_16F           = 64,
	COMPUTE_16F_PEDANTIC  = 65,
	COMPUTE_32F           = 68,
	COMPUTE_32F_PEDANTIC  = 69,
	COMPUTE_32F_FAST_16F  = 74,
	COMPUTE_32F_FAST_16BF = 75,
	COMPUTE_32F_FAST_TF32 = 77,
	COMPUTE_64F           = 70,
	COMPUTE_64F_PEDANTIC  = 71,
	COMPUTE_32I           = 72,
	COMPUTE_32I_PEDANTIC  = 73,
}

MatmulTile :: enum i32 {
	MATMUL_TILE_UNDEFINED = 0,
	MATMUL_TILE_8x8 = 1,
	MATMUL_TILE_8x16 = 2,
	MATMUL_TILE_16x8 = 3,
	MATMUL_TILE_8x32 = 4,
	MATMUL_TILE_16x16 = 5,
	MATMUL_TILE_32x8 = 6,
	MATMUL_TILE_8x64 = 7,
	MATMUL_TILE_16x32 = 8,
	MATMUL_TILE_32x16 = 9,
	MATMUL_TILE_64x8 = 10,
	MATMUL_TILE_32x32 = 11,
	MATMUL_TILE_32x64 = 12,
	MATMUL_TILE_64x32 = 13,
	MATMUL_TILE_32x128 = 14,
	MATMUL_TILE_64x64 = 15,
	MATMUL_TILE_128x32 = 16,
	MATMUL_TILE_64x128 = 17,
	MATMUL_TILE_128x64 = 18,
	MATMUL_TILE_64x256 = 19,
	MATMUL_TILE_128x128 = 20,
	MATMUL_TILE_256x64 = 21,
	MATMUL_TILE_64x512 = 22,
	MATMUL_TILE_128x256 = 23,
	MATMUL_TILE_256x128 = 24,
	MATMUL_TILE_512x64 = 25,
	MATMUL_TILE_64x96 = 26,
	MATMUL_TILE_96x64 = 27,
	MATMUL_TILE_96x128 = 28,
	MATMUL_TILE_128x160 = 29,
	MATMUL_TILE_160x128 = 30,
	MATMUL_TILE_192x128 = 31,
	MATMUL_TILE_128x192 = 32,
	MATMUL_TILE_128x96 = 33,
	MATMUL_TILE_32x256 = 34,
	MATMUL_TILE_256x32 = 35,
	MATMUL_TILE_END,
}

MatmulStages :: enum i32 {
	MATMUL_STAGES_UNDEFINED = 0,
	MATMUL_STAGES_16x1 = 1,
	MATMUL_STAGES_16x2 = 2,
	MATMUL_STAGES_16x3 = 3,
	MATMUL_STAGES_16x4 = 4,
	MATMUL_STAGES_16x5 = 5,
	MATMUL_STAGES_16x6 = 6,
	MATMUL_STAGES_32x1 = 7,
	MATMUL_STAGES_32x2 = 8,
	MATMUL_STAGES_32x3 = 9,
	MATMUL_STAGES_32x4 = 10,
	MATMUL_STAGES_32x5 = 11,
	MATMUL_STAGES_32x6 = 12,
	MATMUL_STAGES_64x1 = 13,
	MATMUL_STAGES_64x2 = 14,
	MATMUL_STAGES_64x3 = 15,
	MATMUL_STAGES_64x4 = 16,
	MATMUL_STAGES_64x5 = 17,
	MATMUL_STAGES_64x6 = 18,
	MATMUL_STAGES_128x1 = 19,
	MATMUL_STAGES_128x2 = 20,
	MATMUL_STAGES_128x3 = 21,
	MATMUL_STAGES_128x4 = 22,
	MATMUL_STAGES_128x5 = 23,
	MATMUL_STAGES_128x6 = 24,
	MATMUL_STAGES_32x10 = 25,
	MATMUL_STAGES_8x4 = 26,
	MATMUL_STAGES_16x10 = 27,
	MATMUL_STAGES_8x5 = 28,
	MATMUL_STAGES_8x3 = 31,
	MATMUL_STAGES_8xAUTO = 32,
	MATMUL_STAGES_16xAUTO = 33,
	MATMUL_STAGES_32xAUTO = 34,
	MATMUL_STAGES_64xAUTO = 35,
	MATMUL_STAGES_128xAUTO = 36,
	MATMUL_STAGES_END,
}

ClusterShape :: enum i32 {
	CLUSTER_SHAPE_AUTO = 0,
	CLUSTER_SHAPE_1x1x1 = 2,
	CLUSTER_SHAPE_2x1x1 = 3,
	CLUSTER_SHAPE_4x1x1 = 4,
	CLUSTER_SHAPE_1x2x1 = 5,
	CLUSTER_SHAPE_2x2x1 = 6,
	CLUSTER_SHAPE_4x2x1 = 7,
	CLUSTER_SHAPE_1x4x1 = 8,
	CLUSTER_SHAPE_2x4x1 = 9,
	CLUSTER_SHAPE_4x4x1 = 10,
	CLUSTER_SHAPE_8x1x1 = 11,
	CLUSTER_SHAPE_1x8x1 = 12,
	CLUSTER_SHAPE_8x2x1 = 13,
	CLUSTER_SHAPE_2x8x1 = 14,
	CLUSTER_SHAPE_16x1x1 = 15,
	CLUSTER_SHAPE_1x16x1 = 16,
	CLUSTER_SHAPE_3x1x1 = 17,
	CLUSTER_SHAPE_5x1x1 = 18,
	CLUSTER_SHAPE_6x1x1 = 19,
	CLUSTER_SHAPE_7x1x1 = 20,
	CLUSTER_SHAPE_9x1x1 = 21,
	CLUSTER_SHAPE_10x1x1 = 22,
	CLUSTER_SHAPE_11x1x1 = 23,
	CLUSTER_SHAPE_12x1x1 = 24,
	CLUSTER_SHAPE_13x1x1 = 25,
	CLUSTER_SHAPE_14x1x1 = 26,
	CLUSTER_SHAPE_15x1x1 = 27,
	CLUSTER_SHAPE_3x2x1 = 28,
	CLUSTER_SHAPE_5x2x1 = 29,
	CLUSTER_SHAPE_6x2x1 = 30,
	CLUSTER_SHAPE_7x2x1 = 31,
	CLUSTER_SHAPE_1x3x1 = 32,
	CLUSTER_SHAPE_2x3x1 = 33,
	CLUSTER_SHAPE_3x3x1 = 34,
	CLUSTER_SHAPE_4x3x1 = 35,
	CLUSTER_SHAPE_5x3x1 = 36,
	CLUSTER_SHAPE_3x4x1 = 37,
	CLUSTER_SHAPE_1x5x1 = 38,
	CLUSTER_SHAPE_2x5x1 = 39,
	CLUSTER_SHAPE_3x5x1 = 40,
	CLUSTER_SHAPE_1x6x1 = 41,
	CLUSTER_SHAPE_2x6x1 = 42,
	CLUSTER_SHAPE_1x7x1 = 43,
	CLUSTER_SHAPE_2x7x1 = 44,
	CLUSTER_SHAPE_1x9x1 = 45,
	CLUSTER_SHAPE_1x10x1 = 46,
	CLUSTER_SHAPE_1x11x1 = 47,
	CLUSTER_SHAPE_1x12x1 = 48,
	CLUSTER_SHAPE_1x13x1 = 49,
	CLUSTER_SHAPE_1x14x1 = 50,
	CLUSTER_SHAPE_1x15x1 = 51,
	CLUSTER_SHAPE_END,
}

MatmulInnerShape :: enum i32 {
	MATMUL_INNER_SHAPE_UNDEFINED = 0,
	MATMUL_INNER_SHAPE_MMA884 = 1,
	MATMUL_INNER_SHAPE_MMA1684 = 2,
	MATMUL_INNER_SHAPE_MMA1688 = 3,
	MATMUL_INNER_SHAPE_MMA16816 = 4,
	MATMUL_INNER_SHAPE_END,
}

PointerMode :: enum i32 {
	POINTER_MODE_HOST                          = 0,
	POINTER_MODE_DEVICE                        = 1,
	POINTER_MODE_DEVICE_VECTOR                 = 2,
	POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3,
	POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = 4,
}

PointerModeMask :: enum i32 {
	POINTER_MODE_MASK_HOST                          = 1,
	POINTER_MODE_MASK_DEVICE                        = 2,
	POINTER_MODE_MASK_DEVICE_VECTOR                 = 4,
	POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO = 8,
	POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST = 16,
}

Order :: enum i32 {
	ORDER_COL          = 0,
	ORDER_ROW          = 1,
	ORDER_COL32        = 2,
	ORDER_COL4_4R2_8C  = 3,
	ORDER_COL32_2R_4R4 = 4,
}

MatrixLayoutAttribute :: enum i32 {
	MATRIX_LAYOUT_TYPE                 = 0,
	MATRIX_LAYOUT_ORDER                = 1,
	MATRIX_LAYOUT_ROWS                 = 2,
	MATRIX_LAYOUT_COLS                 = 3,
	MATRIX_LAYOUT_LD                   = 4,
	MATRIX_LAYOUT_BATCH_COUNT          = 5,
	MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
	MATRIX_LAYOUT_PLANE_OFFSET         = 7,
}

MatmulDescAttributes :: enum i32 {
	COMPUTE_TYPE                     = 0,
	SCALE_TYPE                       = 1,
	POINTER_MODE                     = 2,
	TRANSA                           = 3,
	TRANSB                           = 4,
	TRANSC                           = 5,
	FILL_MODE                        = 6,
	EPILOGUE                         = 7,
	BIAS_POINTER                     = 8,
	BIAS_BATCH_STRIDE                = 10,
	EPILOGUE_AUX_POINTER             = 11,
	EPILOGUE_AUX_LD                  = 12,
	EPILOGUE_AUX_BATCH_STRIDE        = 13,
	ALPHA_VECTOR_BATCH_STRIDE        = 14,
	SM_COUNT_TARGET                  = 15,
	A_SCALE_POINTER                  = 17,
	B_SCALE_POINTER                  = 18,
	C_SCALE_POINTER                  = 19,
	D_SCALE_POINTER                  = 20,
	AMAX_D_POINTER                   = 21,
	EPILOGUE_AUX_DATA_TYPE           = 22,
	EPILOGUE_AUX_SCALE_POINTER       = 23,
	EPILOGUE_AUX_AMAX_POINTER        = 24,
	FAST_ACCUM                       = 25,
	BIAS_DATA_TYPE                   = 26,
	ATOMIC_SYNC_NUM_CHUNKS_D_ROWS    = 27,
	ATOMIC_SYNC_NUM_CHUNKS_D_COLS    = 28,
	ATOMIC_SYNC_IN_COUNTERS_POINTER  = 29,
	ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
}

MatrixTransformDescAttributes :: enum i32 {
	SCALE_TYPE,
	POINTER_MODE,
	TRANSA,
	TRANSB,
}

Epilogue :: enum i32 {
	EPILOGUE_DEFAULT       = 1,
	EPILOGUE_RELU          = 2,
	EPILOGUE_RELU_AUX      = (EPILOGUE_RELU | 128),
	EPILOGUE_BIAS          = 4,
	EPILOGUE_RELU_BIAS     = (EPILOGUE_RELU | EPILOGUE_BIAS),
	EPILOGUE_RELU_AUX_BIAS = (EPILOGUE_RELU_AUX | EPILOGUE_BIAS),
	EPILOGUE_DRELU         = 8 | 128,
	EPILOGUE_DRELU_BGRAD   = EPILOGUE_DRELU | 16,
	EPILOGUE_GELU          = 32,
	EPILOGUE_GELU_AUX      = (EPILOGUE_GELU | 128),
	EPILOGUE_GELU_BIAS     = (EPILOGUE_GELU | EPILOGUE_BIAS),
	EPILOGUE_GELU_AUX_BIAS = (EPILOGUE_GELU_AUX | EPILOGUE_BIAS),
	EPILOGUE_DGELU         = 64 | 128,
	EPILOGUE_DGELU_BGRAD   = EPILOGUE_DGELU | 16,
	EPILOGUE_BGRADA        = 256,
	EPILOGUE_BGRADB        = 512,
}

ReductionScheme :: enum i32 {
	NONE         = 0,
	INPLACE      = 1,
	COMPUTE_TYPE = 2,
	OUTPUT_TYPE  = 4,
	MASK         = 7,
}

MatmulSearch :: enum i32 {
	SEARCH_BEST_FIT           = 0,
	SEARCH_LIMITED_BY_ALGO_ID = 1,
	SEARCH_RESERVED_02        = 2,
	SEARCH_RESERVED_03        = 3,
	SEARCH_RESERVED_04        = 4,
	SEARCH_RESERVED_05        = 5,
}

MatmulPreferenceAttributes :: enum i32 {
	SEARCH_MODE           = 0,
	MAX_WORKSPACE_BYTES   = 1,
	REDUCTION_SCHEME_MASK = 3,
	MIN_ALIGNMENT_A_BYTES = 5,
	MIN_ALIGNMENT_B_BYTES = 6,
	MIN_ALIGNMENT_C_BYTES = 7,
	MIN_ALIGNMENT_D_BYTES = 8,
	MAX_WAVES_COUNT       = 9,
	IMPL_MASK             = 12,
}

MatmulAlgoCapAttributes :: enum i32 {
	ALGO_CAP_SPLITK_SUPPORT              = 0,
	ALGO_CAP_REDUCTION_SCHEME_MASK       = 1,
	ALGO_CAP_CTA_SWIZZLING_SUPPORT       = 2,
	ALGO_CAP_STRIDED_BATCH_SUPPORT       = 3,
	ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT = 4,
	ALGO_CAP_UPLO_SUPPORT                = 5,
	ALGO_CAP_TILE_IDS                    = 6,
	ALGO_CAP_CUSTOM_OPTION_MAX           = 7,
	ALGO_CAP_CUSTOM_MEMORY_ORDER         = 10,
	ALGO_CAP_POINTER_MODE_MASK           = 11,
	ALGO_CAP_EPILOGUE_MASK               = 12,
	ALGO_CAP_STAGES_IDS                  = 13,
	ALGO_CAP_LD_NEGATIVE                 = 14,
	ALGO_CAP_NUMERICAL_IMPL_FLAGS        = 15,
	ALGO_CAP_MIN_ALIGNMENT_A_BYTES       = 16,
	ALGO_CAP_MIN_ALIGNMENT_B_BYTES       = 17,
	ALGO_CAP_MIN_ALIGNMENT_C_BYTES       = 18,
	ALGO_CAP_MIN_ALIGNMENT_D_BYTES       = 19,
	ALGO_CAP_ATOMIC_SYNC                 = 20,
}

MatmulAlgoConfigAttributes :: enum i32 {
	ALGO_CONFIG_ID               = 0,
	ALGO_CONFIG_TILE_ID          = 1,
	ALGO_CONFIG_SPLITK_NUM       = 2,
	ALGO_CONFIG_REDUCTION_SCHEME = 3,
	ALGO_CONFIG_CTA_SWIZZLING    = 4,
	ALGO_CONFIG_CUSTOM_OPTION    = 5,
	ALGO_CONFIG_STAGES_ID        = 6,
	ALGO_CONFIG_INNER_SHAPE_ID   = 7,
	ALGO_CONFIG_CLUSTER_SHAPE_ID = 8,
}
