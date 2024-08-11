package cuda

Result :: enum i32 {
	SUCCESS                        = 0,
	INVALID_VALUE                  = 1,
	OUT_OF_MEMORY                  = 2,
	NOT_INITIALIZED                = 3,
	DEINITIALIZED                  = 4,
	PROFILER_DISABLED              = 5,
	PROFILER_NOT_INITIALIZED       = 6,
	PROFILER_ALREADY_STARTED       = 7,
	PROFILER_ALREADY_STOPPED       = 8,
	STUB_LIBRARY                   = 34,
	DEVICE_UNAVAILABLE             = 46,
	NO_DEVICE                      = 100,
	INVALID_DEVICE                 = 101,
	DEVICE_NOT_LICENSED            = 102,
	INVALID_IMAGE                  = 200,
	INVALID_CONTEXT                = 201,
	CONTEXT_ALREADY_CURRENT        = 202,
	MAP_FAILED                     = 205,
	UNMAP_FAILED                   = 206,
	ARRAY_IS_MAPPED                = 207,
	ALREADY_MAPPED                 = 208,
	NO_BINARY_FOR_GPU              = 209,
	ALREADY_ACQUIRED               = 210,
	NOT_MAPPED                     = 211,
	NOT_MAPPED_AS_ARRAY            = 212,
	NOT_MAPPED_AS_POINTER          = 213,
	ECC_UNCORRECTABLE              = 214,
	UNSUPPORTED_LIMIT              = 215,
	CONTEXT_ALREADY_IN_USE         = 216,
	PEER_ACCESS_UNSUPPORTED        = 217,
	INVALID_PTX                    = 218,
	INVALID_GRAPHICS_CONTEXT       = 219,
	NVLINK_UNCORRECTABLE           = 220,
	JIT_COMPILER_NOT_FOUND         = 221,
	UNSUPPORTED_PTX_VERSION        = 222,
	JIT_COMPILATION_DISABLED       = 223,
	UNSUPPORTED_EXEC_AFFINITY      = 224,
	UNSUPPORTED_DEVSIDE_SYNC       = 225,
	INVALID_SOURCE                 = 300,
	FILE_NOT_FOUND                 = 301,
	SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
	SHARED_OBJECT_INIT_FAILED      = 303,
	OPERATING_SYSTEM               = 304,
	INVALID_HANDLE                 = 400,
	ILLEGAL_STATE                  = 401,
	LOSSY_QUERY                    = 402,
	NOT_FOUND                      = 500,
	NOT_READY                      = 600,
	ILLEGAL_ADDRESS                = 700,
	LAUNCH_OUT_OF_RESOURCES        = 701,
	LAUNCH_TIMEOUT                 = 702,
	LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
	PEER_ACCESS_ALREADY_ENABLED    = 704,
	PEER_ACCESS_NOT_ENABLED        = 705,
	PRIMARY_CONTEXT_ACTIVE         = 708,
	CONTEXT_IS_DESTROYED           = 709,
	ASSERT                         = 710,
	TOO_MANY_PEERS                 = 711,
	HOST_MEMORY_ALREADY_REGISTERED = 712,
	HOST_MEMORY_NOT_REGISTERED     = 713,
	HARDWARE_STACK_ERROR           = 714,
	ILLEGAL_INSTRUCTION            = 715,
	MISALIGNED_ADDRESS             = 716,
	INVALID_ADDRESS_SPACE          = 717,
	INVALID_PC                     = 718,
	LAUNCH_FAILED                  = 719,
	COOPERATIVE_LAUNCH_TOO_LARGE   = 720,
	NOT_PERMITTED                  = 800,
	NOT_SUPPORTED                  = 801,
	SYSTEM_NOT_READY               = 802,
	SYSTEM_DRIVER_MISMATCH         = 803,
	COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
	MPS_CONNECTION_FAILED          = 805,
	MPS_RPC_FAILURE                = 806,
	MPS_SERVER_NOT_READY           = 807,
	MPS_MAX_CLIENTS_REACHED        = 808,
	MPS_MAX_CONNECTIONS_REACHED    = 809,
	MPS_CLIENT_TERMINATED          = 810,
	CDP_NOT_SUPPORTED              = 811,
	CDP_VERSION_MISMATCH           = 812,
	STREAM_CAPTURE_UNSUPPORTED     = 900,
	STREAM_CAPTURE_INVALIDATED     = 901,
	STREAM_CAPTURE_MERGE           = 902,
	STREAM_CAPTURE_UNMATCHED       = 903,
	STREAM_CAPTURE_UNJOINED        = 904,
	STREAM_CAPTURE_ISOLATION       = 905,
	STREAM_CAPTURE_IMPLICIT        = 906,
	CAPTURED_EVENT                 = 907,
	STREAM_CAPTURE_WRONG_THREAD    = 908,
	TIMEOUT                        = 909,
	GRAPH_EXEC_UPDATE_FAILURE      = 910,
	EXTERNAL_DEVICE                = 911,
	INVALID_CLUSTER_SIZE           = 912,
	FUNCTION_NOT_LOADED            = 913,
	INVALID_RESOURCE_TYPE          = 914,
	INVALID_RESOURCE_CONFIGURATION = 915,
	UNKNOWN                        = 999,
}

Device_Attribute :: enum i32 {
    MAX_THREADS_PER_BLOCK = 1,                          /**< Maximum number of threads per block */
    MAX_BLOCK_DIM_X = 2,                                /**< Maximum block dimension X */
    MAX_BLOCK_DIM_Y = 3,                                /**< Maximum block dimension Y */
    MAX_BLOCK_DIM_Z = 4,                                /**< Maximum block dimension Z */
    MAX_GRID_DIM_X = 5,                                 /**< Maximum grid dimension X */
    MAX_GRID_DIM_Y = 6,                                 /**< Maximum grid dimension Y */
    MAX_GRID_DIM_Z = 7,                                 /**< Maximum grid dimension Z */
    MAX_SHARED_MEMORY_PER_BLOCK = 8,                    /**< Maximum shared memory available per block in bytes */
    SHARED_MEMORY_PER_BLOCK = 8,                        /**< Deprecated, use MAX_SHARED_MEMORY_PER_BLOCK */
    TOTAL_CONSTANT_MEMORY = 9,                          /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    WARP_SIZE = 10,                                     /**< Warp size in threads */
    MAX_PITCH = 11,                                     /**< Maximum pitch in bytes allowed by memory copies */
    MAX_REGISTERS_PER_BLOCK = 12,                       /**< Maximum number of 32-bit registers available per block */
    REGISTERS_PER_BLOCK = 12,                           /**< Deprecated, use MAX_REGISTERS_PER_BLOCK */
    CLOCK_RATE = 13,                                    /**< Typical clock frequency in kilohertz */
    TEXTURE_ALIGNMENT = 14,                             /**< Alignment requirement for textures */
    GPU_OVERLAP = 15,                                   /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead ASYNC_ENGINE_COUNT. */
    MULTIPROCESSOR_COUNT = 16,                          /**< Number of multiprocessors on device */
    KERNEL_EXEC_TIMEOUT = 17,                           /**< Specifies whether there is a run time limit on kernels */
    INTEGRATED = 18,                                    /**< Device is integrated with host memory */
    CAN_MAP_HOST_MEMORY = 19,                           /**< Device can map host memory into CUDA address space */
    COMPUTE_MODE = 20,                                  /**< Compute mode (See ::CUcomputemode for details) */
    MAXIMUM_TEXTURE1D_WIDTH = 21,                       /**< Maximum 1D texture width */
    MAXIMUM_TEXTURE2D_WIDTH = 22,                       /**< Maximum 2D texture width */
    MAXIMUM_TEXTURE2D_HEIGHT = 23,                      /**< Maximum 2D texture height */
    MAXIMUM_TEXTURE3D_WIDTH = 24,                       /**< Maximum 3D texture width */
    MAXIMUM_TEXTURE3D_HEIGHT = 25,                      /**< Maximum 3D texture height */
    MAXIMUM_TEXTURE3D_DEPTH = 26,                       /**< Maximum 3D texture depth */
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,               /**< Maximum 2D layered texture width */
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,              /**< Maximum 2D layered texture height */
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,              /**< Maximum layers in a 2D layered texture */
    MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,                 /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,                /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,             /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    SURFACE_ALIGNMENT = 30,                             /**< Alignment requirement for surfaces */
    CONCURRENT_KERNELS = 31,                            /**< Device can possibly execute multiple kernels concurrently */
    ECC_ENABLED = 32,                                   /**< Device has ECC support enabled */
    PCI_BUS_ID = 33,                                    /**< PCI bus ID of the device */
    PCI_DEVICE_ID = 34,                                 /**< PCI device ID of the device */
    TCC_DRIVER = 35,                                    /**< Device is using TCC driver model */
    MEMORY_CLOCK_RATE = 36,                             /**< Peak memory clock frequency in kilohertz */
    GLOBAL_MEMORY_BUS_WIDTH = 37,                       /**< Global memory bus width in bits */
    L2_CACHE_SIZE = 38,                                 /**< Size of L2 cache in bytes */
    MAX_THREADS_PER_MULTIPROCESSOR = 39,                /**< Maximum resident threads per multiprocessor */
    ASYNC_ENGINE_COUNT = 40,                            /**< Number of asynchronous engines */
    UNIFIED_ADDRESSING = 41,                            /**< Device shares a unified address space with the host */
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,               /**< Maximum 1D layered texture width */
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,              /**< Maximum layers in a 1D layered texture */
    CAN_TEX2D_GATHER = 44,                              /**< Deprecated, do not use. */
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,                /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,               /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,             /**< Alternate maximum 3D texture width */
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,            /**< Alternate maximum 3D texture height */
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,             /**< Alternate maximum 3D texture depth */
    PCI_DOMAIN_ID = 50,                                 /**< PCI domain ID of the device */
    TEXTURE_PITCH_ALIGNMENT = 51,                       /**< Pitch alignment requirement for textures */
    MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,                  /**< Maximum cubemap texture width/height */
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,          /**< Maximum cubemap layered texture width/height */
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,         /**< Maximum layers in a cubemap layered texture */
    MAXIMUM_SURFACE1D_WIDTH = 55,                       /**< Maximum 1D surface width */
    MAXIMUM_SURFACE2D_WIDTH = 56,                       /**< Maximum 2D surface width */
    MAXIMUM_SURFACE2D_HEIGHT = 57,                      /**< Maximum 2D surface height */
    MAXIMUM_SURFACE3D_WIDTH = 58,                       /**< Maximum 3D surface width */
    MAXIMUM_SURFACE3D_HEIGHT = 59,                      /**< Maximum 3D surface height */
    MAXIMUM_SURFACE3D_DEPTH = 60,                       /**< Maximum 3D surface depth */
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,               /**< Maximum 1D layered surface width */
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,              /**< Maximum layers in a 1D layered surface */
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,               /**< Maximum 2D layered surface width */
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,              /**< Maximum 2D layered surface height */
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,              /**< Maximum layers in a 2D layered surface */
    MAXIMUM_SURFACECUBEMAP_WIDTH = 66,                  /**< Maximum cubemap surface width */
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,          /**< Maximum cubemap layered surface width */
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,         /**< Maximum layers in a cubemap layered surface */
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,                /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,                /**< Maximum 2D linear texture width */
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,               /**< Maximum 2D linear texture height */
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,                /**< Maximum 2D linear texture pitch in bytes */
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,             /**< Maximum mipmapped 2D texture width */
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,            /**< Maximum mipmapped 2D texture height */
    COMPUTE_CAPABILITY_MAJOR = 75,                      /**< Major compute capability version number */
    COMPUTE_CAPABILITY_MINOR = 76,                      /**< Minor compute capability version number */
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,             /**< Maximum mipmapped 1D texture width */
    STREAM_PRIORITIES_SUPPORTED = 78,                   /**< Device supports stream priorities */
    GLOBAL_L1_CACHE_SUPPORTED = 79,                     /**< Device supports caching globals in L1 */
    LOCAL_L1_CACHE_SUPPORTED = 80,                      /**< Device supports caching locals in L1 */
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,          /**< Maximum shared memory available per multiprocessor in bytes */
    MAX_REGISTERS_PER_MULTIPROCESSOR = 82,              /**< Maximum number of 32-bit registers available per multiprocessor */
    MANAGED_MEMORY = 83,                                /**< Device can allocate managed memory on this system */
    MULTI_GPU_BOARD = 84,                               /**< Device is on a multi-GPU board */
    MULTI_GPU_BOARD_GROUP_ID = 85,                      /**< Unique id for a group of devices on the same multi-GPU board */
    HOST_NATIVE_ATOMIC_SUPPORTED = 86,                  /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,         /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    PAGEABLE_MEMORY_ACCESS = 88,                        /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    CONCURRENT_MANAGED_ACCESS = 89,                     /**< Device can coherently access managed memory concurrently with the CPU */
    COMPUTE_PREEMPTION_SUPPORTED = 90,                  /**< Device supports compute preemption. */
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,       /**< Device can access host registered memory at the same virtual address as the CPU */
    CAN_USE_STREAM_MEM_OPS_V1 = 92,                     /**< Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related APIs are supported. */
    CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,              /**< Deprecated, along with v1 MemOps API, 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
    CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,              /**< Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is supported. */
    COOPERATIVE_LAUNCH = 95,                            /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
    COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,               /**< Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated. */
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,             /**< Maximum optin shared memory per block */
    CAN_FLUSH_REMOTE_WRITES = 98,                       /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. */
    HOST_REGISTER_SUPPORTED = 99,                       /**< Device supports host memory registration via ::cudaHostRegister. */
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100, /**< Device accesses pageable memory via the host's page tables. */
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,          /**< The host can directly access managed memory on the device without migration. */
    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,         /**< Deprecated, Use VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED*/
    VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,         /**< Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs */
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,  /**< Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,           /**< Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,       /**< Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    MAX_BLOCKS_PER_MULTIPROCESSOR = 106,                /**< Maximum number of blocks per multiprocessor */
    GENERIC_COMPRESSION_SUPPORTED = 107,                /**< Device supports compression of memory */
    MAX_PERSISTING_L2_CACHE_SIZE = 108,                 /**< Maximum L2 persisting lines capacity setting in bytes. */
    MAX_ACCESS_POLICY_WINDOW_SIZE = 109,                /**< Maximum value of CUaccessPolicyWindow::num_bytes. */
    GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,      /**< Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate */
    RESERVED_SHARED_MEMORY_PER_BLOCK = 111,             /**< Shared memory reserved by CUDA driver per block in bytes */
    SPARSE_CUDA_ARRAY_SUPPORTED = 112,                  /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    READ_ONLY_HOST_REGISTER_SUPPORTED = 113,            /**< Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU */
    TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,         /**< External timeline semaphore interop is supported on the device */
    MEMORY_POOLS_SUPPORTED = 115,                       /**< Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs */
    GPU_DIRECT_RDMA_SUPPORTED = 116,                    /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,         /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum */
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118,              /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,               /**< Handle types supported with mempool based IPC */
    CLUSTER_LAUNCH = 120,                               /**< Indicates device supports cluster launch */
    DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,        /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    CAN_USE_64_BIT_STREAM_MEM_OPS = 122,                /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs. */
    CAN_USE_STREAM_WAIT_VALUE_NOR = 123,                /**< ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs. */
    DMA_BUF_SUPPORTED = 124,                            /**< Device supports buffer sharing with dma_buf mechanism. */ 
    IPC_EVENT_SUPPORTED = 125,                          /**< Device supports IPC Events. */ 
    MEM_SYNC_DOMAIN_COUNT = 126,                        /**< Number of memory domains the device supports. */
    TENSOR_MAP_ACCESS_SUPPORTED = 127,                  /**< Device supports accessing memory using Tensor Map. */
    HANDLE_TYPE_FABRIC_SUPPORTED = 128,                 /**< Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate() */
    UNIFIED_FUNCTION_POINTERS = 129,                    /**< Device supports unified function pointers. */
    NUMA_CONFIG = 130,                                  /**< NUMA configuration of a device: value is of type ::CUdeviceNumaConfig enum */
    NUMA_ID = 131,                                      /**< NUMA node ID of the GPU memory */
    MULTICAST_SUPPORTED = 132,                          /**< Device supports switch multicast and reduction operations. */
    MPS_ENABLED = 133,                                  /**< Indicates if contexts created on this device will be shared via MPS */
    HOST_NUMA_ID = 134,                                 /**< NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA. */
    D3D12_CIG_SUPPORTED = 135,                          /**< Device supports CIG with D3D12. */
    MAX
}

MemPool_Attribute :: enum i32 {
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    REUSE_ALLOW_OPPORTUNISTIC,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    REUSE_ALLOW_INTERNAL_DEPENDENCIES,

    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    RELEASE_THRESHOLD,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    RESERVED_MEM_CURRENT,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    RESERVED_MEM_HIGH,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    USED_MEM_CURRENT,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    USED_MEM_HIGH
}





