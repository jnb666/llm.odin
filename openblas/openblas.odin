package openblas

@(extra_linker_flags="-L/opt/OpenBLAS/lib")
foreign import "system:openblas"

Order :: enum i32 {
    RowMajor = 101,
    ColMajor = 102,
}

Transpose :: enum i32 {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114,
}

Uplo :: enum i32 {
    Upper = 121,
    Lower = 122,
}

Diag :: enum i32 {
    NonUnit = 131,
    Unit = 132,
}

Side :: enum i32 {
    Left = 141,
    Right = 142,
}

ComplexFloat :: struct {
    real : f32,
    imag : f32,
}

ComplexDouble :: struct {
    real : f64,
    imag : f64,
}

ComplexXdouble :: struct {
    real : f64,
    imag : f64,
}

@(default_calling_convention="c")
foreign openblas {

    @(link_name="openblas_set_num_threads")
    set_num_threads :: proc(num_threads : i32) ---

    @(link_name="openblas_set_num_threads_local")
    set_num_threads_local :: proc(num_threads : i32) -> i32 ---

    @(link_name="openblas_get_num_threads")
    get_num_threads :: proc() -> i32 ---

    @(link_name="openblas_get_num_procs")
    get_num_procs :: proc() -> i32 ---

    @(link_name="openblas_get_config")
    get_config :: proc() -> cstring ---

    @(link_name="openblas_get_corename")
    get_corename :: proc() -> cstring ---

    @(link_name="openblas_get_parallel")
    get_parallel :: proc() -> i32 ---

    @(link_name="cblas_sdsdot")
    sdsdot :: proc(n : i32, alpha : f32, x : ^f32, incx : i32, y : ^f32, incy : i32) -> f32 ---

    @(link_name="cblas_dsdot")
    dsdot :: proc(n : i32, x : ^f32, incx : i32, y : ^f32, incy : i32) -> f64 ---

    @(link_name="cblas_sdot")
    sdot :: proc(n : i32, x : ^f32, incx : i32, y : ^f32, incy : i32) -> f32 ---

    @(link_name="cblas_ddot")
    ddot :: proc(n : i32, x : ^f64, incx : i32, y : ^f64, incy : i32) -> f64 ---

    @(link_name="cblas_cdotu")
    cdotu :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) -> ComplexFloat ---

    @(link_name="cblas_cdotc")
    cdotc :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) -> ComplexFloat ---

    @(link_name="cblas_zdotu")
    zdotu :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) -> ComplexDouble ---

    @(link_name="cblas_zdotc")
    zdotc :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) -> ComplexDouble ---

    @(link_name="cblas_cdotu_sub")
    cdotu_sub :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32, ret : rawptr) ---

    @(link_name="cblas_cdotc_sub")
    cdotc_sub :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32, ret : rawptr) ---

    @(link_name="cblas_zdotu_sub")
    zdotu_sub :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32, ret : rawptr) ---

    @(link_name="cblas_zdotc_sub")
    zdotc_sub :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32, ret : rawptr) ---

    @(link_name="cblas_sasum")
    sasum :: proc(n : i32, x : ^f32, incx : i32) -> f32 ---

    @(link_name="cblas_dasum")
    dasum :: proc(n : i32, x : ^f64, incx : i32) -> f64 ---

    @(link_name="cblas_scasum")
    scasum :: proc(n : i32, x : rawptr, incx : i32) -> f32 ---

    @(link_name="cblas_dzasum")
    dzasum :: proc(n : i32, x : rawptr, incx : i32) -> f64 ---

    @(link_name="cblas_ssum")
    ssum :: proc(n : i32, x : ^f32, incx : i32) -> f32 ---

    @(link_name="cblas_dsum")
    dsum :: proc(n : i32, x : ^f64, incx : i32) -> f64 ---

    @(link_name="cblas_scsum")
    scsum :: proc(n : i32, x : rawptr, incx : i32) -> f32 ---

    @(link_name="cblas_dzsum")
    dzsum :: proc(n : i32, x : rawptr, incx : i32) -> f64 ---

    @(link_name="cblas_snrm2")
    snrm2 :: proc(N : i32, X : ^f32, incX : i32) -> f32 ---

    @(link_name="cblas_dnrm2")
    dnrm2 :: proc(N : i32, X : ^f64, incX : i32) -> f64 ---

    @(link_name="cblas_scnrm2")
    scnrm2 :: proc(N : i32, X : rawptr, incX : i32) -> f32 ---

    @(link_name="cblas_dznrm2")
    dznrm2 :: proc(N : i32, X : rawptr, incX : i32) -> f64 ---

    @(link_name="cblas_isamax")
    isamax :: proc(n : i32, x : ^f32, incx : i32) -> uint ---

    @(link_name="cblas_idamax")
    idamax :: proc(n : i32, x : ^f64, incx : i32) -> uint ---

    @(link_name="cblas_icamax")
    icamax :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_izamax")
    izamax :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_isamin")
    isamin :: proc(n : i32, x : ^f32, incx : i32) -> uint ---

    @(link_name="cblas_idamin")
    idamin :: proc(n : i32, x : ^f64, incx : i32) -> uint ---

    @(link_name="cblas_icamin")
    icamin :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_izamin")
    izamin :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_samax")
    samax :: proc(n : i32, x : ^f32, incx : i32) -> f32 ---

    @(link_name="cblas_damax")
    damax :: proc(n : i32, x : ^f64, incx : i32) -> f64 ---

    @(link_name="cblas_scamax")
    scamax :: proc(n : i32, x : rawptr, incx : i32) -> f32 ---

    @(link_name="cblas_dzamax")
    dzamax :: proc(n : i32, x : rawptr, incx : i32) -> f64 ---

    @(link_name="cblas_samin")
    samin :: proc(n : i32, x : ^f32, incx : i32) -> f32 ---

    @(link_name="cblas_damin")
    damin :: proc(n : i32, x : ^f64, incx : i32) -> f64 ---

    @(link_name="cblas_scamin")
    scamin :: proc(n : i32, x : rawptr, incx : i32) -> f32 ---

    @(link_name="cblas_dzamin")
    dzamin :: proc(n : i32, x : rawptr, incx : i32) -> f64 ---

    @(link_name="cblas_ismax")
    ismax :: proc(n : i32, x : ^f32, incx : i32) -> uint ---

    @(link_name="cblas_idmax")
    idmax :: proc(n : i32, x : ^f64, incx : i32) -> uint ---

    @(link_name="cblas_icmax")
    icmax :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_izmax")
    izmax :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_ismin")
    ismin :: proc(n : i32, x : ^f32, incx : i32) -> uint ---

    @(link_name="cblas_idmin")
    idmin :: proc(n : i32, x : ^f64, incx : i32) -> uint ---

    @(link_name="cblas_icmin")
    icmin :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_izmin")
    izmin :: proc(n : i32, x : rawptr, incx : i32) -> uint ---

    @(link_name="cblas_saxpy")
    saxpy :: proc(n : i32, alpha : f32, x : ^f32, incx : i32, y : ^f32, incy : i32) ---

    @(link_name="cblas_daxpy")
    daxpy :: proc(n : i32, alpha : f64, x : ^f64, incx : i32, y : ^f64, incy : i32) ---

    @(link_name="cblas_caxpy")
    caxpy :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_zaxpy")
    zaxpy :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_caxpyc")
    caxpyc :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_zaxpyc")
    zaxpyc :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_scopy")
    scopy :: proc(n : i32, x : ^f32, incx : i32, y : ^f32, incy : i32) ---

    @(link_name="cblas_dcopy")
    dcopy :: proc(n : i32, x : ^f64, incx : i32, y : ^f64, incy : i32) ---

    @(link_name="cblas_ccopy")
    ccopy :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_zcopy")
    zcopy :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_sswap")
    sswap :: proc(n : i32, x : ^f32, incx : i32, y : ^f32, incy : i32) ---

    @(link_name="cblas_dswap")
    dswap :: proc(n : i32, x : ^f64, incx : i32, y : ^f64, incy : i32) ---

    @(link_name="cblas_cswap")
    cswap :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_zswap")
    zswap :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incy : i32) ---

    @(link_name="cblas_srot")
    srot :: proc(N : i32, X : ^f32, incX : i32, Y : ^f32, incY : i32, c : f32, s : f32) ---

    @(link_name="cblas_drot")
    drot :: proc(N : i32, X : ^f64, incX : i32, Y : ^f64, incY : i32, c : f64, s : f64) ---

    @(link_name="cblas_csrot")
    csrot :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incY : i32, c : f32, s : f32) ---

    @(link_name="cblas_zdrot")
    zdrot :: proc(n : i32, x : rawptr, incx : i32, y : rawptr, incY : i32, c : f64, s : f64) ---

    @(link_name="cblas_srotg")
    srotg :: proc(a : ^f32, b : ^f32, c : ^f32, s : ^f32) ---

    @(link_name="cblas_drotg")
    drotg :: proc(a : ^f64, b : ^f64, c : ^f64, s : ^f64) ---

    @(link_name="cblas_crotg")
    crotg :: proc(a : rawptr, b : rawptr, c : ^f32, s : rawptr) ---

    @(link_name="cblas_zrotg")
    zrotg :: proc(a : rawptr, b : rawptr, c : ^f64, s : rawptr) ---

    @(link_name="cblas_srotm")
    srotm :: proc(N : i32, X : ^f32, incX : i32, Y : ^f32, incY : i32, P : ^f32) ---

    @(link_name="cblas_drotm")
    drotm :: proc(N : i32, X : ^f64, incX : i32, Y : ^f64, incY : i32, P : ^f64) ---

    @(link_name="cblas_srotmg")
    srotmg :: proc(d1 : ^f32, d2 : ^f32, b1 : ^f32, b2 : f32, P : ^f32) ---

    @(link_name="cblas_drotmg")
    drotmg :: proc(d1 : ^f64, d2 : ^f64, b1 : ^f64, b2 : f64, P : ^f64) ---

    @(link_name="cblas_sscal")
    sscal :: proc(N : i32, alpha : f32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dscal")
    dscal :: proc(N : i32, alpha : f64, X : ^f64, incX : i32) ---

    @(link_name="cblas_cscal")
    cscal :: proc(N : i32, alpha : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_zscal")
    zscal :: proc(N : i32, alpha : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_csscal")
    csscal :: proc(N : i32, alpha : f32, X : rawptr, incX : i32) ---

    @(link_name="cblas_zdscal")
    zdscal :: proc(N : i32, alpha : f64, X : rawptr, incX : i32) ---

    @(link_name="cblas_sgemv")
    sgemv :: proc(order : Order, trans : Transpose, m : i32, n : i32, alpha : f32, a : ^f32, lda : i32, x : ^f32, incx : i32, beta : f32, y : ^f32, incy : i32) ---

    @(link_name="cblas_dgemv")
    dgemv :: proc(order : Order, trans : Transpose, m : i32, n : i32, alpha : f64, a : ^f64, lda : i32, x : ^f64, incx : i32, beta : f64, y : ^f64, incy : i32) ---

    @(link_name="cblas_cgemv")
    cgemv :: proc(order : Order, trans : Transpose, m : i32, n : i32, alpha : rawptr, a : rawptr, lda : i32, x : rawptr, incx : i32, beta : rawptr, y : rawptr, incy : i32) ---

    @(link_name="cblas_zgemv")
    zgemv :: proc(order : Order, trans : Transpose, m : i32, n : i32, alpha : rawptr, a : rawptr, lda : i32, x : rawptr, incx : i32, beta : rawptr, y : rawptr, incy : i32) ---

    @(link_name="cblas_sger")
    sger :: proc(order : Order, M : i32, N : i32, alpha : f32, X : ^f32, incX : i32, Y : ^f32, incY : i32, A : ^f32, lda : i32) ---

    @(link_name="cblas_dger")
    dger :: proc(order : Order, M : i32, N : i32, alpha : f64, X : ^f64, incX : i32, Y : ^f64, incY : i32, A : ^f64, lda : i32) ---

    @(link_name="cblas_cgeru")
    cgeru :: proc(order : Order, M : i32, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_cgerc")
    cgerc :: proc(order : Order, M : i32, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_zgeru")
    zgeru :: proc(order : Order, M : i32, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_zgerc")
    zgerc :: proc(order : Order, M : i32, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_strsv")
    strsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : ^f32, lda : i32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtrsv")
    dtrsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : ^f64, lda : i32, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctrsv")
    ctrsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztrsv")
    ztrsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_strmv")
    strmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : ^f32, lda : i32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtrmv")
    dtrmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : ^f64, lda : i32, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctrmv")
    ctrmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztrmv")
    ztrmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_ssyr")
    ssyr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : ^f32, incX : i32, A : ^f32, lda : i32) ---

    @(link_name="cblas_dsyr")
    dsyr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : ^f64, incX : i32, A : ^f64, lda : i32) ---

    @(link_name="cblas_cher")
    cher :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : rawptr, incX : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_zher")
    zher :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : rawptr, incX : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_ssyr2")
    ssyr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : ^f32, incX : i32, Y : ^f32, incY : i32, A : ^f32, lda : i32) ---

    @(link_name="cblas_dsyr2")
    dsyr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : ^f64, incX : i32, Y : ^f64, incY : i32, A : ^f64, lda : i32) ---

    @(link_name="cblas_cher2")
    cher2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_zher2")
    zher2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, A : rawptr, lda : i32) ---

    @(link_name="cblas_sgbmv")
    sgbmv :: proc(order : Order, TransA : Transpose, M : i32, N : i32, KL : i32, KU : i32, alpha : f32, A : ^f32, lda : i32, X : ^f32, incX : i32, beta : f32, Y : ^f32, incY : i32) ---

    @(link_name="cblas_dgbmv")
    dgbmv :: proc(order : Order, TransA : Transpose, M : i32, N : i32, KL : i32, KU : i32, alpha : f64, A : ^f64, lda : i32, X : ^f64, incX : i32, beta : f64, Y : ^f64, incY : i32) ---

    @(link_name="cblas_cgbmv")
    cgbmv :: proc(order : Order, TransA : Transpose, M : i32, N : i32, KL : i32, KU : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_zgbmv")
    zgbmv :: proc(order : Order, TransA : Transpose, M : i32, N : i32, KL : i32, KU : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_ssbmv")
    ssbmv :: proc(order : Order, Uplo : Uplo, N : i32, K : i32, alpha : f32, A : ^f32, lda : i32, X : ^f32, incX : i32, beta : f32, Y : ^f32, incY : i32) ---

    @(link_name="cblas_dsbmv")
    dsbmv :: proc(order : Order, Uplo : Uplo, N : i32, K : i32, alpha : f64, A : ^f64, lda : i32, X : ^f64, incX : i32, beta : f64, Y : ^f64, incY : i32) ---

    @(link_name="cblas_stbmv")
    stbmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : ^f32, lda : i32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtbmv")
    dtbmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : ^f64, lda : i32, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctbmv")
    ctbmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztbmv")
    ztbmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_stbsv")
    stbsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : ^f32, lda : i32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtbsv")
    dtbsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : ^f64, lda : i32, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctbsv")
    ctbsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztbsv")
    ztbsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, K : i32, A : rawptr, lda : i32, X : rawptr, incX : i32) ---

    @(link_name="cblas_stpmv")
    stpmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : ^f32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtpmv")
    dtpmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : ^f64, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctpmv")
    ctpmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztpmv")
    ztpmv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_stpsv")
    stpsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : ^f32, X : ^f32, incX : i32) ---

    @(link_name="cblas_dtpsv")
    dtpsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : ^f64, X : ^f64, incX : i32) ---

    @(link_name="cblas_ctpsv")
    ctpsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_ztpsv")
    ztpsv :: proc(order : Order, Uplo : Uplo, TransA : Transpose, Diag : Diag, N : i32, Ap : rawptr, X : rawptr, incX : i32) ---

    @(link_name="cblas_ssymv")
    ssymv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, A : ^f32, lda : i32, X : ^f32, incX : i32, beta : f32, Y : ^f32, incY : i32) ---

    @(link_name="cblas_dsymv")
    dsymv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, A : ^f64, lda : i32, X : ^f64, incX : i32, beta : f64, Y : ^f64, incY : i32) ---

    @(link_name="cblas_chemv")
    chemv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_zhemv")
    zhemv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_sspmv")
    sspmv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, Ap : ^f32, X : ^f32, incX : i32, beta : f32, Y : ^f32, incY : i32) ---

    @(link_name="cblas_dspmv")
    dspmv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, Ap : ^f64, X : ^f64, incX : i32, beta : f64, Y : ^f64, incY : i32) ---

    @(link_name="cblas_sspr")
    sspr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : ^f32, incX : i32, Ap : ^f32) ---

    @(link_name="cblas_dspr")
    dspr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : ^f64, incX : i32, Ap : ^f64) ---

    @(link_name="cblas_chpr")
    chpr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : rawptr, incX : i32, A : rawptr) ---

    @(link_name="cblas_zhpr")
    zhpr :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : rawptr, incX : i32, A : rawptr) ---

    @(link_name="cblas_sspr2")
    sspr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f32, X : ^f32, incX : i32, Y : ^f32, incY : i32, A : ^f32) ---

    @(link_name="cblas_dspr2")
    dspr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : f64, X : ^f64, incX : i32, Y : ^f64, incY : i32, A : ^f64) ---

    @(link_name="cblas_chpr2")
    chpr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, Ap : rawptr) ---

    @(link_name="cblas_zhpr2")
    zhpr2 :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, X : rawptr, incX : i32, Y : rawptr, incY : i32, Ap : rawptr) ---

    @(link_name="cblas_chbmv")
    chbmv :: proc(order : Order, Uplo : Uplo, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_zhbmv")
    zhbmv :: proc(order : Order, Uplo : Uplo, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_chpmv")
    chpmv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, Ap : rawptr, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_zhpmv")
    zhpmv :: proc(order : Order, Uplo : Uplo, N : i32, alpha : rawptr, Ap : rawptr, X : rawptr, incX : i32, beta : rawptr, Y : rawptr, incY : i32) ---

    @(link_name="cblas_sgemm")
    sgemm :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_dgemm")
    dgemm :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32, beta : f64, C : ^f64, ldc : i32) ---

    @(link_name="cblas_cgemm")
    cgemm :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_cgemm3m")
    cgemm3m :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zgemm")
    zgemm :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zgemm3m")
    zgemm3m :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_sgemmt")
    sgemmt :: proc(Order : Order, Uplo : Uplo, TransA : Transpose, TransB : Transpose, M : i32, K : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_dgemmt")
    dgemmt :: proc(Order : Order, Uplo : Uplo, TransA : Transpose, TransB : Transpose, M : i32, K : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32, beta : f64, C : ^f64, ldc : i32) ---

    @(link_name="cblas_cgemmt")
    cgemmt :: proc(Order : Order, Uplo : Uplo, TransA : Transpose, TransB : Transpose, M : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zgemmt")
    zgemmt :: proc(Order : Order, Uplo : Uplo, TransA : Transpose, TransB : Transpose, M : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_ssymm")
    ssymm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_dsymm")
    dsymm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32, beta : f64, C : ^f64, ldc : i32) ---

    @(link_name="cblas_csymm")
    csymm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zsymm")
    zsymm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_ssyrk")
    ssyrk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f32, A : ^f32, lda : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_dsyrk")
    dsyrk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f64, A : ^f64, lda : i32, beta : f64, C : ^f64, ldc : i32) ---

    @(link_name="cblas_csyrk")
    csyrk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zsyrk")
    zsyrk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_ssyr2k")
    ssyr2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_dsyr2k")
    dsyr2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32, beta : f64, C : ^f64, ldc : i32) ---

    @(link_name="cblas_csyr2k")
    csyr2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zsyr2k")
    zsyr2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_strmm")
    strmm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32) ---

    @(link_name="cblas_dtrmm")
    dtrmm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32) ---

    @(link_name="cblas_ctrmm")
    ctrmm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32) ---

    @(link_name="cblas_ztrmm")
    ztrmm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32) ---

    @(link_name="cblas_strsm")
    strsm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : f32, A : ^f32, lda : i32, B : ^f32, ldb : i32) ---

    @(link_name="cblas_dtrsm")
    dtrsm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : f64, A : ^f64, lda : i32, B : ^f64, ldb : i32) ---

    @(link_name="cblas_ctrsm")
    ctrsm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32) ---

    @(link_name="cblas_ztrsm")
    ztrsm :: proc(Order : Order, Side : Side, Uplo : Uplo, TransA : Transpose, Diag : Diag, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32) ---

    @(link_name="cblas_chemm")
    chemm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zhemm")
    zhemm :: proc(Order : Order, Side : Side, Uplo : Uplo, M : i32, N : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : rawptr, C : rawptr, ldc : i32) ---

    @(link_name="cblas_cherk")
    cherk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f32, A : rawptr, lda : i32, beta : f32, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zherk")
    zherk :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : f64, A : rawptr, lda : i32, beta : f64, C : rawptr, ldc : i32) ---

    @(link_name="cblas_cher2k")
    cher2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : f32, C : rawptr, ldc : i32) ---

    @(link_name="cblas_zher2k")
    zher2k :: proc(Order : Order, Uplo : Uplo, Trans : Transpose, N : i32, K : i32, alpha : rawptr, A : rawptr, lda : i32, B : rawptr, ldb : i32, beta : f64, C : rawptr, ldc : i32) ---

    @(link_name="cblas_xerbla")
    xerbla :: proc(p : i32, rout : cstring, form : cstring) ---

    @(link_name="cblas_saxpby")
    saxpby :: proc(n : i32, alpha : f32, x : ^f32, incx : i32, beta : f32, y : ^f32, incy : i32) ---

    @(link_name="cblas_daxpby")
    daxpby :: proc(n : i32, alpha : f64, x : ^f64, incx : i32, beta : f64, y : ^f64, incy : i32) ---

    @(link_name="cblas_caxpby")
    caxpby :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, beta : rawptr, y : rawptr, incy : i32) ---

    @(link_name="cblas_zaxpby")
    zaxpby :: proc(n : i32, alpha : rawptr, x : rawptr, incx : i32, beta : rawptr, y : rawptr, incy : i32) ---

    @(link_name="cblas_somatcopy")
    somatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : f32, a : ^f32, clda : i32, b : ^f32, cldb : i32) ---

    @(link_name="cblas_domatcopy")
    domatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : f64, a : ^f64, clda : i32, b : ^f64, cldb : i32) ---

    @(link_name="cblas_comatcopy")
    comatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : ^f32, a : ^f32, clda : i32, b : ^f32, cldb : i32) ---

    @(link_name="cblas_zomatcopy")
    zomatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : ^f64, a : ^f64, clda : i32, b : ^f64, cldb : i32) ---

    @(link_name="cblas_simatcopy")
    simatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : f32, a : ^f32, clda : i32, cldb : i32) ---

    @(link_name="cblas_dimatcopy")
    dimatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : f64, a : ^f64, clda : i32, cldb : i32) ---

    @(link_name="cblas_cimatcopy")
    cimatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : ^f32, a : ^f32, clda : i32, cldb : i32) ---

    @(link_name="cblas_zimatcopy")
    zimatcopy :: proc(CORDER : Order, CTRANS : Transpose, crows : i32, ccols : i32, calpha : ^f64, a : ^f64, clda : i32, cldb : i32) ---

    @(link_name="cblas_sgeadd")
    sgeadd :: proc(CORDER : Order, crows : i32, ccols : i32, calpha : f32, a : ^f32, clda : i32, cbeta : f32, c : ^f32, cldc : i32) ---

    @(link_name="cblas_dgeadd")
    dgeadd :: proc(CORDER : Order, crows : i32, ccols : i32, calpha : f64, a : ^f64, clda : i32, cbeta : f64, c : ^f64, cldc : i32) ---

    @(link_name="cblas_cgeadd")
    cgeadd :: proc(CORDER : Order, crows : i32, ccols : i32, calpha : ^f32, a : ^f32, clda : i32, cbeta : ^f32, c : ^f32, cldc : i32) ---

    @(link_name="cblas_zgeadd")
    zgeadd :: proc(CORDER : Order, crows : i32, ccols : i32, calpha : ^f64, a : ^f64, clda : i32, cbeta : ^f64, c : ^f64, cldc : i32) ---

    @(link_name="cblas_sgemm_batch")
    sgemm_batch :: proc(Order : Order, TransA_array : ^Transpose, TransB_array : ^Transpose, M_array : ^i32, N_array : ^i32, K_array : ^i32, alpha_array : ^f32, A_array : ^^f32, lda_array : ^i32, B_array : ^^f32, ldb_array : ^i32, beta_array : ^f32, C_array : ^^f32, ldc_array : ^i32, group_count : i32, group_size : ^i32) ---

    @(link_name="cblas_dgemm_batch")
    dgemm_batch :: proc(Order : Order, TransA_array : ^Transpose, TransB_array : ^Transpose, M_array : ^i32, N_array : ^i32, K_array : ^i32, alpha_array : ^f64, A_array : ^^f64, lda_array : ^i32, B_array : ^^f64, ldb_array : ^i32, beta_array : ^f64, C_array : ^^f64, ldc_array : ^i32, group_count : i32, group_size : ^i32) ---

    @(link_name="cblas_cgemm_batch")
    cgemm_batch :: proc(Order : Order, TransA_array : ^Transpose, TransB_array : ^Transpose, M_array : ^i32, N_array : ^i32, K_array : ^i32, alpha_array : rawptr, A_array : ^rawptr, lda_array : ^i32, B_array : ^rawptr, ldb_array : ^i32, beta_array : rawptr, C_array : ^rawptr, ldc_array : ^i32, group_count : i32, group_size : ^i32) ---

    @(link_name="cblas_zgemm_batch")
    zgemm_batch :: proc(Order : Order, TransA_array : ^Transpose, TransB_array : ^Transpose, M_array : ^i32, N_array : ^i32, K_array : ^i32, alpha_array : rawptr, A_array : ^rawptr, lda_array : ^i32, B_array : ^rawptr, ldb_array : ^i32, beta_array : rawptr, C_array : ^rawptr, ldc_array : ^i32, group_count : i32, group_size : ^i32) ---

    @(link_name="cblas_sbstobf16")
    sbstobf16 :: proc(n : i32, _in : ^f32, incin : i32, out : ^u16, incout : i32) ---

    @(link_name="cblas_sbdtobf16")
    sbdtobf16 :: proc(n : i32, _in : ^f64, incin : i32, out : ^u16, incout : i32) ---

    @(link_name="cblas_sbf16tos")
    sbf16tos :: proc(n : i32, _in : ^u16, incin : i32, out : ^f32, incout : i32) ---

    @(link_name="cblas_dbf16tod")
    dbf16tod :: proc(n : i32, _in : ^u16, incin : i32, out : ^f64, incout : i32) ---

    @(link_name="cblas_sbdot")
    sbdot :: proc(n : i32, x : ^u16, incx : i32, y : ^u16, incy : i32) -> f32 ---

    @(link_name="cblas_sbgemv")
    sbgemv :: proc(order : Order, trans : Transpose, m : i32, n : i32, alpha : f32, a : ^u16, lda : i32, x : ^u16, incx : i32, beta : f32, y : ^f32, incy : i32) ---

    @(link_name="cblas_sbgemm")
    sbgemm :: proc(Order : Order, TransA : Transpose, TransB : Transpose, M : i32, N : i32, K : i32, alpha : f32, A : ^u16, lda : i32, B : ^u16, ldb : i32, beta : f32, C : ^f32, ldc : i32) ---

    @(link_name="cblas_sbgemm_batch")
    sbgemm_batch :: proc(Order : Order, TransA_array : ^Transpose, TransB_array : ^Transpose, M_array : ^i32, N_array : ^i32, K_array : ^i32, alpha_array : ^f32, A_array : ^^u16, lda_array : ^i32, B_array : ^^u16, ldb_array : ^i32, beta_array : ^f32, C_array : ^^f32, ldc_array : ^i32, group_count : i32, group_size : ^i32) ---

}
