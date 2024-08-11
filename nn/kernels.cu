#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32U

#define BF16(x) ( __float2bfloat16_rn(x) )

typedef __nv_bfloat16 bfloat;

typedef __nv_bfloat162 bfloat2;

__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float block_sum(float val) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduce_sum(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : 0.0f;
    float block_val = warp_reduce_sum(warp_val);

    return block_val;
}

struct SoftmaxParams {
	float scale;
	float offset;
};

__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp, size_t idx, const bfloat* inp, int V, int Vp) {
	// this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
	// note that inp is actually (B * T, Vp) but we only use the first V elements
	// this function then calculates:
	// 1) the max value to subtract for numerical stability and
	// 2) the sum normalization factor
	const bfloat* x = inp + idx * Vp;
	// thread coarsening loop, where the 32 threads serially process all V elements
	// thread_rank() is in [0, 31], warp.size() is 32
	float maxval = -3.4e38;
	float sumval = 0.0f;
	for (int i = warp.thread_rank(); i < V; i += warp.size()) {
		float v = x[i];
		float old_maxval = maxval;
		// online softmax recurrence from "Online normalizer calculation for softmax" paper
		maxval = fmaxf(maxval, v);
		sumval *= expf((old_maxval - maxval));
		sumval += expf(v - maxval);
	}
	// warp-level reduction to get the maxval across the 32 threads
	float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
	// all 32 threads do a final shift of the sum considering the global max in this row
	sumval *= expf((maxval - global_maxval));
	// warp-level reduction to get the sumval across the 32 threads
	float global_sumval = cg::reduce(warp, sumval, cg::plus<float>{});
	// the final normalization factor
	float norm = 1.0f / global_sumval;
	return SoftmaxParams{norm, global_maxval};
}

extern "C" {

__global__ void bf16_to_f32(float2* out, const bfloat2* in, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N / 2) {
		bfloat2 v = in[i];
		out[i] = make_float2(v.x, v.y);
	}
}

__global__ void add_f32(float* out, const float* x, const float* y, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		out[i] = x[i] + y[i];
	}
}

__global__ void add_bf16(bfloat2* out, const bfloat2* x, const bfloat2* y, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N/2) {
		out[i] = x[i] + y[i];
	}
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
__global__ void matmul_backward_bias_f32(float* dbias, const float* dout, int BT, int OC) {
    extern __shared__ float smem[];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int tl = blockIdx.x * warpSize;
    const int vstep = blockDim.x / warpSize;

    const float* dout_col = dout + tl + lane_id;
    float dout_sum = 0.0f;
    for (int row = warp_id; row < BT; row += vstep) {
        dout_sum += (float)dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j*warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

__global__ void matmul_backward_bias_bf16(bfloat* dbias, const bfloat* dout, int BT, int OC) {
    extern __shared__ float smem[];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int tl = blockIdx.x * warpSize;
    const int vstep = blockDim.x / warpSize;

    const bfloat* dout_col = dout + tl + lane_id;
    float dout_sum = 0.0f;
    for (int row = warp_id; row < BT; row += vstep) {
        dout_sum += (float)dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j*warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

#define gelu_scale 0.797884560802865f

__global__ void gelu_bf16(const bfloat* in, bfloat* out, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float x  = in[i];
		float cube = 0.044715f * x * x * x;
		float y = 0.5f * x * (1.0f + tanhf(gelu_scale*(x+cube)));  
		out[i] = BF16(y);
	}
}

__global__ void gelu_bwd_bf16(const bfloat* in, bfloat* dout, bfloat* din, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float x = in[i];
		float dy = dout[i];
		float cube = 0.044715f * x * x * x;
		float tanh_arg = gelu_scale * (x + cube);
		float tanh_out = tanhf(tanh_arg);
		float cosh_out = coshf(tanh_arg);
		float sech_out = 1.0f / (cosh_out * cosh_out);
		float local_grad = 0.5f*(1.0f+tanh_out) + x*0.5f*sech_out*gelu_scale*(1.0f + 3.0f*0.044715f*x*x);
		din[i] = local_grad * dy;
	}
}

__global__ void encoder_forward_bf16(bfloat2* out, const int* in, const bfloat2* wte, const bfloat2* wpe, int B, int T, int C) {
	int C2 = C / 2;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < B * T * C2) {
		int bt = i / C2;
		int ix = in[bt];
		int t = bt % T;
		int c = i % C2;
		out[bt*C2 + c] = wte[ix*C2 + c] + wpe[t*C2 + c];
	}
}

__global__ void encoder_backward_bf16(bfloat* dwte, bfloat* dwpe, const bfloat* dout, const int* in, int B, int T, int C) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < B * T * C) {
		int bt = i / C;
		int t = bt % T;
		int c = i % C;
		int ix = in[bt];
		float dout_btc = dout[bt*C + c];
		bfloat* dwte_ix = dwte + ix*C + c;
		bfloat* dwpe_tc = dwpe + t*C + c;
		atomicAdd(dwte_ix, dout_btc);
		atomicAdd(dwpe_tc, dout_btc);
	}
}

__global__ void crossentropy_loss_bf16(bfloat* dlogits, float* losses, const bfloat* logits, const int* targets,
									  float dloss, int BT, int V, int Vp, int train) {
	namespace cg = cooperative_groups;
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
	// example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
	// each block of 4 warps is in charge of 4 rows of the input, one warp per row
	// meta_group_size is the number of warps per block (e.g. 4)
	// meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
	size_t idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
	if (idx >= BT) { 
		return;
	}
	// calculate the offset (maxval) and scale (sumval) for the softmax
	SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, Vp);
	// in each row (handled by one warp), thread 0 calculates the loss
	// calculate the probability needed for the loss and update losses
	if (warp.thread_rank() == 0) {
		int ix = targets[idx];
		float x = logits[idx*Vp + ix];
		float prob = expf(x - sp.offset) * sp.scale;
		losses[idx] = -logf(prob);
	}
	if (train) {
		// finally all threads calculate the gradients
		// prob is only materialized here temporarily and in registers, never
		// as a full tensor that gets written to global memory
		for (int i = warp.thread_rank(); i < V; i += warp.size()) {
			float x = logits[idx*Vp + i];
			float prob = expf(x - sp.offset) * sp.scale;
			int ix = targets[idx];
			float indicator = i == ix ? 1.0f : 0.0f;
			dlogits[idx*Vp + i] = BF16((prob - indicator) * dloss);
		}
	}
}

__global__ void norm_squared_bf16(float* out, const bfloat* data, size_t count) {
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for (size_t i = index; i < count; i += grid_width) {
    	float val = data[i];
        accumulator += val*val;
    }
    // warp-level reduce
    float sum = block_sum(accumulator);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

__global__ void adamw_step_bf16(float* weight32, bfloat* weight, const bfloat* grads, float* ms, float* vs, size_t nparams,
                float learn_rate, float wdecay, float beta1, float beta2, float beta1c, float beta2c, float eps, float scale) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= nparams) return;
   float w = weight32[i];
   float grad = (float)grads[i] * scale;
   float m = ms[i];
   float v = vs[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   ms[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   vs[i] = v;
   m /= beta1c;  // m_hat
   v /= beta2c;  // v_hat
   w -= learn_rate*(m/(sqrtf(v)+eps) + wdecay*w);
   weight32[i] = w;
   weight[i] = BF16(w);
}

}