// fa_nvfp4_common.cuh
//
// Pieces shared by the prefill (Phase 2) and decode (Phase 3) kernels: PTX helpers (mbarrier, TMA, conversions),
// the two-term FP4 Q quantizer, the V scale-max prepass and the host-side tensor-map encoder.

#pragma once

#include "fa_nvfp4_layout.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>

namespace fa_nvfp4 {

#ifdef __CUDACC__

constexpr float LOG2E = 1.4426950408889634f;

// ------------------------------------------------------------------------------------------------
// PTX helpers: mbarrier, bulk copy, named barrier, conversions
// ------------------------------------------------------------------------------------------------
static __device__ __forceinline__ uint32_t smem_u32(const void * p) { return (uint32_t) __cvta_generic_to_shared(p); }

static __device__ __forceinline__ void mbar_init(uint64_t * bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
static __device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void mbar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
static __device__ __forceinline__ void mbar_wait(uint64_t * bar, uint32_t parity) {
    asm volatile(
        "{\n"
        "  .reg .pred P1;\n"
        "LAB_WAIT:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra DONE;\n"
        "  bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}
// 4-D tensor-map box -> this CTA's shared memory, completion counted on bar (bytes). The destination must be
// .shared::cta: with .shared::cluster, ptxas for sm_120a adds a CALL-based remote-CTA path, and any call makes
// it ignore setmaxnreg.
static __device__ __forceinline__ void tma_load_4d(void * dst, const CUtensorMap * tm, int c0, int c1, int c2, int c3,
                                                   uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
        ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar))
        : "memory");
}
static __device__ __forceinline__ void named_bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
static __device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// two UE4M3 codes (low 16 bits; byte 0 -> low half) -> f16x2
static __device__ __forceinline__ __half2 e4m3x2_to_h2(const uint32_t codes) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t) (codes & 0xFFFFu)));
    return *reinterpret_cast<const __half2 *>(&h2);
}
// f16x2 -> two E4M3 bytes (RN, saturating; low half -> low byte)
static __device__ __forceinline__ uint32_t h2_to_e4m3x2(const __half2 x) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(*reinterpret_cast<const uint32_t *>(&x)));
    return r;
}
static __device__ __forceinline__ uint32_t ld_s32(const uint8_t * p) { return *reinterpret_cast<const uint32_t *>(p); }
static __device__ __forceinline__ uint32_t ld_s32_g(const uint8_t * p) { return __ldg(reinterpret_cast<const unsigned int *>(p)); }

// ------------------------------------------------------------------------------------------------
// Q quantization: one warp, one 256-float row -> two FP4 terms in ggml split-half layout + row factor.
// Lane l owns elements 8l..8l+7 (half of sub-block l/2). q == nullptr quantizes a zero row.
// ------------------------------------------------------------------------------------------------
static __device__ __forceinline__ void quantize_q_row(const float * q, uint8_t * t0, uint8_t * t1, float * alpha_out) {
    const int lane = threadIdx.x & 31;
    const int sb   = lane / 2;
    const int b    = sb / 4;
    const int s    = sb % 4;
    float x[8];
    if (q) {
        const float4 v0 = reinterpret_cast<const float4 *>(q)[2 * lane];
        const float4 v1 = reinterpret_cast<const float4 *>(q)[2 * lane + 1];
        x[0] = v0.x; x[1] = v0.y; x[2] = v0.z; x[3] = v0.w;
        x[4] = v1.x; x[5] = v1.y; x[6] = v1.z; x[7] = v1.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = 0.0f;
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    }
    const float alpha     = amax > 0.0f ? amax / (448.0f * 6.0f) : 1.0f;
    const float inv_alpha = 1.0f / alpha;
    const int   qoff      = b * BLOCK_BYTES + 4 + s * 8 + 4 * (lane & 1);

    uint32_t code;
    float    deq[8];
    uint32_t packed = quantize_sub16_pair(x, inv_alpha, alpha, &code, deq);
    *reinterpret_cast<uint32_t *>(t0 + qoff) = packed;
    if ((lane & 1) == 0) {
        t0[b * BLOCK_BYTES + s] = (uint8_t) code;
    }
    float res[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        res[i] = x[i] - deq[i];
    }
    packed = quantize_sub16_pair(res, inv_alpha, alpha, &code, deq);
    *reinterpret_cast<uint32_t *>(t1 + qoff) = packed;
    if ((lane & 1) == 0) {
        t1[b * BLOCK_BYTES + s] = (uint8_t) code;
    }
    if (lane == 0) {
        *alpha_out = alpha;
    }
}

// ------------------------------------------------------------------------------------------------
// Prepass 2: per (seq, kv head) the max UE4M3 scale code of each 16-wide d-block of V over all tokens.
// Output must be zeroed first. grid (chunks, n_head_kv, n_seq), block 256.
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_v_scale_max(const uint8_t * V, const size_t nbv1, const size_t nbv2,
                                                     const size_t nbv3, const int n_kv, int * v_scale_max) {
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const uint8_t * Vg = V + seq * nbv3 + kvh * nbv2;
    uint32_t mx[BLOCKS_PER_ROW] = {0u, 0u, 0u, 0u};
    for (int tok = blockIdx.x * blockDim.x + threadIdx.x; tok < n_kv; tok += gridDim.x * blockDim.x) {
#pragma unroll
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            mx[b] = __vmaxu4(mx[b], ld_s32_g(Vg + tok * nbv1 + sf_word(b)));
        }
    }
#pragma unroll
    for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            mx[b] = __vmaxu4(mx[b], __shfl_xor_sync(0xFFFFFFFFu, mx[b], o));
        }
    }
    if (threadIdx.x % 32 == 0) {
        int * out = v_scale_max + (seq * gridDim.y + kvh) * DBLKS;
#pragma unroll
        for (int db = 0; db < DBLKS; ++db) {
            atomicMax(out + db, (int) ((mx[db / 4] >> (8 * (db % 4))) & 0x7Fu));
        }
    }
}

#endif // __CUDACC__

// ------------------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------------------
// 4-D u8 tensor map over one K or V view: dim 0 = the 144 bytes of a head row, dims 1, 2 = token and
// kv head ordered by increasing stride, dim 3 = sequence. Box = box_rows tokens of one head.
static inline cudaError_t encode_kv_map(CUtensorMap * tm, const uint8_t * base, size_t nb1, size_t nb2, size_t nb3,
                                        int n_kv, int n_head_kv, int n_seq, int box_rows, int * tok_dim, int * head_dim) {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (!encode) {
        cudaDriverEntryPointQueryResult q;
        void * fn = nullptr;
        const cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
        if (err != cudaSuccess || q != cudaDriverEntryPointSuccess || !fn) {
            return err != cudaSuccess ? err : cudaErrorNotSupported;
        }
        encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
    }
    const bool tok_first = nb1 <= nb2;
    *tok_dim  = tok_first ? 1 : 2;
    *head_dim = tok_first ? 2 : 1;
    const cuuint64_t dims[4]    = {(cuuint64_t) ROW_BYTES, (cuuint64_t) (tok_first ? n_kv : n_head_kv),
                                   (cuuint64_t) (tok_first ? n_head_kv : n_kv), (cuuint64_t) n_seq};
    const cuuint64_t strides[3] = {tok_first ? nb1 : nb2, tok_first ? nb2 : nb1, nb3};
    const cuuint32_t box[4]     = {(cuuint32_t) ROW_BYTES, (cuuint32_t) (tok_first ? box_rows : 1),
                                   (cuuint32_t) (tok_first ? 1 : box_rows), 1};
    const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
    const CUresult r = encode(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<uint8_t *>(base), dims, strides, box,
                              elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

} // namespace fa_nvfp4
