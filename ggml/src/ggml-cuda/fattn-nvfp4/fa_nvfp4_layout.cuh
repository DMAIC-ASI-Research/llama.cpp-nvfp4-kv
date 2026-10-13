// fa_nvfp4_layout.cuh
//
// Phase 1 of the native NVFP4 FlashAttention kernel for head dim 256 on consumer Blackwell (sm_120a).
// Compile-time tile configuration, shared-memory maps, warp-level fragment maps and the raw PTX
// building blocks used by the prefill (Phase 2) and decode (Phase 3) kernels.
//
// Design facts this file encodes (see docs/phase1-layout.md for the derivation):
//   * sm_120 has no Tensor Memory and no tcgen05. FP4 tensor-core math is warp-level mma.sync:
//       mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//     It assembles only for sm_120a / sm_120f, never for plain sm_120.
//   * ggml block_nvfp4 = { uint8 d[4] (UE4M3, one per 16 elements), uint8 qs[32] }, 64 elements, 36 bytes.
//     qs byte j of a 16-element sub-block holds element j (low nibble) and element j+8 (high nibble).
//     There is no global FP32 tensor scale in the KV cache format.
//   * Raw K bytes are a valid mxf4nvf4 operand when Q is packed with the same split-half order, and the
//     four scale bytes d[0..3] of a block are exactly the 32-bit scale-factor register for one k64 step.
//   * V's scales run along d, not along tokens, so they cannot be the B-side block scales of P*V.
//     They are folded into P (P' = P * sV[token][d/16]) before P is rounded to E4M3, and the PV product
//     uses kind::f8f6f4 with V's raw e2m1 codes in byte containers.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------------------------------------------
// Architecture gate
// ------------------------------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ == 1200 && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))) \
      || defined(FA_NVFP4_FORCE_PTX)
#    define FA_NVFP4_AVAILABLE 1
#  endif
#endif

namespace fa_nvfp4 {

// ------------------------------------------------------------------------------------------------
// ggml NVFP4 format and model geometry
// ------------------------------------------------------------------------------------------------
constexpr int D              = 256;               // key_length == value_length
constexpr int N_HEAD_KV      = 4;
constexpr int QK             = 64;                // QK_NVFP4
constexpr int QK_SUB         = 16;                // QK_NVFP4_SUB
constexpr int BLOCK_BYTES    = 4 + QK / 2;        // sizeof(block_nvfp4) = 36
constexpr int BLOCKS_PER_ROW = D / QK;            // 4
constexpr int ROW_BYTES      = BLOCKS_PER_ROW * BLOCK_BYTES;   // 144: one token, one KV head
constexpr int TOKEN_BYTES    = N_HEAD_KV * ROW_BYTES;          // 576: K->nb[1] for a contiguous cache
constexpr int DBLKS          = D / QK_SUB;        // 16 scale blocks per head row

static_assert(ROW_BYTES % 16 == 0,   "a head row must be 16-byte aligned for TMA / cp.async.bulk");
static_assert(TOKEN_BYTES % 16 == 0, "TMA global strides must be multiples of 16 bytes");

// Byte offset, inside one 144-byte head row, of the UE4M3 scale of 16-wide d-block dblk (0..15).
__host__ __device__ constexpr int scale_byte(int dblk) { return (dblk / 4) * BLOCK_BYTES + (dblk % 4); }
// Byte offset of the packed byte that holds element d (and its partner d^8 inside the sub-block).
__host__ __device__ constexpr int qs_byte(int d) {
    return (d / QK) * BLOCK_BYTES + 4 + ((d % QK) / QK_SUB) * 8 + (d % 8);
}
// 0: element d is the low nibble of qs_byte(d), 1: high nibble.
__host__ __device__ constexpr int qs_nibble(int d) { return (d % QK_SUB) / 8; }
// Offset of the 32-bit word w (0..7) of block b's qs array. Word w holds the 8 hardware-k positions
// 8w..8w+7 of that k64 step; words 2s and 2s+1 belong to sub-block s, so scale alignment holds.
__host__ __device__ constexpr int qs_word(int b, int w) { return b * BLOCK_BYTES + 4 + 4 * w; }
// Offset of the 32-bit scale-factor register of block b.
__host__ __device__ constexpr int sf_word(int b) { return b * BLOCK_BYTES; }

static_assert(qs_byte(0) == 4 && qs_byte(8) == 4 && qs_byte(16) == 12 && qs_byte(64) == 40, "split-half layout");
static_assert(scale_byte(5) == 37, "scale byte map");

// ------------------------------------------------------------------------------------------------
// Prefill kernel tile configuration (Phase 2): S = Q K^T, O = P V
// ------------------------------------------------------------------------------------------------
namespace prefill {
constexpr int D_SPLIT         = 2;             // two warps per 16-row group, each owns D/2 columns of O
constexpr int NWARPS          = 8;             // MMA (consumer) warps = 2 warpgroups
constexpr int NTHREADS        = (NWARPS + 4) * 32;   // + 1 TMA producer warpgroup = 384
// setmaxnreg works per warpgroup and the register file per SM sub-partition, so 9 warps would cap every
// thread at 168 registers. The producer warpgroup gives its registers back to the MMA warps instead:
// 4 x 32 x PRODUCER_REGS + 8 x 32 x CONSUMER_REGS <= 65536.
constexpr int PRODUCER_REGS   = 40;
constexpr int CONSUMER_REGS   = 232;
static_assert(4 * 32 * PRODUCER_REGS + NWARPS * 32 * CONSUMER_REGS <= 65536, "register split exceeds the SM");
constexpr int ROWS_PER_WARP   = 16;            // one m16 tile per row group
constexpr int BLOCK_M         = NWARPS / D_SPLIT * ROWS_PER_WARP;   // 64 query rows = ncols1 tokens x ncols2 heads
constexpr int D_PER_WARP      = D / D_SPLIT;   // 128 -> O accumulator 64 fp32 registers per thread
constexpr int BLOCK_N         = 64;            // KV tokens per pipeline step
constexpr int STAGES          = 3;             // 3, not 4: the tile-parity V^T double buffer needs the smem
constexpr int VT_TOKENS       = 32;            // one V^T buffer covers one k32 step of the PV product
constexpr int Q_TERMS         = 2;             // Q ~= Q_hi + Q_lo, both FP4 (see docs: precision)
constexpr int MIN_CTAS_PER_SM = 1;

struct alignas(128) Smem {
    // TMA destinations (one 2-D tensor-map box each): raw ggml rows, [stage][0 = K, 1 = V][token][144 B]
    uint8_t kv[STAGES][2][BLOCK_N][ROW_BYTES];
    // V^T: [tile parity][k32 step kk][d][32 logical tokens], e2m1 code << 2 per byte, in vt_offset order.
    // Both steps of a tile are staged together and the
    // buffers alternate by tile parity, so one named barrier per tile and d-half suffices.
    uint8_t vt[2][2][D][VT_TOKENS];
    // Exchange between the two warps of a row group, which each compute S for one k32 half of the tile:
    // P_loc (f16x2, C-fragment order, [word][lane]) and the tile row max (later the final row sum).
    uint32_t px[NWARPS / D_SPLIT][D_SPLIT][BLOCK_N / 16][2][32];
    float    pmx[NWARPS / D_SPLIT][D_SPLIT][ROWS_PER_WARP];
    // mbarriers: full = TMA bytes landed, empty = all consumer warps released the stage
    uint64_t full[STAGES];
    uint64_t empty[STAGES];
};
// MIN_CTAS_PER_SM * (sizeof + 1 KB reserved) <= 100 KB, and one block may use at most 99 KB
static_assert(sizeof(Smem) + 1024 <= (100 * 1024) / MIN_CTAS_PER_SM, "prefill smem exceeds the per-SM budget");
static_assert(sizeof(Smem) <= 99 * 1024, "prefill smem exceeds the per-block limit");
static_assert((BLOCK_N * ROW_BYTES) % 128 == 0, "each TMA destination must stay 128-byte aligned");
// The Q staging area (Q_TERMS x BLOCK_M x 144 B) aliases kv[] before the first K/V copy is issued.
static_assert(Q_TERMS * BLOCK_M * ROW_BYTES <= (int) sizeof(((Smem *) nullptr)->kv), "Q staging alias");
} // namespace prefill

// ------------------------------------------------------------------------------------------------
// Decode kernel tile configuration (Phase 3): swapped orientation, S^T = K Q^T, O^T = V^T P'^T
// ------------------------------------------------------------------------------------------------
namespace decode {
constexpr int NCONSUMERS      = 4;             // consumer warps, each owns whole TILE_N-token steps
constexpr int NWARPS          = NCONSUMERS + 1;   // + 1 TMA producer warp
constexpr int NTHREADS        = NWARPS * 32;   // 160
constexpr int TILE_N          = 32;            // KV tokens per stage (two m16 tiles of S^T)
constexpr int STAGES          = 4;
constexpr int HEADS           = 8;             // MMA N: query heads per KV head, padded to 8
constexpr int Q_TERMS         = 2;
constexpr int MIN_CTAS_PER_SM = 2;

struct alignas(128) Smem {
    uint8_t kv[STAGES][2][TILE_N][ROW_BYTES];      // raw ggml rows
    uint8_t q[Q_TERMS][HEADS][ROW_BYTES];          // quantized Q (ggml layout, split-half nibbles)
    __half  p[NCONSUMERS][HEADS][TILE_N];          // P_loc hand-off per warp: S^T C-fragment layout -> PV B-fragment layout
    float   q_alpha[HEADS];                        // per-head two-level Q factor
    uint64_t full[STAGES];
    uint64_t empty[STAGES];
};
static_assert(sizeof(Smem) + 1024 <= (100 * 1024) / MIN_CTAS_PER_SM, "decode smem exceeds the 2-CTA budget");
static_assert((TILE_N * ROW_BYTES) % 128 == 0, "each TMA destination must stay 128-byte aligned");
// Final cross-warp combine reuses kv[]: NCONSUMERS x (D x HEADS fp32 O^T + 2 x HEADS stats)
static_assert(NCONSUMERS * (D * HEADS + 2 * HEADS) * 4 <= (int) sizeof(((Smem *) nullptr)->kv), "combine alias");
} // namespace decode

// ------------------------------------------------------------------------------------------------
// Warp fragment maps (CUTLASS mma_traits_sm120.hpp SM120_16x8x64_TN_VS and m16n8k32 8-bit layouts)
//   g = lane / 4 (groupID), t = lane % 4 (thread in group)
//   mxf4nvf4 m16n8k64:
//     A: a0 row g, k 8t..8t+7 | a1 row g+8 | a2 row g, k 32+8t.. | a3 row g+8, k 32+8t..   (8 nibbles/reg)
//     B: b0 col g, k 8t..8t+7 | b1 col g, k 32+8t..
//     SFA: row g + 8*(lane & 1), byte i = k-block i (lanes with t >= 2 are ignored by selector 0)
//     SFB: col g, byte i = k-block i
//   f8f6f4 m16n8k32 (8-bit containers, e2m1 code in bits [5:2]):
//     A: a0 row g, k 4t..4t+3 | a1 row g+8 | a2 row g, k 16+4t.. | a3 row g+8, k 16+4t..
//     B: b0 col g, k 4t..4t+3 | b1 col g, k 16+4t..
//   C/D (both): d0 (g, 2t) d1 (g, 2t+1) d2 (g+8, 2t) d3 (g+8, 2t+1)
// ------------------------------------------------------------------------------------------------
__host__ __device__ constexpr int frag_g(int lane)        { return lane >> 2; }
__host__ __device__ constexpr int frag_t(int lane)        { return lane & 3; }
__host__ __device__ constexpr int sfa_row(int lane)       { return (lane >> 2) + 8 * (lane & 1); }

// Prefill PV: logical k (0..31) of k32 step kk -> physical token of the 64-token tile, chosen so that
// the m16n8k64 C fragments of S for n-tiles (2q, 2q+1) are already the A fragment of P for k-group q.
__host__ __device__ constexpr int pv_phys_token(int kk, int logical) {
    return 16 * (2 * kk + logical / 16) + 8 * ((logical % 4) / 2) + 2 * ((logical % 16) / 4) + (logical % 2);
}
static_assert(pv_phys_token(0, 0) == 0 && pv_phys_token(0, 1) == 1 && pv_phys_token(0, 2) == 8 &&
              pv_phys_token(0, 3) == 9 && pv_phys_token(0, 4) == 2 && pv_phys_token(1, 31) == 63, "pv permutation");

// Byte offset of (d, logical k) in one V^T buffer of prefill::Smem::vt. Rows are 32 B; logical k = 16h + 4t + i
// lives at byte 8t + 4h + i, so lane t's B fragment (k 4t..4t+3 and 16+4t..16+4t+3) is one aligned LDS.64,
// and the 8 rows read by one instruction are 256 contiguous bytes (conflict-free without a swizzle).
__host__ __device__ constexpr int vt_offset(int d, int k) {
    return d * prefill::VT_TOKENS + 8 * ((k & 15) >> 2) + 4 * (k >> 4) + (k & 3);
}
static_assert(vt_offset(0, 4) + 4 == vt_offset(0, 20) && vt_offset(1, 0) == 32, "vt layout");

#ifdef __CUDACC__
// ------------------------------------------------------------------------------------------------
// PTX building blocks
// ------------------------------------------------------------------------------------------------

// D += A * B with per-16 UE4M3 block scales on both operands (scale-factor selectors byte 0, thread 0).
static __device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                                const uint32_t sfa, const uint32_t sfb) {
#ifdef FA_NVFP4_AVAILABLE
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
#else
    (void) d; (void) a; (void) b; (void) sfa; (void) sfb;
    __trap();
#endif
}

// Prefill PV: D += A(e4m3, P') * B(e2m1, V codes).
static __device__ __forceinline__ void mma_e4m3_e2m1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#ifdef FA_NVFP4_AVAILABLE
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) d; (void) a; (void) b;
    __trap();
#endif
}

// Decode PV (swapped): D += A(e2m1, V^T codes) * B(e4m3, P'^T).
static __device__ __forceinline__ void mma_e2m1_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#ifdef FA_NVFP4_AVAILABLE
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) d; (void) a; (void) b;
    __trap();
#endif
}

// Four packed split-half bytes (lo nibble = element j, hi = element j+8) -> f8f6f4 e2m1 containers.
static __device__ __forceinline__ uint32_t fp4_lo_to_f8f6f4(const uint32_t x) { return (x & 0x0F0F0F0Fu) << 2; }
static __device__ __forceinline__ uint32_t fp4_hi_to_f8f6f4(const uint32_t x) { return (x >> 2) & 0x3C3C3C3Cu; }

// Round 4 floats to E4M3 (RN, saturating) and pack them little-endian: byte i <- x_i.
static __device__ __forceinline__ uint32_t pack_e4m3x4(const float x0, const float x1, const float x2, const float x3) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));   // upper byte <- first source
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
    return uint32_t(lo) | (uint32_t(hi) << 16);
}

// Round two floats to E2M1 (RN, saturating): low nibble <- lo, high nibble <- hi.
// cvt .e2m1x2 is arch-specific like the block-scaled MMA, so it is gated the same way: nvcc -arch=sm_120a
// also emits a generic compute_120 PTX, and ptxas rejects the instruction there.
static __device__ __forceinline__ uint32_t pack_e2m1x2(const float lo, const float hi) {
#ifdef FA_NVFP4_AVAILABLE
    uint16_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#else
    (void) lo; (void) hi;
    __trap();
    return 0;
#endif
}

// E2M1 code (bit 3 sign, bits 2..1 exponent, bit 0 mantissa) -> float.
static __device__ __forceinline__ float e2m1_value(const uint32_t c) {
    const uint32_t e   = (c >> 1) & 3u;
    const float    mag = e == 0 ? 0.5f * float(c & 1u) : float(1u << (e - 1)) * (1.0f + 0.5f * float(c & 1u));
    return (c & 8u) ? -mag : mag;
}

// Positive float -> UE4M3 scale code (RN, saturating to 448).
static __device__ __forceinline__ uint32_t ue4m3_code(const float x) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(x));
    return r & 0x7Fu;
}

// UE4M3 code -> float (exact).
static __device__ __forceinline__ float ue4m3_to_f32(const uint32_t code) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t) (code & 0xFFu)));
    return __half2float(__ushort_as_half((unsigned short) (h2 & 0xFFFFu)));
}

// Quantize one 16-element sub-block of (x * inv_alpha) into ggml NVFP4 layout.
//   Two lanes cooperate: the even lane of a pair owns elements 0..7, the odd lane elements 8..15.
//   Returns the 4 packed bytes this lane writes: even lane -> bytes 0..3 (j = 0..3), odd lane -> bytes 4..7.
//   *scale_code is valid in both lanes. *deq receives the lane's 8 dequantized values (times alpha) so the
//   caller can form the residual for the second Q term.
static __device__ __forceinline__ uint32_t quantize_sub16_pair(const float (&x)[8], const float inv_alpha,
                                                               const float alpha, uint32_t * scale_code,
                                                               float (&deq)[8]) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    const uint32_t code = ue4m3_code(amax * inv_alpha * (1.0f / 6.0f));
    const float    sc   = ue4m3_to_f32(code);
    const float    inv  = sc > 0.0f ? inv_alpha / sc : 0.0f;
    *scale_code = code;

    // partner values: even lane needs elements 8..11 (odd lane's 0..3), odd lane needs 4..7 (even lane's 4..7)
    const bool odd = threadIdx.x & 1;
    float own[4], other[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = odd ? x[i] : x[4 + i];
        const float recv = __shfl_xor_sync(0xFFFFFFFFu, send, 1);
        own[i]   = odd ? x[4 + i] : x[i];     // even: elements 0..3 ; odd: elements 12..15
        other[i] = recv;                      // even: elements 8..11; odd: elements 4..7
    }
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = odd ? other[i] : own[i];    // element j
        const float hi = odd ? own[i]   : other[i];  // element j + 8
        packed |= pack_e2m1x2(lo * inv, hi * inv) << (8 * i);
    }
    // dequantized copy of this lane's own 8 inputs
    const float step = sc * alpha;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const uint32_t c  = pack_e2m1x2(x[i] * inv, x[i + 1] * inv);
        const float    v0 = e2m1_value(c & 0xF);
        const float    v1 = e2m1_value((c >> 4) & 0xF);
        deq[i]     = v0 * step;
        deq[i + 1] = v1 * step;
    }
    return packed;
}
#endif // __CUDACC__

} // namespace fa_nvfp4
