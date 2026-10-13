// fa_nvfp4_prefill.cuh
//
// Phase 2: fused prefill FlashAttention for head dim 256 with a native NVFP4 (ggml block_nvfp4) KV cache
// on sm_120a. S = Q K^T runs as block-scaled FP4 MMA on the raw K bytes; O = P V runs as E4M3 x E2M1 MMA
// on the raw V codes with V's scales folded into P (docs/phase1-layout.md, option B).
//
// CTA = 8 MMA warps = 4 row groups x D_SPLIT 2. Rows are (query token, head of the GQA group) pairs flattened
// as row = token * gqa + head_in_group, so a GQA group fills the 64 rows without padding. The two warps of a
// row group each compute S for one 32-token half of the tile, exchange tile row maxima and P halves through
// shared memory, and each own 128 columns of O.
// K/V tiles stream through a 3-stage TMA (2-D tensor map per K and V, one box = 64 rows x 144 B) + mbarrier
// pipeline driven by a dedicated producer warpgroup (one elected lane issues, the group donates its registers
// to the MMA warps via setmaxnreg), so the MMA warps never wait on stage release.
//
// Two-level P: E4M3 cannot hold P = exp(s - m_running) for tiles far below the running max (P * sV
// underflows 2^-9 and the tail mass vanishes; with an attention sink this is a 90 % output error).
// Each tile is therefore exponentiated against its own base a_t = max(tile row max, running max - 32),
// so P_loc <= 1 with max 1, and O, l are kept in units of exp2(a_t): moving to the next tile rescales them
// by exp2(a_t - a_t+1) (at most 2^32 up), and O / l is unit-free. P' = P_loc * sV * B with B = 2^k
// constant per (KV head, 16-wide d-block), chosen from the largest V scale code in the cache so that
// max P' <= 240; 1/B is applied once at the end. The PV MMAs accumulate straight into O.
//
// Prepasses: k_tile_states classifies every (row block, KV tile) as skip / unmasked / mixed (only mixed tiles
// load the mask); k_v_scale_max finds the per-d-block max V scale code that sets B.

#pragma once

#include "fa_nvfp4_common.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>

namespace fa_nvfp4 {
namespace prefill {

struct Params {
    // TMA descriptors for the K and V views (filled by launch()): dims ordered by stride, box = 64 rows
    CUtensorMap tm_k, tm_v;
    int tm_tok_dim, tm_head_dim;   // coordinate index of the token and kv-head dims (1 or 2)
    // Q: f32, element (d, token, head, seq) at Q + seq*nbq3 + head*nbq2 + token*nbq1 (bytes) + 4d
    const float * Q;
    size_t nbq1, nbq2, nbq3;
    // K, V: ggml block_nvfp4 rows of 144 bytes, row (token, kv_head, seq) at base + seq*nb3 + kv_head*nb2 + token*nb1
    const uint8_t * K;
    size_t nbk1, nbk2, nbk3;
    const uint8_t * V;
    size_t nbv1, nbv2, nbv3;
    // mask: f16, element (kv, token, seq) at mask + seq*mask_s3 + token*mask_s1 + kv (elements); may be null.
    // mask_s1 must be even (half2 loads). Broadcast over heads.
    const half * mask;
    size_t mask_s1, mask_s3;
    // dst: f32 [D, n_head, n_q, n_seq], contiguous (ggml FLASH_ATTN_EXT output layout)
    float * dst;
    // per (seq, row block, KV tile): TILE_SKIP / TILE_FULL / TILE_MIXED at tile_state[(seq*nblk + blk)*n_tiles + T],
    // filled by k_tile_states. Null = visit every tile without a mask.
    const uint8_t  * tile_state;
    // per (seq, kv head, 16-wide d-block): max UE4M3 V scale code, filled by k_v_scale_max
    const int      * v_scale_max;
    int   n_q, n_kv, n_head, n_head_kv, n_seq;
    float scale;
    float softcap;                 // 0 = off; else logits become softcap * tanh(scale * q.k / softcap) (Gemma 2)
    // head size of Q, K, V and dst: 256, 128 or 64 (0 = 256). Below 256, launch() regroups heads_per_row(head_dim)
    // adjacent KV heads into one virtual 144-byte row (kv_view_supported() must hold for K and V).
    int   head_dim;
    int   hpr;                     // set by launch(): heads per virtual row; n_head_kv and nbk2 / nbv2 are virtual
};

// tile states: fully masked (not visited), unmasked (mask all zero, not loaded), mixed (mask applied per element)
constexpr uint8_t TILE_SKIP = 0, TILE_FULL = 1, TILE_MIXED = 2;

__host__ __device__ constexpr int n_row_blocks(int n_q, int gqa) { return (n_q * gqa + BLOCK_M - 1) / BLOCK_M; }
__host__ __device__ constexpr int n_kv_tiles(int n_kv) { return (n_kv + BLOCK_N - 1) / BLOCK_N; }

// Device scratch for launch(), carved from one allocation of workspace_bytes().
static inline size_t workspace_bytes(int n_q, int n_kv, int n_head, int n_head_kv, int n_seq, int head_dim = D) {
    n_head_kv /= heads_per_row(head_dim);
    const size_t nblk = (size_t) n_row_blocks(n_q, n_head / n_head_kv) * n_seq;
    const size_t states = (nblk * n_kv_tiles(n_kv) + 15) / 16 * 16;
    return states + (size_t) n_seq * n_head_kv * DBLKS * sizeof(int);
}

#ifdef __CUDACC__

constexpr uint32_t STAGE_BYTES = 2 * BLOCK_N * ROW_BYTES;

// ------------------------------------------------------------------------------------------------
// Prepass 1: state of every (row block, 64-token KV tile) for the block's query tokens: TILE_SKIP if every
// entry is -inf, TILE_FULL if every entry is exactly 0, else TILE_MIXED (columns past n_kv count as mixed, so
// the tail tile takes the per-element path). Handles any mask: causal, sliding window, several sequences in
// one batch. One warp per tile; grid (ceil(n_tiles / 32), n_row_blocks, n_seq), block 256.
// ------------------------------------------------------------------------------------------------
constexpr int STATE_TILES_PER_CTA = 32;

__global__ void __launch_bounds__(256) k_tile_states(const half * mask, const size_t mask_s1, const size_t mask_s3,
                                                     const int n_q, const int n_kv, const int gqa, uint8_t * state) {
    const int blk = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tok_lo = (blk * BLOCK_M) / gqa;
    const int tok_hi = min(n_q - 1, (blk * BLOCK_M + BLOCK_M - 1) / gqa);
    const half * m   = mask + seq * mask_s3;
    const int n_t    = n_kv_tiles(n_kv);
    for (int i = warp; i < STATE_TILES_PER_CTA; i += 8) {
        const int T = blockIdx.x * STATE_TILES_PER_CTA + i;
        if (T >= n_t) {
            break;
        }
        const int col = T * BLOCK_N + 2 * lane;   // lane covers columns 2*lane, 2*lane+1
        bool live = false, nonzero = false;
        for (int tok = tok_lo; tok <= tok_hi; ++tok) {
            const half * mr = m + tok * mask_s1;
            if (col + 1 < n_kv) {
                const float2 v = __half22float2(*reinterpret_cast<const half2 *>(mr + col));
                live    |= v.x > -INFINITY || v.y > -INFINITY;
                nonzero |= v.x != 0.0f || v.y != 0.0f;
            } else {
                if (col < n_kv) {
                    live |= __half2float(mr[col]) > -INFINITY;
                }
                nonzero = true;
            }
        }
        live    = __any_sync(0xFFFFFFFFu, live);
        nonzero = __any_sync(0xFFFFFFFFu, nonzero);
        if (lane == 0) {
            state[((size_t) seq * gridDim.y + blk) * n_t + T] = live ? (nonzero ? TILE_MIXED : TILE_FULL) : TILE_SKIP;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Main kernel. grid (n_row_blocks, n_head_kv, n_seq), block NTHREADS, dynamic smem sizeof(Smem).
// ------------------------------------------------------------------------------------------------
constexpr float FLOOR_LOG2 = 32.0f;   // tile base never drops more than 2^32 below the running max
// named barriers: 1, 2 = V^T staging per d-half, 3 = Q staging released, 4..7 = warp pair of row group 0..3
constexpr int   BAR_Q_FREE = 3;
constexpr int   BAR_PAIR0  = 4;

// CAP: logit softcap compiled in or out; HPR: heads per virtual row (1, 2, 4), compile-time so the D = 256 kernel
// is unchanged by the head-size support
template <bool CAP, int HPR>
__global__ void __launch_bounds__(NTHREADS, MIN_CTAS_PER_SM) k_prefill(const __grid_constant__ Params p) {
#ifdef FA_NVFP4_AVAILABLE
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem & sm = *reinterpret_cast<Smem *>(smem_raw);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = frag_g(lane), t = frag_t(lane);
    const int rg = warp / D_SPLIT;   // row group: rows 16*rg .. 16*rg+15
    const int dh = warp % D_SPLIT;   // O columns 128*dh .. 128*dh+127
    const int blk = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    const int gqa    = p.n_head / p.n_head_kv;
    const int n_t    = n_kv_tiles(p.n_kv);
    // Head sizes below 256 (HPR > 1): row group rg serves only head segment part = rg % HPR of the virtual row, so
    // its S skips the other segments' k64 blocks and its two warps split that segment's DR columns of O. Rows of a
    // segment are (token, head of the real GQA group) pairs, q = token * G + head; a block holds RPB of each segment.
    constexpr int DR   = D / HPR;                  // real head size
    constexpr int BPP  = BLOCKS_PER_ROW / HPR;     // k64 blocks per segment
    constexpr int DPW  = DR / D_SPLIT;             // O columns per warp
    constexpr int RPB  = BLOCK_M / HPR;            // rows of one segment per block
    const int     G    = gqa / HPR;                // query heads per real KV head
    const int     part = rg % HPR;
    const int     n_rows = p.n_q * G;              // rows per segment
    const int     cb   = part * DR + dh * DPW;     // first virtual O column of this warp
    // The producer and the MMA warps walk the same tile sequence: every tile whose state is not TILE_SKIP.
    const uint8_t * tstate = p.tile_state ? p.tile_state + ((size_t) seq * gridDim.x + blk) * n_t : nullptr;
    auto tile_state = [&](const int T) -> int {
        if (tstate) {
            return __ldg(tstate + T);
        }
        return T == n_t - 1 && p.n_kv % BLOCK_N ? TILE_MIXED : TILE_FULL;
    };
    auto next_tile = [&](int T) -> int {   // first visited tile >= T, or n_t
        while (T < n_t && tile_state(T) == TILE_SKIP) {
            ++T;
        }
        return T;
    };

    // ---- Q: quantize 64 rows into a staging area that aliases kv[], load fragments into registers
    uint8_t (*qst)[BLOCK_M][ROW_BYTES] = reinterpret_cast<uint8_t (*)[BLOCK_M][ROW_BYTES]>(&sm.kv[0][0][0][0]);
    float * q_alpha = reinterpret_cast<float *>(&sm.vt[0][0][0][0]);
    for (int i = 0; i < (warp < NWARPS ? BLOCK_M / NWARPS : 0); ++i) {
        const int r  = warp * (BLOCK_M / NWARPS) + i;
        const int q = blk * RPB + (r / 16 / HPR) * 16 + r % 16, qpart = (r / 16) % HPR;
        const float * qrow = nullptr;
        if (q < n_rows) {
            const int tok = q / G, h = (kvh * HPR + qpart) * G + q % G;
            qrow = reinterpret_cast<const float *>(reinterpret_cast<const char *>(p.Q) + seq * p.nbq3 + h * p.nbq2 +
                                                   tok * p.nbq1);
        }
        quantize_q_row(qrow, qst[0][r], qst[1][r], &q_alpha[r], HPR, qpart);
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&sm.full[s], 1);
            mbar_init(&sm.empty[s], NWARPS);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp >= NWARPS) {   // ---- TMA producer warpgroup: one elected lane streams every listed tile
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(PRODUCER_REGS));
        named_bar_sync(BAR_Q_FREE, NTHREADS);   // MMA warps hold their Q fragments; staging may be overwritten
        if (warp == NWARPS && lane == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic Q staging writes -> async proxy
            const bool tok_first = p.tm_tok_dim == 1;
            int i = 0;
            for (int T = next_tile(0); T < n_t; T = next_tile(T + 1), ++i) {
                const int st = i % STAGES;
                if (i >= STAGES) {
                    mbar_wait(&sm.empty[st], ((i / STAGES) - 1) & 1);
                }
                const int tok0 = T * BLOCK_N;   // rows past n_kv read as 0
                const int c1 = tok_first ? tok0 : kvh, c2 = tok_first ? kvh : tok0;
                mbar_arrive_expect_tx(&sm.full[st], STAGE_BYTES);
                tma_load_4d(sm.kv[st][0][0], &p.tm_k, 0, c1, c2, seq, &sm.full[st]);
                tma_load_4d(sm.kv[st][1][0], &p.tm_v, 0, c1, c2, seq, &sm.full[st]);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(CONSUMER_REGS));

    uint32_t qa[Q_TERMS][BPP][4], qsf[Q_TERMS][BPP];   // the k64 blocks of this row group's segment
    const int lr0 = 16 * rg + g, lr1 = lr0 + 8;   // local rows of this thread's C fragments
#pragma unroll
    for (int term = 0; term < Q_TERMS; ++term) {
#pragma unroll
        for (int bl = 0; bl < BPP; ++bl) {
            const int b = part * BPP + bl;
            qa[term][bl][0] = ld_s32(&qst[term][lr0][qs_word(b, t)]);
            qa[term][bl][1] = ld_s32(&qst[term][lr1][qs_word(b, t)]);
            qa[term][bl][2] = ld_s32(&qst[term][lr0][qs_word(b, 4 + t)]);
            qa[term][bl][3] = ld_s32(&qst[term][lr1][qs_word(b, 4 + t)]);
            qsf[term][bl]   = ld_s32(&qst[term][16 * rg + sfa_row(lane)][sf_word(b)]);
        }
    }
    // S row factor (log2 domain); with a softcap the factor feeds tanh and softcap * LOG2E scales its output
    constexpr bool has_cap = CAP;
    const float cap_l2  = p.softcap * LOG2E;
    const float rs0 = q_alpha[lr0] * p.scale * (has_cap ? 1.0f / p.softcap : LOG2E);
    const float rs1 = q_alpha[lr1] * p.scale * (has_cap ? 1.0f / p.softcap : LOG2E);

    const int   gr0 = blk * RPB + (rg / HPR) * 16 + g, gr1 = gr0 + 8;   // segment rows of the C fragments
    const half * mrow0 = nullptr, * mrow1 = nullptr;
    if (p.mask) {
        mrow0 = p.mask + seq * p.mask_s3 + (size_t) (min(gr0, n_rows - 1) / G) * p.mask_s1;
        mrow1 = p.mask + seq * p.mask_s3 + (size_t) (min(gr1, n_rows - 1) / G) * p.mask_s1;
    }
    // B = 2^k per local d-block: k = 14 - exponent(max code) puts max(sV) * B in (120, 240]
    uint32_t bexp = 0;   // 8 x 4-bit k (k in -1..14 stored as k + 1)
    {
        const int * vmax = p.v_scale_max + (seq * p.n_head_kv + kvh) * DBLKS + cb / QK_SUB;
#pragma unroll
        for (int i = 0; i < DPW / QK_SUB; ++i) {
            bexp |= (uint32_t) (14 - (__ldg(vmax + i) >> 3) + 1) << (4 * i);
        }
    }
    named_bar_sync(BAR_Q_FREE, NTHREADS);   // release the Q staging area to the producer


    float o[DPW / 8][4];
#pragma unroll
    for (int nt = 0; nt < DPW / 8; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }
    // per row: running max M, current tile base a (O and l are in units of exp2(a)), l
    float M0 = -INFINITY, M1 = -INFINITY, a0 = -INFINITY, a1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    // V^T staging: the warps that read a column range stage it, then sync on that group's barrier. Groups: the
    // d-half dh (HPR 1, 4 warps) or the segment part (HPR 2: 4 warps, HPR 4: 2 warps). Work item of a thread:
    // 8 d x 4 logical tokens; warp sj of the group takes d-groups 4sj..4sj+3 (one per 8-lane quarter vt_r), lane & 7
    // the token quad.
    constexpr int NSG     = HPR == 1 ? D_SPLIT : HPR;      // staging groups
    constexpr int SG_THR  = NWARPS * 32 / NSG;
    const int     sgrp    = HPR == 1 ? dh : part;
    const int     sj      = HPR == 1 ? rg : HPR == 2 ? (rg / 2) * 2 + dh : dh;
    const int     vt_bar  = HPR < 4 ? 1 + sgrp : sgrp < 2 ? 1 + sgrp : 6 + sgrp;   // named barriers 1, 2, 8, 9
    const int vt_r = lane >> 3, tq = lane & 7;
    const int dgrp = sgrp * (32 / NSG) + 4 * sj + vt_r;     // 8-wide virtual d-group, 0..31
    const int vt_b   = dgrp >> 3;
    const int vt_off = vt_b * BLOCK_BYTES + 4 + ((dgrp >> 1) & 3) * 8 + 4 * (dgrp & 1);
    const int vt_d   = vt_b * QK + ((dgrp >> 1) & 3) * QK_SUB + 4 * (dgrp & 1);

    auto stage_vt = [&](const uint8_t * Vsrc, const int buf) {   // V^T staging for this d-half, both k32 steps
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint8_t * vt = sm.vt[buf][kk][0];
            uint32_t lo[4], hi[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t x = ld_s32(Vsrc + pv_phys_token(kk, 4 * tq + i) * ROW_BYTES + vt_off);
                lo[i] = fp4_lo_to_f8f6f4(x);
                hi[i] = fp4_hi_to_f8f6f4(x);
            }
            const uint32_t l01a = __byte_perm(lo[0], lo[1], 0x5140), l01b = __byte_perm(lo[0], lo[1], 0x7362);
            const uint32_t l23a = __byte_perm(lo[2], lo[3], 0x5140), l23b = __byte_perm(lo[2], lo[3], 0x7362);
            const uint32_t h01a = __byte_perm(hi[0], hi[1], 0x5140), h01b = __byte_perm(hi[0], hi[1], 0x7362);
            const uint32_t h23a = __byte_perm(hi[2], hi[3], 0x5140), h23b = __byte_perm(hi[2], hi[3], 0x7362);
            // Store m writes row residue q = (m + r) % 4 so that the four lane quarters hit rows with distinct
            // d % 4 and the 32 lanes cover all 32 banks.
            const int k = 4 * tq;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int      q   = (m + vt_r) & 3;
                const uint32_t sel = (q & 1) ? 0x7632u : 0x5410u;
                *reinterpret_cast<uint32_t *>(vt + vt_offset(vt_d + q, k)) =
                    __byte_perm((q & 2) ? l01b : l01a, (q & 2) ? l23b : l23a, sel);
                *reinterpret_cast<uint32_t *>(vt + vt_offset(vt_d + 8 + q, k)) =
                    __byte_perm((q & 2) ? h01b : h01a, (q & 2) ? h23b : h23a, sel);
            }
        }

    };

    // S = Q K^T for this warp's k32 half (16 rows x 32 tokens: n-tiles 4*dh .. 4*dh+3), raw K rows as B.
    // MMA number idx (0..31) = (term, k-block b, n-tile jj); issued in pieces so it can overlap the PV fold.
    constexpr int NJ = BLOCK_N / 16;   // 4 n-tiles per half
    constexpr int N_SMMA = Q_TERMS * BPP * NJ;
    auto s_mma = [&](float (&acc)[NJ][4], const uint8_t * Ksrc, const int idx) {
        const int term = idx / (BPP * NJ), bl = (idx / NJ) % BPP, jj = idx % NJ;
        const int b = part * BPP + bl;
        const uint8_t * krow = Ksrc + (32 * dh + g + 8 * jj) * ROW_BYTES;
        const uint32_t bb[2] = {ld_s32(krow + qs_word(b, t)), ld_s32(krow + qs_word(b, 4 + t))};
        mma_nvf4(acc[jj], qa[term][bl], bb, qsf[term][bl], ld_s32(krow + sf_word(b)));
    };

    // Software pipeline: S(t+1) and the V^T staging of tile t+1 run during the PV product of tile t.
    float sn[NJ][4];   // S of the next tile
    int T = next_tile(0);
    if (T < n_t) {
        mbar_wait(&sm.full[0], 0);
        stage_vt(sm.kv[0][1][0], 0);
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            sn[jj][0] = sn[jj][1] = sn[jj][2] = sn[jj][3] = 0.0f;
        }
#pragma unroll
        for (int idx = 0; idx < N_SMMA; ++idx) {
            s_mma(sn, sm.kv[0][0][0], idx);
        }
    }
    for (int it = 0; T < n_t; ++it) {
        const int  kv0    = T * BLOCK_N;
        const bool mixed  = tile_state(T) == TILE_MIXED;
        const int  T_next = next_tile(T + 1);
        const int  st     = it % STAGES;
        const uint8_t * Vs = sm.kv[st][1][0];

        float s[NJ][4];
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            s[jj][0] = sn[jj][0]; s[jj][1] = sn[jj][1]; s[jj][2] = sn[jj][2]; s[jj][3] = sn[jj][3];
            sn[jj][0] = sn[jj][1] = sn[jj][2] = sn[jj][3] = 0.0f;
        }

        // ---- scale, mask (only on mixed tiles), half-tile row max. With a softcap, tanh is applied in its own
        //      uniform branch so the no-cap path stays the plain multiply.
        float r0 = rs0, r1 = rs1;
        if constexpr (has_cap) {
#pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                s[jj][0] = cap_l2 * fast_tanh(s[jj][0] * rs0);
                s[jj][1] = cap_l2 * fast_tanh(s[jj][1] * rs0);
                s[jj][2] = cap_l2 * fast_tanh(s[jj][2] * rs1);
                s[jj][3] = cap_l2 * fast_tanh(s[jj][3] * rs1);
            }
            r0 = r1 = 1.0f;
        }
        if (!mixed) {
#pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                s[jj][0] *= r0;
                s[jj][1] *= r0;
                s[jj][2] *= r1;
                s[jj][3] *= r1;
            }
        } else {
#pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                const int col = kv0 + 32 * dh + 8 * jj + 2 * t;
                float2 mk0 = make_float2(0.0f, 0.0f), mk1 = mk0;
                if (col + 1 < p.n_kv) {
                    if (mrow0) {
                        mk0 = __half22float2(*reinterpret_cast<const half2 *>(mrow0 + col));
                        mk1 = __half22float2(*reinterpret_cast<const half2 *>(mrow1 + col));
                    }
                } else {
                    mk0 = mk1 = make_float2(-INFINITY, -INFINITY);
                    if (col < p.n_kv) {
                        mk0.x = mrow0 ? __half2float(mrow0[col]) : 0.0f;
                        mk1.x = mrow1 ? __half2float(mrow1[col]) : 0.0f;
                    }
                }
                s[jj][0] = s[jj][0] * r0 + mk0.x * LOG2E;
                s[jj][1] = s[jj][1] * r0 + mk0.y * LOG2E;
                s[jj][2] = s[jj][2] * r1 + mk1.x * LOG2E;
                s[jj][3] = s[jj][3] * r1 + mk1.y * LOG2E;
            }
        }
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            mt0 = fmaxf(mt0, fmaxf(s[jj][0], s[jj][1]));
            mt1 = fmaxf(mt1, fmaxf(s[jj][2], s[jj][3]));
        }
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFFu, mt0, 1));
        mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFFu, mt0, 2));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFFu, mt1, 1));
        mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFFu, mt1, 2));

        // ---- exchange half-tile row maxima with the partner warp -> full tile row max (bitwise equal in both)
        if (t == 0) {
            sm.pmx[rg][dh][g]     = mt0;
            sm.pmx[rg][dh][g + 8] = mt1;
        }
        named_bar_sync(BAR_PAIR0 + rg, 64);
        mt0 = fmaxf(mt0, sm.pmx[rg][dh ^ 1][g]);
        mt1 = fmaxf(mt1, sm.pmx[rg][dh ^ 1][g + 8]);

        // ---- two-level softmax: new tile base a' = max(mt, M' - FLOOR), rescale O and l by exp2(a - a')
        float R0, R1, b0, b1;
        {
            M0 = fmaxf(M0, mt0);
            M1 = fmaxf(M1, mt1);
            const float na0 = mt0 == -INFINITY ? a0 : fmaxf(mt0, M0 - FLOOR_LOG2);
            const float na1 = mt1 == -INFINITY ? a1 : fmaxf(mt1, M1 - FLOOR_LOG2);
            R0 = a0 == -INFINITY ? 1.0f : fast_exp2(a0 - na0);   // O, l are still 0 while a == -inf
            R1 = a1 == -INFINITY ? 1.0f : fast_exp2(a1 - na1);
            a0 = na0;
            a1 = na1;
            b0 = a0 == -INFINITY ? 0.0f : a0;
            b1 = a1 == -INFINITY ? 0.0f : a1;
        }
        // P_loc as f16x2 [n-tile of the half][row g, row g+8], cols (2t, 2t+1); l is this warp's half-row sum
        __half2 ph_own[NJ][2], ph_oth[NJ][2];
        float ls0 = 0.0f, ls1 = 0.0f;
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const float p0 = fast_exp2(s[jj][0] - b0), p1 = fast_exp2(s[jj][1] - b0);
            const float p2 = fast_exp2(s[jj][2] - b1), p3 = fast_exp2(s[jj][3] - b1);
            ls0 += p0 + p1;
            ls1 += p2 + p3;
            ph_own[jj][0] = __floats2half2_rn(p0, p1);
            ph_own[jj][1] = __floats2half2_rn(p2, p3);
            sm.px[rg][dh][jj][0][lane] = *reinterpret_cast<const uint32_t *>(&ph_own[jj][0]);
            sm.px[rg][dh][jj][1][lane] = *reinterpret_cast<const uint32_t *>(&ph_own[jj][1]);
        }
        l0 = l0 * R0 + ls0;
        l1 = l1 * R1 + ls1;
#pragma unroll
        for (int nt = 0; nt < DPW / 8; ++nt) {
            o[nt][0] *= R0;
            o[nt][1] *= R0;
            o[nt][2] *= R1;
            o[nt][3] *= R1;
        }
        named_bar_sync(BAR_PAIR0 + rg, 64);   // partner's P_loc half is in px
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const uint32_t w0 = sm.px[rg][dh ^ 1][jj][0][lane], w1 = sm.px[rg][dh ^ 1][jj][1][lane];
            ph_oth[jj][0] = *reinterpret_cast<const __half2 *>(&w0);
            ph_oth[jj][1] = *reinterpret_cast<const __half2 *>(&w1);
        }

        named_bar_sync(vt_bar, SG_THR);   // vt[it & 1] of this warp's columns is complete

        // next tile: wait for its data and stage its V^T (buffer (it+1)&1 was last read by PV(t-1), which every
        // warp of the group finished before the barrier above). Without a next tile the S MMAs below re-read the
        // current stage and their result is discarded, so the PV loop stays branch-free.
        const bool has_next = T_next < n_t;
        const uint8_t * Kn = sm.kv[st][0][0];
        if (has_next) {
            const int st1 = (it + 1) % STAGES;
            mbar_wait(&sm.full[st1], ((it + 1) / STAGES) & 1);
            stage_vt(sm.kv[st1][1][0], (it + 1) & 1);
            Kn = sm.kv[st1][0][0];
        }

        // ---- O += P' V, one k32 step (32 permuted tokens) at a time: own half (kk = dh), then the partner's.
        //      The 32 S MMAs of the next tile are interleaved, two per d-block step.
#pragma unroll
        for (int src = 0; src < 2; ++src) {
            const int kk = dh ^ src;
            const __half2 (&ph)[NJ][2] = src ? ph_oth : ph_own;
            const uint8_t * vt = sm.vt[it & 1][kk][0];
            // V scale words of this thread's 8 tokens (q = 2kk + ql; tokens 16q + 2t + {0, 1, 8, 9})
            // for the 64-wide blocks of this warp's columns; byte i of a word = d-block 4*block + i.
            constexpr int NDB = DPW / QK_SUB;               // local d-blocks
            constexpr int NBL = NDB >= 4 ? NDB / 4 : 1;     // 64-wide blocks they touch
            uint32_t sw[2][NBL][4];
#pragma unroll
            for (int ql = 0; ql < 2; ++ql) {
                const int tk0 = 16 * (2 * kk + ql) + 2 * t;
#pragma unroll
                for (int bl = 0; bl < NBL; ++bl) {
                    const int off = sf_word(part * BPP + (dh * DPW) / QK + bl);   // block of column cb
                    sw[ql][bl][0] = ld_s32(Vs + (tk0) * ROW_BYTES + off);
                    sw[ql][bl][1] = ld_s32(Vs + (tk0 + 1) * ROW_BYTES + off);
                    sw[ql][bl][2] = ld_s32(Vs + (tk0 + 8) * ROW_BYTES + off);
                    sw[ql][bl][3] = ld_s32(Vs + (tk0 + 9) * ROW_BYTES + off);
                }
            }
#pragma unroll
            for (int i = 0; i < NDB; ++i) {   // local d-blocks (8 at HPR 1)
                const int bl = i / 4, bi = HPR == 4 ? 2 * dh + i : i % 4;   // compile-time except at HPR 4
                const int kexp = (int) ((bexp >> (4 * i)) & 0xFu) - 1;
                const __half2 Bh = __half2half2(__ushort_as_half((unsigned short) ((kexp + 15) << 10)));
                const uint32_t sel = bi | ((4 + bi) << 4);
                uint32_t a[4];
#pragma unroll
                for (int ql = 0; ql < 2; ++ql) {
                    const __half2 svA = __hmul2(e4m3x2_to_h2(__byte_perm(sw[ql][bl][0], sw[ql][bl][1], sel)), Bh);
                    const __half2 svB = __hmul2(e4m3x2_to_h2(__byte_perm(sw[ql][bl][2], sw[ql][bl][3], sel)), Bh);
#pragma unroll
                    for (int ci = 0; ci < 2; ++ci) {   // n-tiles (2q, 2q+1) of the half, q = 2kk + ql
                        a[2 * ql + ci] = h2_to_e4m3x2(__hmul2(ph[2 * ql][ci], svA)) |
                                         (h2_to_e4m3x2(__hmul2(ph[2 * ql + 1][ci], svB)) << 16);
                    }
                }
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    s_mma(sn, Kn, 2 * (NDB * src + i) + n);   // S of the next tile, overlapped with this fold
                    const int nt = 2 * i + n;
                    const int d  = cb + 8 * nt + g;
                    const uint2    bv    = *reinterpret_cast<const uint2 *>(vt + vt_offset(d, 4 * t));
                    const uint32_t bb[2] = {bv.x, bv.y};
                    mma_e4m3_e2m1(o[nt], a, bb);
                }
            }
        }

        __syncwarp();
        if (lane == 0) {
            mbar_arrive(&sm.empty[st]);   // this warp is done with stage st (K, V rows and scales)
        }
        T = T_next;
    }

    // ---- normalize (O / l, and 1/B per d-block) and store; l = sum of both warps' half-row sums
    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 2);
    named_bar_sync(BAR_PAIR0 + rg, 64);   // partner is done reading pmx for the last tile
    if (t == 0) {
        sm.pmx[rg][dh][g]     = l0;
        sm.pmx[rg][dh][g + 8] = l1;
    }
    named_bar_sync(BAR_PAIR0 + rg, 64);
    l0 += sm.pmx[rg][dh ^ 1][g];
    l1 += sm.pmx[rg][dh ^ 1][g + 8];
    const float il0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, il1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int gr = r ? gr1 : gr0;
        if (gr >= n_rows) {
            continue;
        }
        const int tok = gr / G, h = (kvh * HPR + part) * G + gr % G;
        float * out = p.dst + (((size_t) seq * p.n_q + tok) * p.n_head + h) * DR + dh * DPW + 2 * t;
        const float il = r ? il1 : il0;
#pragma unroll
        for (int nt = 0; nt < DPW / 8; ++nt) {
            const int   kexp = (int) ((bexp >> (4 * (nt / 2))) & 0xFu) - 1;
            const float f    = il * __int_as_float((127 - kexp) << 23);
            *reinterpret_cast<float2 *>(out + 8 * nt) = make_float2(o[nt][2 * r] * f, o[nt][2 * r + 1] * f);
        }
    }
#else
    (void) p;
    __trap();
#endif
}

// ------------------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------------------
// Launch the prepasses and the kernel. workspace: workspace_bytes(...) bytes of device memory (contents
// need not be preserved). Fills the tensor maps, tile_state and v_scale_max of p.
static inline cudaError_t launch(Params p, void * workspace, cudaStream_t stream) {
    p.hpr = heads_per_row(p.head_dim);
    if (p.hpr == 0 || !kv_view_supported(p.head_dim, p.K, p.nbk1, p.nbk2, p.nbk3, p.n_head_kv) ||
        !kv_view_supported(p.head_dim, p.V, p.nbv1, p.nbv2, p.nbv3, p.n_head_kv)) {
        return cudaErrorInvalidValue;
    }
    p.n_head_kv /= p.hpr;   // virtual rows of hpr adjacent heads
    p.nbk2      *= p.hpr;
    p.nbv2      *= p.hpr;
    const int gqa  = p.n_head / p.n_head_kv;
    const int nblk = n_row_blocks(p.n_q, gqa);
    const int n_t  = n_kv_tiles(p.n_kv);
    uint8_t * tile_state = static_cast<uint8_t *>(workspace);
    int     * vmax       = reinterpret_cast<int *>(tile_state + ((size_t) nblk * p.n_seq * n_t + 15) / 16 * 16);

    cudaError_t err = encode_kv_map(&p.tm_k, p.K, p.nbk1, p.nbk2, p.nbk3, p.n_kv, p.n_head_kv, p.n_seq, BLOCK_N,
                                    &p.tm_tok_dim, &p.tm_head_dim);
    int tok_dim_v = 0, head_dim_v = 0;
    if (err == cudaSuccess) {
        err = encode_kv_map(&p.tm_v, p.V, p.nbv1, p.nbv2, p.nbv3, p.n_kv, p.n_head_kv, p.n_seq, BLOCK_N, &tok_dim_v,
                            &head_dim_v);
    }
    if (err == cudaSuccess && (tok_dim_v != p.tm_tok_dim || head_dim_v != p.tm_head_dim)) {
        err = cudaErrorInvalidValue;   // K and V views must order token / head strides the same way
    }
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMemsetAsync(vmax, 0, (size_t) p.n_seq * p.n_head_kv * DBLKS * sizeof(int), stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int vchunks = min(64, (p.n_kv + 255) / 256);
    k_v_scale_max<<<dim3(vchunks, p.n_head_kv, p.n_seq), 256, 0, stream>>>(p.V, p.nbv1, p.nbv2, p.nbv3, p.n_kv, vmax);
    p.v_scale_max = vmax;
    p.tile_state  = nullptr;
    if (p.mask) {
        const dim3 grid((n_t + STATE_TILES_PER_CTA - 1) / STATE_TILES_PER_CTA, nblk, p.n_seq);
        k_tile_states<<<grid, 256, 0, stream>>>(p.mask, p.mask_s1, p.mask_s3, p.n_q, p.n_kv, gqa, tile_state);
        p.tile_state = tile_state;
    }
    void (*kern)(Params) = p.softcap != 0.0f ? (p.hpr == 1 ? k_prefill<true, 1> : p.hpr == 2 ? k_prefill<true, 2> : k_prefill<true, 4>)
                                             : (p.hpr == 1 ? k_prefill<false, 1> : p.hpr == 2 ? k_prefill<false, 2> : k_prefill<false, 4>);
    err = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));   // per device
    if (err != cudaSuccess) {
        return err;
    }
    kern<<<dim3(nblk, p.n_head_kv, p.n_seq), NTHREADS, sizeof(Smem), stream>>>(p);
    return cudaGetLastError();
}

#endif // __CUDACC__

} // namespace prefill
} // namespace fa_nvfp4
