// fa_nvfp4_decode.cuh
//
// Phase 3: split-KV decode FlashAttention for head dim 256 with a native NVFP4 (ggml block_nvfp4) KV cache on
// sm_120a. Swapped orientation (docs/phase1-layout.md section 7), so the query heads of one GQA group fill the
// MMA's N = 8 instead of wasting 16-row M tiles:
//   S^T = K Q^T    mxf4nvf4 m16n8k64, A = raw K rows (16 tokens), B = two-term FP4 Q^T (8 heads)
//   O^T = V^T P'^T f8f6f4 m16n8k32,   A = raw V codes (16 d = one scale block), B = P' (E4M3) for 32 tokens
// Rows (query token, head-in-group) are flattened like the prefill kernel and taken 8 at a time.
//
// CTA = 1 TMA producer warp + 8 consumer warps serving ng = 1..8 row groups of 8 rows: warp w takes row group
// w % ng and tile slot w / ng; tile i of the CTA's KV range goes to stage i % STAGES and slot i % (8 / ng), and
// the ng warps of a slot share that tile, so the KV range is read once for all ng * 8 rows. Each warp keeps its
// own O^T and softmax state; the slots of a row group are merged in shared memory and the CTA writes one
// partial (O, base, l) per row. k_decode_combine merges the KV splits.
// Two-level P as in the prefill kernel: per-tile base, O and l in units of exp2(base). P' is boosted by B = 2^k
// per (warp, d-block), k = 14 - exponent of the largest V scale code the warp has seen so far: a tile with a
// larger scale lowers k and rescales that d-block of O by the exact power of two. No prepass, no cache state.

#pragma once

#include "fa_nvfp4_common.cuh"

namespace fa_nvfp4 {
namespace decode {

struct Params {
    CUtensorMap tm_k, tm_v;        // filled by launch(): box = TILE_N rows of one head
    int tm_tok_dim, tm_head_dim;
    // same tensor conventions as prefill::Params
    const float * Q;
    size_t nbq1, nbq2, nbq3;
    const uint8_t * K;
    size_t nbk1, nbk2, nbk3;
    const uint8_t * V;
    size_t nbv1, nbv2, nbv3;
    const half * mask;             // may be null
    size_t mask_s1, mask_s3;
    float * dst;                   // f32 [D, n_head, n_q, n_seq]
    // filled by launch()
    float     * part_o;            // [seq][kv head][row group][split][HEADS][D]
    float2    * part_al;           // [seq][kv head][row group][split][HEADS]: (base, l)
    int n_splits, tiles_per_split;
    int groups_per_cta;            // row groups served by one CTA (1, 2, 4 or 8), set by launch()
    int   n_q, n_kv, n_head, n_head_kv, n_seq;
    float scale;
};

__host__ __device__ constexpr int n_row_groups(int n_q, int gqa) { return (n_q * gqa + HEADS - 1) / HEADS; }
// Row groups per CTA, from a measured sweep (decode_test groups, 4k-106k caches). L2 already deduplicates CTAs
// that read the same KV tile at the same time, so sharing a tile in shared memory only pays where one group per
// CTA leaves too few KV splits to fill the GPU (about 6-32 row groups, i.e. 8-42 query tokens at GQA 6).
__host__ __device__ constexpr int groups_per_cta(int ngrp) {
    return ngrp <= 5 ? 1 : ngrp <= 8 ? 2 : ngrp <= 16 ? 4 : ngrp <= 32 ? MAX_GROUPS : 1;
}

static inline size_t workspace_bytes(int n_q, int n_head, int n_head_kv, int n_seq, int n_splits) {
    const size_t parts = (size_t) n_seq * n_head_kv * n_row_groups(n_q, n_head / n_head_kv) * n_splits * HEADS;
    return parts * D * sizeof(float) + parts * sizeof(float2);
}

#ifdef __CUDACC__

constexpr uint32_t STAGE_BYTES = 2 * TILE_N * ROW_BYTES;
constexpr float    FLOOR_LOG2  = 32.0f;   // tile base never drops more than 2^32 below the running max
constexpr int      BAR_CONSUMERS = 1;     // named barrier over the consumer warps
constexpr int      BAR_Q_FREE    = 2;     // consumers -> producer: Q fragments loaded, staging area free
constexpr int      Q_STAGES      = 2;     // kv[] stages that hold the Q staging area
static_assert(Q_STAGES * 2 * TILE_N * ROW_BYTES >= Q_TERMS * MAX_GROUPS * HEADS * ROW_BYTES, "Q staging does not fit");

// bar.arrive: count this warp at the named barrier without waiting
static __device__ __forceinline__ void named_bar_arrive(int id, int nthreads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
// bar.sync issued by one lane of the producer warp: barriers count whole warps, so the elected lane's warp is
// counted once; the fence orders the consumers' generic reads of the staging area before the next TMA writes.
static __device__ __forceinline__ void named_bar_sync_warp0(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <int NG>   // row groups per CTA: 1, 2, 4 or MAX_GROUPS (compile-time so the 1-group case stays as tight as before)
__global__ void __launch_bounds__(NTHREADS, MIN_CTAS_PER_SM) k_decode(const __grid_constant__ Params p) {
#ifdef FA_NVFP4_AVAILABLE
    extern __shared__ __align__(128) uint8_t smem_raw[];
    Smem & sm = *reinterpret_cast<Smem *>(smem_raw);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = frag_g(lane), t = frag_t(lane);
    const int gqa    = p.n_head / p.n_head_kv;
    const int n_rows = p.n_q * gqa;
    const int ngrp   = n_row_groups(p.n_q, gqa);
    constexpr int ng     = NG;                      // row groups in this CTA
    constexpr int nslots = NCONSUMERS / NG;         // tile slots: tile i -> slot i % nslots
    const int ncg    = (ngrp + ng - 1) / ng;        // CTA groups per kv head
    const int split = blockIdx.x, cg = blockIdx.y % ncg, kvh = blockIdx.y / ncg, seq = blockIdx.z;
    const int row0   = cg * ng * HEADS;             // first row of this CTA
    const int lgrp   = warp % ng, slot = warp / ng; // consumer warp: local row group, tile slot
    const int n_t    = (p.n_kv + TILE_N - 1) / TILE_N;
    const int t_lo   = split * p.tiles_per_split;
    const int ntiles = max(0, min(n_t, t_lo + p.tiles_per_split) - t_lo);

    // ---- Q: the consumer warps quantize the ng * HEADS rows of this CTA into a staging area that aliases the last
    //      Q_STAGES stages of kv[], so the producer can fill the first stages while Q is still being read
    uint8_t (*qst)[MAX_GROUPS * HEADS][ROW_BYTES] =
        reinterpret_cast<uint8_t (*)[MAX_GROUPS * HEADS][ROW_BYTES]>(&sm.kv[STAGES - Q_STAGES][0][0][0]);
    if (warp < NCONSUMERS) {
        for (int r = warp; r < ng * HEADS; r += NCONSUMERS) {
            const int gr = row0 + r;
            const float * qrow = nullptr;
            if (gr < n_rows) {
                const int tok = gr / gqa, h = kvh * gqa + gr % gqa;
                qrow = reinterpret_cast<const float *>(reinterpret_cast<const char *>(p.Q) + seq * p.nbq3 + h * p.nbq2 +
                                                       tok * p.nbq1);
            }
            quantize_q_row(qrow, qst[0][r], qst[1][r], &sm.q_alpha[r]);
        }
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&sm.full[s], 1);
            mbar_init(&sm.empty[s], ng);   // released by the ng warps of the tile's slot
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == NCONSUMERS) {   // ---- TMA producer
        // The Q staging area is the last Q_STAGES stages: wait for the consumers' release before the first copy
        // into them (always, so that the barrier completes even for short ranges).
        const int i_q = min(ntiles, STAGES - Q_STAGES);
        if (lane == 0) {
            const bool tok_first = p.tm_tok_dim == 1;
            for (int i = 0; i < ntiles; ++i) {
                const int st = i % STAGES;
                if (i == i_q) {
                    named_bar_sync_warp0(BAR_Q_FREE, NTHREADS);
                }
                if (i >= STAGES) {
                    mbar_wait(&sm.empty[st], ((i / STAGES) - 1) & 1);
                }
                const int tok0 = (t_lo + i) * TILE_N;   // rows past n_kv read as 0
                const int c1 = tok_first ? tok0 : kvh, c2 = tok_first ? kvh : tok0;
                mbar_arrive_expect_tx(&sm.full[st], STAGE_BYTES);
                tma_load_4d(sm.kv[st][0][0], &p.tm_k, 0, c1, c2, seq, &sm.full[st]);
                tma_load_4d(sm.kv[st][1][0], &p.tm_v, 0, c1, c2, seq, &sm.full[st]);
            }
            if (ntiles <= i_q) {
                named_bar_sync_warp0(BAR_Q_FREE, NTHREADS);
            }
        }
        return;
    }

    // ---- consumer setup: Q^T B fragments (column g = head g), row factors and mask rows for heads 2t, 2t+1
    uint32_t qb[Q_TERMS][BLOCKS_PER_ROW][2], qsf[Q_TERMS][BLOCKS_PER_ROW];
#pragma unroll
    for (int term = 0; term < Q_TERMS; ++term) {
#pragma unroll
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            qb[term][b][0] = ld_s32(&qst[term][lgrp * HEADS + g][qs_word(b, t)]);
            qb[term][b][1] = ld_s32(&qst[term][lgrp * HEADS + g][qs_word(b, 4 + t)]);
            qsf[term][b]   = ld_s32(&qst[term][lgrp * HEADS + g][sf_word(b)]);
        }
    }
    float rs[2];
    const half * mrow[2] = {nullptr, nullptr};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int r = lgrp * HEADS + 2 * t + c;   // row of this CTA
        rs[c] = sm.q_alpha[r] * p.scale * LOG2E;
        if (p.mask) {
            mrow[c] = p.mask + seq * p.mask_s3 + (size_t) (min(row0 + r, n_rows - 1) / gqa) * p.mask_s1;
        }
    }
    named_bar_arrive(BAR_Q_FREE, NTHREADS);   // release the Q staging area to the producer (non-blocking)

    // running B = 2^k per d-block, 4 bits each; k starts at 14 (the boost for an all-subnormal block)
    uint32_t bexp[2] = {0xEEEEEEEEu, 0xEEEEEEEEu};

    float o[DBLKS][4];
#pragma unroll
    for (int db = 0; db < DBLKS; ++db) {
        o[db][0] = o[db][1] = o[db][2] = o[db][3] = 0.0f;
    }
    float M[2] = {-INFINITY, -INFINITY}, a[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (int i = slot; i < ntiles; i += nslots) {
        const int st  = i % STAGES;
        const int kv0 = (t_lo + i) * TILE_N;

        // mask values first so their latency overlaps the MMAs: tokens kv0 + 16mt + g + 8r, heads 2t + c
        float mk[2][2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int tok = kv0 + 16 * mt + g + 8 * r;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    mk[mt][r][c] = tok >= p.n_kv ? -INFINITY : (mrow[c] ? __half2float(mrow[c][tok]) * LOG2E : 0.0f);
                }
            }
        }

        mbar_wait(&sm.full[st], (i / STAGES) & 1);
        const uint8_t * Ks = sm.kv[st][0][0];
        const uint8_t * Vs = sm.kv[st][1][0];

        // ---- S^T (32 tokens x 8 heads): C rows = tokens 16mt + g (+8), cols = heads 2t, 2t+1
        float s[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.0f;
            const uint8_t * k0  = Ks + (16 * mt + g) * ROW_BYTES;
            const uint8_t * k1  = k0 + 8 * ROW_BYTES;
            const uint8_t * ksf = Ks + (16 * mt + sfa_row(lane)) * ROW_BYTES;
#pragma unroll
            for (int term = 0; term < Q_TERMS; ++term) {
#pragma unroll
                for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
                    const uint32_t ka[4] = {ld_s32(k0 + qs_word(b, t)), ld_s32(k1 + qs_word(b, t)),
                                            ld_s32(k0 + qs_word(b, 4 + t)), ld_s32(k1 + qs_word(b, 4 + t))};
                    mma_nvf4(s[mt], ka, qb[term][b], ld_s32(ksf + sf_word(b)), qsf[term][b]);
                }
            }
        }

        // ---- scale, mask, per-head tile max (over tokens: rows of this thread, then lanes with the same t)
        float mt_h[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float & v = s[mt][2 * r + c];
                    v = v * rs[c] + mk[mt][r][c];
                    mt_h[c] = fmaxf(mt_h[c], v);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            mt_h[c] = fmaxf(mt_h[c], __shfl_xor_sync(0xFFFFFFFFu, mt_h[c], 4));
            mt_h[c] = fmaxf(mt_h[c], __shfl_xor_sync(0xFFFFFFFFu, mt_h[c], 8));
            mt_h[c] = fmaxf(mt_h[c], __shfl_xor_sync(0xFFFFFFFFu, mt_h[c], 16));
        }

        // ---- two-level softmax per head: base a' = max(tile max, M' - FLOOR); O, l rescaled by exp2(a - a')
        float R[2], base[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            M[c] = fmaxf(M[c], mt_h[c]);
            const float na = mt_h[c] == -INFINITY ? a[c] : fmaxf(mt_h[c], M[c] - FLOOR_LOG2);
            R[c]    = a[c] == -INFINITY ? 1.0f : fast_exp2(a[c] - na);
            a[c]    = na;
            base[c] = a[c] == -INFINITY ? 0.0f : a[c];
            l[c]   *= R[c];
        }
#pragma unroll
        for (int db = 0; db < DBLKS; ++db) {
            o[db][0] *= R[0];
            o[db][1] *= R[1];
            o[db][2] *= R[0];
            o[db][3] *= R[1];
        }
        __half * pw = sm.p[warp][0];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const float pv = fast_exp2(s[mt][2 * r + c] - base[c]);
                    l[c] += pv;
                    pw[(2 * t + c) * TILE_N + 16 * mt + g + 8 * r] = __float2half_rn(pv);
                }
            }
        }
        __syncwarp();
        // B fragment source: head g, tokens 4t..4t+3 and 16+4t..16+4t+3
        const uint2 pa = *reinterpret_cast<const uint2 *>(pw + g * TILE_N + 4 * t);
        const uint2 pb = *reinterpret_cast<const uint2 *>(pw + g * TILE_N + 16 + 4 * t);
        __syncwarp();
        const __half2 pA01 = *reinterpret_cast<const __half2 *>(&pa.x), pA23 = *reinterpret_cast<const __half2 *>(&pa.y);
        const __half2 pB01 = *reinterpret_cast<const __half2 *>(&pb.x), pB23 = *reinterpret_cast<const __half2 *>(&pb.y);

        // ---- O^T += V^T P'^T, one 16-wide d-block (= one m16 tile and one V scale) at a time
#pragma unroll
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            uint32_t w[8];   // scale word of block b for tokens 4t+i (w[0..3]) and 16+4t+i (w[4..7])
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                w[i]     = ld_s32(Vs + (4 * t + i) * ROW_BYTES + sf_word(b));
                w[4 + i] = ld_s32(Vs + (16 + 4 * t + i) * ROW_BYTES + sf_word(b));
            }
            // max scale code per d-block over the tile's 32 tokens (the 4 t-lanes cover them all)
            uint32_t mx = __vmaxu4(__vmaxu4(__vmaxu4(w[0], w[1]), __vmaxu4(w[2], w[3])),
                                   __vmaxu4(__vmaxu4(w[4], w[5]), __vmaxu4(w[6], w[7])));
            mx = __vmaxu4(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
            mx = __vmaxu4(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
#pragma unroll
            for (int bi = 0; bi < 4; ++bi) {
                const int db = 4 * b + bi;
                const int sh = 4 * (db % 8);
                const int kold = (int) ((bexp[db / 8] >> sh) & 0xFu);
                const int ktile = 14 - (int) (((mx >> (8 * bi)) & 0x7Fu) >> 3);
                if (ktile < kold) {   // larger scale than seen so far: lower B, rescale O exactly
                    const float f = __int_as_float((127 + ktile - kold) << 23);
                    o[db][0] *= f; o[db][1] *= f; o[db][2] *= f; o[db][3] *= f;
                    bexp[db / 8] = (bexp[db / 8] & ~(0xFu << sh)) | ((uint32_t) ktile << sh);
                }
                const int kexp = ktile < kold ? ktile : kold;
                const __half2 Bh = __half2half2(__ushort_as_half((unsigned short) ((kexp + 15) << 10)));
                const uint32_t sel = bi | ((4 + bi) << 4);
                const __half2 sA01 = __hmul2(e4m3x2_to_h2(__byte_perm(w[0], w[1], sel)), Bh);
                const __half2 sA23 = __hmul2(e4m3x2_to_h2(__byte_perm(w[2], w[3], sel)), Bh);
                const __half2 sB01 = __hmul2(e4m3x2_to_h2(__byte_perm(w[4], w[5], sel)), Bh);
                const __half2 sB23 = __hmul2(e4m3x2_to_h2(__byte_perm(w[6], w[7], sel)), Bh);
                const uint32_t bb[2] = {h2_to_e4m3x2(__hmul2(pA01, sA01)) | (h2_to_e4m3x2(__hmul2(pA23, sA23)) << 16),
                                        h2_to_e4m3x2(__hmul2(pB01, sB01)) | (h2_to_e4m3x2(__hmul2(pB23, sB23)) << 16)};
                // A: d = 16db+g (low nibble) and 16db+g+8 (high nibble) share one byte per token
                const int off = qs_byte(16 * db + g);
                uint32_t x0 = 0, x1 = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    x0 |= uint32_t(Vs[(4 * t + k) * ROW_BYTES + off]) << (8 * k);
                    x1 |= uint32_t(Vs[(16 + 4 * t + k) * ROW_BYTES + off]) << (8 * k);
                }
                const uint32_t va[4] = {fp4_lo_to_f8f6f4(x0), fp4_hi_to_f8f6f4(x0), fp4_lo_to_f8f6f4(x1), fp4_hi_to_f8f6f4(x1)};
                mma_e2m1_e4m3(o[db], va, bb);
            }
        }

        __syncwarp();
        if (lane == 0) {
            mbar_arrive(&sm.empty[st]);
        }
    }

    // ---- merge the 4 consumer warps (reusing kv[] once every warp is done with its stages)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        l[c] += __shfl_xor_sync(0xFFFFFFFFu, l[c], 4);
        l[c] += __shfl_xor_sync(0xFFFFFFFFu, l[c], 8);
        l[c] += __shfl_xor_sync(0xFFFFFFFFu, l[c], 16);
    }
    named_bar_sync(BAR_CONSUMERS, NCONSUMERS * 32);
    float  * comb_o  = reinterpret_cast<float *>(&sm.kv[0][0][0][0]);   // [warp][head][d]
    float2 * comb_al = reinterpret_cast<float2 *>(comb_o + NCONSUMERS * HEADS * D);   // [warp][head]
#pragma unroll
    for (int db = 0; db < DBLKS; ++db) {
        const float invB = __int_as_float((127 - (int) ((bexp[db / 8] >> (4 * (db % 8))) & 0xFu)) << 23);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                comb_o[(warp * HEADS + 2 * t + c) * D + 16 * db + g + 8 * r] = o[db][2 * r + c] * invB;
            }
        }
    }
    if (g == 0) {
        comb_al[warp * HEADS + 2 * t]     = make_float2(a[0], l[0]);
        comb_al[warp * HEADS + 2 * t + 1] = make_float2(a[1], l[1]);
    }
    named_bar_sync(BAR_CONSUMERS, NCONSUMERS * 32);

    // combine the nslots warps of each local row group; warp w serves group w % ng
    for (int idx = threadIdx.x; idx < ng * HEADS * D; idx += NCONSUMERS * 32) {
        const int lg = idx / (HEADS * D), h = (idx / D) % HEADS, d = idx % D;
        const int grp = cg * ng + lg;   // global row group
        if (grp >= ngrp) {
            break;
        }
        float amax = -INFINITY;
        for (int s = 0; s < nslots; ++s) {
            amax = fmaxf(amax, comb_al[(s * ng + lg) * HEADS + h].x);
        }
        float acc = 0.0f, lsum = 0.0f;
        if (amax != -INFINITY) {
            for (int s = 0; s < nslots; ++s) {
                const int    w  = s * ng + lg;
                const float2 al = comb_al[w * HEADS + h];
                const float  f  = al.x == -INFINITY ? 0.0f : fast_exp2(al.x - amax);
                acc  += f * comb_o[(w * HEADS + h) * D + d];
                lsum += f * al.y;
            }
        }
        const size_t part = (((size_t) seq * p.n_head_kv + kvh) * ngrp + grp) * p.n_splits + split;
        p.part_o[part * HEADS * D + h * D + d] = acc;
        if (d == 0) {
            p.part_al[part * HEADS + h] = make_float2(amax, lsum);
        }
    }
#else
    (void) p;
    __trap();
#endif
}

// Merge the KV splits of every (seq, kv head, row group, row) and write dst. grid (n_head_kv * ngrp, n_seq), block D.
__global__ void __launch_bounds__(D) k_decode_combine(const __grid_constant__ Params p) {
    const int gqa  = p.n_head / p.n_head_kv;
    const int ngrp = n_row_groups(p.n_q, gqa);
    const int grp = blockIdx.x % ngrp, kvh = blockIdx.x / ngrp, seq = blockIdx.y;
    const int d   = threadIdx.x;
    const size_t part0 = (((size_t) seq * p.n_head_kv + kvh) * ngrp + grp) * p.n_splits;
    for (int h = 0; h < HEADS; ++h) {
        const int gr = grp * HEADS + h;
        if (gr >= p.n_q * gqa) {
            break;
        }
        float amax = -INFINITY;
        for (int s = 0; s < p.n_splits; ++s) {
            amax = fmaxf(amax, p.part_al[(part0 + s) * HEADS + h].x);
        }
        float acc = 0.0f, lsum = 0.0f;
        if (amax != -INFINITY) {
            for (int s = 0; s < p.n_splits; ++s) {
                const float2 al = p.part_al[(part0 + s) * HEADS + h];
                const float  f  = al.x == -INFINITY ? 0.0f : exp2f(al.x - amax);
                acc  += f * p.part_o[((part0 + s) * HEADS + h) * D + d];
                lsum += f * al.y;
            }
        }
        const int tok = gr / gqa, head = kvh * gqa + gr % gqa;
        p.dst[(((size_t) seq * p.n_q + tok) * p.n_head + head) * D + d] = lsum > 0.0f ? acc / lsum : 0.0f;
    }
}

#endif // __CUDACC__

// Split count for the device: at most one wave of CTAs (one per SM), and at least ~12 tiles per split (measured: fewer,
// longer splits win at every cache length; extra splits only add per-CTA setup and merge work).
static inline int choose_splits(int n_kv, int n_q, int n_head, int n_head_kv, int n_seq, int n_sms, int ng = 0) {
    const int n_t   = (n_kv + TILE_N - 1) / TILE_N;
    const int ngrp  = n_row_groups(n_q, n_head / n_head_kv);
    ng = ng > 0 ? ng : groups_per_cta(ngrp);
    const int other = n_head_kv * ((ngrp + ng - 1) / ng) * n_seq;
    int s = MIN_CTAS_PER_SM * n_sms / other;   // round down: one extra CTA would start a second wave
    s = s > n_t / 12 ? n_t / 12 : s;
    return s < 1 ? 1 : s;
}

#ifdef __CUDACC__
// Launch the decode kernel and the split combine. workspace: workspace_bytes(...) bytes for p.n_splits (set it with
// choose_splits).
static inline cudaError_t launch(Params p, void * workspace, cudaStream_t stream) {
    const int gqa  = p.n_head / p.n_head_kv;
    const int ngrp = n_row_groups(p.n_q, gqa);
    const int n_t  = (p.n_kv + TILE_N - 1) / TILE_N;
    p.groups_per_cta  = p.groups_per_cta > 0 ? p.groups_per_cta : groups_per_cta(ngrp);   // > 0: caller's choice
    if (p.groups_per_cta != 1 && p.groups_per_cta != 2 && p.groups_per_cta != 4 && p.groups_per_cta != MAX_GROUPS) {
        return cudaErrorInvalidValue;
    }
    p.tiles_per_split = (n_t + p.n_splits - 1) / p.n_splits;
    p.n_splits        = (n_t + p.tiles_per_split - 1) / p.tiles_per_split;

    const size_t parts = (size_t) p.n_seq * p.n_head_kv * ngrp * p.n_splits * HEADS;
    p.part_o  = static_cast<float *>(workspace);
    p.part_al = reinterpret_cast<float2 *>(p.part_o + parts * D);

    cudaError_t err = encode_kv_map(&p.tm_k, p.K, p.nbk1, p.nbk2, p.nbk3, p.n_kv, p.n_head_kv, p.n_seq, TILE_N,
                                    &p.tm_tok_dim, &p.tm_head_dim);
    int tok_dim_v = 0, head_dim_v = 0;
    if (err == cudaSuccess) {
        err = encode_kv_map(&p.tm_v, p.V, p.nbv1, p.nbv2, p.nbv3, p.n_kv, p.n_head_kv, p.n_seq, TILE_N, &tok_dim_v,
                            &head_dim_v);
    }
    if (err == cudaSuccess && (tok_dim_v != p.tm_tok_dim || head_dim_v != p.tm_head_dim)) {
        err = cudaErrorInvalidValue;
    }
    if (err != cudaSuccess) {
        return err;
    }

    void (*kern)(Params) = p.groups_per_cta == 1 ? k_decode<1> : p.groups_per_cta == 2 ? k_decode<2> :
                           p.groups_per_cta == 4 ? k_decode<4> : k_decode<MAX_GROUPS>;
    err = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));   // per device
    if (err != cudaSuccess) {
        return err;
    }
    const int ncg = (ngrp + p.groups_per_cta - 1) / p.groups_per_cta;
    kern<<<dim3(p.n_splits, p.n_head_kv * ncg, p.n_seq), NTHREADS, sizeof(Smem), stream>>>(p);
    k_decode_combine<<<dim3(p.n_head_kv * ngrp, p.n_seq), D, 0, stream>>>(p);
    return cudaGetLastError();
}
#endif // __CUDACC__

} // namespace decode
} // namespace fa_nvfp4
