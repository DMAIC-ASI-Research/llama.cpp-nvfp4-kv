#include "fattn-nvfp4.cuh"
#include "fattn-nvfp4/fa_nvfp4_decode.cuh"
#include "fattn-nvfp4/fa_nvfp4_prefill.cuh"

#include <algorithm>
#include <cstdlib>

// Up to this many (query token, head) rows per KV head the split-KV decode kernel is faster than prefill
// (crossover_bench: decode wins at 384 rows, prefill at 516, for 4k-106k caches).
static constexpr int NVFP4_DECODE_MAX_ROWS = 448;

static bool nvfp4_strides16(const ggml_tensor * t) {
    return t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

// The prefill split count from DMAIC_PREFILL_K: unset or "1" is k = 1 (no split); "0" asks choose_k(); any other
// positive integer is used as k. Pure, so the workspace sizing and the launch agree.
static int nvfp4_prefill_k(int n_q, int n_kv, int n_head, int n_head_kv, int n_seq, int nsm, int head_dim) {
    static const char * k_env = getenv("DMAIC_PREFILL_K");
    if (!k_env) {
        return 1;
    }
    const int k_val = atoi(k_env);
    if (k_val > 0) {
        return k_val;
    }
    if (k_env[0] == '0' && k_env[1] == '\0') {
        return fa_nvfp4::prefill::choose_k(n_q, n_kv, n_head, n_head_kv, n_seq, nsm, 0, head_dim);
    }
    return 1;
}

// One fixed workspace per device for both attention paths (DMAIC, owner 2026-11-11): the workspace used to come
// from the CUDA pool, which keeps its largest size for good, so the engine's VRAM grew in steps through the day
// (+92 MiB served, while the engine was sized at load). Now the first call allocates, once, the most any call can
// need for this cache: every n_q up to DMAIC_PREFILL_NQ_MAX (the server's -ub; 512, llama.cpp's default, when
// unset) and every n_kv up to the cache's full size. The pointer never changes (CUDA graphs keep it). A call that
// needs more (a bigger cache or n_q) takes the pool as before and is logged once.
static size_t nvfp4_ws_most(int kv_size, int n_head, int n_head_kv, int n_seq, int head_dim, int nsm, int nq_cap) {
    const int hpr = fa_nvfp4::heads_per_row(head_dim);
    size_t most = 0;
    for (int n_q = 1; n_q <= nq_cap; ++n_q) {
        if (n_q * (n_head / (n_head_kv / hpr)) <= NVFP4_DECODE_MAX_ROWS) {
            for (int n_kv = fa_nvfp4::prefill::BLOCK_N; n_kv <= kv_size + fa_nvfp4::prefill::BLOCK_N - 1;
                 n_kv += fa_nvfp4::prefill::BLOCK_N) {
                const int s = fa_nvfp4::decode::choose_splits(n_kv, n_q, n_head, n_head_kv, n_seq, nsm, 0, head_dim);
                most = std::max(most, fa_nvfp4::decode::workspace_bytes(n_q, n_head, n_head_kv, n_seq, s, head_dim));
            }
            continue;
        }
        for (int n_kv = fa_nvfp4::prefill::BLOCK_N; n_kv <= kv_size + fa_nvfp4::prefill::BLOCK_N - 1;
             n_kv += fa_nvfp4::prefill::BLOCK_N) {
            const int k = nvfp4_prefill_k(n_q, n_kv, n_head, n_head_kv, n_seq, nsm, head_dim);
            most = std::max(most, fa_nvfp4::prefill::workspace_bytes(n_q, n_kv, n_head, n_head_kv, n_seq, head_dim, k));
        }
    }
    return most;
}

static char * nvfp4_fixed_ws(ggml_backend_cuda_context & ctx, const ggml_tensor * K, int n_head, int n_seq,
                             size_t need) {
    struct slot { char * p = nullptr; size_t bytes = 0; bool tried = false; bool over_logged = false; };
    static slot slots[GGML_CUDA_MAX_DEVICES];
    slot & s = slots[ctx.device];
    if (!s.tried) {
        s.tried = true;
        const ggml_tensor * cache = K->view_src ? K->view_src : K;
        const int kv_size = (int) cache->ne[1];
        const char * cap_env = getenv("DMAIC_PREFILL_NQ_MAX");
        const int nq_cap = cap_env && atoi(cap_env) > 0 ? atoi(cap_env) : 512;
        const size_t most = nvfp4_ws_most(kv_size, n_head, (int) K->ne[2], n_seq, (int) K->ne[0],
                                          ggml_cuda_info().devices[ctx.device].nsm, nq_cap);
        ggml_cuda_set_device(ctx.device);
        if (most > 0 && cudaMalloc((void **) &s.p, most) == cudaSuccess) {
            s.bytes = most;
            GGML_LOG_INFO("DMAIC nvfp4 FA workspace: %.1f MiB fixed (cache %d, n_q <= %d)\n",
                          most / 1048576.0, kv_size, nq_cap);
        } else {
            (void) cudaGetLastError();
            s.p = nullptr;
            GGML_LOG_WARN("DMAIC nvfp4 FA workspace: could not allocate %.1f MiB fixed; using the pool\n",
                          most / 1048576.0);
        }
    }
    if (s.p && need <= s.bytes) {
        return s.p;
    }
    if (s.p && !s.over_logged) {
        s.over_logged = true;
        GGML_LOG_WARN("DMAIC nvfp4 FA workspace: a call needs %.1f MiB, over the fixed %.1f MiB; it uses the pool\n",
                      need / 1048576.0, s.bytes / 1048576.0);
    }
    return nullptr;
}

// Shapes, types and strides only (no data pointers), so the answer is the same when the graph allocator sizes the
// output (ggml_cuda_flash_attn_ext_get_alloc_size) and when the op runs.
bool ggml_cuda_flash_attn_ext_nvfp4_usable(const int device, const ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_NVFP4_FA") != nullptr;
    const int cc = ggml_cuda_info().devices[device].cc;
    if (disabled || cc != GGML_CUDA_CC_BLACKWELL || !blackwell_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (K->type != GGML_TYPE_NVFP4 || V->type != GGML_TYPE_NVFP4 || Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // head sizes 256, 128 and 64 (below 256, adjacent KV heads are read as one 256-wide virtual row)
    const int head_dim = (int) K->ne[0];
    if (fa_nvfp4::heads_per_row(head_dim) == 0 || Q->ne[0] != head_dim || V->ne[0] != head_dim || sinks ||
        max_bias != 0.0f) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) { // sparse n_kv_max hint
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
        K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (!nvfp4_strides16(Q) || Q->nb[0] != sizeof(float) || !ggml_is_contiguous(dst) ||
        !fa_nvfp4::kv_view_supported(head_dim, nullptr, K->nb[1], K->nb[2], K->nb[3], (int) K->ne[2]) ||
        !fa_nvfp4::kv_view_supported(head_dim, nullptr, V->nb[1], V->nb[2], V->nb[3], (int) V->ne[2])) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[1] % 4 != 0 || mask->ne[0] < K->ne[1] ||
                 mask->ne[1] < Q->ne[1] || (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3]))) {
        return false;
    }
    return true;
}

bool ggml_cuda_flash_attn_ext_nvfp4_aligned(const ggml_tensor * dst) {   // 16-byte data pointers (TMA, float4 Q loads)
    return (uintptr_t) dst->src[0]->data % 16 == 0 && (uintptr_t) dst->src[1]->data % 16 == 0 &&
           (uintptr_t) dst->src[2]->data % 16 == 0;
}

void ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int n_q = Q->ne[1], n_head = Q->ne[2], n_seq = Q->ne[3];
    const int n_kv = K->ne[1], n_head_kv = K->ne[2], head_dim = K->ne[0];
    const int hpr  = fa_nvfp4::heads_per_row(head_dim);   // KV heads per 256-wide virtual row
    const half * mask_d  = mask ? (const half *) mask->data : nullptr;
    const size_t mask_s1 = mask ? mask->nb[1] / sizeof(half) : 0;
    const size_t mask_s3 = mask && mask->ne[3] > 1 ? mask->nb[3] / sizeof(half) : 0;
    cudaStream_t stream  = ctx.stream();

    if (n_q * (n_head / (n_head_kv / hpr)) <= NVFP4_DECODE_MAX_ROWS) {   // rows per virtual KV head
        fa_nvfp4::decode::Params p{};
        p.Q = (const float *) Q->data;   p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
        p.K = (const uint8_t *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
        p.V = (const uint8_t *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
        p.mask = mask_d; p.mask_s1 = mask_s1; p.mask_s3 = mask_s3;
        p.dst = (float *) dst->data;
        p.n_q = n_q; p.n_kv = n_kv; p.n_head = n_head; p.n_head_kv = n_head_kv; p.n_seq = n_seq;
        p.scale = scale;
        p.softcap = logit_softcap;
        p.head_dim = head_dim;
        p.n_splits = fa_nvfp4::decode::choose_splits(n_kv, n_q, n_head, n_head_kv, n_seq, ggml_cuda_info().devices[ctx.device].nsm,
                                                     0, head_dim);
        const size_t need = fa_nvfp4::decode::workspace_bytes(n_q, n_head, n_head_kv, n_seq, p.n_splits, head_dim);
        ggml_cuda_pool_alloc<char> pool_ws(ctx.pool());
        char * ws = nvfp4_fixed_ws(ctx, K, n_head, n_seq, need);
        if (!ws) {
            ws = pool_ws.alloc(need);
        }
        CUDA_CHECK(fa_nvfp4::decode::launch(p, ws, stream));
    } else {
        fa_nvfp4::prefill::Params p{};
        p.Q = (const float *) Q->data;   p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
        p.K = (const uint8_t *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
        p.V = (const uint8_t *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
        p.mask = mask_d; p.mask_s1 = mask_s1; p.mask_s3 = mask_s3;
        p.dst = (float *) dst->data;
        p.n_q = n_q; p.n_kv = n_kv; p.n_head = n_head; p.n_head_kv = n_head_kv; p.n_seq = n_seq;
        p.scale = scale;
        p.softcap = logit_softcap;
        p.head_dim = head_dim;
        const int k_prefill = nvfp4_prefill_k(n_q, n_kv, n_head, n_head_kv, n_seq,
                                              ggml_cuda_info().devices[ctx.device].nsm, head_dim);
        p.k     = k_prefill;
        p.n_sms = ggml_cuda_info().devices[ctx.device].nsm;
        const size_t need = fa_nvfp4::prefill::workspace_bytes(n_q, n_kv, n_head, n_head_kv, n_seq, head_dim, k_prefill);
        ggml_cuda_pool_alloc<char> pool_ws(ctx.pool());
        char * ws = nvfp4_fixed_ws(ctx, K, n_head, n_seq, need);
        if (!ws) {
            ws = pool_ws.alloc(need);
        }
        if (k_prefill > 1) {
            // Split partials sit after the two prepass buffers; same carve as tests/prefill_test.cu:76-80.
            const size_t nblk   = (size_t) fa_nvfp4::prefill::n_row_blocks(n_q, n_head / (n_head_kv / hpr)) * n_seq;
            const size_t states = (nblk * fa_nvfp4::prefill::n_kv_tiles(n_kv) + 15) / 16 * 16;
            p.part_ws = ws + states + (size_t) n_seq * (n_head_kv / hpr) * fa_nvfp4::DBLKS * sizeof(int);
        }
        CUDA_CHECK(fa_nvfp4::prefill::launch(p, ws, stream));
    }
}
