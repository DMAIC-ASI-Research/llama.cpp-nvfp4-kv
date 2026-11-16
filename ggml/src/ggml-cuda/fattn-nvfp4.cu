#include "fattn-nvfp4.cuh"
#include "fattn-nvfp4/fa_nvfp4_decode.cuh"
#include "fattn-nvfp4/fa_nvfp4_prefill.cuh"

#include <cstdlib>

// Up to this many (query token, head) rows per KV head the split-KV decode kernel is faster than prefill
// (crossover_bench: decode wins at 384 rows, prefill at 516, for 4k-106k caches).
static constexpr int NVFP4_DECODE_MAX_ROWS = 448;

static bool nvfp4_strides16(const ggml_tensor * t) {
    return t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
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
        ggml_cuda_pool_alloc<char> ws(ctx.pool(), fa_nvfp4::decode::workspace_bytes(n_q, n_head, n_head_kv, n_seq, p.n_splits, head_dim));
        CUDA_CHECK(fa_nvfp4::decode::launch(p, ws.get(), stream));
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
        ggml_cuda_pool_alloc<char> ws(ctx.pool(), fa_nvfp4::prefill::workspace_bytes(n_q, n_kv, n_head, n_head_kv, n_seq, head_dim));
        CUDA_CHECK(fa_nvfp4::prefill::launch(p, ws.get(), stream));
    }
}
