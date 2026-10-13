#include "fattn-nvfp4.cuh"
#include "fattn-nvfp4/fa_nvfp4_decode.cuh"
#include "fattn-nvfp4/fa_nvfp4_prefill.cuh"

#include <cstdlib>

// Up to this many (query token, head) rows per KV head the split-KV decode kernel is faster than prefill.
static constexpr int NVFP4_DECODE_MAX_ROWS = 256;

static bool nvfp4_aligned16(const ggml_tensor * t) {
    return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

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
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || sinks || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) { // sparse n_kv_max hint
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
        K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (!nvfp4_aligned16(Q) || !nvfp4_aligned16(K) || !nvfp4_aligned16(V) || Q->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[1] % 4 != 0 || mask->ne[0] < K->ne[1] ||
                 mask->ne[1] < Q->ne[1] || (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3]))) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_q = Q->ne[1], n_head = Q->ne[2], n_seq = Q->ne[3];
    const int n_kv = K->ne[1], n_head_kv = K->ne[2];
    const half * mask_d  = mask ? (const half *) mask->data : nullptr;
    const size_t mask_s1 = mask ? mask->nb[1] / sizeof(half) : 0;
    const size_t mask_s3 = mask && mask->ne[3] > 1 ? mask->nb[3] / sizeof(half) : 0;
    cudaStream_t stream  = ctx.stream();

    if (n_q * (n_head / n_head_kv) <= NVFP4_DECODE_MAX_ROWS) {
        fa_nvfp4::decode::Params p{};
        p.Q = (const float *) Q->data;   p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
        p.K = (const uint8_t *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
        p.V = (const uint8_t *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
        p.mask = mask_d; p.mask_s1 = mask_s1; p.mask_s3 = mask_s3;
        p.dst = (float *) dst->data;
        p.n_q = n_q; p.n_kv = n_kv; p.n_head = n_head; p.n_head_kv = n_head_kv; p.n_seq = n_seq;
        p.scale = scale;
        p.n_splits = fa_nvfp4::decode::choose_splits(n_kv, n_q, n_head, n_head_kv, n_seq, ggml_cuda_info().devices[ctx.device].nsm);
        ggml_cuda_pool_alloc<char> ws(ctx.pool(), fa_nvfp4::decode::workspace_bytes(n_q, n_head, n_head_kv, n_seq, p.n_splits));
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
        ggml_cuda_pool_alloc<char> ws(ctx.pool(), fa_nvfp4::prefill::workspace_bytes(n_q, n_kv, n_head, n_head_kv, n_seq));
        CUDA_CHECK(fa_nvfp4::prefill::launch(p, ws.get(), stream));
    }
}
