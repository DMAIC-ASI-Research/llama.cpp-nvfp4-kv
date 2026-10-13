#include "common.cuh"

// Native NVFP4 FlashAttention (D = 256, sm_120a): block-scaled FP4 K*Q and E4M3 x E2M1 P*V on the raw cache bytes.
bool ggml_cuda_flash_attn_ext_nvfp4_usable(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
