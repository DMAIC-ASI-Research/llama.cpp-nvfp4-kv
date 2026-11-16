# NVFP4 KV cache for llama.cpp: how to run it

This fork adds an **NVFP4 KV cache** to llama.cpp, with native FlashAttention kernels for consumer Blackwell GPUs
(RTX 50-series, `sm_120a`). The KV cache takes about a quarter of the memory of f16, so a 27B model fits a long
context on a 16 GB card.

It was built and tested with **Qwen3.8-27B-DMAIC_ASI-S16**
([Hugging Face](https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF)) on an RTX 5070 Ti 16 GB. On that
card it serves about **120k tokens of context** with multi-token-prediction speculative decoding.

**You need two things, from two places:**

| What | Where |
|---|---|
| The model (one 13.2 GB GGUF file) | Hugging Face: https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF |
| This llama.cpp build (NVFP4 KV cache) | GitHub: https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv |

If you want an AI assistant to set this up for you, point it at [SETUP-AGENT.md](SETUP-AGENT.md).

## What you need

| | |
|---|---|
| GPU | NVIDIA RTX 50-series (Blackwell, compute capability 12.0). Tested: RTX 5070 Ti 16 GB |
| Driver | 570 or newer |
| CUDA toolkit | 12.8 or newer (tested 12.9) |
| Build tools | git, CMake 3.21+, a C++17 compiler |
| OS | Linux (tested Ubuntu 24.04) |
| Disk | about 14 GB for the model, about 2 GB for the build |

Other GPUs: the NVFP4 FlashAttention path is compiled for `sm_120a` only. On other cards this fork has not been
tested; use upstream llama.cpp and an ordinary KV cache type instead.

## 1. Build

```bash
git clone https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv
cd llama.cpp-nvfp4-kv
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(command -v nvcc)
cmake --build build --config Release -j 4 --target llama-server llama-cli
```

`-DCMAKE_CUDA_COMPILER` makes sure CMake uses your CUDA 12.8+ `nvcc`, not an older one from the distro (check with
`command -v nvcc && nvcc --version`).

`-j 4` runs four compile jobs at once. Each CUDA job can take 4-5 GB of RAM, so keep jobs at or below your RAM in GB
divided by 5 (`-j 2` on 16 GB). Never use `-j` without a number: it starts unlimited jobs and can run the machine
out of memory.

The CUDA build takes a while (tens of minutes on a 10-thread CPU). When it ends, `build/bin/llama-server` exists.

## 2. Download the model

```bash
pip install -U huggingface_hub
hf download DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF Qwen3.8-27B-DMAIC_ASI-S16.gguf --local-dir models
```

## 3. Run

```bash
build/bin/llama-server -m models/Qwen3.8-27B-DMAIC_ASI-S16.gguf \
  -ngl 99 -c 120064 --fit off -fa on -ctk nvfp4 -ctv nvfp4 -b 2048 -ub 256 -np 1 -kvu \
  --spec-type draft-mtp,ngram-simple -ctkd nvfp4 -ctvd nvfp4 --spec-draft-n-ubatch 64 \
  --host 127.0.0.1 --port 8080
```

Then open http://127.0.0.1:8080, or use the OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

What the flags do:
- `-ctk nvfp4 -ctv nvfp4`: the NVFP4 KV cache (the point of this fork). `-fa on` is required for it.
- `-c 120064`: the context that fits a 16 GB card with nothing else on it. If loading fails with an out-of-memory
  error (a desktop on the same GPU takes some memory), lower it in steps of 8192.
- `--spec-type draft-mtp,ngram-simple`: speculative decoding with the model's own multi-token-prediction head, plus
  n-gram lookup. `-ctkd/-ctvd nvfp4` keep the draft's cache in NVFP4 too. Remove these four flags to run without it.
- `--fit off`: use the settings as given (llama.cpp's auto-fit would otherwise warn and give up, since `-ngl` is set).
- `-ub 256`, `--spec-draft-n-ubatch 64`: the batch sizes that fit with the full context.

## Quality

The NVFP4 KV writer searches the per-16 scale (it keeps the best of the UE4M3 codes around amax/6, amax/5 and
amax/4), so it is never worse than plain rounding; on real KV data the attention-output error drops about 15 %.

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). The NVFP4 KV cache and its kernels are by
DMAIC ASI Research.
