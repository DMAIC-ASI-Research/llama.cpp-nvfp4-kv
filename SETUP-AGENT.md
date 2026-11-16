# Setup guide for AI agents: Qwen3.8-27B-DMAIC_ASI-S16 on llama.cpp with an NVFP4 KV cache

You are an AI agent asked to set up and run this model on the user's machine. Follow the steps in order. Each step
has a command, what success looks like, and what to do if it fails. Do not skip the checks. Stop and tell the user
when a check fails and the fix needs them (installing a driver, sudo, buying hardware).

**You need two things, from two places:**

| What | Where |
|---|---|
| The model (one 13.2 GB GGUF file) | Hugging Face: https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF |
| This llama.cpp build (NVFP4 KV cache) | GitHub: https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv |

Rules for you:
- Run nothing with `sudo` yourself. If a step needs root, give the user the exact command and wait.
- Do not change the user's driver, CUDA install or system settings.
- Work in one directory. Default: `~/llm` (ask the user if they want another place). Below, `$WORK` means it.
- Report at the end: what you installed, where, the launch command, and the check results.

## Step 0: checks before anything else

Run each; compare with the expected result.

| Check | Command | Expected |
|---|---|---|
| GPU | `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader` | an RTX 50-series card, compute_cap `12.0`, memory ≥ 16 GB, driver ≥ 570 |
| CUDA toolkit | `command -v nvcc && nvcc --version` | release 12.8 or newer, from that path (see Step 1 if several nvcc exist) |
| CMake | `cmake --version` | 3.21 or newer |
| Compiler | `g++ --version` | any version with C++17 |
| git | `git --version` | present |
| Disk | `df -h $HOME` | at least 20 GB free |
| Python | `python3 --version` | 3.9 or newer (for the download tool) |

If the GPU is not compute capability 12.0: stop. This build's NVFP4 attention kernels run only on consumer Blackwell
(`sm_120a`). Tell the user to use upstream llama.cpp with `-ctk q8_0 -ctv q8_0` instead.

If memory is below 16 GB: stop; the model file alone is 13.2 GB.

If `nvcc` is missing: the user must install the CUDA toolkit 12.8+ (NVIDIA's installer or their distro's
`cuda-toolkit` package). Give them that, and wait.

## Step 1: get and build the code

```bash
mkdir -p $WORK && cd $WORK
git clone https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv
cd llama.cpp-nvfp4-kv
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(command -v nvcc)
cmake --build build --config Release -j 4 --target llama-server llama-cli
```

- Success: `ls build/bin/llama-server` prints the path. The build takes tens of minutes; let it finish (run it in
  the background and check on it if your tool has a time limit).
- The configure line names the compiler (`-DCMAKE_CUDA_COMPILER=$(command -v nvcc)`) because an old distro `nvcc`
  (for example `/usr/bin/nvcc` from Ubuntu's `nvidia-cuda-toolkit`) can be found first and fails with errors like
  `'_Float32' does not name a type`. `command -v nvcc` must print the 12.8+ toolkit's nvcc (usually
  `/usr/local/cuda/bin/nvcc`); if it does not, put that directory first in PATH. Delete `build/` before configuring again.
- `unsupported gpu architecture 'compute_120a'`: the CUDA toolkit is older than 12.8. Back to Step 0.
- Job count: each CUDA compile job can take 4-5 GB of RAM. Before building, read total RAM (`free -g`) and use
  `-j N` with N = RAM in GB divided by 5, at least 1, at most `nproc` (`-j 4` fits 24-32 GB; `-j 2` for 16 GB).
  Never run `-j` without a number: it starts unlimited jobs and can run the machine out of memory, which kills
  the user's desktop session. If compiling is killed for memory, rerun with a smaller N; the build resumes.

Check the binary knows NVFP4:

```bash
build/bin/llama-server --help | grep -c nvfp4
```

Expected: a number above 0.

## Step 2: download the model

```bash
cd $WORK
python3 -m pip install -U huggingface_hub
hf download DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF Qwen3.8-27B-DMAIC_ASI-S16.gguf --local-dir models
```

- Success: `ls -l models/Qwen3.8-27B-DMAIC_ASI-S16.gguf` shows 13,207,095,840 bytes.
- `hf: command not found`: use `python3 -m huggingface_hub.cli download ...` with the same arguments, or add
  `~/.local/bin` to PATH.

## Step 3: start the server

Check nothing else is using the GPU memory first: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.
A desktop on the same GPU typically uses 0.3-1.5 GB; that is fine, but it lowers the context that fits.

```bash
cd $WORK
llama.cpp-nvfp4-kv/build/bin/llama-server -m models/Qwen3.8-27B-DMAIC_ASI-S16.gguf \
  -ngl 99 -c 120064 --fit off -fa on -ctk nvfp4 -ctv nvfp4 -b 2048 -ub 256 -np 1 -kvu \
  --spec-type draft-mtp,ngram-simple -ctkd nvfp4 -ctvd nvfp4 --spec-draft-n-ubatch 64 \
  --host 127.0.0.1 --port 8080 > server.log 2>&1 &
```

- Success: within about a minute `server.log` contains `server is listening` and
  `curl -s http://127.0.0.1:8080/health` returns `{"status":"ok"}`.
- `out of memory` / `failed to allocate` in the log: stop the server, lower `-c` by 8192 (112000, 104000, ...) and
  start again. Tell the user the context you ended at.
- `nvfp4` reported as an unknown cache type: the binary is not this fork's. Recheck Step 1.
- To run without speculative decoding (simpler, a little slower), drop
  `--spec-type draft-mtp,ngram-simple -ctkd nvfp4 -ctvd nvfp4 --spec-draft-n-ubatch 64`.

## Step 4: final check

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":512}'
```

Expected: a JSON reply whose message content contains `ready` (the model may think first; the thinking is in a
separate field). Report the launch command you used to the user, and that the API is at
`http://127.0.0.1:8080/v1` (OpenAI-compatible; any model name works).

## Reference

- Model card: https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF
- Human guide: [NVFP4-HOWTO.md](NVFP4-HOWTO.md)
- Sampling the base model recommends: thinking mode temperature 1.0, top_p 0.95, top_k 20; non-thinking temperature
  0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
