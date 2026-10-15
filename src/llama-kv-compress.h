#pragma once

// Lossless packing of NVFP4 KV-cache rows for host copies of the state (prompt cache, checkpoints, slot saves).
// The UE4M3 scale bytes (4 of every 36) are split out and Huffman coded; the E2M1 code bytes are stored as they are.
// On real Qwen3.8 KV data the scales code to about half their size, so a saved NVFP4 cache is ~5 % smaller.
// DMAIC_AI docs/research/kvquant-results.md ("Compression headroom").

#include <cstddef>
#include <cstdint>
#include <vector>

// Appends the packed form of n_blocks block_nvfp4 (36 bytes each) to out.
void llama_nvfp4_pack(const uint8_t * blocks, size_t n_blocks, std::vector<uint8_t> & out);

// Size of the packed form, without building it.
size_t llama_nvfp4_packed_size(const uint8_t * blocks, size_t n_blocks);

// Unpacks into n_blocks block_nvfp4; false if the data is malformed or not n_blocks long.
bool llama_nvfp4_unpack(const uint8_t * data, size_t size, uint8_t * blocks, size_t n_blocks);
