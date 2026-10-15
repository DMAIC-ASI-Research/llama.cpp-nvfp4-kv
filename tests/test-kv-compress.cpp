// Round trips of the NVFP4 state packer (src/llama-kv-compress.cpp).
#include "../src/llama-kv-compress.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int check(const char * name, const std::vector<uint8_t> & blocks) {
    const size_t n = blocks.size() / 36;
    std::vector<uint8_t> packed;
    llama_nvfp4_pack(blocks.data(), n, packed);
    std::vector<uint8_t> back(blocks.size(), 0xAA);
    if (!llama_nvfp4_unpack(packed.data(), packed.size(), back.data(), n) || back != blocks) {
        fprintf(stderr, "%s: round trip FAILED\n", name);
        return 1;
    }
    // truncated or wrong-length input must be rejected, not crash
    if (n > 0 && (llama_nvfp4_unpack(packed.data(), packed.size() - 1, back.data(), n) ||
                  llama_nvfp4_unpack(packed.data(), packed.size(), back.data(), n - 1))) {
        fprintf(stderr, "%s: malformed input accepted\n", name);
        return 1;
    }
    printf("%-14s %8zu blocks: %8zu -> %8zu bytes (%.2f %%)\n", name, n, blocks.size(), packed.size(),
           blocks.empty() ? 0.0 : 100.0 * (1.0 - (double) packed.size() / blocks.size()));
    return 0;
}

int main() {
    std::mt19937 rng(42);
    int fails = 0;

    fails += check("empty", {});

    std::vector<uint8_t> b(36 * 1000);
    for (auto & x : b) x = rng() & 0xFF;
    fails += check("uniform", b);           // incompressible scales: stored raw

    for (size_t i = 0; i < b.size(); ++i) {
        if (i % 36 < 4) b[i] = 0x30;
    }
    fails += check("one scale", b);         // a single symbol

    std::normal_distribution<double> g(0x34, 3.0);
    for (size_t i = 0; i < b.size(); ++i) {
        if (i % 36 < 4) b[i] = (uint8_t) std::min(0x7E, std::max(0, (int) g(rng)));
    }
    fails += check("realistic", b);

    std::vector<uint8_t> big(36 * 200000);
    for (size_t i = 0; i < big.size(); ++i) {
        // a long tail of rare scales forces the 12-bit length limit
        big[i] = i % 36 < 4 ? (uint8_t) ((rng() % 1000 == 0) ? rng() % 256 : (int) g(rng) & 0x7F) : (uint8_t) rng();
    }
    fails += check("long tail", big);

    // corrupt Huffman data must fail cleanly or decode to something, never overrun
    std::vector<uint8_t> packed;
    llama_nvfp4_pack(b.data(), b.size() / 36, packed);
    std::vector<uint8_t> back(b.size());
    for (int t = 0; t < 2000; ++t) {
        std::vector<uint8_t> bad = packed;
        bad[rng() % bad.size()] ^= (uint8_t) (1 + rng() % 255);
        llama_nvfp4_unpack(bad.data(), bad.size(), back.data(), back.size() / 36);
    }

    printf(fails ? "FAILED\n" : "OK\n");
    return fails ? 1 : 0;
}
