#include "llama-kv-compress.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>
#include <thread>

// Packed layout (little endian):
//   u32 magic "NVZ2" | u64 n_blocks | u32 n_chunks | u64 chunk size x n_chunks | chunks
// The blocks are split evenly into chunks, coded independently (so in parallel). A chunk:
//   u8 mode
//   mode 0: scales raw (4 bytes per block)
//   mode 1: 128 bytes of 4-bit code lengths (symbol 2i low nibble, 2i+1 high nibble; 0 = absent)
//           | u64 bitstream bytes | bitstream (canonical Huffman, MSB first)
//   then the code bytes (32 per block), in block order

namespace {

constexpr uint32_t MAGIC    = 0x325A564E; // "NVZ2"
constexpr size_t   MIN_CHUNK = 16384;     // blocks
constexpr int      MAX_CHUNKS = 8;
constexpr int      BLOCK    = 36;
constexpr int      N_SCALE  = 4;
constexpr int      N_CODE   = 32;
constexpr int      MAX_LEN  = 12;

template <typename T> bool get(const uint8_t *& p, const uint8_t * end, T & v) {
    if ((size_t) (end - p) < sizeof(T)) {
        return false;
    }
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return true;
}

// Huffman code lengths for 256 symbols, at most MAX_LEN bits; frequencies are flattened until the limit holds.
std::array<uint8_t, 256> code_lengths(const std::array<uint64_t, 256> & hist) {
    std::array<uint8_t, 256> len{};
    std::array<uint64_t, 256> f = hist;
    uint64_t total = 0;
    int n_sym = 0;
    for (int s = 0; s < 256; ++s) {
        total += f[s];
        n_sym += f[s] > 0;
    }
    if (n_sym == 0) {
        return len;
    }
    if (n_sym == 1) {
        for (int s = 0; s < 256; ++s) {
            len[s] = f[s] ? 1 : 0;
        }
        return len;
    }
    for (int shift = MAX_LEN - 1;; --shift) {
        // nodes: 0..255 leaves, then internal nodes; parent links give each leaf's depth
        struct node { uint64_t w; int id; };
        auto cmp = [](const node & a, const node & b) { return a.w != b.w ? a.w > b.w : a.id > b.id; };
        std::priority_queue<node, std::vector<node>, decltype(cmp)> pq(cmp);
        std::vector<int> parent(512, -1);
        for (int s = 0; s < 256; ++s) {
            if (f[s]) {
                pq.push({f[s], s});
            }
        }
        int next = 256;
        while (pq.size() > 1) {
            const node a = pq.top(); pq.pop();
            const node b = pq.top(); pq.pop();
            parent[a.id] = parent[b.id] = next;
            pq.push({a.w + b.w, next++});
        }
        int max_len = 0;
        for (int s = 0; s < 256; ++s) {
            int d = 0;
            if (f[s]) {
                for (int p = parent[s]; p != -1; p = parent[p]) {
                    ++d;
                }
            }
            len[s] = (uint8_t) d;
            max_len = std::max(max_len, d);
        }
        if (max_len <= MAX_LEN) {
            return len;
        }
        // too deep: raise the rare symbols' weights and try again
        const uint64_t floor_w = std::max<uint64_t>(1, total >> std::max(shift, 1));
        for (int s = 0; s < 256; ++s) {
            if (f[s]) {
                f[s] = std::max(f[s], floor_w);
            }
        }
    }
}

// Canonical codes from lengths: symbols ordered by (length, symbol).
std::array<uint16_t, 256> canonical(const std::array<uint8_t, 256> & len) {
    std::array<uint16_t, 256> code{};
    uint16_t c = 0;
    for (int l = 1; l <= MAX_LEN; ++l) {
        for (int s = 0; s < 256; ++s) {
            if (len[s] == l) {
                code[s] = c++;
            }
        }
        c <<= 1;
    }
    return code;
}

} // namespace

namespace {

struct scale_code {
    std::array<uint64_t, 256> hist{};
    std::array<uint8_t, 256>  len{};
    uint64_t bits    = 0;
    bool     huffman = false;
};

scale_code plan(const uint8_t * blocks, size_t n_blocks) {
    scale_code sc;
    for (size_t b = 0; b < n_blocks; ++b) {
        for (int i = 0; i < N_SCALE; ++i) {
            sc.hist[blocks[b*BLOCK + i]]++;
        }
    }
    sc.len = code_lengths(sc.hist);
    for (int s = 0; s < 256; ++s) {
        sc.bits += sc.hist[s] * sc.len[s];
    }
    sc.huffman = n_blocks > 0 && 128 + 8 + (sc.bits + 7) / 8 < (uint64_t) N_SCALE * n_blocks;
    return sc;
}

size_t chunk_size(const scale_code & sc, size_t n_blocks) {
    const size_t scales = sc.huffman ? 128 + 8 + (sc.bits + 7) / 8 : (size_t) N_SCALE * n_blocks;
    return 1 + scales + (size_t) N_CODE * n_blocks;
}

void pack_chunk(const uint8_t * blocks, size_t n_blocks, const scale_code & sc, uint8_t * dst) {
    *dst++ = sc.huffman ? 1 : 0;
    if (!sc.huffman) {
        for (size_t b = 0; b < n_blocks; ++b) {
            memcpy(dst, blocks + b*BLOCK, N_SCALE);
            dst += N_SCALE;
        }
    } else {
        for (int s = 0; s < 256; s += 2) {
            *dst++ = (uint8_t) (sc.len[s] | (sc.len[s + 1] << 4));
        }
        const uint64_t n_bytes = (sc.bits + 7) / 8;
        memcpy(dst, &n_bytes, 8);
        dst += 8;
        const auto code = canonical(sc.len);
        uint64_t acc = 0;
        int n_acc = 0;
        for (size_t b = 0; b < n_blocks; ++b) {
            for (int i = 0; i < N_SCALE; ++i) {
                const uint8_t s = blocks[b*BLOCK + i];
                acc = (acc << sc.len[s]) | code[s];
                n_acc += sc.len[s];
            }
            while (n_acc >= 8) {   // at most 7 + 4 x 12 bits are pending, so acc never overflows
                n_acc -= 8;
                *dst++ = (uint8_t) (acc >> n_acc);
            }
        }
        if (n_acc > 0) {
            *dst++ = (uint8_t) (acc << (8 - n_acc));
        }
    }
    for (size_t b = 0; b < n_blocks; ++b) {
        memcpy(dst, blocks + b*BLOCK + N_SCALE, N_CODE);
        dst += N_CODE;
    }
}

bool unpack_chunk(const uint8_t * p, const uint8_t * end, uint8_t * blocks, size_t n_blocks) {
    uint8_t mode;
    if (!get(p, end, mode)) {
        return false;
    }
    if (mode == 0) {
        if ((size_t) (end - p) < (size_t) N_SCALE * n_blocks) {
            return false;
        }
        for (size_t b = 0; b < n_blocks; ++b) {
            memcpy(blocks + b*BLOCK, p, N_SCALE);
            p += N_SCALE;
        }
    } else if (mode == 1) {
        if (end - p < 128) {
            return false;
        }
        std::array<uint8_t, 256> len{};
        for (int s = 0; s < 256; s += 2) {
            len[s]     = p[s/2] & 0xF;
            len[s + 1] = p[s/2] >> 4;
        }
        p += 128;
        uint64_t n_bytes;
        if (!get(p, end, n_bytes) || (uint64_t) (end - p) < n_bytes) {
            return false;
        }
        // the lengths must form a prefix code (Kraft sum <= 1)
        uint64_t kraft = 0;
        for (int s = 0; s < 256; ++s) {
            if (len[s] > MAX_LEN) {
                return false;
            }
            if (len[s]) {
                kraft += 1ull << (MAX_LEN - len[s]);
            }
        }
        if (kraft == 0 || kraft > (1ull << MAX_LEN)) {
            return false;
        }
        const auto code = canonical(len);
        std::vector<uint16_t> table(1 << MAX_LEN, 0); // (len << 8) | symbol; len 0 = no code
        for (int s = 0; s < 256; ++s) {
            if (len[s]) {
                const uint32_t first = (uint32_t) code[s] << (MAX_LEN - len[s]);
                for (uint32_t k = 0; k < (1u << (MAX_LEN - len[s])); ++k) {
                    table[first + k] = (uint16_t) ((len[s] << 8) | s);
                }
            }
        }
        const uint8_t * bp   = p;
        const uint8_t * bend = p + n_bytes;
        uint64_t acc = 0;
        int n_acc = 0;
        for (size_t b = 0; b < n_blocks; ++b) {
            for (int i = 0; i < N_SCALE; ++i) {
                while (n_acc <= 56) {
                    acc = (acc << 8) | (bp < bend ? *bp++ : 0);
                    n_acc += 8;
                }
                const uint16_t e = table[(acc >> (n_acc - MAX_LEN)) & ((1u << MAX_LEN) - 1)];
                const int l = e >> 8;
                if (l == 0) {
                    return false;
                }
                blocks[b*BLOCK + i] = (uint8_t) (e & 0xFF);
                n_acc -= l;
            }
        }
        p = bend;
    } else {
        return false;
    }
    if ((size_t) (end - p) != (size_t) N_CODE * n_blocks) {
        return false;
    }
    for (size_t b = 0; b < n_blocks; ++b) {
        memcpy(blocks + b*BLOCK + N_SCALE, p, N_CODE);
        p += N_CODE;
    }
    return true;
}

size_t n_chunks_for(size_t n_blocks) {
    const size_t hw = std::max(1u, std::thread::hardware_concurrency() / 2);
    return std::max<size_t>(1, std::min<size_t>({ n_blocks / MIN_CHUNK, (size_t) MAX_CHUNKS, hw }));
}

size_t chunk_first(size_t c, size_t n_chunks, size_t n_blocks) {
    return c * n_blocks / n_chunks;
}

// runs f(c) for every chunk, on worker threads for all but the first
template <typename F> void for_chunks(size_t n_chunks, F && f) {
    std::vector<std::thread> workers;
    for (size_t c = 1; c < n_chunks; ++c) {
        workers.emplace_back(f, c);
    }
    f(0);
    for (auto & w : workers) {
        w.join();
    }
}

size_t header_size(size_t n_chunks) {
    return 4 + 8 + 4 + 8 * n_chunks;
}

} // namespace

size_t llama_nvfp4_packed_size(const uint8_t * blocks, size_t n_blocks) {
    const size_t nc = n_chunks_for(n_blocks);
    std::vector<size_t> sizes(nc);
    for_chunks(nc, [&](size_t c) {
        const size_t b0 = chunk_first(c, nc, n_blocks), b1 = chunk_first(c + 1, nc, n_blocks);
        sizes[c] = chunk_size(plan(blocks + b0*BLOCK, b1 - b0), b1 - b0);
    });
    size_t total = header_size(nc);
    for (size_t s : sizes) {
        total += s;
    }
    return total;
}

void llama_nvfp4_pack(const uint8_t * blocks, size_t n_blocks, std::vector<uint8_t> & out) {
    const size_t nc = n_chunks_for(n_blocks);
    std::vector<scale_code> sc(nc);
    std::vector<size_t> sizes(nc);
    for_chunks(nc, [&](size_t c) {
        const size_t b0 = chunk_first(c, nc, n_blocks), b1 = chunk_first(c + 1, nc, n_blocks);
        sc[c] = plan(blocks + b0*BLOCK, b1 - b0);
        sizes[c] = chunk_size(sc[c], b1 - b0);
    });

    const size_t start = out.size();
    size_t total = header_size(nc);
    std::vector<size_t> offs(nc);
    for (size_t c = 0; c < nc; ++c) {
        offs[c] = total;
        total += sizes[c];
    }
    out.resize(start + total);
    uint8_t * dst = out.data() + start;
    const uint64_t n64 = n_blocks;
    const uint32_t nc32 = (uint32_t) nc;
    memcpy(dst, &MAGIC, 4);
    memcpy(dst + 4, &n64, 8);
    memcpy(dst + 12, &nc32, 4);
    for (size_t c = 0; c < nc; ++c) {
        const uint64_t s64 = sizes[c];
        memcpy(dst + 16 + 8*c, &s64, 8);
    }
    for_chunks(nc, [&](size_t c) {
        const size_t b0 = chunk_first(c, nc, n_blocks), b1 = chunk_first(c + 1, nc, n_blocks);
        pack_chunk(blocks + b0*BLOCK, b1 - b0, sc[c], dst + offs[c]);
    });
}

bool llama_nvfp4_unpack(const uint8_t * data, size_t size, uint8_t * blocks, size_t n_blocks) {
    const uint8_t * p   = data;
    const uint8_t * end = data + size;
    uint32_t magic;
    uint64_t n;
    uint32_t nc;
    if (!get(p, end, magic) || magic != MAGIC || !get(p, end, n) || n != n_blocks || !get(p, end, nc) ||
            nc == 0 || nc > MAX_CHUNKS || nc > std::max<size_t>(1, n_blocks)) {
        return false;
    }
    std::vector<const uint8_t *> starts(nc + 1);
    starts[0] = data + header_size(nc);
    for (uint32_t c = 0; c < nc; ++c) {
        uint64_t s;
        if (!get(p, end, s) || s > (uint64_t) (end - starts[c])) {
            return false;
        }
        starts[c + 1] = starts[c] + s;
    }
    if (starts[nc] != end) {
        return false;
    }
    std::vector<char> ok(nc, 0);
    for_chunks(nc, [&](size_t c) {
        const size_t b0 = chunk_first(c, nc, n_blocks), b1 = chunk_first(c + 1, nc, n_blocks);
        ok[c] = unpack_chunk(starts[c], starts[c + 1], blocks + b0*BLOCK, b1 - b0);
    });
    return std::all_of(ok.begin(), ok.end(), [](char v) { return v != 0; });
}
