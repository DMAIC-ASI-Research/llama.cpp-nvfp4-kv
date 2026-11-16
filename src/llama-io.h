#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    // true for the size pass, which counts bytes and ignores the data
    virtual bool size_only() const { return false; }

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // pointer to the next size bytes, valid until the reader is destroyed; nullptr if the reader cannot do that
    virtual const uint8_t * read_ptr(size_t size) { (void) size; return nullptr; }

    // set tensor data from host memory owned by keep (e.g. unpacked KV rows); readers that defer read_tensor also defer this
    virtual void set_tensor(ggml_tensor * tensor, size_t offset, const uint8_t * data, size_t size, std::shared_ptr<void> keep);

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
