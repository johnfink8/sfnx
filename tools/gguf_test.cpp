// Host-only test of the GGUF reader's header checks: writes tiny GGUF files, one valid and several malformed, and
// expects the valid one to load and each malformed one to be refused with a message naming the tensor.
// make bin/gguf_test && bin/gguf_test [DIR [MODEL.gguf ...]]   (DIR for the files, default /tmp; each MODEL, a real
// model's first shard, must load too: only its headers are read)
#include "../src/gguf.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// one F32 tensor "w" with the given n_dims and sizes, its data at offset 0 of the data section, data_bytes of data
std::vector<uint8_t> make(uint32_t n_dims, const std::vector<uint64_t>& ne, uint64_t data_bytes, uint64_t offset = 0) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    u32(0x46554747u);   // "GGUF"
    u32(3);             // version
    u64(1);             // tensors
    u64(0);             // metadata keys
    const std::string name = "w";
    u64(name.size());
    put(name.data(), name.size());
    u32(n_dims);
    for (uint64_t v : ne) u64(v);
    u32(0);             // F32
    u64(offset);
    while (b.size() % 32) b.push_back(0);
    b.resize(b.size() + data_bytes, 0);
    return b;
}

std::string write(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b) {
    const std::string path = dir + "/fnx_gguf_test_" + tag + ".gguf";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size() || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
    return path;
}

int failures = 0;

void expect_refused(const std::string& dir, const std::string& tag, const std::vector<uint8_t>& b, const char* want) {
    const std::string path = write(dir, tag, b);
    try {
        fnx::Gguf g(path);
        std::printf("FAIL %s: loaded\n", tag.c_str());
        ++failures;
    } catch (const std::runtime_error& e) {
        const bool ok = std::strstr(e.what(), "tensor w") && std::strstr(e.what(), want);
        std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", tag.c_str(), e.what());
        failures += !ok;
    }
    std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp";

    {
        const std::string path = write(dir, "valid", make(2, {8, 3}, 8 * 3 * 4));
        try {
            fnx::Gguf g(path);
            const fnx::Tensor& t = g.t("w");
            const bool ok = t.n_dims == 2 && t.ne[0] == 8 && t.ne[1] == 3 && t.ne[2] == 1 && t.ne[3] == 1 &&
                            t.nbytes == 96 && t.row_bytes() == 32;
            std::printf("%s valid: n_dims %d, ne %lld x %lld, %zu bytes\n", ok ? "ok  " : "FAIL", t.n_dims,
                        (long long) t.ne[0], (long long) t.ne[1], t.nbytes);
            failures += !ok;
        } catch (const std::runtime_error& e) {
            std::printf("FAIL valid: %s\n", e.what());
            ++failures;
        }
        std::remove(path.c_str());
    }

    expect_refused(dir, "dims0", make(0, {}, 64), "0 dimensions");
    expect_refused(dir, "dims5", make(5, {1, 1, 1, 1, 1}, 64), "5 dimensions");
    expect_refused(dir, "dims_huge", make(0xffffffffu, {1}, 64), "4294967295 dimensions");
    expect_refused(dir, "ne_zero", make(2, {8, 0}, 64), "size 0 in dimension 1");
    expect_refused(dir, "ne_negative", make(1, {~0ull}, 64), "in dimension 0");
    // 2^32 x 2^32 elements: 2^64 wraps to 0, which would pass the end check without the overflow test
    expect_refused(dir, "count_overflow", make(2, {1ull << 32, 1ull << 32}, 64), "element count overflows");
    // 2^62 elements of 4 bytes: the count fits, the byte count does not
    expect_refused(dir, "bytes_overflow", make(1, {1ull << 62}, 64), "runs past the end");
    expect_refused(dir, "past_end", make(2, {8, 3}, 95), "runs past the end");
    expect_refused(dir, "offset_wraps", make(1, {8}, 64, ~0ull - 8), "runs past the end");

    for (int a = 2; a < argc; ++a) {
        try {
            fnx::Gguf g(argv[a]);
            std::printf("ok   %s: loads\n", argv[a]);
        } catch (const std::runtime_error& e) {
            std::printf("FAIL %s: %s\n", argv[a], e.what());
            ++failures;
        }
    }

    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
