// Host-only check and timing of the loader's reads (Stager, src/gguf.hpp): layers [A, B) of a model staged as the
// loader stages them, each tensor compared byte for byte with the mapping; or the old pattern for comparison, three
// threads slicing every layer's routed experts out of the mapping as load_card did (card ff ranges 0/224/448/640).
//   make bin/stage_test && bin/stage_test MODEL-00001.gguf A B [stage|mmap]
// All of [A, B) is held in RAM at once (a Q8_0 layer: 2.9 GB). The tensors that were out of the page cache at the
// start are dropped from it again at the end, so a run leaves the page cache as it found it and can be repeated on
// other cold layers; a layer partly cached at the start is read partly from the cache (the line says how much).
// Measured 2026-10-25, Q8_0 layers 1-22 (cold, 2.67 GB of experts each), Kingston NV3 on Gen3 x4, prod running:
//   old pattern (three threads slicing from the mapping): 0.71-0.84 GB/s; one thread faulting pages in order 1.1-1.2;
//   16 threads faulting (populating) parts side by side 2.3; buffered pread 1 thread 1.3, 8-16 threads 2.5-2.9;
//   O_DIRECT pread 3-8 threads x 16-64 MiB 3.3-3.4 (the drive's rate); slicing a staged layer for three cards ~0.4 s.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/gguf.hpp"

namespace {

size_t resident(const fnx::Tensor& t) {
    const uintptr_t a = (uintptr_t) t.data / 4096 * 4096, e = (uintptr_t) t.data + t.nbytes;
    std::vector<unsigned char> v((e - a + 4095) / 4096);
    if (mincore((void*) a, e - a, v.data())) throw std::runtime_error("mincore");
    size_t n = 0;
    for (unsigned char c : v) n += c & 1;
    return n * 4096;
}

void drop(const fnx::Gguf& g, const fnx::Tensor& t) {
    const uintptr_t a = (uintptr_t) t.data / 4096 * 4096;
    madvise((void*) a, (uintptr_t) t.data + t.nbytes - a, MADV_DONTNEED);   // this process's mappings first: mapped pages stay
    const int fd = ::open(g.shard_path(t.shard).c_str(), O_RDONLY);
    posix_fadvise(fd, (off_t) t.offset, (off_t) t.nbytes, POSIX_FADV_DONTNEED);
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: stage_test MODEL-00001.gguf A B [stage|mmap]\n");
        return 2;
    }
    try {
        fnx::Gguf g(argv[1]);
        const int a = std::atoi(argv[2]), b = std::atoi(argv[3]);
        const std::string mode = argc > 4 ? argv[4] : "stage";
        std::vector<fnx::Stager::Group> groups((size_t) (b - a));
        std::vector<const fnx::Tensor*> cold;
        size_t bytes = 0, cached0 = 0;
        for (const auto& kv : g.tensors()) {
            int l = -1, n = 0;
            if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) != 1 || n == 0 || l < a || l >= b) continue;
            if (mode == "mmap" && kv.first.find("_exps.") == std::string::npos) continue;
            groups[(size_t) (l - a)].names.push_back(kv.first);
            const size_t r = resident(kv.second);
            if (r == 0) cold.push_back(&kv.second);
            bytes += kv.second.nbytes;
            cached0 += r;
        }
        const auto t0 = std::chrono::steady_clock::now();
        double secs = 0;
        if (mode == "stage") {
            // every group held at once and compared only once all are read: the comparison reads the mapping, which
            // would take the disk from the reads being timed
            fnx::Stager st(g, groups, 1, b - a);
            st.t(b - a - 1, groups.back().names.front());
            size_t diff = 0;
            for (int i = 0; i < b - a; ++i) {
                for (const std::string& n : groups[(size_t) i].names) {
                    const fnx::Tensor& s = st.t(i, n);
                    const fnx::Tensor& m = g.t(n);
                    if (s.nbytes != m.nbytes || std::memcmp(s.data, m.data, m.nbytes)) {
                        std::fprintf(stderr, "MISMATCH: %s\n", n.c_str());
                        ++diff;
                    }
                }
                st.done(i);
            }
            secs = st.read_seconds();
            std::printf("stage: layers %d-%d, %.2f GB (%.2f GB cached at the start): read in %.2f s = %.2f GB/s (%.2f GB from the disk, %.2f from "
                        "the page cache); %zu tensors differ from the mapping\n",
                        a, b - 1, bytes / 1e9, cached0 / 1e9, secs, bytes / 1e9 / secs, st.disk_bytes() / 1e9, st.cache_bytes() / 1e9, diff);
            for (const fnx::Tensor* t : cold) drop(g, *t);
            return diff ? 1 : 0;
        }
        if (mode != "mmap") throw std::runtime_error("mode: stage or mmap");
        const int ff[4] = {0, 224, 448, 640};
        std::vector<std::thread> th;
        for (int c = 0; c < 3; ++c)
            th.emplace_back([&, c] {
                std::vector<uint8_t> h;
                for (int l = a; l < b; ++l) {
                    const std::string p = "blk." + std::to_string(l) + ".";
                    const fnx::Tensor &tg = g.t(p + "ffn_gate_exps.weight"), &tu = g.t(p + "ffn_up_exps.weight"), &td = g.t(p + "ffn_down_exps.weight");
                    const size_t rb = tg.row_bytes(), half = (size_t) (ff[c + 1] - ff[c]) * rb, mat = (size_t) tg.ne[1] * rb;
                    h.resize((size_t) tg.ne[2] * 2 * half);
                    for (int64_t e = 0; e < tg.ne[2]; ++e) {
                        std::memcpy(&h[(size_t) e * 2 * half], tg.data + e * mat + (size_t) ff[c] * rb, half);
                        std::memcpy(&h[(size_t) e * 2 * half + half], tu.data + e * mat + (size_t) ff[c] * rb, half);
                    }
                    const fnx::TypeInfo ti = fnx::type_info(td.type);
                    const size_t drb = (size_t) (ff[c + 1] - ff[c]) / ti.block * ti.bytes, src = td.row_bytes();
                    h.resize((size_t) td.ne[2] * td.ne[1] * drb);
                    for (int64_t e = 0; e < td.ne[2]; ++e)
                        for (int64_t r = 0; r < td.ne[1]; ++r)
                            std::memcpy(&h[(size_t) (e * td.ne[1] + r) * drb], td.data + (size_t) (e * td.ne[1] + r) * src + (size_t) ff[c] / ti.block * ti.bytes, drb);
                }
            });
        for (auto& t : th) t.join();
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("mmap: layers %d-%d, %.2f GB of routed experts (%.2f GB cached at the start): sliced for three cards in %.2f s = %.2f GB/s\n",
                    a, b - 1, bytes / 1e9, cached0 / 1e9, secs, bytes / 1e9 / secs);
        for (const fnx::Tensor* t : cold) drop(g, *t);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "stage_test: %s\n", e.what());
        return 1;
    }
}
