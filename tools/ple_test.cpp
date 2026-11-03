// FNX_NGRAM=disk against the mapping, CPU only (runs beside a serving engine): the table rows of every prompt's every
// position, gathered as the prompt path gathers them (4,096 positions a call) and as decode does (one a call), must be
// the mapping's bytes exactly. Prints the reads and their time, the row cache cold at each prompt.
//   bin/ple_test MODEL.gguf PROMPT.bin...      FNX_NGRAM_THREADS, FNX_NGRAM_CACHE_MIB as the engine reads them
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../src/gguf.hpp"
#include "../src/ple_io.hpp"
#include "ref_tokens.hpp"

using namespace fnx;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// one prompt in calls of `per` positions through a fresh disk reader; every call's bytes against the mapping's.
// Returns false on a mismatch
bool run(const Gguf& g, const Tensor& t, const PleHash& ph, const std::vector<int>& toks, int per, const char* what, int max_pos) {
    PleRows ram(g, t, PleRows::Mode::RAM), disk(g, t, PleRows::Mode::DISK);
    const int heads = (int) ph.off.size(), T = std::min((int) toks.size(), max_pos);
    const size_t rb = disk.row_bytes();
    std::vector<uint64_t> rows;
    std::vector<uint8_t> a, b;
    std::vector<double> call_ms;
    double busy = 0;
    for (int p0 = 0; p0 < T; p0 += per) {
        const int n = std::min(per, T - p0);
        rows.resize((size_t) n * heads);
        ple_window_rows(ph, p0, n, [&](int q) -> int64_t { return toks[(size_t) q]; }, rows.data());
        a.resize(rows.size() * rb);
        b.resize(rows.size() * rb);
        ram.gather(rows.data(), rows.size(), a.data());
        const double t0 = now_s();
        disk.gather(rows.data(), rows.size(), b.data());
        const double dt = now_s() - t0;
        busy += dt;
        call_ms.push_back(dt * 1e3);
        if (std::memcmp(a.data(), b.data(), a.size())) {
            for (size_t i = 0; i < rows.size(); ++i)
                if (std::memcmp(&a[i * rb], &b[i * rb], rb)) {
                    std::printf("MISMATCH %s: position %d head %zu row %llu\n", what, p0 + (int) (i / heads), i % heads,
                                (unsigned long long) rows[i]);
                    break;
                }
            return false;
        }
    }
    const PleRows::Stats s = disk.stats();
    std::sort(call_ms.begin(), call_ms.end());
    std::printf("  %-7s %7d positions, %4d a call: %5.1f%% from the cache, %8llu reads, %7.0f reads/s; %.3f s in gathers, "
                "a call p50 %.3f ms p99 %.3f ms\n",
                what, T, per, 100.0 * s.hits / s.lookups, (unsigned long long) s.reads, s.reads / busy, busy,
                call_ms[call_ms.size() / 2], call_ms[call_ms.size() * 99 / 100]);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf PROMPT.bin...\n", argv[0]);
        return 2;
    }
    try {
        Gguf g(argv[1]);
        const Tensor& t = g.t("per_layer_token_embd.weight");
        const std::string p = "qwen4exp.";
        PleHash ph;
        for (int64_t v : g.ia(p + "ple.layer_multipliers")) ph.mult.push_back((uint64_t) v);
        for (int64_t v : g.ia(p + "ple.head_offsets")) ph.off.push_back((uint64_t) v);
        for (int64_t v : g.ia(p + "ple.head_vocab_sizes")) ph.vocab.push_back((uint64_t) v);
        ph.eos = g.i(p + "ple.eos_token_id");
        if (ph.mult.size() != 3 || ph.off.size() != 16 || ph.vocab.size() != 16)
            throw std::runtime_error("the PLE metadata is not 3 multipliers and 16 heads");
        {
            PleRows probe(g, t, PleRows::Mode::DISK);
            std::printf("%s\n%s\n", probe.describe().c_str(), probe.probe(4096).c_str());
        }
        bool ok = true;
        for (int i = 2; i < argc; ++i) {
            const std::vector<int> toks = ref_tokens(argv[i]);
            std::printf("%s\n", argv[i]);
            ok &= run(g, t, ph, toks, 4096, "prompt", (int) toks.size());
            ok &= run(g, t, ph, toks, 1, "decode", 4000);
        }
        std::printf(ok ? "PASS: every row the mapping's bytes\n" : "FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ple_test: %s\n", e.what());
        return 1;
    }
}
