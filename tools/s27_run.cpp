// s27_run: the 27B's Engine (make FNX_MODEL=27b) on one prompt, greedy decode with drafts.
//   s27_run MODEL.gguf PROMPT.bin GEN [K=4] [LOOKUP=15] [OUT.ids]
// The prompt but its last token through prefill (the draft layer over it); decode verifies the last token plus drafts
// a round: n-gram lookup's when it offers at least 3 that agree with the draft layer's first, else a chain of K from
// the draft layer (K 0: no drafts). OUT.ids: the generated ids, one per line. The draft vocabulary: S27_DRAFT_VOCAB
// (default data/draft_vocab_en.bin, prod's). S27_BENCH=1: window times by row count instead of decoding
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/engine.hpp"
#include "../src/env.hpp"
#include "../src/model.hpp"
#include "ref_tokens.hpp"

using namespace fnx;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: s27_run MODEL.gguf PROMPT.bin GEN [K=4] [LOOKUP=15] [OUT.ids]\n");
        return 2;
    }
    const std::vector<int> prompt = ref_tokens(argv[2]);
    auto arg = [&](int i, const char* name, long long lo, long long hi, long long def) -> int {
        if (i >= argc) return (int) def;
        long long v = 0;
        if (!parse_whole(argv[i], v) || v < lo || v > hi)
            throw std::runtime_error(std::string(name) + ": " + std::to_string(lo) + ".." + std::to_string(hi) + ", not '" + argv[i] + "'");
        return (int) v;
    };
    const int gen = arg(3, "GEN", 0, 1 << 20, 0);
    const int K = arg(4, "K", 0, MAXT - 1, 4);
    const int LK = arg(5, "LOOKUP", 0, MAXT - 1, 15), LNG = 3;
    if (prompt.size() < 2) throw std::runtime_error("the prompt needs two tokens or more");
    const int ctx_len = std::max(32768, (int) prompt.size() + gen + 64);
    const char* dv = std::getenv("S27_DRAFT_VOCAB");
    const auto t0 = std::chrono::steady_clock::now();
    Model m(argv[1]);
    Mtp mtp(m, "nextn", dv ? dv : "data/draft_vocab_en.bin");
    Engine e(m, ctx_len, 1);
    if (K) e.mtp_attach(mtp);
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "s27: loaded in %.1f s; prompt %zu tokens\n", std::chrono::duration<double>(t1 - t0).count(), prompt.size());
    e.prefill(std::vector<int>(prompt.begin(), prompt.end() - 1), prompt.back());
    const auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "s27: prompt read in %.2f s (%.0f tok/s)\n", std::chrono::duration<double>(t2 - t1).count(),
                 (prompt.size() - 1) / std::chrono::duration<double>(t2 - t1).count());

    if (const char* bs = std::getenv("S27_BENCH"); bs && !std::strcmp(bs, "1")) {
        for (int T : {1, 2, 3, 4, 5, 6, 8, 12, 16}) {
            std::vector<double> ms, card[NCARD];
            for (int r = 0; r < 15; ++r) {
                e.verify(std::vector<int>(T, prompt.back()));
                ms.push_back(e.last_gpu_ms);
                for (int c = 0; c < NCARD; ++c) card[c].push_back(e.last_card_ms[c]);
                e.commit(1);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("window T=%2d  %6.2f ms (cards", T, ms[ms.size() / 2]);
            for (int c = 0; c < NCARD; ++c) {
                std::sort(card[c].begin(), card[c].end());
                std::printf(" %.2f", card[c][card[c].size() / 2]);
            }
            std::printf(")\n");
        }
        return 0;
    }
    std::vector<int> ctx(prompt), out;
    auto lookup = [&]() {
        std::vector<int> r;
        if (LK == 0 || (int) ctx.size() <= LNG) return r;
        const size_t n0 = ctx.size() - LNG;
        for (size_t st = n0; st-- > 0;)
            if (std::equal(ctx.begin() + st, ctx.begin() + st + LNG, ctx.begin() + n0)) {
                r.assign(ctx.begin() + st + LNG, ctx.begin() + std::min(ctx.size(), st + LNG + LK));
                break;
            }
        return r;
    };
    int rounds = 0, lookup_rounds = 0;
    long offered = 0, accepted = 0;
    double gpu_ms = 0, draft_ms = 0;
    int d1 = -1, d1_row = 0;   // the draft layer's guess for the next token, and the row of its pass it came from
    const auto t3 = std::chrono::steady_clock::now();
    while ((int) out.size() < gen) {
        const int p = e.pos();
        std::vector<int> d = lookup();
        const auto td = std::chrono::steady_clock::now();
        if (K && d1 >= 0 && (d.size() < 3 || d[0] != d1)) {
            d.assign(1, d1);
            int row = d1_row;
            while ((int) d.size() < K) {
                d.push_back(e.draft(1, false, row, {d.back()}, p + (int) d.size() - 1));
                row = 0;
            }
        } else if (d.size() >= 3 && (LK > 0)) {
            ++lookup_rounds;
        } else {
            d.clear();
        }
        draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        std::vector<int> win{ctx.back()};
        win.insert(win.end(), d.begin(), d.end());
        const std::vector<int> o = e.verify(win);
        gpu_ms += e.last_gpu_ms;
        int a = 0;
        while (a < (int) d.size() && d[a] == o[a]) ++a;
        e.commit(a + 1);
        ++rounds;
        offered += (long) d.size();
        accepted += a;
        for (int i = 0; i <= a; ++i) {
            out.push_back(o[i]);
            ctx.push_back(o[i]);
        }
        if (o[a] == 248046 || o[a] == 248044) break;   // eos
        const auto tc = std::chrono::steady_clock::now();
        if (K) d1 = e.draft(a + 1, true, 0, std::vector<int>(o.begin(), o.begin() + a + 1), p), d1_row = a;
        draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
    std::printf("decode: %zu tokens in %.0f ms, %.1f tok/s, %.2f tokens a round, %.2f ms a round (verify gpu %.2f, drafting %.2f), "
                "drafts %ld of %ld, %d of %d rounds lookup\n",
                out.size(), ms, out.size() * 1000.0 / ms, (double) out.size() / rounds, ms / rounds, gpu_ms / rounds, draft_ms / rounds,
                accepted, offered, lookup_rounds, rounds);
    if (argc > 6) {
        std::FILE* f = std::fopen(argv[6], "w");
        if (!f) throw std::runtime_error(std::string(argv[6]) + ": cannot write");
        for (int t : out) std::fprintf(f, "%d\n", t);
        std::fclose(f);
    }
    return 0;
}
