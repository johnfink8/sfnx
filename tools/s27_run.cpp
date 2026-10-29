// s27_run: the 27B on the cards, one prompt, greedy decode with drafts.
//   s27_run MODEL.gguf PROMPT.bin GEN [K=4] [LOOKUP=15] [OUT.ids]
// The prompt is read in windows of MAXT rows (the draft layer over each). Decode verifies the last token plus drafts a
// round: n-gram lookup's when it offers at least 3 that agree with the draft layer's first, else a chain of K from the
// draft layer (K 0: no drafts). OUT.ids: the generated ids, one per line.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/model.hpp"
#include "../src/s27.hpp"
#include "ref_tokens.hpp"

using namespace fnx;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: s27_run MODEL.gguf PROMPT.bin GEN [K=4] [LOOKUP=15] [OUT.ids]\n");
        return 2;
    }
    const std::vector<int> prompt = ref_tokens(argv[2]);
    const int gen = std::atoi(argv[3]);
    const int K = argc > 4 ? std::atoi(argv[4]) : 4;
    const int LK = argc > 5 ? std::atoi(argv[5]) : 15, LNG = 3;
    if (K < 0 || K > MAXT - 1) throw std::runtime_error("K: 0.." + std::to_string(MAXT - 1));
    if (LK < 0 || LK > MAXT - 1) throw std::runtime_error("LOOKUP: 0.." + std::to_string(MAXT - 1));
    if (prompt.size() < 2) throw std::runtime_error("the prompt needs two tokens or more");
    const auto t0 = std::chrono::steady_clock::now();
    Model m(argv[1]);
    S27 e(m, 32768);
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "s27: loaded in %.1f s; prompt %zu tokens\n", std::chrono::duration<double>(t1 - t0).count(), prompt.size());

    // the prompt but its last token, MAXT rows a window, the draft layer over each (row r with the token after it)
    int d1 = -1;
    float d1_p = 0.f;
    // S27_DRAFT_P=p: a draft is kept only when the draft layer gives it at least p (0: every one of the K). On prose2 it
    // never paid: p 0.35 with K 4 111 tok/s, p 0.5 110.6, against 117.6 for every one of K 3
    const float draft_p = std::getenv("S27_DRAFT_P") ? (float) std::atof(std::getenv("S27_DRAFT_P")) : 0.f;
    for (size_t i = 0; i + 1 < prompt.size(); i += MAXT) {
        const size_t n = std::min<size_t>(MAXT, prompt.size() - 1 - i);
        e.verify(std::vector<int>(prompt.begin() + i, prompt.begin() + i + n));
        e.commit((int) n);
        if (K) d1 = e.draft(std::vector<int>(prompt.begin() + i + 1, prompt.begin() + i + 1 + n), (int) i, true), d1_p = e.last_draft_p;
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "s27: prompt read in %.2f s (%.0f tok/s)\n", std::chrono::duration<double>(t2 - t1).count(),
                 (prompt.size() - 1) / std::chrono::duration<double>(t2 - t1).count());

    // S27_BENCH=1: windows of 1..16 rows instead of decoding, their GPU time (the median of 15 each); the draft pass's too
    if (const char* bs = std::getenv("S27_BENCH"); bs && !std::strcmp(bs, "1")) {
        for (int T : {1, 2, 3, 4, 5, 6, 8, 12, 16}) {
            std::vector<float> ms, card[NCARD];
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
        std::vector<double> dms;
        for (int r = 0; r < 15; ++r) {
            const auto a = std::chrono::steady_clock::now();
            e.draft({prompt.back()}, e.pos(), false);
            dms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        std::sort(dms.begin(), dms.end());
        std::printf("draft pass (1 row, host round trip included)  %6.2f ms\n", dms[dms.size() / 2]);
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
    const auto t3 = std::chrono::steady_clock::now();
    while ((int) out.size() < gen) {
        const int p = e.pos();
        std::vector<int> d = lookup();
        const auto td = std::chrono::steady_clock::now();
        if (K && (d.size() < 3 || d[0] != d1)) {
            // the draft layer's chain: up to K, each kept only while the draft before it was at least S27_DRAFT_P likely
            d.clear();
            float pp = d1_p;
            if (pp >= draft_p) d.push_back(d1);
            while (!d.empty() && (int) d.size() < K && pp >= draft_p) {
                const int nx = e.draft({d.back()}, p + (int) d.size() - 1, false);
                pp = e.last_draft_p;
                if (pp >= draft_p) d.push_back(nx);
            }
        } else if (!K && LK == 0) {
            d.clear();
        } else {
            ++lookup_rounds;
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
        // o[0..a]: the next a + 1 tokens; the last becomes the next window's first row (not yet in the cache)
        for (int i = 0; i <= a; ++i) {
            out.push_back(o[i]);
            ctx.push_back(o[i]);
        }
        if (o[a] == 248046 || o[a] == 248044) break;   // eos
        const auto tc = std::chrono::steady_clock::now();
        if (K) d1 = e.draft(std::vector<int>(o.begin(), o.begin() + a + 1), p, true), d1_p = e.last_draft_p;
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
