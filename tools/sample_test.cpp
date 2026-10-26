// The speculative sampler against plain sampling: the same prompt, SEEDS runs of N tokens each way (drafts on: MTP
// chain + lookup, rejection sampling; drafts off: one row a window), then per position the two token frequency tables,
// their total variation distance and a chi-square test (pooling tokens seen fewer than 5 times). A run that ended
// (end of reply) before a position counts there as the outcome "ended", so every position's tables hold SEEDS runs a
// mode and a sampler that ends replies at another rate differs too; positions no run of either mode reached are not
// compared and the summary says how many were. Also: temperature 0 against fnx_run mtp's greedy tokens is checked by
// sfnx_probe.py. With SAMPLE_TEST_LANES=L (2..6) the speculative side runs as prod's batched windows do: L lanes on the
// prompt at once through BatchDecoder (drafts, the row plan, its rejection sampler), each lane its own seed and run.
//
//   sample_test MODEL.gguf RT_DIR VOCAB.bin PROMPT.bin SEEDS N [TEMP TOP_P TOP_K PRESENCE]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../src/batch.hpp"
#include "../src/engine.hpp"
#include "../src/env.hpp"
#include "../src/session.hpp"
#include "ref_tokens.hpp"

// the outcome of a run that ended before a position
constexpr int ENDED = -1;

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "sample_test MODEL.gguf RT_DIR VOCAB.bin PROMPT.bin SEEDS N [TEMP TOP_P TOP_K PRESENCE]\n"); return 2; }
    fnx::Model m(argv[1]);
    const long long lanes_n = fnx::env_int("SAMPLE_TEST_LANES", 1);
    if (lanes_n < 1 || lanes_n > 6) { std::fprintf(stderr, "SAMPLE_TEST_LANES: 1 (Session::generate) or 2..6 (batched windows)\n"); return 2; }
    const int nl = (int) lanes_n;
    const fnx::Mtp mtp(m, argv[2], argv[3]);
    fnx::Engine e(m, 32768, nl);
    e.mtp_attach(mtp);
    const std::vector<int> prompt = ref_tokens(argv[4]);
    const int seeds = std::atoi(argv[5]), n = std::atoi(argv[6]);
    if (seeds < 2 || n < 1) { std::fprintf(stderr, "sample_test: SEEDS >= 2, N >= 1\n"); return 2; }
    fnx::Sampling s;
    s.temperature = argc > 7 ? std::atof(argv[7]) : 0.6f;
    s.top_p = argc > 8 ? std::atof(argv[8]) : 0.95f;
    s.top_k = argc > 9 ? std::atoi(argv[9]) : 20;
    s.pen_present = argc > 10 ? std::atof(argv[10]) : 1.0f;
    const std::vector<int> eos{248044, 248046};
    fnx::Session sess(e, 4, 7, eos);
    std::vector<std::map<int, int>> freq[2];
    if (nl > 1) {   // the speculative side as batched windows: nl lanes at a time, each run one lane's
        std::vector<int> lane{0};
        while ((int) lane.size() < nl) lane.push_back(e.add_lane());
        fnx::BatchDecoder dec(e, 6, 15, eos, fnx::batch_drafts());
        freq[0].assign(n, {});
        double ms = 0;
        long toks = 0;
        for (int k0 = 0; k0 < seeds; k0 += nl) {
            const int g = std::min(nl, seeds - k0);
            std::vector<fnx::BatchLane> bl((size_t) g);
            for (int r = 0; r < g; ++r) {
                e.use_lane(lane[(size_t) r]);
                e.reset();
                fnx::Session ls(e, 4, 7, eos, 248045, 0, lane[(size_t) r]);
                fnx::GenStats st;
                ls.read(prompt, st);
                bl[(size_t) r].start(lane[(size_t) r], s, std::mt19937_64(1000003ull * (k0 + r + 1)), prompt);
            }
            std::vector<int> pos((size_t) g, 0);
            std::vector<bool> ended((size_t) g, false);
            for (;;) {
                std::vector<fnx::BatchLane*> ls;
                std::vector<int> idx, limit;
                for (int r = 0; r < g; ++r)
                    if (!ended[(size_t) r] && pos[(size_t) r] < n) {
                        ls.push_back(&bl[(size_t) r]);
                        idx.push_back(r);
                        limit.push_back(n - pos[(size_t) r]);
                    }
                if (ls.empty()) break;
                fnx::BatchDecoder::fit(ls);
                const auto out = dec.window(ls, limit, [&](int i, int t) {
                    int& p = pos[(size_t) idx[(size_t) i]];
                    if (p < n) ++freq[0][(size_t) p++][t];
                    ++toks;
                });
                ms += dec.last_ms;
                for (size_t i = 0; i < out.size(); ++i)
                    if (out[i].eos) ended[(size_t) idx[i]] = true;
            }
            for (int r = 0; r < g; ++r)
                for (int p = pos[(size_t) r]; p < n; ++p) ++freq[0][(size_t) p][ENDED];
        }
        for (int i : lane) {
            e.use_lane(i);
            e.reset();
        }
        e.use_lane(0);
        std::printf("batched (%d lanes): %d runs, %.1f tok/s\n", nl, seeds, toks * 1000.0 / ms);
    }
    for (int mode = nl > 1 ? 1 : 0; mode < 2; ++mode) {
        sess.speculate = mode == 0;
        freq[mode].assign(n, {});
        double ms = 0;
        long toks = 0;
        for (int k = 0; k < seeds; ++k) {
            s.seed = 1000003ull * (k + 1) + (uint64_t) mode * 7919ull;
            fnx::GenStats st;
            sess.read(prompt, st);
            int pos = 0;
            sess.generate(n, s, [&](int t) { if (pos < n) ++freq[mode][pos++][t]; }, nullptr, st);
            for (int p = pos; p < n; ++p) ++freq[mode][p][ENDED];
            ms += st.decode_ms;
            toks += st.generated;
            // the next run starts from scratch: a prompt the engine's history no longer matches
            e.reset();
        }
        std::printf("%s: %d runs, %.1f tok/s\n", mode == 0 ? "speculative" : "plain", seeds, toks * 1000.0 / ms);
    }
    bool ok = true;
    int compared = 0;
    for (int p = 0; p < n; ++p) {
        if (freq[0][p].size() == 1 && freq[1][p].size() == 1 && freq[0][p].count(ENDED) && freq[1][p].count(ENDED)) continue;
        ++compared;
        std::map<int, int> all;
        for (int mode = 0; mode < 2; ++mode)
            for (auto& kv : freq[mode][p]) all[kv.first] += kv.second;
        double tv = 0, chi = 0;
        int df = 0;
        double pool[2] = {0, 0};
        long tot[2] = {0, 0};
        for (int mode = 0; mode < 2; ++mode)
            for (auto& kv : freq[mode][p]) tot[mode] += kv.second;
        auto add = [&](double a, double b) {   // a 2 x k contingency table's cell pair
            const double ra = (double) tot[0] / (tot[0] + tot[1]), rb = 1.0 - ra, c = a + b;
            if (c <= 0) return;
            chi += (a - c * ra) * (a - c * ra) / (c * ra) + (b - c * rb) * (b - c * rb) / (c * rb);
            ++df;
        };
        for (auto& kv : all) {
            const double a = freq[0][p].count(kv.first) ? freq[0][p][kv.first] : 0, b = freq[1][p].count(kv.first) ? freq[1][p][kv.first] : 0;
            tv += std::fabs(a / tot[0] - b / tot[1]) / 2;
            if (a + b < 5) {
                pool[0] += a;
                pool[1] += b;
            } else {
                add(a, b);
            }
        }
        add(pool[0], pool[1]);
        df = std::max(1, df - 1);
        // chi-square's upper tail by the Wilson-Hilferty normal approximation
        const double z = (std::cbrt(chi / df) - (1 - 2.0 / (9 * df))) / std::sqrt(2.0 / (9 * df));
        const double pval = 0.5 * std::erfc(z / std::sqrt(2.0));
        std::printf("position %d: %zu distinct outcomes, total variation %.3f, chi2 %.1f on %d df, p = %.3f%s\n", p, all.size(), tv, chi,
                    df, pval, pval < 0.001 ? "  <-- differs" : "");
        if (pval < 0.001) ok = false;
    }
    if (compared == 0) {
        std::printf("MISMATCH: no position compared (every run of both modes ended at once)\n");
        return 1;
    }
    std::printf("%s (%d of %d positions reached by a run, each compared)\n",
                ok ? "the speculative sampler matches plain sampling at every position (p >= 0.001)" : "MISMATCH", compared, n);
    return ok ? 0 : 1;
}
