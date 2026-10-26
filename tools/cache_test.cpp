// The conversation cache against reading from scratch. `cached` walks one Session through five prompts:
//   1. A1 (a long single-message prompt)                  read from scratch
//   2. A1 + its reply + code1's message                   extends the live sequence
//   3. A1 up to its last message boundary + edit1         a checkpoint of the live sequence
//   4. open1                                              another conversation: the first is parked
//   5. A1 up to its last message boundary + code1         the parked conversation's checkpoint
// recording for each the first generated token's top-64 logits and GEN greedy tokens (one row a window), and writing
// the prompts to PROMPTS.txt. `fresh` reads each prompt of PROMPTS.txt from scratch and records the same. compare: KL of
// the first token's top-64 distributions (cached against fresh) and the greedy tokens' agreement; it exits 1 when a
// prompt's first-token KL is above MAX_KL (default 1.5e-3, the worst measured) or its top-1 differs.
// Greedy tokens may part at a near-tie, so their agreement is printed, not judged.
//
//   cache_test MODEL.gguf RT_DIR VOCAB.bin cached A1.bin CODE1.bin EDIT1.bin OPEN1.bin PROMPTS.txt OUT.txt [GEN]
//   cache_test MODEL.gguf RT_DIR VOCAB.bin fresh PROMPTS.txt OUT.txt [GEN]
//   cache_test compare CACHED_OUT.txt FRESH_OUT.txt [MAX_KL]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/engine.hpp"
#include "../src/session.hpp"
#include "ref_tokens.hpp"

namespace {


constexpr int TURN = 248045;

struct Result {
    std::vector<std::pair<float, int>> top;
    std::vector<int> gen;
    double read_ms = 0;
    int reused = 0;
};

// after Session::read: the last prompt token's window (its top-64), then greedy tokens a row at a time
Result finish(fnx::Engine& e, const std::vector<int>& prompt, int n_gen) {
    Result r;
    std::vector<int> o = e.verify({prompt.back()});
    r.top = e.top_tokens(1, {})[0];
    e.commit(1);
    int x = o[0];
    r.gen.push_back(x);
    while ((int) r.gen.size() < n_gen) {
        o = e.verify({x});
        e.commit(1);
        x = o[0];
        r.gen.push_back(x);
    }
    return r;
}

void write_result(std::FILE* f, const Result& r) {
    std::fprintf(f, "reused %d read_ms %.1f\ntop", r.reused, r.read_ms);
    for (const auto& t : r.top) std::fprintf(f, " %d:%.6f", t.second, t.first);
    std::fprintf(f, "\ngen");
    for (int t : r.gen) std::fprintf(f, " %d", t);
    std::fprintf(f, "\n");
}

std::vector<Result> read_results(const char* path) {
    std::ifstream f(path);
    std::vector<Result> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("reused", 0) == 0) {
            out.emplace_back();
            std::sscanf(line.c_str(), "reused %d read_ms %lf", &out.back().reused, &out.back().read_ms);
        } else if (line.rfind("top", 0) == 0) {
            std::istringstream is(line.substr(3));
            std::string w;
            while (is >> w) {
                const size_t c = w.find(':');
                out.back().top.emplace_back(std::stof(w.substr(c + 1)), std::stoi(w.substr(0, c)));
            }
        } else if (line.rfind("gen", 0) == 0) {
            std::istringstream is(line.substr(3));
            for (int t; is >> t;) out.back().gen.push_back(t);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "compare") {
        const auto a = read_results(argv[2]), b = read_results(argv[3]);
        if (a.size() != b.size()) { std::fprintf(stderr, "different numbers of prompts\n"); return 1; }
        char* end = nullptr;
        const double max_kl = argc >= 5 ? std::strtod(argv[4], &end) : 1.5e-3;
        if (argc >= 5 && (end == argv[4] || *end != '\0' || !(max_kl >= 0))) {
            std::fprintf(stderr, "MAX_KL: want a number >= 0, got '%s'\n", argv[4]);
            return 2;
        }
        double worst = 0;
        bool top1_differs = false;
        for (size_t i = 0; i < a.size(); ++i) {
            // KL(fresh || cached) over fresh's top 64 (a token missing from cached's list: cached's 64th logit, a bound)
            auto lse = [](const std::vector<std::pair<float, int>>& t) {
                double m = t[0].first, s = 0;
                for (const auto& x : t) s += std::exp(x.first - m);
                return m + std::log(s);
            };
            const double la = lse(a[i].top), lb = lse(b[i].top);
            double kl = 0;
            for (const auto& x : b[i].top) {
                const double pb = std::exp(x.first - lb);
                double qa = a[i].top.back().first;
                for (const auto& y : a[i].top)
                    if (y.second == x.second) qa = y.first;
                kl += pb * ((x.first - lb) - (qa - la));
            }
            size_t same = 0;
            while (same < a[i].gen.size() && same < b[i].gen.size() && a[i].gen[same] == b[i].gen[same]) ++same;
            std::printf("prompt %zu: cached reused %d (read %.0f ms) / fresh %.0f ms; first token KL %.2e, top-1 %s, greedy tokens agree for %zu of %zu\n",
                        i + 1, a[i].reused, a[i].read_ms, b[i].read_ms, kl, a[i].top[0].second == b[i].top[0].second ? "same" : "DIFFERENT",
                        same, a[i].gen.size());
            worst = std::max(worst, kl);
            top1_differs = top1_differs || a[i].top[0].second != b[i].top[0].second;
        }
        const bool ok = worst <= max_kl && !top1_differs;
        std::printf("worst first-token KL %.2e (at most %.2e)%s: %s\n", worst, max_kl, top1_differs ? ", a top-1 differs" : "",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 6) { std::fprintf(stderr, "see the header for the usage\n"); return 2; }
    fnx::Model m(argv[1]);
    fnx::Engine e(m, 65536);
    const fnx::Mtp mtp(m, argv[2], argv[3]);
    e.mtp_attach(mtp);
    const std::string mode = argv[4];
    fnx::Session sess(e, 4, 7, {248044, 248046}, TURN, (size_t) 32 << 30);
    std::vector<std::vector<int>> prompts;
    std::vector<Result> res;
    const char* out_path;
    int n_gen;
    auto run = [&](const std::vector<int>& p) {
        fnx::GenStats st;
        const auto t0 = std::chrono::steady_clock::now();
        sess.read(p, st);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        Result r = finish(e, p, n_gen);
        r.read_ms = ms;
        r.reused = st.reused;
        res.push_back(r);
        prompts.push_back(p);
        const auto cs = sess.cache_stats();
        std::printf("prompt %zu: %zu tokens, reused %d, read in %.0f ms; cache: %d checkpoints, %d parked, %zu MiB\n", res.size(),
                    p.size(), st.reused, ms, cs.checkpoints, cs.parked, cs.bytes >> 20);
        return r;
    };
    if (mode == "cached") {
        if (argc < 11) { std::fprintf(stderr, "cached needs A1 CODE1 EDIT1 OPEN1 PROMPTS OUT\n"); return 2; }
        n_gen = argc > 11 ? std::atoi(argv[11]) : 32;
        out_path = argv[10];
        const auto a1 = ref_tokens(argv[5]), code1 = ref_tokens(argv[6]), edit1 = ref_tokens(argv[7]), open1 = ref_tokens(argv[8]);
        size_t ti = 0;
        for (size_t i = a1.size() - 1; i-- > 0;)
            if (a1[i] == TURN) { ti = i; break; }
        const Result r1 = run(a1);
        std::vector<int> p2(a1);
        p2.insert(p2.end(), r1.gen.begin(), r1.gen.end());
        p2.insert(p2.end(), code1.begin(), code1.end());
        run(p2);
        std::vector<int> p3(a1.begin(), a1.begin() + ti);
        p3.insert(p3.end(), edit1.begin(), edit1.end());
        run(p3);
        run(open1);
        std::vector<int> p5(a1.begin(), a1.begin() + ti);
        p5.insert(p5.end(), code1.begin(), code1.end());
        run(p5);
        std::FILE* pf = std::fopen(argv[9], "w");
        for (const auto& p : prompts) {
            for (int t : p) std::fprintf(pf, "%d ", t);
            std::fprintf(pf, "\n");
        }
        std::fclose(pf);
    } else if (mode == "fresh") {
        if (argc < 7) { std::fprintf(stderr, "fresh needs PROMPTS OUT\n"); return 2; }
        n_gen = argc > 7 ? std::atoi(argv[7]) : 32;
        out_path = argv[6];
        std::ifstream pf(argv[5]);
        std::string line;
        std::vector<std::vector<int>> ps;
        while (std::getline(pf, line)) {
            std::istringstream is(line);
            ps.emplace_back();
            for (int t; is >> t;) ps.back().push_back(t);
        }
        fnx::Session cold(e, 4, 7, {248044, 248046}, TURN, 0);
        for (const auto& p : ps) {
            e.reset();
            fnx::GenStats st;
            const auto t0 = std::chrono::steady_clock::now();
            cold.read(p, st);
            Result r = finish(e, p, n_gen);
            r.read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            res.push_back(r);
        }
    } else {
        std::fprintf(stderr, "mode: cached, fresh or compare\n");
        return 2;
    }
    std::FILE* f = std::fopen(out_path, "w");
    for (const auto& r : res) write_result(f, r);
    std::fclose(f);
    return 0;
}
