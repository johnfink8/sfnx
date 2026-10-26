// Lanes are separate sequences: three conversations alone (one after another on lane 0, from scratch), then on
// three lanes at once, interleaved: each reads its prompt, generates GEN greedy tokens, then a second turn (its prompt +
// its reply + a few more of its own tokens) - read, generate - in a rotating order. Every lane's tokens must equal the
// same conversation's alone, and the second turns must reuse their lane's prefix.
//
//   lanes_test MODEL.gguf RT_DIR VOCAB.bin A.bin B.bin C.bin [GEN]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/engine.hpp"
#include "../src/env.hpp"
#include "../src/session.hpp"
#include "ref_tokens.hpp"

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "lanes_test MODEL.gguf RT_DIR VOCAB.bin A.bin B.bin C.bin [GEN]\n"); return 2; }
    fnx::Model m(argv[1]);
    const fnx::Mtp mtp(m, argv[2], argv[3]);
    fnx::Engine e(m, (int) fnx::env_int("FNX_MAX_CTX", 32768));
    e.mtp_attach(mtp);
    const int gen = argc > 7 ? std::atoi(argv[7]) : 48;
    const std::vector<std::vector<int>> prompts{ref_tokens(argv[4]), ref_tokens(argv[5]), ref_tokens(argv[6])};
    const std::vector<int> eos{248044, 248046};
    fnx::Sampling greedy;
    auto turn = [&](fnx::Session& s, const std::vector<int>& p, std::vector<int>& out, fnx::GenStats& st) {
        s.read(p, st);
        out.clear();
        s.generate(gen, greedy, [&](int t) { out.push_back(t); }, nullptr, st);
    };
    // alone
    std::vector<std::vector<int>> solo1(3), solo2(3);
    {
        fnx::Session s(e, 4, 7, eos);
        s.speculate = std::getenv("LANES_NOSPEC") == nullptr;
        for (int i = 0; i < 3; ++i) {
            e.use_lane(0);
            e.reset();
            fnx::GenStats st;
            turn(s, prompts[i], solo1[i], st);
            std::vector<int> p2(prompts[i]);
            p2.insert(p2.end(), solo1[i].begin(), solo1[i].end());
            p2.insert(p2.end(), prompts[i].end() - 6, prompts[i].end());
            turn(s, p2, solo2[i], st);
        }
    }
    if (std::getenv("LANES_SOLO2")) {   // the solo phase again: is it reproducible on its own?
        if (std::getenv("LANES_ADD_FIRST")) e.add_lane();
        fnx::Session s(e, 4, 7, eos);
        s.speculate = std::getenv("LANES_NOSPEC") == nullptr;
        for (int i = 0; i < 3; ++i) {
            e.use_lane(0);
            e.reset();
            fnx::GenStats st;
            std::vector<int> a, b;
            turn(s, prompts[i], a, st);
            std::vector<int> p2(prompts[i]);
            p2.insert(p2.end(), a.begin(), a.end());
            p2.insert(p2.end(), prompts[i].end() - 6, prompts[i].end());
            turn(s, p2, b, st);
            std::printf("solo again, conversation %d: turn 1 %s, turn 2 %s\n", i, a == solo1[i] ? "same" : "DIFFERENT",
                        b == solo2[i] ? "same" : "DIFFERENT");
        }
    }
    e.use_lane(0);
    e.reset();
    // on three lanes, interleaved
    std::vector<int> lane{0, e.add_lane(), e.add_lane()};
    std::printf("lanes %d, %d, %d; %d pages free\n", lane[0], lane[1], lane[2], e.free_pages());
    const bool seq = std::getenv("LANES_SEQ") != nullptr, nospec = std::getenv("LANES_NOSPEC") != nullptr;
    std::vector<fnx::Session> ss;
    for (int i = 0; i < 3; ++i) {
        ss.emplace_back(e, 4, 7, eos, 248045, 0, lane[i]);
        ss.back().speculate = !nospec;
    }
    std::vector<std::vector<int>> got1(3), got2(3);
    std::vector<fnx::GenStats> st1(3), st2(3);
    auto second = [&](int i) {
        std::vector<int> p2(prompts[i]);
        p2.insert(p2.end(), got1[i].begin(), got1[i].end());
        p2.insert(p2.end(), prompts[i].end() - 6, prompts[i].end());
        ss[i].read(p2, st2[i]);
    };
    if (seq) {   // each lane's two turns back to back
        for (int i : {0, 1, 2}) {
            ss[i].read(prompts[i], st1[i]);
            ss[i].generate(gen, greedy, [&](int t) { got1[i].push_back(t); }, nullptr, st1[i]);
            second(i);
            ss[i].generate(gen, greedy, [&](int t) { got2[i].push_back(t); }, nullptr, st2[i]);
        }
    } else {
        for (int i = 0; i < 3; ++i) ss[i].read(prompts[i], st1[i]);   // all three read first: their states coexist
        for (int i : {1, 0, 2}) ss[i].generate(gen, greedy, [&](int t) { got1[i].push_back(t); }, nullptr, st1[i]);
        for (int i : {2, 1, 0}) second(i);
        for (int i : {0, 2, 1}) ss[i].generate(gen, greedy, [&](int t) { got2[i].push_back(t); }, nullptr, st2[i]);
    }
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        const bool a = got1[i] == solo1[i], b = got2[i] == solo2[i];
        std::printf("conversation %d (lane %d): turn 1 %s (%zu tokens), turn 2 %s (%zu tokens, reused %d of %d)\n", i, lane[i],
                    a ? "same" : "DIFFERENT", got1[i].size(), b ? "same" : "DIFFERENT", got2[i].size(), st2[i].reused, st2[i].prompt_tokens);
        ok = ok && a && b && st2[i].reused > 0;
    }
    std::printf("%s; %d pages free\n", ok ? "lanes keep their sequences apart" : "FAILED", e.free_pages());
    return ok ? 0 : 1;
}
