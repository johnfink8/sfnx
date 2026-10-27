// The prompt path's copy-engine ring plan (src/pf_ring_plan.hpp), on the CPU alone:
//   1. two and three cards: the plan is exactly the step sequence prefill.cu ran for them as hand-written code (written
//      out below), for every card;
//   2. every count card_split.hpp allows: the n cards' plans run against each other finish one sum with every part
//      holding every card's addend once, nothing read on a card before the flag that covers it, every receive slot and
//      gathered part written once, no deadlock, and no more flags than PF_RING_STEPS holds.
// Exit 0 when all pass; each failure printed.
#include <cstdio>
#include <string>
#include <vector>

#include "../src/model.hpp"   // card_split.hpp: the supported counts
#include "../src/pf_ring_plan.hpp"

using fnx::PfRingOp;
using K = PfRingOp::Kind;

static int fails = 0;
static void fail(const std::string& m) {
    std::printf("FAIL: %s\n", m.c_str());
    ++fails;
}

static PfRingOp op(K k, int part = -1, int hop = -1, int step = -1, bool fin = false) {
    PfRingOp o{k, part, hop, step};
    o.fin = fin;
    return o;
}

// prefill.cu's pf_ring_dir as written for three cards (a = g, b = g - 1, c = g + 1) and two (c = b = the other)
static std::vector<PfRingOp> written(int n, int g) {
    const int a = g, b = (g + n - 1) % n, c = (g + 1) % n;
    std::vector<PfRingOp> v = {op(K::PACK, a), op(K::SEND, a, 0), op(K::SIGNAL, -1, -1, 0)};
    if (n == 2) {
        for (const PfRingOp& o : {op(K::WAIT, -1, -1, 0), op(K::ADD, c, 0, -1, true), op(K::PASS, c), op(K::SIGNAL, -1, -1, 1),
                                  op(K::WAIT, -1, -1, 1), op(K::OUT)})
            v.push_back(o);
        return v;
    }
    for (const PfRingOp& o : {op(K::WAIT, -1, -1, 0), op(K::ADD, b, 0), op(K::SEND, b, 1), op(K::SIGNAL, -1, -1, 1),
                              op(K::WAIT, -1, -1, 1), op(K::ADD, c, 1, -1, true), op(K::PASS, c), op(K::SIGNAL, -1, -1, 2),
                              op(K::WAIT, -1, -1, 2), op(K::PASS, a), op(K::SIGNAL, -1, -1, 3), op(K::WAIT, -1, -1, 3), op(K::OUT)})
        v.push_back(o);
    return v;
}

// a buffer's contents in the simulation: which part, whose addends (bit per card), and the flag that publishes it on
// the card holding it (-1: written there by the card itself)
struct Slot {
    bool full = false;
    int part = -1;
    unsigned addends = 0;
    int step = -1;
};

static void simulate(int n) {
    const std::string tag = std::to_string(n) + " cards: ";
    const unsigned all = (1u << n) - 1;
    std::vector<std::vector<PfRingOp>> plan(n);
    for (int g = 0; g < n; ++g) plan[g] = fnx::pf_ring_plan(n, g);
    struct Card {
        size_t pc = 0;
        std::vector<bool> flag, waited;
        std::vector<Slot> in, ag;
        Slot send;
        std::vector<int> pending_in, pending_ag;   // receive slots and parts copied to the next card since the last signal
        bool done = false;
    };
    std::vector<Card> cd(n);
    const int steps = fnx::pf_ring_steps(n);
    for (Card& c : cd) {
        c.flag.assign(steps, false);
        c.waited.assign(steps, false);
        c.in.assign(n - 1, {});
        c.ag.assign(n, {});
    }
    // own writes (-1) are readable; a copy from the previous card (-2 until its signal) once its flag was waited on
    auto readable = [&](const Card& c, const Slot& s) { return s.full && (s.step == -1 || (s.step >= 0 && c.waited[(size_t) s.step])); };
    for (bool moved = true; moved;) {
        moved = false;
        for (int g = 0; g < n; ++g) {
            Card& c = cd[g];
            Card& nx = cd[(g + 1) % n];
            while (!c.done && c.pc < plan[g].size()) {
                const PfRingOp& o = plan[g][c.pc];
                const std::string at = tag + "card " + std::to_string(g) + " op " + std::to_string(c.pc) + ": ";
                if (o.k == K::WAIT) {
                    if (o.step < 0 || o.step >= steps) { fail(at + "a flag past PF_RING_STEPS"); return; }
                    if (!c.flag[(size_t) o.step]) break;
                    c.waited[(size_t) o.step] = true;
                } else if (o.k == K::PACK) {
                    c.send = {true, o.part, 1u << g, -1};
                } else if (o.k == K::ADD) {
                    const Slot& r = c.in[(size_t) o.hop];
                    if (!readable(c, r)) { fail(at + "adds a receive slot not yet covered by a flag it waited on"); return; }
                    if (r.part != o.part) fail(at + "the receive slot holds another part");
                    if (r.addends & (1u << g)) fail(at + "adds its own addend twice");
                    const Slot s{true, o.part, r.addends | (1u << g), -1};
                    if (o.fin) {
                        if (s.addends != all) fail(at + "finishes a part without every card's addend");
                        if (c.ag[(size_t) o.part].full) fail(at + "a gathered part written twice");
                        c.ag[(size_t) o.part] = s;
                    } else {
                        c.send = s;
                    }
                } else if (o.k == K::SEND) {
                    if (!c.send.full || c.send.part != o.part) { fail(at + "sends what the send buffer does not hold"); return; }
                    if (nx.in[(size_t) o.hop].full) fail(at + "a receive slot written twice");
                    Slot s = c.send;
                    s.step = -2;   // set at the signal
                    nx.in[(size_t) o.hop] = s;
                    c.pending_in.push_back(o.hop);
                } else if (o.k == K::PASS) {
                    const Slot& f = c.ag[(size_t) o.part];
                    if (!readable(c, f)) { fail(at + "passes a part not here yet"); return; }
                    if (f.addends != all) fail(at + "passes an unfinished part");
                    if (nx.ag[(size_t) o.part].full) fail(at + "a gathered part written twice on the next card");
                    Slot s = f;
                    s.step = -2;
                    nx.ag[(size_t) o.part] = s;
                    c.pending_ag.push_back(o.part);
                } else if (o.k == K::SIGNAL) {
                    if (o.step < 0 || o.step >= steps) { fail(at + "a flag past PF_RING_STEPS"); return; }
                    if (nx.flag[(size_t) o.step]) fail(at + "a flag set twice");
                    nx.flag[(size_t) o.step] = true;
                    for (const int h : c.pending_in) nx.in[(size_t) h].step = o.step;
                    for (const int k : c.pending_ag) nx.ag[(size_t) k].step = o.step;
                    c.pending_in.clear();
                    c.pending_ag.clear();
                } else if (o.k == K::OUT) {
                    for (int k = 0; k < n; ++k)
                        if (!readable(c, c.ag[(size_t) k]) || c.ag[(size_t) k].addends != all)
                            fail(at + "the sum read out without part " + std::to_string(k) + " finished and covered");
                    c.done = true;
                }
                ++c.pc;
                moved = true;
            }
        }
    }
    for (int g = 0; g < n; ++g) {
        if (!cd[g].done) fail(tag + "card " + std::to_string(g) + " stuck at op " + std::to_string(cd[g].pc));
        if (!cd[g].pending_in.empty() || !cd[g].pending_ag.empty()) fail(tag + "card " + std::to_string(g) + " copied after its last flag");
    }
}

int main() {
    for (int n : {2, 3})
        for (int g = 0; g < n; ++g)
            if (fnx::pf_ring_plan(n, g) != written(n, g))
                fail(std::to_string(n) + " cards, card " + std::to_string(g) + ": the plan is not the written sequence");
    int counts = 0;
    for (int n = 2; n <= fnx::NCARD_MAX; ++n)
        if (fnx::ncard_ok(n)) {
            simulate(n);
            ++counts;
        }
    if (fails) {
        std::printf("%d failures\n", fails);
        return 1;
    }
    std::printf("pf ring plan: two and three cards as written; %d counts (2 to %d) sum complete\n", counts, fnx::NCARD_MAX);
    return 0;
}
