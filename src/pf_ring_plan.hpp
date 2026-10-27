// The prompt path's chunk sum on the copy engines (prefill.cu, pf_ring_dir) as a list of steps for card g of n: a
// reduce-scatter then an all-gather round the ring, each card sending only to the next. The rows are cut in n parts
// (part k: rows [k C / n, (k + 1) C / n)); card g sends its own addend of part g to g + 1, each card after adds its own
// and sends it on (hop s carries part g - s), so card g finishes part g + 1 after n - 1 hops; the finished parts then go
// round n - 1 times more (forwarded n - 2 times). Two cards: halves, one hop each way. Plain C++ (no CUDA), so
// tools/pf_ring_plan_test.cpp checks it on the CPU: the same steps as the two- and three-card sequences, and every
// count's sum complete with nothing read before its flag.
#pragma once

#include <vector>

namespace fnx {

struct PfRingOp {
    enum Kind {
        PACK,      // this card's addend of part `part` (x's rows) to the send buffer as f16
        ADD,       // receive slot `hop` + this card's addend of `part`: into the send buffer, or (fin) the gathered
                   // rows' place for that part (finished)
        SEND,      // the send buffer to the next card's receive slot `hop` (part's rows)
        PASS,      // the finished part `part` from the gathered rows to the same place on the next card
        SIGNAL,    // flag `step` on the next card (after this card's copies so far)
        WAIT,      // until this card's flag `step` is set
        OUT        // the gathered rows (every part finished) back into x
    };
    Kind k;
    int part = -1, hop = -1, step = -1;
    bool fin = false;
    bool operator==(const PfRingOp& o) const { return k == o.k && part == o.part && hop == o.hop && step == o.step && fin == o.fin; }
};

// flags a set holds: a sum's 2 n - 2 steps, four at least (two cards' set laid out as three cards')
constexpr int pf_ring_steps(int n) { return 2 * n - 2 > 4 ? 2 * n - 2 : 4; }

inline std::vector<PfRingOp> pf_ring_plan(int n, int g) {
    using K = PfRingOp;
    auto mod = [n](int v) { return ((v % n) + n) % n; };
    std::vector<PfRingOp> ops;
    ops.push_back({K::PACK, g});
    ops.push_back({K::SEND, g, 0});
    ops.push_back({K::SIGNAL, -1, -1, 0});
    for (int s = 1; s < n - 1; ++s) {   // hop s: part g - s with s addends; add mine, on to g + 1
        ops.push_back({K::WAIT, -1, -1, s - 1});
        ops.push_back({K::ADD, mod(g - s), s - 1});
        ops.push_back({K::SEND, mod(g - s), s});
        ops.push_back({K::SIGNAL, -1, -1, s});
    }
    {   // part g + 1 with the other cards' addends: add mine, finished; keep it, send it on
        const int c = mod(g + 1);
        ops.push_back({K::WAIT, -1, -1, n - 2});
        PfRingOp add{K::ADD, c, n - 2};
        add.fin = true;
        ops.push_back(add);
        ops.push_back({K::PASS, c});
        ops.push_back({K::SIGNAL, -1, -1, n - 1});
    }
    for (int f = 1; f < n - 1; ++f) {   // part g - f + 1 arrived finished from g - 1: forward it
        ops.push_back({K::WAIT, -1, -1, n - 2 + f});
        ops.push_back({K::PASS, mod(g - f + 1)});
        ops.push_back({K::SIGNAL, -1, -1, n - 1 + f});
    }
    ops.push_back({K::WAIT, -1, -1, 2 * n - 3});   // the last part arrived: all here
    ops.push_back({K::OUT});
    return ops;
}

}  // namespace fnx
