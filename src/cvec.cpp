#include "cvec.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include "env.hpp"
#include "gguf.hpp"

namespace fnx {

bool cvec_from_env(CvecSpec& out) {
    const char* spec = std::getenv("FNX_CVEC");
    if (!spec) return false;
    // FILE:SCALE: a path may itself hold a colon, so the last one splits
    const std::string s(spec);
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos)
        throw std::runtime_error("FNX_CVEC: want FILE:SCALE, got '" + s + "'");
    out.path = s.substr(0, colon);
    if (out.path.empty()) throw std::runtime_error("FNX_CVEC: no file before the ':' in '" + s + "'");
    double sc = 0.0;
    if (!parse_number(s.substr(colon + 1).c_str(), sc))
        throw std::runtime_error("FNX_CVEC: the scale in '" + s + "' is not a number");
    if (sc < 0.0 || sc > 4.0)
        throw std::runtime_error("FNX_CVEC: a scale from 0 to 4 (1: the component removed, past 1 it inverts it), "
                                 "got '" + s + "'");
    out.scale = sc;
    if (const char* r = std::getenv("FNX_CVEC_RANGE")) {
        const std::string rv(r);
        const size_t dash = rv.find(':');
        if (dash == std::string::npos) throw std::runtime_error("FNX_CVEC_RANGE: want START:END, got '" + rv + "'");
        long long lo = 0, hi = 0;
        if (!parse_whole(rv.substr(0, dash).c_str(), lo) || !parse_whole(rv.substr(dash + 1).c_str(), hi) || lo > hi)
            throw std::runtime_error("FNX_CVEC_RANGE: whole numbers START:END, got '" + rv + "'");
        out.range_lo = (int) lo;
        out.range_hi = (int) hi;
    }
    return true;
}

Cvec load_cvec(const CvecSpec& spec, const std::string& hint, int n_embd, int nl) {
    Gguf g(spec.path);
    std::map<int, std::vector<float>> found;   // direction.N by layer index
    for (const auto& [name, t] : g.tensors()) {
        if (name.compare(0, 10, "direction.") != 0) continue;
        long long l = 0;
        if (!parse_whole(name.c_str() + 10, l))
            throw std::runtime_error(spec.path + ": tensor '" + name + "': the layer index is not a whole number");
        if (l < 1 || l >= nl)
            throw std::runtime_error(spec.path + ": " + name + ": this model's blocks are 0.." + std::to_string(nl - 1) +
                                     " and no direction belongs to block 0 or to the draft layer");
        if (t.type != T_F32)
            throw std::runtime_error(spec.path + ": " + name + " is not F32");
        if (t.n_dims != 1 || t.ne[0] != n_embd)
            throw std::runtime_error(spec.path + ": " + name + " is " + std::to_string(t.ne[0]) + " wide; the residuals "
                                     "here are " + std::to_string(n_embd) + " (a direction is a per-layer vector, not a "
                                     "matrix)");
        std::vector<float> v((size_t) n_embd);
        std::memcpy(v.data(), t.data, (size_t) n_embd * 4);
        double norm = 0.0;
        for (float x : v) norm += (double) x * (double) x;
        norm = std::sqrt(norm);
        // subtracting a multiple of (x . v) v removes the component along v only when v is unit: a direction off by
        // more than rounding was made some other way, and this engine will not guess at its meaning
        if (!(norm > 1.0 - 1e-3 && norm < 1.0 + 1e-3))
            throw std::runtime_error(spec.path + ": " + name + " has norm " + std::to_string(norm) + ", not unit");
        found[(int) l] = std::move(v);
    }
    if (found.empty())
        throw std::runtime_error(spec.path + ": no direction.N tensors (a control vector file holds one per layer)");
    if (g.has_key("general.architecture")) {
        const std::string arch = g.s("general.architecture");
        if (arch != "controlvector")
            throw std::runtime_error(spec.path + ": general.architecture is '" + arch + "', not controlvector");
    }
    if (g.has_key("controlvector.model_hint")) {
        const std::string h = g.s("controlvector.model_hint");
        if (h != hint)
            throw std::runtime_error(spec.path + ": its model hint is '" + h + "'; this build runs " + hint);
    }
    const int file_lo = found.begin()->first, file_hi = found.rbegin()->first;
    if (g.has_key("controlvector.layer_count")) {
        const long long lc = g.i("controlvector.layer_count");
        if (lc != (long long) found.size())
            throw std::runtime_error(spec.path + ": it says " + std::to_string(lc) + " layers and holds " +
                                     std::to_string(found.size()));
    }
    Cvec c;
    c.spec = spec;
    c.n = n_embd;
    c.nl = nl;
    c.lo = file_lo;
    c.hi = file_hi;
    if (spec.range_lo >= 0 || spec.range_hi >= 0) {   // a range narrows what the file carries; it cannot extend it
        const int lo = spec.range_lo >= 0 ? std::max(spec.range_lo, file_lo) : file_lo;
        const int hi = spec.range_hi >= 0 ? std::min(spec.range_hi, file_hi) : file_hi;
        if (lo > hi)
            throw std::runtime_error(spec.path + ": FNX_CVEC_RANGE " + std::to_string(spec.range_lo) + ":" +
                                     std::to_string(spec.range_hi) + " covers none of the file's " +
                                     std::to_string(file_lo) + ".." + std::to_string(file_hi));
        c.lo = lo;
        c.hi = hi;
    }
    c.rows.assign((size_t) nl * n_embd, 0.0f);
    c.carried.assign((size_t) nl, 0);
    for (const auto& [l, v] : found) {
        if (l < c.lo || l > c.hi) continue;
        std::memcpy(c.rows.data() + (size_t) l * n_embd, v.data(), (size_t) n_embd * 4);
        c.carried[(size_t) l] = 1;
    }
    int kept = 0;
    for (char x : c.carried) kept += x != 0;
    std::fprintf(stderr, "fnx: control vector %s: %d of %d blocks (%d..%d), scale %.4f\n", spec.path.c_str(), kept, nl,
                 c.lo, c.hi, c.spec.scale);
    return c;
}

}  // namespace fnx
