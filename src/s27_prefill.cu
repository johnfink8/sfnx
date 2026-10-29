// The 27B's prompt path (Engine::prefill for FNX_MODEL=27b): the prompt's rows through windows of MAXT rows, each kept
// whole, the draft layer over each window's rows (its K/V) when attached. The state it leaves is a run of windows'.
#include "engine.hpp"
#include "s27_impl.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace fnx {

void Engine::prefill_alloc() {}
void Engine::prefill_free() {}

void Engine::prefill(const std::vector<int>& tokens, int after) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("prefill: a window is not committed");
    if (mtp_ && after < 0) throw std::runtime_error("prefill: the draft layer needs the token after the last row");
    const size_t n = tokens.size();
    if (pos_ + (int) n > max_ctx_) throw Refused("prefill: context full");
    for (size_t i = 0; i < n; i += MAXT) {
        const size_t e = std::min(n, i + MAXT);
        const int T = (int) (e - i), p0 = pos_;
        verify(std::vector<int>(tokens.begin() + (long) i, tokens.begin() + (long) e));
        commit(T);
        if (mtp_) {
            std::vector<int> next(tokens.begin() + (long) i + 1, tokens.begin() + (long) e);
            next.push_back(e < n ? tokens[e] : after);
            draft(T, true, 0, next, p0);
        }
        if (on_prefill_chunk && (e % (size_t) pf_c_ == 0 || e == n)) on_prefill_chunk((int) e);
    }
}

}  // namespace fnx
