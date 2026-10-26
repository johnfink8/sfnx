// topk_test: fnx::top_tokens (the sampler's candidates) against a CPU reference on rows built to tie: every logit
// equal, a tie at the threshold wider than the kernel's first pass holds, ties straddling its chunks, penalties that
// make ties, and plain random rows. Exit 0 when every row's TOPK_N (logit, id) pairs match exactly. Needs one GPU.
//
//   bin/topk_test
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../src/kernels.hpp"

using namespace fnx;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "topk_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Pen {
    std::vector<int> tok, cnt;
    float repeat = 1.f, freq = 0.f, present = 0.f;
};

// the kernel's penalties and order, on the host: logit descending, id ascending on ties
std::vector<std::pair<float, int>> reference(std::vector<float> row, int v0, const Pen& p) {
    for (size_t i = 0; i < p.tok.size(); ++i) {
        const int t = p.tok[i] - v0;
        if (t < 0 || t >= (int) row.size() || p.cnt[i] <= 0) continue;
        float l = row[(size_t) t];
        l = l <= 0.f ? l * p.repeat : l / p.repeat;
        row[(size_t) t] = l - (float) p.cnt[i] * p.freq - p.present;
    }
    std::vector<int> idx(row.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + TOPK_N, idx.end(),
                      [&](int a, int b) { return row[(size_t) a] > row[(size_t) b] || (row[(size_t) a] == row[(size_t) b] && a < b); });
    std::vector<std::pair<float, int>> out;
    for (int k = 0; k < TOPK_N; ++k) out.emplace_back(row[(size_t) idx[(size_t) k]], v0 + idx[(size_t) k]);
    return out;
}

bool run(const std::string& name, const std::vector<std::vector<float>>& rows, int v0, const std::vector<Pen>& pens) {
    const int T = (int) rows.size(), n = (int) rows[0].size();
    int stride = 1;
    for (const Pen& p : pens) stride = std::max(stride, (int) p.tok.size());
    std::vector<float> h_l((size_t) T * n), h_par((size_t) T * 3);
    std::vector<int> h_tok((size_t) T * stride), h_cnt((size_t) T * stride), h_n(T);
    for (int t = 0; t < T; ++t) {
        std::copy(rows[(size_t) t].begin(), rows[(size_t) t].end(), h_l.begin() + (size_t) t * n);
        const Pen& p = pens[(size_t) t];
        h_par[(size_t) t * 3] = p.repeat, h_par[(size_t) t * 3 + 1] = p.freq, h_par[(size_t) t * 3 + 2] = p.present;
        h_n[(size_t) t] = (int) p.tok.size();
        for (size_t i = 0; i < p.tok.size(); ++i) h_tok[(size_t) t * stride + i] = p.tok[i], h_cnt[(size_t) t * stride + i] = p.cnt[i];
    }
    float *d_l, *d_par, *d_v;
    int *d_tok, *d_cnt, *d_n, *d_i;
    unsigned* d_err;
    ck(cudaMalloc(&d_l, h_l.size() * 4), "malloc");
    ck(cudaMalloc(&d_par, h_par.size() * 4), "malloc");
    ck(cudaMalloc(&d_tok, h_tok.size() * 4), "malloc");
    ck(cudaMalloc(&d_cnt, h_cnt.size() * 4), "malloc");
    ck(cudaMalloc(&d_n, h_n.size() * 4), "malloc");
    ck(cudaMalloc(&d_v, (size_t) T * TOPK_N * 4), "malloc");
    ck(cudaMalloc(&d_i, (size_t) T * TOPK_N * 4), "malloc");
    ck(cudaMalloc(&d_err, 4), "malloc");
    ck(cudaMemcpy(d_l, h_l.data(), h_l.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_par, h_par.data(), h_par.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_cnt, h_cnt.data(), h_cnt.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_n, h_n.data(), h_n.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemset(d_err, 0, 4), "memset");
    top_tokens(d_l, n, T, v0, d_tok, d_cnt, d_n, stride, d_par, d_v, d_i, d_err, nullptr);
    ck(cudaDeviceSynchronize(), "top_tokens");
    std::vector<float> got_v((size_t) T * TOPK_N);
    std::vector<int> got_i((size_t) T * TOPK_N);
    unsigned err = 0;
    ck(cudaMemcpy(got_v.data(), d_v, got_v.size() * 4, cudaMemcpyDeviceToHost), "copy");
    ck(cudaMemcpy(got_i.data(), d_i, got_i.size() * 4, cudaMemcpyDeviceToHost), "copy");
    ck(cudaMemcpy(&err, d_err, 4, cudaMemcpyDeviceToHost), "copy");
    for (void* p : {(void*) d_l, (void*) d_par, (void*) d_tok, (void*) d_cnt, (void*) d_n, (void*) d_v, (void*) d_i, (void*) d_err}) cudaFree(p);
    bool ok = err == 0;
    if (err) std::printf("%s: err bits %#x\n", name.c_str(), err);
    for (int t = 0; t < T && ok; ++t) {
        const auto want = reference(rows[(size_t) t], v0, pens[(size_t) t]);
        for (int k = 0; k < TOPK_N; ++k) {
            const size_t at = (size_t) t * TOPK_N + k;
            if (got_v[at] != want[(size_t) k].first || got_i[at] != want[(size_t) k].second) {
                std::printf("%s: row %d, candidate %d: got (%g, %d), want (%g, %d)\n", name.c_str(), t, k, got_v[at], got_i[at],
                            want[(size_t) k].first, want[(size_t) k].second);
                ok = false;
                break;
            }
        }
    }
    std::printf("%-34s %s\n", name.c_str(), ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    constexpr int N = 82944, V0 = 82944;   // a card's slice of the vocabulary and where it starts
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    auto random_row = [&] {
        std::vector<float> r(N);
        for (float& v : r) v = nd(rng);
        return r;
    };
    bool ok = true;
    const Pen none;
    // plain rows: the first pass only
    ok &= run("random", {random_row(), random_row(), random_row()}, V0, {none, none, none});
    // every logit the same: the lowest TOPK_N ids
    ok &= run("all equal", {std::vector<float>(N, 1.5f)}, V0, {none});
    ok &= run("all zero", {std::vector<float>(N, 0.f)}, V0, {none});
    // 10 above the threshold, 300 at it scattered over the row (more than the first pass's 256)
    {
        std::vector<float> r(N, -5.f);
        for (int k = 0; k < 10; ++k) r[(size_t) (k * 7919 % N)] = 9.f + k;
        for (int k = 0; k < 300; ++k) r[(size_t) ((k * 104729 + 13) % N)] = 2.f;
        ok &= run("300 tied at the threshold", {r}, V0, {none});
    }
    // 63 above, 5000 tied at it: one candidate from the ties
    {
        std::vector<float> r(N, -1.f);
        for (int k = 0; k < 5000; ++k) r[(size_t) ((k * 31 + 5) % N)] = 0.25f;
        for (int k = 0; k < 63; ++k) r[(size_t) ((k * 1299709 + 3) % N)] = 4.f + 0.01f * k;
        ok &= run("63 above, 5000 tied", {r}, V0, {none});
    }
    // ties around the 1,024-logit chunks' edges, the threshold reached part way through a chunk
    {
        std::vector<float> r(N, -2.f);
        for (int c = 0; c < 40; ++c)
            for (int d = -4; d < 4; ++d) r[(size_t) (1024 * (c + 1) + d)] = 1.f;
        ok &= run("ties straddling chunks", {r}, V0, {none});
    }
    // a penalty that pulls 400 tokens down onto a tie wider than the first pass
    {
        std::vector<float> r(N, -3.f);
        Pen p;
        p.present = 1.f;
        for (int k = 0; k < 400; ++k) {
            r[(size_t) (k * 97)] = 3.f;
            p.tok.push_back(V0 + k * 97);
            p.cnt.push_back(1);
        }
        for (int k = 0; k < 300; ++k) r[(size_t) (k * 97 + 50)] = 2.f;
        ok &= run("penalties making a wide tie", {r}, V0, {p});
    }
    // several rows at once, each its own kind
    ok &= run("mixed rows", {random_row(), std::vector<float>(N, 7.f), random_row()}, V0, {none, none, none});
    std::printf(ok ? "topk_test: all ok\n" : "topk_test: FAILED\n");
    return ok ? 0 : 1;
}
