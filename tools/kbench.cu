// kbench: time single fnx kernels on random inputs (CUDA events, median of many launches)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include "../src/kernels.hpp"

using namespace fnx;

template <typename T> T* dev_rand(size_t n, float scale, std::mt19937& rng) {
    std::vector<float> h(n);
    std::normal_distribution<float> d(0.f, scale);
    for (auto& v : h) v = d(rng);
    T* p;
    cudaMalloc(&p, n * sizeof(float));
    cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return p;
}

// L2 is emptied before every timed launch (a 24 MB read; a write would leave dirty lines to write back during the
// kernel): in the engine each layer's weights and state arrive cold, and a kernel that chains dependent loads is far
// slower then than when it repeats on a warm cache
template <typename F> float time_us(F f, int reps = 200) {
    static void* flush = nullptr;
    constexpr size_t FLUSH = 24u << 20;
    if (!flush) {
        cudaMalloc(&flush, FLUSH);
        cudaMemset(flush, 0, FLUSH);
    }
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        l2_prefetch(flush, FLUSH, st);
        cudaEventRecord(a, st);
        f(st);
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        ts.push_back(ms * 1000.f);
    }
    std::sort(ts.begin(), ts.end());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::printf("error: %s\n", cudaGetErrorString(e));
    return ts[ts.size() / 2];
}

// the routing on the host, from logits rows of NEXP + 1 on the device: top 10 per row (ties: the lower index), jobs as
// the engine builds them (one per distinct expert, first-pick order); fills topi/topw/jobs/njobs on the device
static void route_from_logits(const float* logits, int T, int* topi, float* topw, Job* gu, Job* dn, int* njobs) {
    std::vector<float> l((size_t) T * (NEXP + 1));
    cudaMemcpy(l.data(), logits, l.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<int> ti(T * TOPK);
    std::vector<float> tw(T * TOPK);
    for (int t = 0; t < T; ++t) {
        std::vector<int> idx(NEXP);
        for (int i = 0; i < NEXP; ++i) idx[i] = i;
        const float* r = &l[(size_t) t * (NEXP + 1)];
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
        float z = 0.f;
        for (int k = 0; k < TOPK; ++k) z += std::exp(r[idx[k]] - r[idx[0]]);
        for (int k = 0; k < TOPK; ++k) {
            ti[t * TOPK + k] = idx[k];
            tw[t * TOPK + k] = std::exp(r[idx[k]] - r[idx[0]]) / z;
        }
    }
    std::vector<Job> g, d;
    for (int p = 0; p < T * TOPK; ++p) {
        size_t j = 0;
        while (j < g.size() && g[j].w != ti[p]) ++j;
        if (j == g.size()) {
            g.push_back(Job{});
            d.push_back(Job{});
            g[j].w = d[j].w = ti[p];
        }
        g[j].y[g[j].n] = p / TOPK;
        g[j].out[g[j].n] = d[j].y[d[j].n] = d[j].out[d[j].n] = p;
        ++g[j].n;
        ++d[j].n;
    }
    const int nj[2] = {(int) g.size(), 0};
    cudaMemcpy(topi, ti.data(), ti.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(topw, tw.data(), tw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(gu, g.data(), g.size() * sizeof(Job), cudaMemcpyHostToDevice);
    cudaMemcpy(dn, d.data(), d.size() * sizeof(Job), cudaMemcpyHostToDevice);
    cudaMemcpy(njobs, nj, sizeof nj, cudaMemcpyHostToDevice);
}

int main() {
    std::mt19937 rng(1);
    cudaSetDevice(2);
    hc_prepare();
    for (int T : {1, 2, 3, 4, 5, 8, 16}) {
        {   // dense: card 2's qkvz (5 key heads: 5120 rows) and ssm_out (2560 x 1920), Q8_0; the f32 router and ab
            void* xq;
            cudaMalloc(&xq, (size_t) MAXT * q8_1_bytes(N));
            cudaMemset(xq, 0, (size_t) MAXT * q8_1_bytes(N));
            Job jh{};
            jh.n = T;
            for (int t = 0; t < T; ++t) jh.y[t] = jh.out[t] = t;
            Job* job;
            cudaMalloc(&job, sizeof(Job));
            cudaMemcpy(job, &jh, sizeof jh, cudaMemcpyHostToDevice);
            float* out;
            cudaMalloc(&out, (size_t) MAXT * 82774 * 4);
            for (auto [rows, cols, label] : {std::tuple{5120, N, "qkvz"}, std::tuple{N, 1920, "ssm_out"}, std::tuple{82774, N, "head"}}) {
                DW w;
                w.type = T_Q8_0;
                w.n_in = cols;
                w.n_out = rows;
                w.row_bytes = (size_t) (cols / 32) * 34;
                cudaMalloc(&w.p, w.row_bytes * rows);
                cudaMemset(w.p, 0, w.row_bytes * rows);
                const double mb = w.row_bytes * rows / 1e6;
                const float us = time_us([&](cudaStream_t s) { mmvq(w, 0, job, nullptr, 1, xq, out, rows, s); });
                std::printf("T=%d mmvq %-8s %7.1f us (%5.1f MB, %4.0f GB/s)\n", T, label, us, mb, mb / us * 1e3);
                cudaFree(w.p);
                Q8S s;
                cudaMalloc(&s.q, (size_t) rows * cols);
                cudaMalloc(&s.d, (size_t) rows * cols / 32 * 2);
                cudaMemset(s.q, 1, (size_t) rows * cols);
                cudaMemset(s.d, 0, (size_t) rows * cols / 32 * 2);
                const double mbs = rows * (double) cols * (1 + 2.0 / 32) / 1e6;
                const float us2 = time_us([&](cudaStream_t st) { q8s_gemv(s, rows, cols, xq, out, rows, T, st); });
                std::printf("T=%d q8s  %-8s %7.1f us (%5.1f MB, %4.0f GB/s)\n", T, label, us2, mbs, mbs / us2 * 1e3);
                cudaFree(s.q);
                cudaFree(s.d);
            }
            for (auto [rows, label] : {std::pair{NEXP + 1, "router"}, std::pair{30, "ab"}}) {
                float* W = dev_rand<float>((size_t) rows * N, 0.02f, rng);
                float* x = dev_rand<float>((size_t) MAXT * N, 1.f, rng);
                const double mb = (double) rows * N * 4 / 1e6;
                const float us = time_us([&](cudaStream_t s) { gemv_f32(W, x, out, rows, N, T, s); });
                std::printf("T=%d gemv %-8s %7.1f us (%5.1f MB, %4.0f GB/s)\n", T, label, us, mb, mb / us * 1e3);
                cudaFree(W);
                cudaFree(x);
            }
        }
        {   // the hyper-connection read with the previous write folded in
            float* R = dev_rand<float>((size_t) MAXT * D, 1.f, rng);
            float* bo = dev_rand<float>((size_t) MAXT * N, 1.f, rng);
            float* inj = dev_rand<float>(MAXT * HC, 1.f, rng);
            float* wn = dev_rand<float>(D, 1.f, rng);
            Q8S wd, wu;
            cudaMalloc(&wd.q, (size_t) LR * D);
            cudaMalloc(&wd.d, (size_t) LR * D / 32 * 2);
            cudaMalloc(&wu.q, (size_t) LR * D);
            cudaMalloc(&wu.d, (size_t) LR * D / 32 * 2);
            cudaMemset(wd.q, 1, (size_t) LR * D);
            cudaMemset(wd.d, 0, (size_t) LR * D / 32 * 2);
            cudaMemset(wu.q, 1, (size_t) LR * D);
            cudaMemset(wu.d, 0, (size_t) LR * D / 32 * 2);
            uint16_t* wi;
            cudaMalloc(&wi, (size_t) HC * D * 2);
            cudaMemset(wi, 0, (size_t) HC * D * 2);
            float *mixed, *injo, *scratch;
            cudaMalloc(&mixed, (size_t) MAXT * N * 4);
            cudaMalloc(&injo, MAXT * HC * 4);
            cudaMalloc(&scratch, hc_scratch_floats(MAXT) * 4);
            cudaMemset(scratch, 0, hc_scratch_floats(MAXT) * 4);
            unsigned* herr;
            cudaMalloc(&herr, 4);
            cudaMemset(herr, 0, 4);
            void* q8;
            cudaMalloc(&q8, (size_t) MAXT * q8_1_bytes(N));
            const double mb = (2.0 * LR * D * (1 + 2.0 / 32) + HC * D * 2) / 1e6;
            const float us = time_us([&](cudaStream_t s) {
                hc_read(R, bo, inj, wn, wd, wu, wi, 1e-6f, mixed, injo, scratch, herr, T, s, q8);
            });
            std::printf("T=%d hc_read       %7.1f us (%5.1f MB, %4.0f GB/s)\n", T, us, mb, mb / us * 1e3);
            for (const int g : {72, 64, 40}) {   // as a card of g SMs runs it (a 72-SM PG500-216)
                hc_prepare(g);
                cudaMemset(scratch, 0, hc_scratch_floats(MAXT) * 4);   // its barrier counter, in generations of 80
                const float ug = time_us([&](cudaStream_t s) {
                    hc_read(R, bo, inj, wn, wd, wu, wi, 1e-6f, mixed, injo, scratch, herr, T, s, q8);
                });
                std::printf("T=%d hc_read on %d %7.1f us (x%.2f)\n", T, g, ug, ug / us);
            }
            hc_prepare();
            cudaMemset(scratch, 0, hc_scratch_floats(MAXT) * 4);
            {   // the same with the down weights pulled into L2 first (not timed)
                cudaStream_t st;
                cudaStreamCreate(&st);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                std::vector<float> ts;
                for (int i = 0; i < 200; ++i) {
                    l2_prefetch(wd.q, (size_t) LR * D, st);
                    l2_prefetch(wd.d, (size_t) LR * D / 32 * 2, st);
                    cudaEventRecord(e0, st);
                    hc_read(R, bo, inj, wn, wd, wu, wi, 1e-6f, mixed, injo, scratch, herr, T, st, q8);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    ts.push_back(ms * 1000.f);
                }
                std::sort(ts.begin(), ts.end());
                std::printf("T=%d hc_read, down weights in L2: %7.1f us\n", T, ts[ts.size() / 2]);
            }
            unsigned herr_h = 0;
            cudaMemcpy(&herr_h, herr, 4, cudaMemcpyDeviceToHost);
            if (herr_h & 4u) throw std::runtime_error("kbench: an hc read's grid barrier timed out (its times above are not the read's)");
        }
        {   // recurrent core, a card's share: 6 key heads, 16 value heads (card 0's map)
            const int nk = 6, nv = 16, ch = (2 * nk + nv) * HS;
            int* vmap;
            {
                const int hv[16] = {0, 1, 2, 3, 4, 5 | 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
                int h[16];
                for (int i = 0; i < 16; ++i) h[i] = hv[i] | (i < 6 ? 1 << 8 : 0);
                cudaMalloc(&vmap, sizeof h);
                cudaMemcpy(vmap, h, sizeof h, cudaMemcpyHostToDevice);
            }
            float* qkvz = dev_rand<float>((size_t) MAXT * (ch + nv * HS), 1.f, rng);
            float* ab = dev_rand<float>((size_t) MAXT * 2 * nv, 1.f, rng);
            float* hist = dev_rand<float>((size_t) 2 * 3 * ch, 1.f, rng);
            float* convw = dev_rand<float>((size_t) ch * 4, 0.3f, rng);
            float* dt = dev_rand<float>(nv, 1.f, rng);
            float* a = dev_rand<float>(nv, 1.f, rng);
            float* nw = dev_rand<float>(HS, 1.f, rng);
            float* S = dev_rand<float>((size_t) nv * HS * HS, 0.1f, rng);
            void* yq;
            cudaMalloc(&yq, (size_t) MAXT * q8_1_bytes(nv * HS));
            int* ctl;
            cudaMalloc(&ctl, 8);
            cudaMemset(ctl, 0, 8);
            float* gs;
            cudaMalloc(&gs, gdn_scratch_floats(nv) * 4);
            cudaMemset(gs, 0, gdn_scratch_floats(nv) * 4);
            float *snap, *hsnap;
            cudaMalloc(&snap, (size_t) (MAXT - 1) * nv * HS * HS * 4);
            cudaMalloc(&hsnap, (size_t) MAXT * 3 * ch * 4);
            std::printf("T=%d gdn_core   %7.1f us\n", T, time_us([&](cudaStream_t s) {
                            gdn_core(qkvz, ab, hist, convw, dt, a, nw, S, yq, ctl, gs, snap, hsnap, nk, nv, vmap, T, 1e-6f, s);
                        }));
        }
        {   // routing
            float* logits = dev_rand<float>((size_t) MAXT * (NEXP + 1), 1.f, rng);
            int *topi, *njobs;
            float* topw;
            Job *gu, *dn;
            cudaMalloc(&topi, MAXT * TOPK * 4);
            cudaMalloc(&topw, MAXT * TOPK * 4);
            cudaMalloc(&njobs, 8);
            cudaMemset(njobs, 0, 8);
            cudaMalloc(&gu, (MAXT * TOPK + 1) * sizeof(Job));
            cudaMalloc(&dn, (MAXT * TOPK + 1) * sizeof(Job));
            float* Wr = dev_rand<float>((size_t) (NEXP + 1) * N, 0.02f, rng);
            float* xr = dev_rand<float>((size_t) MAXT * N, 1.f, rng);
            std::printf("T=%d router_route %7.1f us\n", T,
                        time_us([&](cudaStream_t s) { router_route(Wr, xr, logits, T, topi, topw, gu, dn, njobs, s); }));
        }
        {   // experts, card 2's slice (ff 192): routing rows that share experts the way decode rows do
            const int ffs = 192;
            std::vector<float> lh((size_t) MAXT * (NEXP + 1));
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> base(NEXP + 1);
            for (auto& v : base) v = nd(rng);
            for (int t = 0; t < MAXT; ++t)
                for (int i = 0; i <= NEXP; ++i) lh[(size_t) t * (NEXP + 1) + i] = base[i] + 0.6f * nd(rng);
            float* logits;
            cudaMalloc(&logits, lh.size() * 4);
            cudaMemcpy(logits, lh.data(), lh.size() * 4, cudaMemcpyHostToDevice);
            int *topi, *njobs;
            float* topw;
            Job *jg, *jd;
            cudaMalloc(&topi, MAXT * TOPK * 4);
            cudaMalloc(&topw, MAXT * TOPK * 4);
            cudaMalloc(&njobs, 8);
            cudaMemset(njobs, 0, 8);
            cudaMalloc(&jg, (MAXT * TOPK + 1) * sizeof(Job));
            cudaMalloc(&jd, (MAXT * TOPK + 1) * sizeof(Job));
            route_from_logits(logits, T, topi, topw, jg, jd, njobs);
            int nj = 0;
            cudaMemcpy(&nj, njobs, 4, cudaMemcpyDeviceToHost);
            auto dw = [&](uint32_t type, int n_in, int n_out, size_t rb, int mats, size_t& stride) {
                DW w;
                w.type = type;
                w.n_in = n_in;
                w.n_out = n_out;
                w.row_bytes = rb;
                stride = (size_t) n_out * rb;
                cudaMalloc(&w.p, stride * mats);
                cudaMemset(w.p, 0, stride * mats);
                return w;
            };
            size_t sgu, sdn;
            const DW gu_e = dw(T_Q4_K, N, 2 * ffs, (N / 256) * 144, NEXP, sgu);
            const DW dn_e = dw(T_Q5_1, ffs, N, (ffs / 32) * 24, NEXP, sdn);
            void* xq;
            cudaMalloc(&xq, (size_t) MAXT * q8_1_bytes(N));
            cudaMemset(xq, 0, (size_t) MAXT * q8_1_bytes(N));
            float *g, *part;
            cudaMalloc(&g, (size_t) (MAXT * TOPK + MAXT) * 2 * ffs * 4);
            cudaMemset(g, 0, (size_t) (MAXT * TOPK + MAXT) * 2 * ffs * 4);
            cudaMalloc(&part, (size_t) (MAXT * TOPK + MAXT) * N * 4);
            const int P = T * TOPK;
            const double mb_gu = nj * sgu / 1e6, mb_dn = nj * sdn / 1e6;
            const float tg = time_us([&](cudaStream_t s) { moe_gate_up(gu_e, sgu, jg, njobs, P, xq, g, s); });
            const float td = time_us([&](cudaStream_t s) { moe_down(dn_e, sdn, jd, njobs, P, g, part, s); });
            std::printf("T=%d moe %2d experts: gate_up %6.1f us (%5.1f MB, %4.0f GB/s)  down %6.1f us (%5.1f MB, %4.0f GB/s)\n", T, nj,
                        tg, mb_gu, mb_gu / tg * 1e3, td, mb_dn, mb_dn / td * 1e3);
        }
    }
}
