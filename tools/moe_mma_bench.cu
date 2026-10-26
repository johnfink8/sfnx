// The prompt path's routed experts: the decode-arithmetic kernels (moe_gate_up_big / moe_down_big, q8_1 activations)
// against the tensor-core ones (moe_gate_up_mma / moe_down_mma, f16 activations) on random experts of the real shapes
// (512 experts; gate|up Q4_K 448 x 2560, down Q5_1 2560 x 224) with C tokens routed to 10 random experts each. Both are
// checked on sampled outputs against a host product of the f16-rounded weights and activations (in double).
//   make bin/moe_mma_bench && bin/moe_mma_bench [C...]
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "../src/kernels.hpp"

using namespace fnx;

static float time_ms(const std::function<void()>& f, int reps = 10) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

static float h2f(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}
static float f16r(float v) { return __half2float(__float2half(v)); }

static void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
// one row, dequantized and rounded to f16 as llama.cpp's dequantize_block_* does
static void deq_q4k(const uint8_t* row, int ncols, float* y) {
    for (int b = 0; b < ncols / 256; ++b) {
        const uint8_t* p = row + b * 144;
        uint16_t dh, mh;
        std::memcpy(&dh, p, 2);
        std::memcpy(&mh, p + 2, 2);
        const float d = h2f(dh), dmin = h2f(mh);
        for (int s = 0; s < 8; ++s) {
            int sc, m;
            scale_min_k4(s, p + 4, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            const uint8_t* q = p + 16 + 32 * (s / 2);
            for (int l = 0; l < 32; ++l) y[b * 256 + s * 32 + l] = f16r(d1 * ((q[l] >> (4 * (s % 2))) & 0xF) - m1);
        }
    }
}
static void deq_q51(const uint8_t* row, int ncols, float* y) {
    for (int b = 0; b < ncols / 32; ++b) {
        const uint8_t* p = row + b * 24;
        uint16_t dh, mh;
        std::memcpy(&dh, p, 2);
        std::memcpy(&mh, p + 2, 2);
        const float d = h2f(dh), m = h2f(mh);
        uint32_t qh;
        std::memcpy(&qh, p + 4, 4);
        for (int j = 0; j < 16; ++j) {
            const int x0 = (p[8 + j] & 0xF) | (((qh >> j) << 4) & 0x10);
            const int x1 = (p[8 + j] >> 4) | ((qh >> (j + 12)) & 0x10);
            y[b * 32 + j] = f16r(d * x0 + m);
            y[b * 32 + j + 16] = f16r(d * x1 + m);
        }
    }
}

struct Experts {
    DW w;
    size_t stride;
    std::vector<uint8_t> h;
};
// random blocks with small finite scales (the first two halves of every block)
static Experts make_experts(std::mt19937& rng, uint32_t type, int rows, int cols) {
    Experts e;
    const size_t bb = type == T_Q4_K ? 144 : 24, bw = type == T_Q4_K ? 256 : 32;
    const size_t rb = (size_t) cols / bw * bb;
    e.stride = (size_t) rows * rb;
    e.h.resize((size_t) NEXP * e.stride);
    for (auto& b : e.h) b = (uint8_t) rng();
    const uint16_t dh = type == T_Q4_K ? 0x2000 : 0x1c00, mh = type == T_Q4_K ? 0x1c00 : 0x9c00;
    for (size_t blk = 0; blk < e.h.size() / bb; ++blk) {
        std::memcpy(&e.h[blk * bb], &dh, 2);
        std::memcpy(&e.h[blk * bb + 2], &mh, 2);
    }
    e.w.type = type;
    e.w.n_in = cols;
    e.w.n_out = rows;
    e.w.row_bytes = rb;
    cudaMalloc(&e.w.p, e.h.size());
    cudaMemcpy(e.w.p, e.h.data(), e.h.size(), cudaMemcpyHostToDevice);
    return e;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {512, 1024, 2048};
    cudaSetDevice(2);
    std::mt19937 rng(7);
    const int ffs = 224;
    Experts gu = make_experts(rng, T_Q4_K, 2 * ffs, N), dn = make_experts(rng, T_Q5_1, N, ffs);
    for (int C : sizes) {
        const int P = C * TOPK;
        // activations: f32 rows of N, their q8_1 and f16 forms
        std::vector<float> xh((size_t) C * N);
        std::normal_distribution<float> nd(0.f, 0.02f);   // small enough that silu(gate) * up stays inside f16
        for (auto& v : xh) v = nd(rng);
        std::vector<uint16_t> x16(xh.size());
        for (size_t i = 0; i < xh.size(); ++i) {
            const __half v = __float2half(xh[i]);
            std::memcpy(&x16[i], &v, 2);
        }
        float* xf;
        void* xq;
        uint16_t* xd;
        cudaMalloc(&xf, xh.size() * 4);
        cudaMalloc(&xq, (size_t) C * q8_1_bytes(N));
        cudaMalloc(&xd, x16.size() * 2);
        cudaMemcpy(xf, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(xd, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice);
        quantize_q8_1(xf, xq, N, C, nullptr);
        // routing: 10 distinct experts per token; gate|up jobs read token rows, down jobs read pair rows
        std::vector<std::vector<int>> per(NEXP);
        std::vector<int> pair_expert(P);
        for (int t = 0; t < C; ++t) {
            std::vector<int> e(NEXP);
            for (int i = 0; i < NEXP; ++i) e[i] = i;
            std::shuffle(e.begin(), e.end(), rng);
            for (int k = 0; k < TOPK; ++k) {
                per[e[k]].push_back(t * TOPK + k);
                pair_expert[t * TOPK + k] = e[k];
            }
        }
        std::vector<BigJob> jgu, jdn;
        for (int e = 0; e < NEXP; ++e)
            for (size_t i = 0; i < per[e].size(); i += PF_JOB) {
                BigJob a{}, b{};
                a.w = b.w = e;
                a.n = b.n = (int) std::min<size_t>(PF_JOB, per[e].size() - i);
                for (int k = 0; k < a.n; ++k) {
                    a.out[k] = b.out[k] = b.y[k] = per[e][i + k];
                    a.y[k] = per[e][i + k] / TOPK;
                }
                jgu.push_back(a);
                jdn.push_back(b);
            }
        const int nj = (int) jgu.size();
        BigJob *dgu, *ddn;
        int* dnj;
        cudaMalloc(&dgu, nj * sizeof(BigJob));
        cudaMalloc(&ddn, nj * sizeof(BigJob));
        cudaMemcpy(dgu, jgu.data(), nj * sizeof(BigJob), cudaMemcpyHostToDevice);
        cudaMemcpy(ddn, jdn.data(), nj * sizeof(BigJob), cudaMemcpyHostToDevice);
        cudaMalloc(&dnj, 4);
        cudaMemcpy(dnj, &nj, 4, cudaMemcpyHostToDevice);
        float *g1, *g2, *o1, *o2;
        uint16_t* h16;
        cudaMalloc(&g1, (size_t) P * 2 * ffs * 4);
        cudaMalloc(&g2, (size_t) P * 2 * ffs * 4);
        cudaMalloc(&o1, (size_t) P * N * 4);
        cudaMalloc(&o2, (size_t) P * N * 4);
        cudaMalloc(&h16, (size_t) P * ffs * 2);
        const float tg1 = time_ms([&] { moe_gate_up_big(gu.w, gu.stride, dgu, dnj, nj, xq, g1, nullptr); });
        const float tg2 = time_ms([&] { moe_gate_up_mma(gu.w, gu.stride, dgu, dnj, nj, xd, g2, nullptr); });
        const float td1 = time_ms([&] { moe_down_big(dn.w, dn.stride, ddn, dnj, nj, g2, o1, nullptr); });
        const float ts = time_ms([&] { swiglu_h16(g2, ffs, P, h16, nullptr); });
        const float td2 = time_ms([&] { moe_down_mma(dn.w, dn.stride, ddn, dnj, nj, h16, o2, nullptr); });
        const cudaError_t err = cudaDeviceSynchronize();
        std::vector<float> hg1((size_t) P * 2 * ffs), hg2(hg1.size()), ho1((size_t) P * N), ho2(ho1.size());
        std::vector<uint16_t> hh((size_t) P * ffs);
        cudaMemcpy(hg1.data(), g1, hg1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hg2.data(), g2, hg2.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ho1.data(), o1, ho1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ho2.data(), o2, ho2.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hh.data(), h16, hh.size() * 2, cudaMemcpyDeviceToHost);
        // sampled outputs against the host product: the largest error relative to the sample's largest value
        double eg1 = 0, eg2 = 0, mg = 0, ed1 = 0, ed2 = 0, md = 0;
        std::vector<float> wr(N);
        for (int s = 0; s < 400; ++s) {
            const int pr = (int) (rng() % P), e = pair_expert[pr], t = pr / TOPK;
            {
                const int row = (int) (rng() % (2 * ffs));
                deq_q4k(gu.h.data() + (size_t) e * gu.stride + (size_t) row * gu.w.row_bytes, N, wr.data());
                double ref = 0;
                for (int k = 0; k < N; ++k) ref += (double) wr[k] * h2f(x16[(size_t) t * N + k]);
                mg = std::max(mg, std::fabs(ref));
                eg1 = std::max(eg1, std::fabs(ref - hg1[(size_t) pr * 2 * ffs + row]));
                eg2 = std::max(eg2, std::fabs(ref - hg2[(size_t) pr * 2 * ffs + row]));
            }
            {
                const int row = (int) (rng() % N);
                deq_q51(dn.h.data() + (size_t) e * dn.stride + (size_t) row * dn.w.row_bytes, ffs, wr.data());
                double ref = 0;
                for (int k = 0; k < ffs; ++k) ref += (double) wr[k] * h2f(hh[(size_t) pr * ffs + k]);
                md = std::max(md, std::fabs(ref));
                ed1 = std::max(ed1, std::fabs(ref - ho1[(size_t) pr * N + row]));
                ed2 = std::max(ed2, std::fabs(ref - ho2[(size_t) pr * N + row]));
            }
        }
        std::printf("C=%5d (%4d jobs) gate|up: q8_1 %6.2f ms (err %.2g), mma %6.2f ms (err %.2g) of %.3g | down: q8_1 %6.2f ms (err %.2g), "
                    "swiglu %.2f + mma %6.2f ms (err %.2g) of %.3g %s\n",
                    C, nj, tg1, eg1, tg2, eg2, mg, td1, ed1, ts, td2, ed2, md, err == cudaSuccess ? "" : cudaGetErrorString(err));
        cudaFree(xf);
        cudaFree(xq);
        cudaFree(xd);
        cudaFree(dgu);
        cudaFree(ddn);
        cudaFree(dnj);
        cudaFree(g1);
        cudaFree(g2);
        cudaFree(o1);
        cudaFree(o2);
        cudaFree(h16);
    }
    return 0;
}
