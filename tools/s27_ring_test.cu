// s27_ring_test: ring_sum at the build's N on its cards, alone: card c holds x[t][i] = (c + 1) * (i % 97 + t) / 64,
// the sum must be 6x that on three cards (n(n+1)/2 x). Prints the worst relative error per T and the error bits.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/kernels.hpp"
#include "../src/s27.hpp"
#include "../src/s27_ring3.cuh"

using namespace fnx;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    std::vector<Ring> rings(NCARD);
    std::vector<float*> x(NCARD);
    std::vector<unsigned*> tag(NCARD);
    std::vector<cudaStream_t> st(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(c), "set");
        for (int p = 0; p < NCARD; ++p)
            if (p != c) {
                cudaDeviceEnablePeerAccess(p, 0);
                cudaGetLastError();
            }
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
        Ring& r = rings[c];
        r.card = c;
        ck(cudaMalloc(&r.in, ring_in_words(MAXT) * 8), "in");
        ck(cudaMemset(r.in, 0, ring_in_words(MAXT) * 8), "in0");
        ck(cudaMalloc(&r.ag, ring_ag_words(MAXT) * 8), "ag");
        ck(cudaMemset(r.ag, 0, ring_ag_words(MAXT) * 8), "ag0");
        ck(cudaMalloc(&r.err, 4), "err");
        ck(cudaMemset(r.err, 0, 4), "err0");
        ck(cudaMalloc(&x[c], (size_t) MAXT * N * 4), "x");
        ck(cudaMalloc(&tag[c], 4), "tag");
        ck(cudaDeviceSynchronize(), "sync");
    }
    for (int c = 0; c < NCARD; ++c) {
        rings[c].next_in = rings[(c + 1) % NCARD].in;
        rings[c].next_ag = rings[(c + 1) % NCARD].ag;
    }
    // 128 sums back to back on every card at once (as a window's), their time per sum
    for (int T : {1, 2, 4, 8, 16}) {
        cudaEvent_t a, b2;
        ck(cudaSetDevice(0), "set");
        cudaEventCreate(&a);
        cudaEventCreate(&b2);
        unsigned base = 1000 + T * 10;
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(c), "set");
            ck(cudaMemcpy(tag[c], &base, 4, cudaMemcpyHostToDevice), "tag up");
        }
        ck(cudaSetDevice(0), "set");
        cudaEventRecord(a, st[0]);
        for (int call = 0; call < 128; ++call)
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(c), "set");
                ring_sum(rings[c], x[c], T, tag[c], call, st[c]);
            }
        ck(cudaSetDevice(0), "set");
        cudaEventRecord(b2, st[0]);
        ck(cudaEventSynchronize(b2), "time");
        for (int c = 1; c < NCARD; ++c) {
            ck(cudaSetDevice(c), "set");
            ck(cudaStreamSynchronize(st[c]), "sync");
        }
        float ms;
        cudaEventElapsedTime(&ms, a, b2);
        std::printf("T=%2d: %.1f us a sum (128 back to back)\n", T, ms * 1000 / 128);
    }
    {   // the three-values-a-word ring (s27_ring3.cuh), the same way
        std::vector<R3> r3(NCARD);
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(c), "set");
            r3[c].card = c;
            ck(cudaMalloc(&r3[c].in, r3_in_words() * 8), "in3");
            ck(cudaMemset(r3[c].in, 0, r3_in_words() * 8), "in3 0");
            ck(cudaMalloc(&r3[c].ag, r3_ag_words() * 8), "ag3");
            ck(cudaMemset(r3[c].ag, 0, r3_ag_words() * 8), "ag3 0");
            r3[c].err = rings[c].err;
            ck(cudaDeviceSynchronize(), "sync");
        }
        for (int c = 0; c < NCARD; ++c) r3[c].next_in = r3[(c + 1) % NCARD].in, r3[c].next_ag = r3[(c + 1) % NCARD].ag;
        for (int T : {1, 2, 4, 8, 16}) {
            unsigned base = 3000 + T * 10;
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(c), "set");
                ck(cudaMemcpy(tag[c], &base, 4, cudaMemcpyHostToDevice), "tag up");
                ck(cudaDeviceSynchronize(), "sync");
            }
            cudaEvent_t a, b2;
            ck(cudaSetDevice(0), "set");
            cudaEventCreate(&a);
            cudaEventCreate(&b2);
            cudaEventRecord(a, st[0]);
            for (int call = 0; call < 128; ++call)
                for (int c = 0; c < NCARD; ++c) {
                    ck(cudaSetDevice(c), "set");
                    k_ring3<<<R3_BLOCKS, 256, 0, st[c]>>>(r3[c], x[c], T, tag[c], call);
                }
            ck(cudaSetDevice(0), "set");
            cudaEventRecord(b2, st[0]);
            ck(cudaEventSynchronize(b2), "time");
            float ms;
            cudaEventElapsedTime(&ms, a, b2);
            std::printf("ring3 T=%2d: %.1f us a sum (128 back to back)\n", T, ms * 1000 / 128);
        }
    }
    unsigned tg = 2;
    for (int T : {1, 2, 5, 16}) {
        for (int call = 0; call < 4; ++call, ++tg) {
            for (int c = 0; c < NCARD; ++c) {
                std::vector<float> h((size_t) T * N);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < N; ++i) h[(size_t) t * N + i] = (c + 1) * (float) (i % 97 + t) / 64.f;
                ck(cudaSetDevice(c), "set");
                ck(cudaMemcpy(x[c], h.data(), h.size() * 4, cudaMemcpyHostToDevice), "x up");
                ck(cudaMemcpy(tag[c], &tg, 4, cudaMemcpyHostToDevice), "tag up");
            }
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(c), "set");
                ring_sum(rings[c], x[c], T, tag[c], call, st[c]);
            }
            double worst = 0;
            unsigned err = 0;
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(c), "set");
                ck(cudaStreamSynchronize(st[c]), "sync");
                std::vector<float> h((size_t) T * N);
                ck(cudaMemcpy(h.data(), x[c], h.size() * 4, cudaMemcpyDeviceToHost), "x down");
                unsigned e = 0;
                ck(cudaMemcpy(&e, rings[c].err, 4, cudaMemcpyDeviceToHost), "err down");
                err |= e;
                const float k = NCARD * (NCARD + 1) / 2.f;
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < N; ++i) {
                        const float want = k * (float) (i % 97 + t) / 64.f;
                        worst = std::max(worst, (double) std::fabs(h[(size_t) t * N + i] - want) / (std::fabs(want) + 1e-3));
                    }
            }
            std::printf("T=%2d call %d: worst rel %.2e, err bits %u\n", T, call, worst, err);
        }
    }
}
