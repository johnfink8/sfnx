// s27_ring_test: the 27B's sums across the cards at the build's N, alone. Times kernels.cu's ring_sum (the 27B's ring
// with FNX_RING_F32=1) and s27_ring3.cuh's k_ring3 (three fp16 a word, its default), 128 sums back to back; then checks both:
// card c holds x[t][i] = (c + 1) * (i % 97 + t) / 64, the sum must be n(n+1)/2 x on n cards. Exits 1 when a sum is off
// by more than fp16's rounding of it or sets an error bit.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "../src/kernels.hpp"
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
    std::vector<R3> r3(NCARD);
    std::vector<float*> x(NCARD);
    std::vector<unsigned*> tag(NCARD);
    std::vector<cudaStream_t> st(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(c), "set");
        for (int p = 0; p < NCARD; ++p)
            if (p != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
        Ring& r = rings[c];
        r.card = c;
        ck(cudaMalloc(&r.in, ring_in_words(MAXT) * 8), "in");
        ck(cudaMalloc(&r.ag, ring_ag_words(MAXT) * 8), "ag");
        ck(cudaMalloc(&r.err, 4), "err");
        r3[c].card = c;
        ck(cudaMalloc(&r3[c].in, r3_in_words() * 8), "in3");
        ck(cudaMalloc(&r3[c].ag, r3_ag_words() * 8), "ag3");
        r3[c].err = r.err;
        ck(cudaMalloc(&x[c], (size_t) MAXT * N * 4), "x");
        ck(cudaMemset(x[c], 0, (size_t) MAXT * N * 4), "x0");   // the timed sums add zeros
        ck(cudaMalloc(&tag[c], 4), "tag");
    }
    for (int c = 0; c < NCARD; ++c) {
        rings[c].next_in = rings[(c + 1) % NCARD].in;
        rings[c].next_ag = rings[(c + 1) % NCARD].ag;
        r3[c].next_in = r3[(c + 1) % NCARD].in;
        r3[c].next_ag = r3[(c + 1) % NCARD].ag;
    }
    // every buffer and error word zeroed with all cards idle, so the tags below start over from the lowest
    auto clear = [&] {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(c), "set");
            ck(cudaDeviceSynchronize(), "sync");
            ck(cudaMemset(rings[c].in, 0, ring_in_words(MAXT) * 8), "in0");
            ck(cudaMemset(rings[c].ag, 0, ring_ag_words(MAXT) * 8), "ag0");
            ck(cudaMemset(r3[c].in, 0, r3_in_words() * 8), "in3 0");
            ck(cudaMemset(r3[c].ag, 0, r3_ag_words() * 8), "ag3 0");
            ck(cudaMemset(rings[c].err, 0, 4), "err0");
            ck(cudaDeviceSynchronize(), "sync");
        }
    };
    using Sum = std::function<void(int c, int T, int call)>;
    const Sum fp32 = [&](int c, int T, int call) { ring_sum(rings[c], x[c], T, tag[c], call, st[c]); };
    const Sum fp16x3 = [&](int c, int T, int call) { k_ring3<<<R3_BLOCKS, 256, 0, st[c]>>>(r3[c], x[c], T, tag[c], call); };
    auto set_tag = [&](unsigned t) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(c), "set");
            // on the card's own stream and waited for: a pageable cudaMemcpy can return before its DMA lands, and the
            // sums' non-blocking streams do not wait for the legacy stream it runs on
            ck(cudaMemcpyAsync(tag[c], &t, 4, cudaMemcpyHostToDevice, st[c]), "tag up");
            ck(cudaStreamSynchronize(st[c]), "tag up");
        }
    };
    int fails = 0;
    for (const auto& [name, sum] : {std::pair{"ring_sum", fp32}, std::pair{"k_ring3", fp16x3}}) {
        clear();
        unsigned tg = 1;
        for (int T : {1, 2, 4, 8, 16}) {   // the time
            set_tag(tg++);
            cudaEvent_t a, b;
            ck(cudaSetDevice(0), "set");
            ck(cudaEventCreate(&a), "event");
            ck(cudaEventCreate(&b), "event");
            ck(cudaEventRecord(a, st[0]), "record");
            for (int call = 0; call < 128; ++call)
                for (int c = 0; c < NCARD; ++c) {
                    ck(cudaSetDevice(c), "set");
                    sum(c, T, call);
                }
            ck(cudaSetDevice(0), "set");
            ck(cudaEventRecord(b, st[0]), "record");
            for (int c = 0; c < NCARD; ++c) {
                ck(cudaSetDevice(c), "set");
                ck(cudaStreamSynchronize(st[c]), "sync");
            }
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
            std::printf("%s T=%2d: %.1f us a sum (128 back to back)\n", name, T, ms * 1000 / 128);
        }
        clear();
        for (int T : {1, 2, 5, 16}) {   // the sums
            for (int call = 0; call < 4; ++call) {
                set_tag(tg++);
                for (int c = 0; c < NCARD; ++c) {
                    std::vector<float> h((size_t) T * N);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < N; ++i) h[(size_t) t * N + i] = (c + 1) * (float) (i % 97 + t) / 64.f;
                    ck(cudaSetDevice(c), "set");
                    ck(cudaMemcpyAsync(x[c], h.data(), h.size() * 4, cudaMemcpyHostToDevice, st[c]), "x up");
                    ck(cudaStreamSynchronize(st[c]), "x up");
                }
                for (int c = 0; c < NCARD; ++c) {
                    ck(cudaSetDevice(c), "set");
                    sum(c, T, call);
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
                // each of the n - 1 steps rounds a partial to fp16 (2^-11 relative), and the gather the sum once more
                const double bar = NCARD * std::ldexp(1.0, -11);
                const bool ok = worst <= bar && !err;
                if (!ok) ++fails;
                std::printf("%s T=%2d call %d: worst rel %.2e (bar %.2e), err bits %u%s\n", name, T, call, worst, bar, err,
                            ok ? "" : "  FAIL");
            }
        }
    }
    if (fails) {
        std::printf("s27_ring_test: %d sums failed\n", fails);
        return 1;
    }
    std::printf("s27_ring_test: ok\n");
}
