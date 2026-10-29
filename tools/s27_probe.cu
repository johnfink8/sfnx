// s27_probe: the dense 27B's decode window floor on the build's cards: every Q8_0 matrix one card reads in a window (64
// layers at card 0's slice, then its third of the output head), through q8s_gemv at T rows, each card at once in a
// CUDA graph. No sums between the cards, no recurrent or attention core: the weight reading alone.
//   s27_probe [T...]
#include <cuda_runtime.h>

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "../src/env.hpp"
#include "../src/kernels.hpp"
#include "../src/s27_attn.cuh"

using namespace fnx;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// card 0's recurrent slice (card_split.hpp): its key heads, its share of the value heads
constexpr int KH0 = khead_count(NCARD, 0), VH0 = NVH / NCARD;
constexpr int VOCAB = 248320;   // Qwen3.8-27B's output head rows
struct Mat {
    int rows, cols;
};

int main(int argc, char** argv) {
    std::vector<int> Ts;
    for (int i = 1; i < argc; ++i) {
        long long T = 0;
        if (!parse_whole(argv[i], T) || T < 1 || T > MAXT) {
            std::fprintf(stderr, "s27_probe: T is 1..%d, not '%s'\n", MAXT, argv[i]);
            return 2;
        }
        Ts.push_back((int) T);
    }
    if (Ts.empty()) Ts = {1, 2, 3, 4, 6, 8, 12, 16};
    // card 0's slice: KH0 of the key heads, VH0 of the value heads, its FF blocks, S27_NQ query heads and the S27_NKV KV
    // heads they read
    std::vector<Mat> mats;
    for (int l = 0; l < NL; ++l) {
        if ((l + 1) % 4) {
            mats.push_back({2 * KH0 * HS + 2 * VH0 * HS, N});   // qkv + z
            mats.push_back({N, VH0 * HS});                       // ssm_out
        } else {
            mats.push_back({S27_QKV, N});                        // q + gate, k, v
            mats.push_back({N, S27_NQ * HD});                    // wo
        }
        mats.push_back({2 * FFD_CARD_MAX, N});                   // gate | up
        mats.push_back({N, FFD_CARD_MAX});                       // down
    }
    mats.push_back({VOCAB / NCARD + 1, N});                      // output head
    // S27_PROBE_ONE=1: the same bytes as a few 2 GB matrices (the kernel's rate without the boundaries between the
    // window's 257; one matrix would pass q8s's 32-bit offsets)
    if (std::getenv("S27_PROBE_ONE")) {
        double b0 = 0;
        for (auto& m : mats) b0 += (double) m.rows * m.cols;
        const int rows = (int) (b0 / N), step = 400000;
        mats.clear();
        for (int r = 0; r < rows; r += step) mats.push_back({std::min(step, rows - r), N});
    }
    double bytes = 0;
    for (auto& m : mats) bytes += (double) m.rows * m.cols * (1 + 2.0 / 32);
    std::printf("per card: %zu matrices, %.2f GB\n", mats.size(), bytes / 1e9);

    constexpr int NC = NCARD;
    std::vector<std::vector<Q8S>> w(NC);
    std::vector<void*> xq(NC);
    std::vector<float*> out(NC);
    std::vector<cudaStream_t> st(NC);
    for (int c = 0; c < NC; ++c) {
        ck(cudaSetDevice(c), "set");
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
        ck(cudaMalloc(&xq[c], (size_t) MAXT * q8_1_bytes(17408)), "xq");
        ck(cudaMemset(xq[c], 0, (size_t) MAXT * q8_1_bytes(17408)), "xq0");
        {
            int8_t* tq;
            float* td;
            ck(cudaMalloc(&tq, (size_t) MAXT * 17408), "tq");
            ck(cudaMalloc(&td, (size_t) MAXT * 17408 / 32 * 4), "td");
            q8_twin(xq[c], (size_t) MAXT * q8_1_bytes(17408), tq, td);
        }
        ck(cudaMalloc(&out[c], (size_t) MAXT * 400000 * 4), "out");
        for (auto& m : mats) {
            Q8S s;
            ck(cudaMalloc(&s.q, (size_t) m.rows * m.cols), "q");
            ck(cudaMalloc(&s.d, (size_t) m.rows * m.cols / 32 * 2), "d");
            ck(cudaMemset(s.q, 1, (size_t) m.rows * m.cols), "q0");
            ck(cudaMemset(s.d, 0, (size_t) m.rows * m.cols / 32 * 2), "d0");
            w[c].push_back(s);
        }
    }
    ck(cudaSetDevice(0), "set");
    l2_prefetch(w[0][0].q, 32, st[0]);   // its sink allocated now, not inside a capture
    ck(cudaStreamSynchronize(st[0]), "prefetch warm");
    for (int T : Ts) {
        std::vector<cudaGraphExec_t> ex(NC);
        for (int c = 0; c < NC; ++c) {
            ck(cudaSetDevice(c), "set");
            cudaGraph_t g;
            ck(cudaStreamBeginCapture(st[c], cudaStreamCaptureModeThreadLocal), "capture");
            // S27_PF_MB=m: beside each matrix, the first m MB of the next pulled into L2 on a second stream
            static const double pf_mb = std::getenv("S27_PF_MB") ? std::atof(std::getenv("S27_PF_MB")) : 0.0;
            static cudaStream_t st2[NCARD];
            static cudaEvent_t fork[NCARD], join[NCARD];
            if (pf_mb > 0 && !st2[c]) {
                ck(cudaStreamCreateWithFlags(&st2[c], cudaStreamNonBlocking), "st2");
                ck(cudaEventCreateWithFlags(&fork[c], cudaEventDisableTiming), "fork");
                ck(cudaEventCreateWithFlags(&join[c], cudaEventDisableTiming), "join");
            }
            for (size_t i = 0; i < mats.size(); ++i) {
                const bool pf = pf_mb > 0 && i + 1 < mats.size();
                if (pf) {
                    ck(cudaEventRecord(fork[c], st[c]), "fork");
                    ck(cudaStreamWaitEvent(st2[c], fork[c], 0), "fork wait");
                    const size_t nb = std::min((size_t) (pf_mb * 1048576), (size_t) mats[i + 1].rows * mats[i + 1].cols);
                    l2_prefetch(w[c][i + 1].q, nb, st2[c]);
                    ck(cudaEventRecord(join[c], st2[c]), "join");
                }
                q8s_gemv(w[c][i], mats[i].rows, mats[i].cols, xq[c], out[c], mats[i].rows, T, st[c]);
                if (pf) ck(cudaStreamWaitEvent(st[c], join[c], 0), "join wait");
            }
            ck(cudaStreamEndCapture(st[c], &g), "end");
            ck(cudaGraphInstantiate(&ex[c], g, 0), "inst");
            ck(cudaGraphLaunch(ex[c], st[c]), "warm");
            ck(cudaStreamSynchronize(st[c]), "sync");
        }
        constexpr int REPS = 20;
        std::vector<float> ms(NC);
        std::vector<std::thread> th;
        for (int c = 0; c < NC; ++c)
            th.emplace_back([&, c] {
                ck(cudaSetDevice(c), "set");
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                cudaEventRecord(a, st[c]);
                for (int r = 0; r < REPS; ++r) cudaGraphLaunch(ex[c], st[c]);
                cudaEventRecord(b, st[c]);
                ck(cudaEventSynchronize(b), "time");
                cudaEventElapsedTime(&ms[c], a, b);
                ms[c] /= REPS;
            });
        for (auto& t : th) t.join();
        std::printf("T=%2d  window", T);
        for (int c = 0; c < NC; ++c) std::printf("%s%6.2f", c ? " / " : " ", ms[c]);
        std::printf(" ms  (%4.0f GB/s on card 0)\n", bytes / ms[0] / 1e6);
    }
}
