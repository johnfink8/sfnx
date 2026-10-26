// Copy-engine P2P in the ring's pattern: every card copies BYTES to the next card at once (cudaMemcpyPeerAsync), REPS
// times, each step waiting for the previous step on all cards (as a ring's steps do). Prints the time per step and
// the bandwidth per card's outgoing link.
//   p2p_ring_bench [BYTES...]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    constexpr int NC = 3, REPS = 200;
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::strtoull(argv[i], nullptr, 10));
    if (sizes.empty()) sizes = {262144, 1835008, 5505024};
    void* src[NC];
    void* dst[NC];
    cudaStream_t st[NC];
    cudaEvent_t done[NC];
    const size_t maxb = 5505024 * 2;
    for (int c = 0; c < NC; ++c) {
        ck(cudaSetDevice(c), "set");
        for (int o = 0; o < NC; ++o)
            if (o != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(o, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaMalloc(&src[c], maxb), "malloc");
        ck(cudaMalloc(&dst[c], maxb), "malloc");
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
        ck(cudaEventCreateWithFlags(&done[c], cudaEventDisableTiming), "event");
    }
    for (size_t bytes : sizes) {
        if (bytes > maxb) continue;
        for (int pass = 0; pass < 2; ++pass) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r) {
                for (int c = 0; c < NC; ++c) {
                    ck(cudaSetDevice(c), "set");
                    if (r > 0)   // the step waits for the previous step's copy into this card
                        ck(cudaStreamWaitEvent(st[c], done[(c + NC - 1) % NC], 0), "wait");
                    ck(cudaMemcpyPeerAsync(dst[(c + 1) % NC], (c + 1) % NC, src[c], c, bytes, st[c]), "copy");
                    ck(cudaEventRecord(done[c], st[c]), "record");
                }
            }
            for (int c = 0; c < NC; ++c) {
                ck(cudaSetDevice(c), "set");
                ck(cudaStreamSynchronize(st[c]), "sync");
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (pass == 1)
                std::printf("%8.2f MB per step: %7.1f us per step, %5.2f GB/s per link\n", bytes / 1e6, s / REPS * 1e6,
                            bytes / (s / REPS) / 1e9);
        }
    }
    return 0;
}
