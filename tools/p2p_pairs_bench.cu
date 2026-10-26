// Copy-engine P2P bandwidth for a set of copies run at once: each argument "a>b" is a copy from card a to card b, all
// of them started together and repeated REPS times (each repetition waits for the last on every stream).
//   p2p_pairs_bench BYTES a>b [a>b ...]
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
    if (argc < 3) {
        std::fprintf(stderr, "p2p_pairs_bench BYTES a>b [a>b ...]\n");
        return 2;
    }
    constexpr int NC = 3, REPS = 100;
    const size_t bytes = std::strtoull(argv[1], nullptr, 10);
    struct Cp { int a, b; void* src; void* dst; cudaStream_t st; };
    std::vector<Cp> cps;
    for (int i = 2; i < argc; ++i) {
        Cp c{};
        if (std::sscanf(argv[i], "%d>%d", &c.a, &c.b) != 2 || c.a == c.b || c.a < 0 || c.b < 0 || c.a >= NC || c.b >= NC) {
            std::fprintf(stderr, "bad pair %s\n", argv[i]);
            return 2;
        }
        cps.push_back(c);
    }
    for (int c = 0; c < NC; ++c) {
        ck(cudaSetDevice(c), "set");
        for (int o = 0; o < NC; ++o)
            if (o != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(o, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
    }
    for (Cp& c : cps) {
        ck(cudaSetDevice(c.a), "set");
        ck(cudaMalloc(&c.src, bytes), "malloc");
        ck(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream");
        ck(cudaSetDevice(c.b), "set");
        ck(cudaMalloc(&c.dst, bytes), "malloc");
    }
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < REPS; ++r) {
            for (Cp& c : cps) {
                ck(cudaSetDevice(c.a), "set");
                ck(cudaMemcpyPeerAsync(c.dst, c.b, c.src, c.a, bytes, c.st), "copy");
            }
            for (Cp& c : cps) ck(cudaStreamSynchronize(c.st), "sync");
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pass == 1) {
            std::printf("%.2f MB:", bytes / 1e6);
            for (int i = 2; i < argc; ++i) std::printf(" %s", argv[i]);
            std::printf(": %.2f GB/s per copy\n", bytes / (s / REPS) / 1e9);
        }
    }
    return 0;
}
