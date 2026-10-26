// P2P bandwidth of SM stores in the ring's pattern: on every card at once, a kernel of BLOCKS x 256 threads copies
// BYTES from local memory into the next card's memory, in stores of W bytes per thread (4, 8 or 16).
//   p2p_store_bench BYTES BLOCKS W
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename V> __global__ void k_push(const V* __restrict__ src, V* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "p2p_store_bench BYTES BLOCKS W\n");
        return 2;
    }
    constexpr int NC = 3, REPS = 100;
    const size_t bytes = std::strtoull(argv[1], nullptr, 10);
    const int blocks = std::atoi(argv[2]), w = std::atoi(argv[3]);
    void* src[NC];
    void* dst[NC];
    cudaStream_t st[NC];
    for (int c = 0; c < NC; ++c) {
        ck(cudaSetDevice(c), "set");
        for (int o = 0; o < NC; ++o)
            if (o != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(o, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaMalloc(&src[c], bytes), "malloc");
        ck(cudaMalloc(&dst[c], bytes), "malloc");
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
    }
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < REPS; ++r) {
            for (int c = 0; c < NC; ++c) {
                ck(cudaSetDevice(c), "set");
                void* d = dst[(c + 1) % NC];
                if (w == 16) k_push<<<blocks, 256, 0, st[c]>>>((const uint4*) src[c], (uint4*) d, bytes / 16);
                else if (w == 8) k_push<<<blocks, 256, 0, st[c]>>>((const uint2*) src[c], (uint2*) d, bytes / 8);
                else k_push<<<blocks, 256, 0, st[c]>>>((const unsigned*) src[c], (unsigned*) d, bytes / 4);
            }
            for (int c = 0; c < NC; ++c) ck(cudaStreamSynchronize(st[c]), "sync");
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pass == 1)
            std::printf("%.2f MB, %d blocks, %d-byte stores: %.2f GB/s per card\n", bytes / 1e6, blocks, w, bytes / (s / REPS) / 1e9);
    }
    return 0;
}
