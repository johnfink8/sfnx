// P2P stores from a kernel, per directed pair of cards: the time for blocks of a kernel on card a to write B bytes
// into card b's memory with 8-byte stores (as the ring's LL words) and then have one 8-byte flag seen by a kernel on
// card b (the round trip of a hop). Peer access is enabled both ways.
//   nvcc -O3 -arch=sm_70 -o bin/p2p_store tools/p2p_store.cu && bin/p2p_store
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

// writes n words (value = tag) into dst, then the last block's... every thread's words carry the tag, so the reader
// spins on each word: the time until the reader has seen all n words
__global__ void k_write(unsigned long long* dst, size_t n, unsigned long long tag) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        *(volatile unsigned long long*) (dst + i) = tag;
}
__global__ void k_wait(const unsigned long long* src, size_t n, unsigned long long tag, unsigned* fails) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const long long t0 = clock64();
        while (*(const volatile unsigned long long*) (src + i) != tag)
            if (clock64() - t0 > 2000000000LL) {
                atomicAdd(fails, 1u);
                break;
            }
    }
}

int main() {
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    std::vector<unsigned long long*> buf(n_dev);
    std::vector<unsigned*> fails(n_dev);
    std::vector<cudaStream_t> st(n_dev);
    const size_t maxw = 1 << 20;
    for (int d = 0; d < n_dev; ++d) {
        CK(cudaSetDevice(d));
        for (int p = 0; p < n_dev; ++p)
            if (p != d) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
            }
        CK(cudaMalloc(&buf[d], maxw * 8));
        CK(cudaMemset(buf[d], 0, maxw * 8));
        CK(cudaMalloc(&fails[d], 4));
        CK(cudaMemset(fails[d], 0, 4));
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    }
    unsigned long long tag = 1;
    std::printf("from -> to   bytes   us (write until the reader saw every word; median of 50)\n");
    for (size_t words : {size_t(1024), size_t(3200), size_t(12800), size_t(65536)})
        for (int a = 0; a < n_dev; ++a)
            for (int b = 0; b < n_dev; ++b) {
                if (a == b) continue;
                std::vector<double> us;
                for (int rep = 0; rep < 60; ++rep) {
                    ++tag;
                    CK(cudaSetDevice(b));
                    k_wait<<<20, 256, 0, st[b]>>>(buf[b], words, tag, fails[b]);
                    CK(cudaSetDevice(a));
                    CK(cudaStreamSynchronize(st[a]));
                    const auto t0 = std::chrono::steady_clock::now();
                    k_write<<<20, 256, 0, st[a]>>>(buf[b], words, tag);
                    CK(cudaSetDevice(b));
                    CK(cudaStreamSynchronize(st[b]));
                    const auto t1 = std::chrono::steady_clock::now();
                    if (rep >= 10) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                std::sort(us.begin(), us.end());
                std::printf("  %d -> %d  %7zu  %8.1f\n", a, b, words * 8, us[us.size() / 2]);
            }
    unsigned f = 0;
    for (int d = 0; d < n_dev; ++d) {
        unsigned h;
        CK(cudaSetDevice(d));
        CK(cudaMemcpy(&h, fails[d], 4, cudaMemcpyDeviceToHost));
        f += h;
    }
    std::printf("timeouts: %u\n", f);
    return 0;
}
