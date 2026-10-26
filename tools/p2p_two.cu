// Kernel P2P stores to two peers at once: card 0 writes B bytes into card 1 and B into card 2, (a) every thread
// alternating between the destinations word by word, (b) the first half of the blocks to card 1 and the second to
// card 2, (c) one destination after the other. Time until both readers saw every word.
//   nvcc -O3 -arch=sm_70 -ccbin g++-14 -include algorithm -o bin/p2p_two tools/p2p_two.cu && bin/p2p_two
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

__global__ void k_write2(unsigned long long* d1, unsigned long long* d2, size_t n, unsigned long long tag, int mode) {
    const size_t nt = (size_t) gridDim.x * blockDim.x, tid = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (mode == 0) {
        for (size_t i = tid; i < n; i += nt) {
            *(volatile unsigned long long*) (d1 + i) = tag;
            *(volatile unsigned long long*) (d2 + i) = tag;
        }
    } else if (mode == 1) {
        const bool first = blockIdx.x < gridDim.x / 2;
        unsigned long long* d = first ? d1 : d2;
        const size_t half = (size_t) (gridDim.x / 2) * blockDim.x, t2 = first ? tid : tid - half;
        for (size_t i = t2; i < n; i += half) *(volatile unsigned long long*) (d + i) = tag;
    } else {
        for (size_t i = tid; i < n; i += nt) *(volatile unsigned long long*) (d1 + i) = tag;
        for (size_t i = tid; i < n; i += nt) *(volatile unsigned long long*) (d2 + i) = tag;
    }
}
__global__ void k_wait(const unsigned long long* src, size_t n, unsigned long long tag) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        while (*(const volatile unsigned long long*) (src + i) != tag) {
        }
}

int main() {
    std::vector<unsigned long long*> buf(3);
    std::vector<cudaStream_t> st(3);
    const size_t maxw = 1 << 18;
    for (int d = 0; d < 3; ++d) {
        CK(cudaSetDevice(d));
        for (int p = 0; p < 3; ++p)
            if (p != d) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
            }
        CK(cudaMalloc(&buf[d], maxw * 8));
        CK(cudaMemset(buf[d], 0, maxw * 8));
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    }
    unsigned long long tag = 1;
    const char* names[3] = {"alternating", "split blocks", "one then other"};
    for (size_t words : {size_t(1024), size_t(2133), size_t(4266), size_t(12800)})
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<double> us;
            for (int rep = 0; rep < 60; ++rep) {
                ++tag;
                for (int b = 1; b < 3; ++b) {
                    CK(cudaSetDevice(b));
                    k_wait<<<20, 256, 0, st[b]>>>(buf[b], words, tag);
                }
                CK(cudaSetDevice(0));
                const auto t0 = std::chrono::steady_clock::now();
                k_write2<<<20, 256, 0, st[0]>>>(buf[1], buf[2], words, tag, mode);
                for (int b = 1; b < 3; ++b) {
                    CK(cudaSetDevice(b));
                    CK(cudaStreamSynchronize(st[b]));
                }
                const auto t1 = std::chrono::steady_clock::now();
                if (rep >= 10) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(us.begin(), us.end());
            std::printf("%7zu bytes to each of two peers, %-15s %8.1f us\n", words * 8, names[mode], us[us.size() / 2]);
        }
    return 0;
}
