// Host-to-VRAM bandwidth of one card three ways: cudaMemcpyAsync from page-locked memory (the copy engine), and a
// kernel reading mapped page-locked memory in 16-byte loads (moe_stage's copy), at several grid sizes and copy sizes.
//   bin/h2d_probe [DEVICE]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void k_copy(const uint4* src, uint4* dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    ck(cudaSetDevice(dev), "set");
    const size_t max = (size_t) 256 << 20;
    void *h = nullptr, *dh = nullptr, *d = nullptr;
    ck(cudaHostAlloc(&h, max, cudaHostAllocMapped), "host");
    ck(cudaHostGetDevicePointer(&dh, h, 0), "mapped");
    ck(cudaMalloc(&d, max), "dev");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (size_t bytes : {(size_t) 1 << 20, (size_t) 8 << 20, (size_t) 64 << 20, max}) {
        auto time = [&](auto&& f) {
            f();
            cudaEventRecord(a, s);
            for (int r = 0; r < 10; ++r) f();
            cudaEventRecord(b, s);
            ck(cudaEventSynchronize(b), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return bytes * 10 / (ms * 1e6);
        };
        std::printf("%4zu MiB: memcpy %.1f GB/s", bytes >> 20, time([&] { cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s); }));
        for (int g : {80, 160, 320, 640, 1280})
            std::printf(", kernel %d blocks %.1f", g,
                        time([&] { k_copy<<<g, 256, 0, s>>>((const uint4*) dh, (uint4*) d, bytes / 16); }));
        std::printf(" GB/s\n");
    }
    return 0;
}
