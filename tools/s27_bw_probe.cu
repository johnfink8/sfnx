// s27_bw_probe: the most a card reads from its own memory, one big buffer, int4 loads summed so nothing is elided;
// a few grid sizes. The ceiling the window's weight reading is measured against.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void k_read(const int4* __restrict__ p, size_t n, int* out) {
    int s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = __ldg(p + i);
        s += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x7fffffff) *out = s;
}

int main() {
    const size_t bytes = (size_t) 4 << 30;
    int4* p;
    int* o;
    ck(cudaMalloc(&p, bytes), "malloc");
    ck(cudaMemset(p, 1, bytes), "memset");
    ck(cudaMalloc(&o, 4), "o");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int blocks : {160, 320, 640, 1280, 2560}) {
        for (int threads : {256, 512, 1024}) {
            k_read<<<blocks, threads>>>(p, bytes / 16, o);
            cudaEventRecord(a);
            for (int r = 0; r < 5; ++r) k_read<<<blocks, threads>>>(p, bytes / 16, o);
            cudaEventRecord(b);
            ck(cudaEventSynchronize(b), "sync");
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("%5d x %4d: %.0f GB/s\n", blocks, threads, 5.0 * bytes / (ms / 1e3) / 1e9);
        }
    }
}
