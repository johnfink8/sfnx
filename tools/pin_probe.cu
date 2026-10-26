// Does page-locked host memory cost VRAM? A context on every card, then GIB GiB of cudaHostAlloc(Portable) in 1 GiB
// pieces, each card's free memory before and after.
//   pin_probe GIB
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

int main(int argc, char** argv) {
    const int gib = argc > 1 ? std::atoi(argv[1]) : 16;
    int n = 0;
    cudaGetDeviceCount(&n);
    std::vector<size_t> before((size_t) n), after((size_t) n);
    for (int d = 0; d < n; ++d) {
        cudaSetDevice(d);
        cudaFree(nullptr);
        size_t t = 0;
        cudaMemGetInfo(&before[(size_t) d], &t);
    }
    std::vector<void*> p;
    for (int i = 0; i < gib; ++i) {
        void* q = nullptr;
        if (cudaHostAlloc(&q, (size_t) 1 << 30, cudaHostAllocPortable) != cudaSuccess) { std::printf("alloc %d failed\n", i); break; }
        p.push_back(q);
    }
    for (int d = 0; d < n; ++d) {
        cudaSetDevice(d);
        size_t t = 0;
        cudaMemGetInfo(&after[(size_t) d], &t);
        std::printf("card %d: %zu GiB page-locked: %.1f MiB less free VRAM\n", d, p.size(), (double) (before[(size_t) d] - after[(size_t) d]) / (1 << 20));
    }
    for (void* q : p) cudaFreeHost(q);
    return 0;
}
