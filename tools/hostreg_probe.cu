// What pinning shared host memory costs a process (the lab's worker would pin the holder's expert copies): a memfd of
// GIB GiB, filled, then mapped and registered (portable, mapped: every card's context) as a second mapping would be.
//   nvcc -o bin/hostreg_probe tools/hostreg_probe.cu && bin/hostreg_probe 16
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    const size_t bytes = (size_t) (argc > 1 ? std::atof(argv[1]) : 16) * (1ull << 30);
    const int fd = memfd_create("hostreg_probe", 0);
    if (fd < 0 || ftruncate(fd, (off_t) bytes) != 0) { std::perror("memfd"); return 1; }
    auto t = std::chrono::steady_clock::now();
    void* a = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (a == MAP_FAILED) { std::perror("mmap"); return 1; }
    std::memset(a, 1, bytes);
    std::printf("filled %.1f GiB in %.2f s\n", bytes / 1073741824.0, since(t));
    // a second mapping of the same pages, as another process would have
    t = std::chrono::steady_clock::now();
    void* b = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (b == MAP_FAILED) { std::perror("mmap 2"); return 1; }
    std::printf("second mapping (populated) in %.2f s\n", since(t));
    cudaFree(nullptr);
    for (int d = 0; d < 3; ++d) { cudaSetDevice(d); cudaFree(nullptr); }
    cudaSetDevice(0);
    t = std::chrono::steady_clock::now();
    const cudaError_t e = cudaHostRegister(b, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    std::printf("cudaHostRegister: %s in %.2f s (%.1f GiB/s)\n", cudaGetErrorString(e), since(t), bytes / 1073741824.0 / since(t));
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, b, 0);
    std::printf("device pointer %s the host pointer\n", dp == b ? "equals" : "differs from");
    t = std::chrono::steady_clock::now();
    cudaHostUnregister(b);
    std::printf("unregister in %.2f s\n", since(t));
    return 0;
}
