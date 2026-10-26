// Can a second process map a first one's memory at the same addresses without pinning or copying it (the lab's
// holder / worker split)? The parent allocates GIB GiB of host-NUMA memory and a device allocation on card 0 with the
// CUDA virtual memory API, exports both as file descriptors, maps them at fixed addresses and fills them; a child
// (forked before any CUDA call) gets the descriptors over a socket, maps them at the same addresses, and checks them
// from a kernel on every card.
//   nvcc -o bin/vmm_probe tools/vmm_probe.cu -lcuda && bin/vmm_probe 8
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                                                    \
    do {                                                                                         \
        CUresult r_ = (x);                                                                       \
        if (r_ != CUDA_SUCCESS) {                                                                \
            const char* s_ = nullptr;                                                            \
            cuGetErrorString(r_, &s_);                                                           \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_ ? s_ : "?");      \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
static void send_fds(int sock, const int* fds, int n) {
    char buf[CMSG_SPACE(sizeof(int) * 4)] = {};
    char b = 'x';
    iovec io{&b, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = buf;
    m.msg_controllen = CMSG_SPACE(sizeof(int) * n);
    cmsghdr* c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * n);
    std::memcpy(CMSG_DATA(c), fds, sizeof(int) * n);
    if (sendmsg(sock, &m, 0) != 1) std::perror("sendmsg");
}
static void recv_fds(int sock, int* fds, int n) {
    char buf[CMSG_SPACE(sizeof(int) * 4)] = {};
    char b;
    iovec io{&b, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = buf;
    m.msg_controllen = sizeof buf;
    if (recvmsg(sock, &m, 0) != 1) std::perror("recvmsg");
    std::memcpy(fds, CMSG_DATA(CMSG_FIRSTHDR(&m)), sizeof(int) * n);
}

__global__ void k_sum(const unsigned* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) s += p[i];
    atomicAdd(out, s);
}
static unsigned long long sum_on(int dev, CUdeviceptr p, size_t bytes) {
    cudaSetDevice(dev);
    unsigned long long* d;
    cudaMalloc(&d, 8);
    cudaMemset(d, 0, 8);
    k_sum<<<160, 256>>>((const unsigned*) p, bytes / 4, d);
    unsigned long long h = 0;
    const cudaError_t e = cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) std::fprintf(stderr, "card %d: %s\n", dev, cudaGetErrorString(e));
    cudaFree(d);
    return h;
}

static const CUdeviceptr HOST_VA = 0x600000000000ull, DEV_VA = 0x640000000000ull;

static void access_all(CUdeviceptr va, size_t bytes, bool host) {
    CUmemAccessDesc a[4] = {};
    int n = 0;
    for (int d = 0; d < 3; ++d) {
        a[n].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a[n].location.id = d;
        a[n].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ++n;
    }
    if (host) {
        a[n].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        a[n].location.id = 0;
        a[n].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ++n;
    }
    CK(cuMemSetAccess(va, bytes, a, n));
}

int main(int argc, char** argv) {
    const size_t want = (size_t) (argc > 1 ? std::atof(argv[1]) : 8) * (1ull << 30);
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    const pid_t pid = fork();
    if (pid == 0) {   // the worker
        close(sv[0]);
        int fds[2];
        size_t sizes[2];
        recv_fds(sv[1], fds, 2);
        if (read(sv[1], sizes, sizeof sizes) != sizeof sizes) return 1;
        auto t = std::chrono::steady_clock::now();
        CK(cuInit(0));
        for (int d = 0; d < 3; ++d) { cudaSetDevice(d); cudaFree(nullptr); }
        std::printf("child: CUDA up in %.2f s\n", since(t));
        t = std::chrono::steady_clock::now();
        CUmemGenericAllocationHandle hh, hd;
        CK(cuMemImportFromShareableHandle(&hh, (void*) (uintptr_t) fds[0], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CK(cuMemImportFromShareableHandle(&hd, (void*) (uintptr_t) fds[1], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUdeviceptr vh = 0, vd = 0;
        CK(cuMemAddressReserve(&vh, sizes[0], 0, HOST_VA, 0));
        CK(cuMemAddressReserve(&vd, sizes[1], 0, DEV_VA, 0));
        std::printf("child: reserved at the parent's addresses: host %s, device %s\n", vh == HOST_VA ? "yes" : "NO", vd == DEV_VA ? "yes" : "NO");
        CK(cuMemMap(vh, sizes[0], 0, hh, 0));
        CK(cuMemMap(vd, sizes[1], 0, hd, 0));
        access_all(vh, sizes[0], true);
        access_all(vd, sizes[1], false);
        std::printf("child: imported and mapped %.1f GiB host + %.1f GiB device in %.3f s\n", sizes[0] / 1073741824.0,
                    sizes[1] / 1073741824.0, since(t));
        for (int d = 0; d < 3; ++d)
            std::printf("child: card %d reads host sum %llu, device sum %llu\n", d, sum_on(d, vh, sizes[0]), sum_on(d, vd, sizes[1]));
        std::printf("child: CPU reads host word %u\n", ((const unsigned*) vh)[12345]);
        return 0;
    }
    close(sv[1]);
    CK(cuInit(0));
    CUdevice dev0;
    CK(cuDeviceGet(&dev0, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev0));
    CK(cuCtxSetCurrent(ctx));
    int numa = -1;
    cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev0);
    std::printf("parent: card 0's host NUMA id %d\n", numa);
    CUmemAllocationProp ph{}, pd{};
    ph.type = pd.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ph.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    ph.location.id = 0;
    pd.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pd.location.id = 0;
    ph.requestedHandleTypes = pd.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gh = 0, gd = 0;
    CK(cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CK(cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const size_t hb = (want + gh - 1) / gh * gh, db = ((size_t) 1 << 30) / gd * gd;
    auto t = std::chrono::steady_clock::now();
    CUmemGenericAllocationHandle hh, hd;
    CK(cuMemCreate(&hh, hb, &ph, 0));
    std::printf("parent: %.1f GiB of host-NUMA memory created in %.2f s (granularity %zu)\n", hb / 1073741824.0, since(t), gh);
    CK(cuMemCreate(&hd, db, &pd, 0));
    CUdeviceptr vh = 0, vd = 0;
    CK(cuMemAddressReserve(&vh, hb, 0, HOST_VA, 0));
    CK(cuMemAddressReserve(&vd, db, 0, DEV_VA, 0));
    CK(cuMemMap(vh, hb, 0, hh, 0));
    CK(cuMemMap(vd, db, 0, hd, 0));
    for (int d = 1; d < 3; ++d) { cudaSetDevice(d); cudaFree(nullptr); }
    cudaSetDevice(0);
    access_all(vh, hb, true);
    access_all(vd, db, false);
    t = std::chrono::steady_clock::now();
    unsigned* hp = (unsigned*) vh;
    for (size_t i = 0; i < hb / 4; ++i) hp[i] = (unsigned) i;
    std::printf("parent: filled the host memory from the CPU in %.2f s\n", since(t));
    CK(cuMemsetD32(vd, 3, db / 4));
    CK(cuCtxSynchronize());
    int fds[2];
    CK(cuMemExportToShareableHandle(&fds[0], hh, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    CK(cuMemExportToShareableHandle(&fds[1], hd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    send_fds(sv[0], fds, 2);
    const size_t sizes[2] = {hb, db};
    if (write(sv[0], sizes, sizeof sizes) != sizeof sizes) return 1;
    unsigned long long ref = 0;
    for (size_t i = 0; i < hb / 4; ++i) ref += (unsigned) i;
    std::printf("parent: expected host sum %llu, device sum %llu\n", ref, (unsigned long long) 3 * (db / 4));
    int st = 0;
    waitpid(pid, &st, 0);
    std::printf("parent: child exited %d\n", WEXITSTATUS(st));
    return 0;
}
