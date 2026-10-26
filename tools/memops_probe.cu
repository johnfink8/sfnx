// Whether the driver allows stream memory operations (cuStreamWaitValue32 / cuStreamWriteValue32) on each card, and
// how long a cross-card signal takes through them: card a writes a flag in card b's memory after a copy, card b's
// stream waits on it. Prints the round trip of REPS ping-pongs between cards 0 and 1.
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
static void cu(CUresult e, const char* what) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        std::fprintf(stderr, "%s: %s\n", what, s ? s : "?");
        std::exit(1);
    }
}

int main() {
    ck(cudaFree(nullptr), "init");
    for (int c = 0; c < 3; ++c) {
        int v1 = 0, v2 = 0;
        cu(cuDeviceGetAttribute(&v1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, c), "attr");
        cu(cuDeviceGetAttribute(&v2, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1, c), "attr nor");
        std::printf("card %d: stream mem ops %d (wait NOR %d)\n", c, v1, v2);
    }
    constexpr int REPS = 1000;
    unsigned* flag[2];
    cudaStream_t st[2];
    for (int c = 0; c < 2; ++c) {
        ck(cudaSetDevice(c), "set");
        const cudaError_t e = cudaDeviceEnablePeerAccess(1 - c, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer");
        cudaGetLastError();
        ck(cudaMalloc(&flag[c], 4), "malloc");
        ck(cudaMemset(flag[c], 0, 4), "memset");
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
    }
    for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "set"); ck(cudaDeviceSynchronize(), "sync"); }
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned r = 1; r <= REPS; ++r) {
        // card 0: signal card 1, wait for its answer; card 1: wait, answer
        ck(cudaSetDevice(0), "set");
        cu(cuStreamWriteValue32((CUstream) st[0], (CUdeviceptr) flag[1], r, 0), "write 0->1");
        cu(cuStreamWaitValue32((CUstream) st[0], (CUdeviceptr) flag[0], r, CU_STREAM_WAIT_VALUE_EQ), "wait 0");
        ck(cudaSetDevice(1), "set");
        cu(cuStreamWaitValue32((CUstream) st[1], (CUdeviceptr) flag[1], r, CU_STREAM_WAIT_VALUE_EQ), "wait 1");
        cu(cuStreamWriteValue32((CUstream) st[1], (CUdeviceptr) flag[0], r, 0), "write 1->0");
    }
    for (int c = 0; c < 2; ++c) { ck(cudaSetDevice(c), "set"); ck(cudaStreamSynchronize(st[c]), "sync"); }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("ping-pong 0 <-> 1: %.1f us per round trip\n", s / REPS * 1e6);
    return 0;
}
