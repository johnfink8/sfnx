// The prompt path's dense GEMM shapes (pf_gemm: out[t][r] = W[r] . X[t], f16 in, f32 out, C = 4,096 rows; logged by
// FNX_PF_GEMM_LOG) under cuBLAS's default algorithm and each tensor-op algorithm by number: is the default the fastest?
//   make bin/gemm_bench && bin/gemm_bench [C]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float time_ms(cublasHandle_t h, int algo, int rows, int cols, int C, const __half* W, const __half* X, float* out) {
    const float one = 1.f, zero = 0.f;
    auto run = [&] {
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, out,
                            CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) algo);
    };
    if (run() != CUBLAS_STATUS_SUCCESS) return -1.f;
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) run();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / 10;
}

int main(int argc, char** argv) {
    const int C = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int shapes[][2] = {{6528, 2560}, {5760, 2560}, {3712, 2560}, {5632, 2560}, {2560, 2560}, {2560, 2048}, {320, 10240},
                             {512, 2560},  {448, 2560},  {384, 2560},  {2560, 224}, {2560, 192},
                             // k and v as one GEMM (two of the rows above stacked)
                             {1024, 2560}, {896, 2560},  {768, 2560}};
    cudaSetDevice(2);
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    __half *W, *X;
    float* out;
    cudaMalloc(&W, (size_t) 6528 * 10240 * 2);
    cudaMalloc(&X, (size_t) C * 10240 * 2);
    cudaMalloc(&out, (size_t) C * 6528 * 4);
    cudaMemset(W, 0, (size_t) 6528 * 10240 * 2);
    cudaMemset(X, 0, (size_t) C * 10240 * 2);
    for (auto& s : shapes) {
        const int rows = s[0], cols = s[1];
        const float def = time_ms(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, rows, cols, C, W, X, out);
        float best = 1e9f;
        int best_algo = -1;
        std::string all;   // each algorithm's time, for picking one that suits several shapes
        for (int a = CUBLAS_GEMM_ALGO0_TENSOR_OP; a <= CUBLAS_GEMM_ALGO15_TENSOR_OP; ++a) {
            const float t = time_ms(h, a, rows, cols, C, W, X, out);
            char buf[16];
            std::snprintf(buf, sizeof buf, " %.3f", t);
            all += buf;
            if (t > 0 && t < best) best = t, best_algo = a - CUBLAS_GEMM_ALGO0_TENSOR_OP;
        }
        if (std::getenv("ALL")) std::printf("rows %5d cols %5d algos 0-15:%s\n", rows, cols, all.c_str());
        const double flop = 2.0 * rows * cols * C;
        std::printf("rows %5d cols %5d: default %6.3f ms (%5.1f TFLOPS), best algo %2d %6.3f ms (%5.1f TFLOPS) %s\n", rows, cols, def,
                    flop / def / 1e9, best_algo, best, flop / best / 1e9, best < 0.95f * def ? "<- faster" : "");
    }
    return 0;
}
