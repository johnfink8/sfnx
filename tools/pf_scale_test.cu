// Check the prompt dense activation scale against a float reference, including
// a captured graph and two GEMMs reusing the same converted K/V activations.
#include "../src/pf_common.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    using namespace fnx;
    try {
        const int card = argc > 1 ? std::stoi(argv[1]) : 0;
        ckc(cudaSetDevice(card), "select card");
        constexpr int cols = 128, rows = 8, tokens = 4, ldo = 12;
        std::vector<float> x(cols * tokens, 1.0f);
        std::vector<float> initial(ldo * tokens, -913.0f);
        std::vector<__half> weight(cols * rows, __float2half(0.125f));
        float *dx = nullptr, *scales = nullptr, *a = nullptr, *b = nullptr;
        __half *xh = nullptr, *w = nullptr;
        ckc(cudaMalloc(&dx, x.size() * sizeof(float)), "allocate input");
        ckc(cudaMalloc(&xh, x.size() * sizeof(__half)), "allocate converted input");
        ckc(cudaMalloc(&w, weight.size() * sizeof(__half)), "allocate weights");
        ckc(cudaMalloc(&scales, tokens * sizeof(float)), "allocate scales");
        ckc(cudaMalloc(&a, ldo * tokens * sizeof(float)), "allocate output A");
        ckc(cudaMalloc(&b, ldo * tokens * sizeof(float)), "allocate output B");
        ckc(cudaMemcpy(w, weight.data(), weight.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy weights");
        cudaStream_t stream;
        ckc(cudaStreamCreate(&stream), "create stream");
        cublasHandle_t blas;
        ckb(cublasCreate(&blas), "create handle");
        ckb(cublasSetStream(blas, stream), "set stream");
        for (const bool graph : {false, true}) {
            for (const int magnitude : {0, 1, 2}) {
                const float outlier = magnitude == 0 ? 17.0f : magnitude == 1 ? 117144.703f : 3.0e30f;
                x[0] = 17.0f;
                x[cols] = outlier;
                x[cols * 2] = -outlier;
                x[cols * 3] = 0.0f;
                ckc(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy input");
                ckc(cudaMemcpyAsync(a, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "clear output A");
                ckc(cudaMemcpyAsync(b, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "clear output B");
                ckc(cudaStreamSynchronize(stream), "input ready");
                if (graph) ckc(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
                pf_f32_f16_scaled(dx, cols, tokens, xh, scales, stream);
                pf_scaled_gemm(blas, w, rows, cols, xh, tokens, a, ldo, scales, stream);
                pf_scaled_gemm(blas, w, rows, cols, xh, tokens, b, ldo, scales, stream);
                if (graph) {
                    cudaGraph_t captured;
                    cudaGraphExec_t executable;
                    ckc(cudaStreamEndCapture(stream, &captured), "end capture");
                    ckc(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "instantiate graph");
                    ckc(cudaGraphLaunch(executable, stream), "launch graph");
                    ckc(cudaStreamSynchronize(stream), "graph done");
                    ckc(cudaGraphExecDestroy(executable), "destroy graph executable");
                    ckc(cudaGraphDestroy(captured), "destroy graph");
                }
                std::vector<float> out_a(ldo * tokens), out_b(ldo * tokens);
                float scale[tokens] = {};
                ckc(cudaMemcpyAsync(out_a.data(), a, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy output A");
                ckc(cudaMemcpyAsync(out_b.data(), b, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy output B");
                ckc(cudaMemcpyAsync(scale, scales, sizeof(scale), cudaMemcpyDeviceToHost, stream), "copy scales");
                ckc(cudaStreamSynchronize(stream), "outputs ready");
                if (scale[0] != 1.0f || scale[3] != 1.0f || scale[1] != scale[2] ||
                    (magnitude == 0 && scale[1] != 1.0f) || (magnitude == 1 && scale[1] != 4.0f) ||
                    (magnitude == 2 && scale[1] <= 4.0f))
                    throw std::runtime_error("unexpected per-row scale");
                for (int t = 0; t < tokens; ++t) {
                    double ref = 0;
                    for (int c = 0; c < cols; ++c) ref += x[t * cols + c] * 0.125;
                    for (int r = 0; r < rows; ++r) {
                        for (float v : {out_a[t * ldo + r], out_b[t * ldo + r]}) {
                            if (!std::isfinite(v) || std::abs(v - ref) > 0.001 * (1.0 + std::abs(ref)))
                                throw std::runtime_error("scaled result disagrees with float reference");
                        }
                    }
                    for (int r = rows; r < ldo; ++r)
                        if (out_a[t * ldo + r] != -913.0f || out_b[t * ldo + r] != -913.0f)
                            throw std::runtime_error("output padding was modified");
                }
                cublasPointerMode_t mode;
                ckb(cublasGetPointerMode(blas, &mode), "check pointer mode");
                if (mode != CUBLAS_POINTER_MODE_HOST) throw std::runtime_error("pointer mode not restored");
                std::printf("PASS card=%d graph=%d magnitude=%d scale=%.3g\n", card, graph, magnitude, scale[1]);
            }
        }
        ckb(cublasDestroy(blas), "destroy handle");
        ckc(cudaStreamDestroy(stream), "destroy stream");
        cudaFree(b); cudaFree(a); cudaFree(scales); cudaFree(w); cudaFree(xh); cudaFree(dx);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
