// llama.cpp's own expert product on this box: ggml_mul_mat_id of Q4_K experts of the real gate|up shape (512 experts,
// 448 rows of 2560) with C tokens routed to 10 random experts each, on the CUDA backend; time per call.
//   mmid_bench [C...]
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {512, 1024, 2048};
    const int dev = std::getenv("DEV") ? std::atoi(std::getenv("DEV")) : 2;
    const int rows = std::getenv("ROWS") ? std::atoi(std::getenv("ROWS")) : 448;
    const int cols = std::getenv("COLS") ? std::atoi(std::getenv("COLS")) : 2560;
    const int NE = 512, K = 10;
    ggml_backend_t be = ggml_backend_cuda_init(dev);
    std::mt19937 rng(7);
    for (int C : sizes) {
        ggml_init_params ip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, cols, rows, NE);
        ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, 1, C);
        ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, C);
        ggml_tensor* y = ggml_mul_mat_id(ctx, w, x, ids);
        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<uint8_t> wh(ggml_nbytes(w));
        for (auto& b : wh) b = (uint8_t) rng();
        const uint16_t dh = 0x2000, mh = 0x1c00;
        for (size_t blk = 0; blk < wh.size() / 144; ++blk) {
            std::memcpy(&wh[blk * 144], &dh, 2);
            std::memcpy(&wh[blk * 144 + 2], &mh, 2);
        }
        ggml_backend_tensor_set(w, wh.data(), 0, wh.size());
        std::vector<float> xh((size_t) cols * C);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xh) v = nd(rng);
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);
        std::vector<int32_t> ih((size_t) K * C);
        std::vector<int> e(NE);
        for (int t = 0; t < C; ++t) {
            std::iota(e.begin(), e.end(), 0);
            std::shuffle(e.begin(), e.end(), rng);
            for (int k = 0; k < K; ++k) ih[(size_t) t * K + k] = e[k];
        }
        ggml_backend_tensor_set(ids, ih.data(), 0, ih.size() * 4);
        ggml_backend_graph_compute(be, g);
        ggml_backend_synchronize(be);
        const int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, g);
        ggml_backend_synchronize(be);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("C=%5d rows=%d cols=%d: %.2f ms (%.2f TMAC/s)\n", C, rows, cols, ms, 1e-9 * C * K * rows * (double) cols / ms);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
