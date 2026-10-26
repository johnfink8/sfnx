// Loads the model onto the 3 cards and reports each card's bytes and free VRAM.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

#include "../src/model.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: load_probe MODEL.gguf\n"); return 2; }
    const auto t0 = std::chrono::steady_clock::now();
    fnx::Model m(argv[1]);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (const auto& c : m.cards) {
        size_t fr, tot;
        cudaSetDevice(c.dev);
        cudaMemGetInfo(&fr, &tot);
        std::printf("card %d: %.2f GiB of weights, %.2f GiB free; k heads %d+%d, q heads %d+%d, kv %d+%d, ff [%d, %d), vocab [%d, %d)\n",
                    c.dev, c.bytes / 1073741824.0, fr / 1073741824.0, c.k0, c.nk, c.q0, c.nq, c.kv0, c.nkv, c.ff0, c.ff1, c.v0, c.v1);
    }
    std::printf("loaded in %.1f s\n", s);
}
