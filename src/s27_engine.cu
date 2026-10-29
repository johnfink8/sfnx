// The Engine (engine.hpp) for Qwen3.8-27B (make FNX_MODEL=27b): dense, split across the cards as Flash-Next is. Every
// layer on every card: the recurrent heads, the FFN by FF width (ffd_lo), the attention by heads (s27_attn.cuh: each
// card's K/V for its own two KV heads, no exchange), the vocabulary; the residual and the norms copied, two sums a
// layer (s27_ring3.cuh). The draft layer is the GGUF's nextn block (CardW::mtp), its head the Mtp's rows.
// Lanes, pages, states, the sampler's candidates and the window graphs work as engine.cu's.
#include "engine.hpp"
#include "env.hpp"
#include "peer_probe.hpp"
#include "s27_impl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fnx {

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
float h2f(uint16_t h) {
    __half_raw r;
    r.x = h;
    return __half2float(__half(r));
}
void install_graph(CardBuf::GraphSlot& s, cudaGraph_t g, const std::vector<int>& key) {
    if (s.exec) {
        cudaGraphExecUpdateResultInfo info{};
        if (cudaGraphExecUpdate(s.exec, g, &info) == cudaSuccess) {
            s.key = key;
            ck(cudaGraphDestroy(g), "graph destroy");
            return;
        }
        cudaGetLastError();   // a different shape: a new executable in its place
        ck(cudaGraphExecDestroy(s.exec), "graph exec destroy");
        s.exec = nullptr;
    }
    ck(cudaGraphInstantiate(&s.exec, g, 0), "instantiate");
    s.key = key;
    ck(cudaGraphDestroy(g), "graph destroy");
}

template <typename T> T* dalloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, n * sizeof(T));
    if (e != cudaSuccess) {
        cudaGetLastError();
        int dev = -1;
        size_t free_b = 0, total_b = 0;
        cudaGetDevice(&dev);
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error("cudaMalloc of " + std::to_string((n * sizeof(T)) >> 20) + " MiB on card " + std::to_string(dev) + ": " +
                                 cudaGetErrorString(e) + " (" + std::to_string(free_b >> 20) + " MiB free)");
    }
    ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    own_dev(p);
    return (T*) p;
}

long host_available_mib() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

template <typename T> T* halloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("cudaHostAlloc of " + std::to_string((n * sizeof(T)) >> 20) + " MiB: " + cudaGetErrorString(e) + " (" +
                                 std::to_string(host_available_mib()) + " MiB available)");
    }
    own_pinned(p);
    return (T*) p;
}

__device__ float block_sum1024(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}
// R[t] += pend[t] (a half's sum, when pend is not null), then y[t] = rms(R[t]) * w: the next half's input; with tq, also
// quantized as quantize_q8_1 does into the split twin (tq, td) of a q8_1 buffer of N-wide rows, which is all q8s reads
__global__ void k_add_norm(float* R, const float* pend, const float* w, float* y, int8_t* tq, float* td, float eps) {
    __shared__ float sh[32];
    float* r = R + (size_t) blockIdx.x * N;
    float q = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = r[i];
        if (pend) r[i] = x += pend[(size_t) blockIdx.x * N + i];
        q += x * x;
    }
    const float s = rsqrtf(block_sum1024(q, sh) / (float) N + eps);
    const int lane = threadIdx.x & 31;
    for (int bl = threadIdx.x >> 5; bl < N / 32; bl += blockDim.x >> 5) {
        const int i = bl * 32 + lane;
        const float v = r[i] * s * w[i];
        y[(size_t) blockIdx.x * N + i] = v;
        if (!tq) continue;
        float amax = fabsf(v);
        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        tq[(size_t) blockIdx.x * N + i] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (lane == 0) td[(size_t) blockIdx.x * (N / 32) + bl] = __half2float(__float2half(d));
    }
}
void add_norm(CardBuf& b, float* R, const float* pend, const float* w, int T, float eps, bool twin) {
    k_add_norm<<<T, 1024, 0, b.st>>>(R, pend, w, b.mixed, twin ? b.xtq : nullptr, twin ? b.xtd : nullptr, eps);
    ck(cudaGetLastError(), "add_norm");
}
// out[t] = [rms(e[t]) * we | rms(h[t]) * wh]: the draft layer's input before its projection
__global__ void k_norm_cat(const float* e, const float* h, const float* we, const float* wh, float* out, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x[2] = {e + (size_t) t * N, h + (size_t) t * N};
    const float* w[2] = {we, wh};
    for (int s = 0; s < 2; ++s) {
        float q = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) q += x[s][i] * x[s][i];
        const float r = rsqrtf(block_sum1024(q, sh) / (float) N + eps);
        for (int i = threadIdx.x; i < N; i += blockDim.x) out[(size_t) t * 2 * N + s * N + i] = x[s][i] * r * w[s][i];
    }
}

// x summed across the cards (every card ends with the sum), three fp16 values a word (s27_ring3.cuh). On the box GPU 0
// sits on the other socket from 1 and 2: storing straight to both peers (2 hops, not the ring's 4) measured slower at
// every window size, 28.9 against 19.4 ms at 4 rows, from the traffic it sends across the sockets; the ring with two
// values a word (kernels.cu's) took 19.4 ms against 18.5 at 4 rows, 25.9 against 23.8 at 8
void sum_cards(CardBuf& b, float* x, int T, const int* ctl, int call) {
    ring_call_check(call, "sum_cards");
    k_ring3<<<R3_BLOCKS, 256, 0, b.st>>>(b.r3, x, T, (const unsigned*) (ctl + 1), call);
    ck(cudaGetLastError(), "ring3");
}

// out[t][r] = w[r] . y[t] for a Q8_0 matrix (the 27B's are all Q8_0: the loader checks). q8s beat three kernels written
// for these shapes (activations in registers with split columns; a lane on 64 columns; activations in shared memory,
// whole rows a warp): at 4 rows 13.8 ms of matrices a window against 14.9, 21.9 and ~15.2. The rest of the gap to the
// same bytes in a few big matrices (11.5 ms at 4 rows) is at the boundaries between the window's 257 dependent
// matrices, and stayed there as one persistent launch with a grid barrier between matrices (13.8 ms), with each block
// prefetching its next rows into L2 before the barrier (15.1), with an L2 prefetch on a second stream (14.8 to 15.7)
// and with capped persistent grids (13.2 to 16.8)
void mv(CardBuf& b, const DW& w, const void* y, float* out, int T) { q8s_gemv(w.s, w.n_out, w.n_in, y, out, w.n_out, T, b.st); }

// the rows of a window: one sequence's at *ctl + t on the current lane's pages (lr empty), or a batched window's
struct Rows {
    const int* ctl;
    LaneRows lr;
};

// an attention layer's mixer half on one card: the last half's sum folded into R (pend) and the norm, the card's q heads
// and their two KV heads projected, the window's K/V stored, attention over every position, the card's heads through
// wo, summed across the cards into b.bo (folded into R by the next half)
void attn_half(CardBuf& b, const LayerW& L, float* R, bool pend, int slot, const Rows& rows, int q0, int kv0, int call, int T, float eps,
               const Rope& rope) {
    add_norm(b, R, pend ? b.bo : nullptr, L.attn_norm, T, eps, true);
    if (L.wq.n_out != S27_QKV) throw std::runtime_error("27b: an attention layer's q|k|v is not S27_QKV rows");
    mv(b, L.wq, b.xq, b.qkv, T);
    k_s27_prep<<<dim3(S27_NQ + 2 * S27_NKV, T), HD, 0, b.st>>>(b.qkv, S27_QKV, L.q_norm, L.k_norm, b.pool, slot, rows.ctl, b.pt, rows.lr,
                                                                eps, rope);
    k_s27_attn<<<dim3(S27_ASPLIT, T, S27_NQ), 128, 0, b.st>>>(b.qkv, S27_QKV, b.pool, slot, rows.ctl, b.pt, rows.lr, kv0, q0, b.apart);
    k_s27_merge<<<dim3(T, S27_NQ), HD, 0, b.st>>>(b.qkv, S27_QKV, b.apart, b.att);
    ck(cudaGetLastError(), "attention");
    quantize_q8_1(b.att, b.xq, S27_NQ * HD, T, b.st);
    mv(b, L.wo, b.xq, b.bo, T);
    sum_cards(b, b.bo, T, rows.ctl, call);
}

void ffn27(CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl, int call, int T, float eps) {
    const int ffs = w.fd1 - w.fd0;
    add_norm(b, R, b.bo, L.ffn_norm, T, eps, true);
    mv(b, L.ffn_gu, b.xq, b.gu, T);
    swiglu_stacked_q8_1(b.gu, b.hq, ffs, T, b.st);
    mv(b, L.ffn_down, b.hq, b.bo, T);
    sum_cards(b, b.bo, T, ctl, call);
}
}  // namespace

std::vector<std::function<void()>>& owned() {
    static std::vector<std::function<void()>> list;
    return list;
}

Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    rope_.base = m_.rope_base;
    long long env_orig = 0;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        env_orig = env_int("FNX_YARN_ORIG_CTX", 0);
        if (env_orig <= 0 || env_orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(env_orig));
    }
    const RopeChoice rc = choose_rope(m_.yarn_factor, m_.yarn_orig, m_.ctx_train, std::getenv("FNX_ROPE_SCALE") ? rope_scale() : -1.0, env_orig);
    if (max_ctx > rc.limit)
        throw std::runtime_error("context " + std::to_string(max_ctx) + " is past the " + std::to_string(rc.limit) + " the rotary covers (" +
                                 (rc.S ? "trained " + std::to_string(rc.orig) + " x YaRN " + std::to_string(rc.S)
                                       : "the model's trained length; FNX_ROPE_SCALE=S (--rope-scale S) covers trained x S") +
                                 ")");
    if (rc.S) {
        rope_ = yarn_rope(m_.rope_base, rc.orig, rc.S, NROT);
        std::fprintf(stderr, "27b: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
    }
    if (std::getenv("FNX_KV8") && env_int("FNX_KV8", 0) != 0)
        throw std::runtime_error("FNX_KV8: the 27B keeps its K/V in f16 (llama.cpp's default); int8 K/V is Flash-Next's");
    const long long pf_rows = env_int("FNX_PF_ROWS", pf_c_);
    if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
        throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " +
                                 std::to_string(pf_rows));
    pf_c_ = (int) pf_rows;
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        CardBuf* b = new CardBuf();
        b->dev = w.dev;
        ck(cudaSetDevice(w.dev), "cudaSetDevice");
        for (int p = 0; p < NCARD; ++p)
            if (p != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaStreamCreateWithFlags(&b->st, cudaStreamNonBlocking), "stream");
        own_stream(b->st);
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        if (w.nq != S27_NQ || w.nkv != S27_NKV || w.q0 != c * S27_NQ || w.kv0 != w.q0 / (NQH / NKVH))
            throw std::runtime_error("27b: card " + std::to_string(c) + "'s attention heads are not the split s27_attn.cuh runs");
        const int ffs = w.fd1 - w.fd0;
        b->R = dalloc<float>((size_t) MAXT * N);
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo = dalloc<float>((size_t) MAXT * N);
        b->hfin = dalloc<float>((size_t) MAXT * N);
        b->xq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(S27_XQ_COLS));
        b->xtq = dalloc<int8_t>((size_t) MAXT * S27_XQ_COLS);
        b->xtd = dalloc<float>((size_t) MAXT * S27_XQ_COLS / 32);
        q8_twin(b->xq, (size_t) MAXT * q8_1_bytes(S27_XQ_COLS), b->xtq, b->xtd);
        b->hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        q8_twin(b->hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
        for (const void* q : {(const void*) b->xq, (const void*) b->hq}) owned().push_back([q] { q8_untwin(q); });
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qkv = dalloc<float>((size_t) MAXT * S27_QKV);
        b->att = dalloc<float>((size_t) MAXT * S27_NQ * HD);
        b->apart = dalloc<float>((size_t) MAXT * S27_NQ * S27_ASPLIT * (HD + 2));
        b->gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        b->err = dalloc<unsigned>(1);
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l)
            if (!is_attn(l)) {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        b->ctl = dalloc<int>(2);
        b->r3.card = c;
        b->r3.in = dalloc<unsigned long long>(r3_in_words());
        b->r3.ag = dalloc<unsigned long long>(r3_ag_words());
        b->r3.err = b->err;
        cb_.push_back(b);
    }
    {   // every link carries a probe's stores before the ring's (src/peer_probe.hpp)
        std::vector<int> devs;
        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);
        for (const std::string& line : peer_report(probe_peer_stores(devs), NCARD)) std::fprintf(stderr, "%s\n", line.c_str());
    }
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->r3.next_in = cb_[(c + 1) % NCARD]->r3.in;
        cb_[c]->r3.next_ag = cb_[(c + 1) % NCARD]->r3.ag;
    }
    // the prompt path's buffers now, so the pool below is sized from what is left after them
    prefill_alloc();
    // the K/V pool: as many pages as every card's free VRAM holds beside FNX_KV_RESERVE_MIB (default 1024: the window
    // graphs captured at first use, cuBLAS's workspaces, the lanes' page tables and rotary tables), at least the whole
    // context and the extra lanes' blocks. A page is 8.9 MB a card (34.8 KB a position: 16 attention layers and the
    // draft layer, two KV heads of 256 f16 for K and V)
    const long long reserve_mib = env_int("FNX_KV_RESERVE_MIB", 1024);
    if (reserve_mib < 0) throw std::runtime_error("FNX_KV_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
    const size_t page_bytes = S27_PAGE * sizeof(__half);
    const int ctx_pages = max_ctx / S27_PP + 2;
    int n_pages = INT32_MAX;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
        const size_t reserve = (size_t) reserve_mib << 20;
        const int fit = fr > reserve ? (int) ((fr - reserve) / page_bytes) : 0;
        if (fit < ctx_pages + extra_pages)
            throw std::runtime_error("27b: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: " +
                                     std::to_string(ctx_pages + extra_pages) + " K/V pages (" + std::to_string(max_ctx) + " positions and " +
                                     std::to_string(std::max(lanes, 1) - 1) + " more lanes' states) and " + std::to_string(reserve_mib) +
                                     " MiB in reserve (FNX_KV_RESERVE_MIB) do not fit");
        n_pages = std::min(n_pages, fit);
    }
    if (std::getenv("FNX_KV_PAGES")) {
        const long long want = env_int("FNX_KV_PAGES", 0);
        if (want < ctx_pages + extra_pages || want > n_pages)
            throw std::runtime_error("FNX_KV_PAGES: want " + std::to_string(ctx_pages + extra_pages) + " to " + std::to_string(n_pages));
        n_pages = (int) want;
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        b->n_pages = n_pages;
        b->pool = dalloc<__half>((size_t) n_pages * S27_PAGE);
        b->pt = dalloc<int>((size_t) n_pages);
        LaneDev L;
        L.pt = b->pt;
        L.S = b->S;
        L.chist = b->chist;
        L.ctl = dalloc<int>(2);
        b->lanes.push_back(L);
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
        std::fprintf(stderr, "27b: card %d: %d K/V pages (%d positions), %zu MiB free after them\n", c, n_pages, n_pages * S27_PP, fr >> 20);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    for (int i = n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
}

// the ring words' 16-bit tags (s27_ring3.cuh) wrap every 256 tags: every card idle, every buffer zeroed each time the
// tags cross a multiple of 128, so no word left in a slot is 256 tags old
void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemset(b->r3.in, 0, r3_in_words() * 8), "ring clear");
        ck(cudaMemset(b->r3.ag, 0, r3_ag_words() * 8), "ring clear");
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
}

struct CardPool {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable go, done;
    const std::function<void(int)>* job = nullptr;
    unsigned gen = 0;
    int left = 0;
    bool quit = false;
    std::exception_ptr err[NCARD];
    int dev0 = 0;
    explicit CardPool(const std::vector<int>& devs) : dev0(devs[0]) {
        for (int c = 1; c < NCARD; ++c)
            th.emplace_back([this, c, d = devs[(size_t) c]] {
                const cudaError_t set = cudaSetDevice(d);
                unsigned seen = 0;
                for (;;) {
                    const std::function<void(int)>* f;
                    {
                        std::unique_lock<std::mutex> l(mu);
                        go.wait(l, [&] { return quit || gen != seen; });
                        if (quit) return;
                        seen = gen;
                        f = job;
                    }
                    try {
                        if (set != cudaSuccess) throw std::runtime_error(std::string("card pool: cudaSetDevice: ") + cudaGetErrorString(set));
                        (*f)(c);
                    } catch (...) { err[c] = std::current_exception(); }
                    std::lock_guard<std::mutex> l(mu);
                    if (--left == 0) done.notify_one();
                }
            });
    }
    ~CardPool() {
        {
            std::lock_guard<std::mutex> l(mu);
            quit = true;
        }
        go.notify_all();
        for (auto& t : th) t.join();
    }
    void run(const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> l(mu);
            job = &f;
            left = NCARD - 1;
            ++gen;
            for (auto& e : err) e = nullptr;
        }
        go.notify_all();
        std::exception_ptr e0;
        try {
            ck(cudaSetDevice(dev0), "cudaSetDevice");
            f(0);
        } catch (...) { e0 = std::current_exception(); }
        {
            std::unique_lock<std::mutex> l(mu);
            done.wait(l, [&] { return left == 0; });
        }
        if (e0) std::rethrow_exception(e0);
        for (auto& e : err)
            if (e) std::rethrow_exception(e);
    }
};

void Engine::on_cards(const std::function<void(int)>& f) {
    if (!pool_) {
        std::vector<int> devs;
        for (CardBuf* b : cb_) devs.push_back(b->dev);
        pool_ = std::make_shared<CardPool>(devs);
    }
    pool_->run(f);
}

void Engine::tags_room(unsigned n) {
    if (tag_max_ + n + 1u >= TAG_END) {
        clear_rings();
        std::fprintf(stderr, "27b: ring tags reached %u: tags start again\n", tag_max_);
        tag_max_ = 1;
        return;
    }
    if ((tag_max_ + n) / 128u != tag_max_ / 128u) clear_rings();
}

unsigned Engine::ring_tag() {
    tags_room(1);
    return ++tag_max_;
}

unsigned Engine::take_tags(int n) {
    tags_room((unsigned) n + 1u);
    unsigned base = tag_max_ + 1u;
    if ((base & 1u) != ((tag_ + 1u) & 1u)) ++base;
    tag_ = base + (unsigned) n - 1u;
    tag_max_ = tag_;
    return base;
}

// the 27B has no expert arena: the pool's pages are its own
void Engine::arena_reclaim(double) {}
void Engine::arena_churn() {}
std::vector<int> Engine::arena_lent() const { return {}; }

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / S27_PP + 1;
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && on_pool_short && !in_pool_short_ && pending_.empty()) {
        const int was = cur_;
        in_pool_short_ = true;
        try {
            on_pool_short(lane, short_by);
        } catch (...) {
            in_pool_short_ = false;
            throw;
        }
        in_pool_short_ = false;
        if (cur_ != was) throw std::runtime_error("on_pool_short left lane " + std::to_string(cur_) + " current, not " + std::to_string(was));
    }
    if (need - (int) had > (int) free_pages_.size())
        throw Refused("the K/V pool is full (" + std::to_string(n_pos) + " positions on this lane, " + std::to_string(need - (int) had) +
                      " more pages wanted, " + std::to_string(free_pages_.size()) + " free)");
    while ((int) L.pages.size() < need) {
        L.pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (L.pages.size() == had) return;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemcpy(b->lanes[(size_t) lane].pt + had, L.pages.data() + had, (L.pages.size() - had) * sizeof(int), cudaMemcpyHostToDevice),
           "page table");
    }
}

void Engine::release_pages() {
    LaneHost& L = lanes_[(size_t) cur_];
    for (int p : L.pages) free_pages_.push_back(p);
    L.pages.clear();
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
}

void Engine::use_lane(int i) {
    if (i < 0 || i >= (int) lanes_.size()) throw std::runtime_error("use_lane: no lane " + std::to_string(i));
    if (i == cur_) return;
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("use_lane: a window is not committed");
    LaneHost& was = lanes_[(size_t) cur_];
    was.pos = pos_;
    was.tag = tag_;
    was.history = std::move(history_);
    cur_ = i;
    LaneHost& now = lanes_[(size_t) i];
    pos_ = now.pos;
    tag_ = now.tag;
    history_ = std::move(now.history);
    now.history.clear();
    for (CardBuf* b : cb_) {
        const LaneDev& L = b->lanes[(size_t) i];
        b->pt = L.pt;
        b->S = L.S;
        b->chist = L.chist;
        b->mrope = L.mrope;
    }
}

void Engine::set_images(const std::vector<Image>& images) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("set_images: a window is not committed");
    int end = 0;
    for (const Image& im : images) {
        if (im.nx < 1 || im.ny < 1 || im.n != im.nx * im.ny) throw std::runtime_error("set_images: an image's cells are not its grid");
        if (im.cell < end || im.cell + im.n > max_ctx_) throw std::runtime_error("set_images: images out of order or past the context");
        end = im.cell + im.n;
    }
    LaneHost& L = lanes_[(size_t) cur_];
    const bool same = L.images.size() == images.size() &&
                      std::equal(images.begin(), images.end(), L.images.begin(), [](const Image& a, const Image& b) {
                          return a.cell == b.cell && a.n == b.n && a.nx == b.nx && a.ny == b.ny;
                      });
    L.images = images;
    if (same) return;   // the rows change; the positions do not
    if (images.empty()) {
        for (CardBuf* b : cb_) b->mrope = b->lanes[(size_t) cur_].mrope = nullptr;
        return;
    }
    std::vector<int> tab((size_t) max_ctx_ * 3);
    int p = 0;
    size_t k = 0;
    for (int c = 0; c < max_ctx_;) {
        if (k < images.size() && c == images[k].cell) {
            const Image& im = images[k++];
            for (int j = 0; j < im.n; ++j) {
                tab[(size_t) (c + j) * 3] = p;
                tab[(size_t) (c + j) * 3 + 1] = p + j / im.nx;
                tab[(size_t) (c + j) * 3 + 2] = p + j % im.nx;
            }
            c += im.n;
            p += std::max(im.nx, im.ny);
        } else {
            tab[(size_t) c * 3] = tab[(size_t) c * 3 + 1] = tab[(size_t) c * 3 + 2] = p++;
            ++c;
        }
    }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        LaneDev& D = b->lanes[(size_t) cur_];
        if (!D.mrope_buf) D.mrope_buf = dalloc<int>(tab.size());
        ck(cudaMemcpy(D.mrope_buf, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice), "rotary table");
        b->mrope = D.mrope = D.mrope_buf;
    }
}

void Engine::warm_graphs() {
    const int was = cur_;
    std::vector<int> lanes;
    for (int i = 0; i < (int) lanes_.size(); ++i) lanes.push_back(i);
    const int tok = 248045;   // any token: every lane is reset after
    use_lane(0);
    reset();
    prefill(std::vector<int>(1100, tok), tok);
    reset();
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        for (LaneDev& D : b->lanes)
            if (!D.mrope_buf) D.mrope_buf = dalloc<int>((size_t) max_ctx_ * 3);
    }
    for (int T = 1; T <= MAXT; ++T) {
        verify(std::vector<int>((size_t) T, tok));
        commit(1);
    }
    if (mtp_)
        for (int T = 1; T <= MAXT; ++T) draft(T, true, 0, std::vector<int>((size_t) T, tok), 0);
    for (int n = 1; n <= std::min((int) lanes.size(), MAXT); ++n)
        step(std::vector<int>(lanes.begin(), lanes.begin() + n), std::vector<int>((size_t) n, tok));
    if (batch_drafts() != BatchDrafts::off)
        for (int T = 2; T <= MAXT; ++T) {
            const int n = std::min((int) lanes.size(), T);
            std::vector<int> counts((size_t) n, T / n);
            for (int i = 0; i < T % n; ++i) ++counts[(size_t) i];
            verify_rows(std::vector<int>(lanes.begin(), lanes.begin() + n), counts, std::vector<int>((size_t) T, tok));
            commit_rows(std::vector<int>((size_t) n, 1));
        }
    for (int i : lanes) {
        use_lane(i);
        reset();
    }
    use_lane(was);
}

std::vector<long> Engine::free_vram_mib() const {
    std::vector<long> v;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        size_t f = 0, t = 0;
        ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        v.push_back((long) (f >> 20));
    }
    return v;
}

// the 27B routes no experts
std::vector<unsigned long long> Engine::probe_counts() const { return {}; }
std::vector<unsigned> Engine::route_counts() const { return {}; }
std::vector<unsigned> Engine::route_trace(size_t) const { return {}; }

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves)
int Engine::lane_block_pages() const {
    int n = 0;
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = 256 * 64;   // alignment
        for (int l = 0; l < NL; ++l)
            if (!is_attn(l)) bytes += (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4 + 512;
        const size_t page = S27_PAGE * sizeof(__half);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const int n = lane_block_pages();
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    if (start < 0) throw std::runtime_error("add_lane: no run of " + std::to_string(n) + " free pages in the K/V pool");
    free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int p) { return p >= start && p < start + n; }),
                      free_pages_.end());
    LaneHost h;
    h.block0 = start;
    h.block_n = n;
    lanes_.push_back(h);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        char* at = (char*) (b.pool + (size_t) start * S27_PAGE);
        ck(cudaMemset(at, 0, (size_t) n * S27_PAGE * sizeof(__half)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l)
            if (!is_attn(l)) {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

Engine::~Engine() {
    for (CardBuf* b : cb_) {
        cudaSetDevice(b->dev);
        cudaDeviceSynchronize();
        for (auto* slots : {&b->win_graphs, &b->mtp_graphs, &b->batch_graphs})
            for (CardBuf::GraphSlot& s : *slots)
                if (s.exec) cudaGraphExecDestroy(s.exec);
    }
    prefill_free();
    std::vector<std::function<void()>>& o = owned();
    for (auto it = o.rbegin(); it != o.rend(); ++it) (*it)();
    o.clear();
    cudaGetLastError();
    for (CardBuf* b : cb_) delete b;
}

void Engine::embed_token(int tok, int cell, float* out) const {
    if (tok >= IMAGE_TOK) {
        for (const Image& im : lanes_[(size_t) cur_].images)
            if (cell >= im.cell && cell < im.cell + im.n) {
                if (!im.rows) break;
                std::memcpy(out, im.rows + (size_t) (cell - im.cell) * N, N * sizeof(float));
                return;
            }
        throw std::runtime_error("embed: image cell " + std::to_string(cell) + " without its embedding row");
    }
    if (tok < 0 || tok >= m_.vocab) throw std::runtime_error("embed: token " + std::to_string(tok) + " outside the vocabulary");
    const Tensor& te = *m_.tok_embd;
    const uint8_t* row = te.data + (size_t) tok * te.row_bytes();
    for (int b = 0; b < N / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, row + b * 34, 2);
        const float d = h2f(dh);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float) (int8_t) row[b * 34 + 2 + j];
    }
}

// the 27B has no PLE rows: ple must be null
void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    if (ple) throw std::runtime_error("27b: embed_rows takes no PLE rows");
    for (int t = 0; t < T; ++t) embed_token(tok_at_cell(p0 + t), p0 + t, emb + (size_t) t * N);
}

void Engine::dump_dev(const std::string& name, int layer, const float* p, int T, int n) {
    if (!dump) return;
    std::vector<float> h((size_t) T * n);
    ck(cudaSetDevice(cb_[0]->dev), "set");
    ck(cudaStreamSynchronize(cb_[0]->st), ("dump sync before " + name + " of layer " + std::to_string(layer)).c_str());
    ck(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost), "dump copy");
    dump(name, layer, T, n, h);
}

std::vector<int> Engine::window(const std::vector<int>& tokens) {
    std::vector<int> out = verify(tokens);
    commit((int) tokens.size());
    return out;
}

size_t Engine::SeqState::total() const {
    size_t t = 0;
    for (size_t b : bytes) t += b;
    return t;
}

Engine::SeqState::~SeqState() {
    for (void* p : host) std::free(p);
}

namespace {
// a sequence's device state on one card, in a fixed order: (pointer, bytes) pieces. hist_half: the conv history half
// the next window reads
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, int pos, bool kv, unsigned hist_half, const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    for (int l = 0; l < NL; ++l)
        if (!is_attn(l)) {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    if (kv)   // the pages holding positions 0 .. pos (every layer's K/V, the draft layer's too)
        for (int i = 0; i <= pos / S27_PP && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * S27_PAGE), S27_PAGE * sizeof(__half));
    return v;
}

constexpr size_t STAGE_BYTES = (size_t) 32 << 20;
void stage_copy(CardBuf& b, const std::vector<std::pair<char*, size_t>>& pieces, char* host, size_t total, bool to_host) {
    if (!b.stage[0]) {
        for (int k = 0; k < 2; ++k) {
            const cudaError_t e = cudaHostAlloc((void**) &b.stage[k], STAGE_BYTES, cudaHostAllocPortable);
            if (e != cudaSuccess) {
                cudaGetLastError();
                b.stage[k] = nullptr;
                throw std::runtime_error("state staging: page-locked host memory for 32 MiB: " + std::string(cudaGetErrorString(e)));
            }
            ck(cudaEventCreateWithFlags(&b.stage_ev[k], cudaEventDisableTiming), "event");
            own_pinned(b.stage[k]);
            own_event(b.stage_ev[k]);
        }
    }
    const size_t nblk = (total + STAGE_BYTES - 1) / STAGE_BYTES;
    size_t pi = 0, poff = 0;
    auto dma = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        char* half = b.stage[k % 2];
        for (size_t done = 0; done < blk;) {
            const size_t take = std::min(blk - done, pieces[pi].second - poff);
            char* dev = pieces[pi].first + poff;
            ck(cudaMemcpyAsync(to_host ? half + done : dev, to_host ? dev : half + done, take,
                               to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, b.st), "state copy");
            done += take;
            poff += take;
            if (poff == pieces[pi].second) {
                ++pi;
                poff = 0;
            }
        }
        ck(cudaEventRecord(b.stage_ev[k % 2], b.st), "event");
    };
    auto host_part = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        if (to_host) std::memcpy(host + k * STAGE_BYTES, b.stage[k % 2], blk);
        else std::memcpy(b.stage[k % 2], host + k * STAGE_BYTES, blk);
    };
    for (size_t k = 0; k < nblk; ++k) {
        if (to_host) {
            dma(k);
            if (k > 0) {
                ck(cudaEventSynchronize(b.stage_ev[(k - 1) % 2]), "state copy");
                host_part(k - 1);
            }
        } else {
            if (k > 1) ck(cudaEventSynchronize(b.stage_ev[k % 2]), "state copy");
            host_part(k);
            dma(k);
        }
    }
    if (to_host && nblk > 0) {
        ck(cudaEventSynchronize(b.stage_ev[(nblk - 1) % 2]), "state copy");
        host_part(nblk - 1);
    }
    ck(cudaStreamSynchronize(b.st), "state copy");
}

void each_card_thread(int n, const std::function<void(int)>& f) {
    std::vector<std::exception_ptr> err((size_t) n);
    std::vector<std::thread> th;
    for (int c = 0; c < n; ++c)
        th.emplace_back([&, c] {
            try {
                f(c);
            } catch (...) {
                err[(size_t) c] = std::current_exception();
            }
        });
    for (std::thread& t : th) t.join();
    for (const std::exception_ptr& e : err)
        if (e) std::rethrow_exception(e);
}
}  // namespace

std::shared_ptr<Engine::SeqState> Engine::save_state(bool kv) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("save_state: a window is not committed");
    auto s = std::make_shared<SeqState>();
    s->tokens = history_;
    for (const Image& im : lanes_[(size_t) cur_].images)
        if (im.cell + im.n <= pos_) s->images.push_back(Image{im.cell, im.n, im.nx, im.ny, nullptr});
    if (kv) ensure_pages(pos_);
    s->kv = kv;
    s->host.assign(NCARD, nullptr);
    s->bytes.assign(NCARD, 0);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], pos_, kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        s->host[c] = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
        if (!s->host[c])
            throw std::runtime_error("save_state: host memory for " + std::to_string(n >> 20) + " MiB (" + std::to_string(host_available_mib()) +
                                     " MiB available)");
        s->bytes[c] = n;
    }
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s->host[c], s->bytes[c], true);
    });
    return s;
}

void Engine::load_state(const SeqState& s) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("load_state: a window is not committed");
    if ((int) s.tokens.size() + 1 > max_ctx_) throw std::runtime_error("load_state: the state does not fit the context");
    if (!s.kv && (history_.size() < s.tokens.size() || !std::equal(s.tokens.begin(), s.tokens.end(), history_.begin())))
        throw std::runtime_error("load_state: a checkpoint of another sequence (the engine does not hold its K/V)");
    const int pos = (int) s.tokens.size();
    if (s.kv) ensure_pages(pos);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], pos, s.kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        if (n != s.bytes[c]) throw std::runtime_error("load_state: a state saved with another layout");
    }
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s.host[c], s.bytes[c], false);
    });
    history_ = s.tokens;
    pos_ = pos;
    set_images(s.images);
    LaneHost& L = lanes_[(size_t) cur_];
    const size_t keep = (size_t) (pos / S27_PP + 1);
    if (L.pages.size() > keep) {
        free_pages_.insert(free_pages_.end(), L.pages.begin() + (long) keep, L.pages.end());
        L.pages.resize(keep);
        std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    }
}

void Engine::reset() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("reset: a window is not committed");
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) continue;
            ck(cudaMemsetAsync(b.S[l], 0, (size_t) w.nv() * HS * HS * 4, b.st), "reset state");
            ck(cudaMemsetAsync(b.chist[l], 0, (size_t) 2 * 3 * w.conv_ch() * 4, b.st), "reset conv history");
        }
        ck(cudaStreamSynchronize(b.st), "reset sync");
    }
    release_pages();
    history_.clear();
    pos_ = 0;
    set_images({});
}

void Engine::commit(int n) {
    const int T = (int) pending_.size();
    if (T == 0) throw std::runtime_error("commit: no window to commit");
    if (n < 1 || n > T) throw std::runtime_error("commit: keep 1..T rows");
    if (n < T) {
        // the window read history parity tag & 1 and wrote the other: that one becomes the kept count's history
        const unsigned wrote = (tag_ & 1u) ^ 1u;
        constexpr int CJ = 2 * NL;
        if (!h_.commit_jobs) h_.commit_jobs = halloc<CopyJob>((size_t) NCARD * CJ);
        for (CardBuf* b : cb_)
            if (!b->commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->commit_jobs = dalloc<CopyJob>(CJ);
                ck(cudaEventCreateWithFlags(&b->commit_ev, cudaEventDisableTiming), "commit event");
                own_event(b->commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& w = m_.cards[c];
            const size_t sbytes = (size_t) w.nv() * HS * HS * 4, hfloats = (size_t) 3 * w.conv_ch();
            ck(cudaEventSynchronize(b.commit_ev), "commit table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.commit_jobs) + (size_t) c * CJ;
            int k = 0;
            for (int l = 0; l < NL; ++l) {
                if (is_attn(l)) continue;
                jobs[k++] = {b.S[l], b.snap[l] + (size_t) (n - 1) * w.nv() * HS * HS, sbytes};
                jobs[k++] = {b.chist[l] + wrote * hfloats, b.hsnap[l] + (size_t) (n - 1) * hfloats, hfloats * 4};
            }
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit jobs");
            ck(cudaEventRecord(b.commit_ev, b.st), "commit event");
            copy_many(b.commit_jobs, k, b.st);
        });
    }
    history_.insert(history_.end(), pending_.begin(), pending_.begin() + n);
    pos_ += n;
    pending_.clear();
}

std::vector<float> Engine::last_logits(int T) const {
    const int vocab = m_.cards[NCARD - 1].v1;
    std::vector<float> out((size_t) T * vocab);
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(cb_[c]->dev), "set");
        const int nv = w.v1 - w.v0;
        for (int t = 0; t < T; ++t)
            ck(cudaMemcpy(out.data() + (size_t) t * vocab + w.v0, cb_[c]->head_logits + (size_t) t * nv, (size_t) nv * 4,
                          cudaMemcpyDeviceToHost), "logits");
    }
    return out;
}

std::vector<std::vector<std::pair<float, int>>> Engine::top_tokens(int T, const Penalties& pen) {
    if (T < 1 || T > MAXT) throw std::runtime_error("top_tokens: 1..MAXT rows");
    if (!pen.rows.empty() && (int) pen.rows.size() != T) throw std::runtime_error("top_tokens: one penalty list per row");
    if (!tk_.pen_tok) {
        tk_.pen_tok = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_cnt = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_n = halloc<int>(MAXT);
        tk_.v = halloc<float>((size_t) NCARD * MAXT * TOPK_N);
        tk_.i = halloc<int>((size_t) NCARD * MAXT * TOPK_N);
        tk_.err = halloc<unsigned>(NCARD);
        tk_.par = halloc<float>((size_t) MAXT * 3);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            b->tk_pen_tok = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_cnt = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_n = dalloc<int>(MAXT);
            b->tk_v = dalloc<float>((size_t) MAXT * TOPK_N);
            b->tk_i = dalloc<int>((size_t) MAXT * TOPK_N);
            b->tk_err = dalloc<unsigned>(1);
            b->tk_par = dalloc<float>((size_t) MAXT * 3);
        }
    }
    static const Penalties::Row none;
    int stride = 1;
    for (int t = 0; t < T; ++t) stride = std::max(stride, (int) (pen.rows.empty() ? none : pen.rows[(size_t) t]).counts.size());
    if (stride > PEN_CAP) throw std::runtime_error("top_tokens: more penalized tokens than " + std::to_string(PEN_CAP));
    for (int t = 0; t < T; ++t) {
        const Penalties::Row& row = pen.rows.empty() ? none : pen.rows[(size_t) t];
        const auto& r = row.counts;
        tk_.par[3 * t] = row.repeat;
        tk_.par[3 * t + 1] = row.freq;
        tk_.par[3 * t + 2] = row.present;
        tk_.pen_n[t] = (int) r.size();
        for (size_t k = 0; k < r.size(); ++k) {
            tk_.pen_tok[(size_t) t * stride + k] = r[k].first;
            tk_.pen_cnt[(size_t) t * stride + k] = r[k].second;
        }
    }
    on_cards([&](int c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaMemcpyAsync(b.tk_pen_tok, tk_.pen_tok, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_cnt, tk_.pen_cnt, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_n, tk_.pen_n, (size_t) T * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_par, tk_.par, (size_t) T * 3 * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemsetAsync(b.tk_err, 0, 4, b.st), "top err");
        fnx::top_tokens(b.head_logits, w.v1 - w.v0, T, w.v0, b.tk_pen_tok, b.tk_pen_cnt, b.tk_pen_n, stride, b.tk_par, b.tk_v, b.tk_i,
                        b.tk_err, b.st);
        ck(cudaMemcpyAsync(tk_.v + (size_t) c * MAXT * TOPK_N, b.tk_v, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top v");
        ck(cudaMemcpyAsync(tk_.i + (size_t) c * MAXT * TOPK_N, b.tk_i, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top i");
        ck(cudaMemcpyAsync(tk_.err + c, b.tk_err, 4, cudaMemcpyDeviceToHost, b.st), "top err");
        ck(cudaStreamSynchronize(b.st), "top sync");
    });
    for (int c = 0; c < NCARD; ++c)
        if (tk_.err[c] & 16u) throw std::runtime_error("top_tokens: the head's logits hold a NaN or inf");
    std::vector<std::vector<std::pair<float, int>>> out(T);
    for (int t = 0; t < T; ++t) {
        auto& r = out[t];
        r.reserve((size_t) NCARD * TOPK_N);
        for (int c = 0; c < NCARD; ++c)
            for (int k = 0; k < TOPK_N; ++k) {
                const size_t at = ((size_t) c * MAXT + t) * TOPK_N + k;
                r.emplace_back(tk_.v[at], tk_.i[at]);
            }
        std::sort(r.begin(), r.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        r.resize(TOPK_N);
    }
    return out;
}

namespace {
const char* window_error(unsigned err) {
    if (err & 1u) return "a ring sum timed out";
    if (err & 2u) return "a ring sum met a NaN or inf (or a value past fp16's range)";
    if (err & 16u) return "the head's logits hold a NaN or inf";
    return nullptr;
}
}  // namespace

std::vector<int> Engine::verify(const std::vector<int>& tokens) {
    const int T = (int) tokens.size();
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify: the last window is not committed");
    if (T < 1 || T > MAXT) throw std::runtime_error("window: 1..MAXT tokens");
    if (pos_ + T > max_ctx_) throw std::runtime_error("window: context full");
    ensure_pages(pos_ + T);
    const auto t0 = std::chrono::steady_clock::now();
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * N);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    const int h0 = (int) history_.size();
    embed_rows(pos_, T, [&](int q) -> int { return q < h0 ? history_[(size_t) q] : tokens[(size_t) (q - h0)]; }, h_.R0, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    h_.ctl[0] = pos_;
    h_.ctl[1] = (int) take_tags(1);
    float* bv = h_.bv;
    int* bi = h_.bi;
    auto upload = [&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
    };
    auto download = [&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], b.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    if (dump) {
        each(upload);
        dump_dev("R0", -1, cb_[0]->R, T, N);
        enqueue(T, true);
        each(download);
    } else {
        const std::vector<int> key = graph_key({cur_});
        if (cb_[0]->win_graphs[T].key != key || !cb_[0]->win_graphs[T].exec) {
            each([&](int, CardBuf& b, const CardW&) { ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture"); });
            enqueue(T, false);
            each([&](int, CardBuf& b, const CardW&) {
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                install_graph(b.win_graphs[T], g, key);
            });
        }
        const auto l0 = std::chrono::steady_clock::now();
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            upload(c, b, m_.cards[c]);
            ck(cudaGraphLaunch(b.win_graphs[T].exec, b.st), "graph launch");
            download(c, b, m_.cards[c]);
        });
        last_launch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count();
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "window sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
        last_card_ms[c] = ms;
    });
    if (const char* e = window_error(err)) throw std::runtime_error(std::string("window: ") + e);
    std::vector<int> out(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (bv[c * T + t] > bv[best * T + t]) best = c;
        out[t] = bi[best * T + t];
    }
    pending_ = tokens;
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

// the layers of a window of T rows (one sequence's, or a batched window's from lanes: b.lane_rows[T], with the
// recurrent core in gdn), then the head: the output norm with the last sum folded in (kept in hfin for the draft
// layer), this card's vocabulary, its argmax
void Engine::enqueue_layers(int T, bool lanes, const GdnFn& gdn, bool dumps) {
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    auto dd = [&](const std::string& name, int layer, const float* p, int n) {
        if (dumps) dump_dev(name, layer, p, T, n);
    };
    // a card's rows: its window's at *b.ctl + t on its current lane's pages, or a batched window's from b.lane_rows
    auto rows_of = [&](CardBuf& b) {
        return lanes ? Rows{b.ctl, LaneRows{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT}} : Rows{b.ctl, LaneRows{}};
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            if (!is_attn(l)) {
                const int nv = w.nv();
                add_norm(b, b.R, l > 0 ? b.bo : nullptr, L.attn_norm, T, eps, true);
                q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                gdn(b, w, L, l);
                mv(b, L.ssm_out, b.xq, b.bo, T);
                sum_cards(b, b.bo, T, b.ctl, call);
            } else {
                attn_half(b, L, b.R, l > 0, s27_slot(l), rows_of(b), w.q0, w.kv0, call, T, eps, rope_of(b.mrope));
            }
        });
        ++call;
        dd("attn_out", l, cb_[0]->bo, N);
        each([&](int, CardBuf& b, const CardW& w) { ffn27(b, w, w.L[l], b.R, b.ctl, call, T, eps); });
        ++call;
        dd("ffn_out", l, cb_[0]->bo, N);
    }
    each([&](int, CardBuf& b, const CardW& w) {
        add_norm(b, b.R, b.bo, w.out_norm, T, eps, true);
        ck(cudaMemcpyAsync(b.hfin, b.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "hfin");
        mv(b, w.out, b.xq, b.head_logits, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.err, b.st);
    });
    dd("result_norm", -1, cb_[0]->hfin, N);
}

// every card's work for one window of T rows of the current lane, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
    enqueue_layers(
        T, false,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                     b.hsnap[l], w.nk, w.nv(), w.vmap, T, m_.eps, b.st);
        },
        dumps);
}

double rope_scale() {
    const char* e = std::getenv("FNX_ROPE_SCALE");
    if (!e) return 0.0;
    const double S = env_num("FNX_ROPE_SCALE", 0.0);
    if (S == 1.0) return 0.0;
    if (!(S > 1.0)) throw std::runtime_error(std::string("FNX_ROPE_SCALE: want 1 (off) or a scale above 1, got ") + e);
    return S;
}

bool ring_f32() {
    // the 27B's sums are three fp16 values a word (s27_ring3.cuh): no fp32 ring
    const long long v = env_int("FNX_RING_F32", 0);
    if (v != 0) throw std::runtime_error("FNX_RING_F32: the 27B's ring is fp16 only, got " + std::to_string(v));
    return false;
}

BatchDrafts batch_drafts() {
    static const BatchDrafts mode = [] {
        const char* e = std::getenv("FNX_BATCH_DRAFTS");
        if (!e || !std::strcmp(e, "1")) return BatchDrafts::mtp;
        if (!std::strcmp(e, "0")) return BatchDrafts::off;
        if (!std::strcmp(e, "lookup")) return BatchDrafts::lookup;
        throw std::runtime_error(std::string("FNX_BATCH_DRAFTS: 0, lookup or 1, not '") + e + "'");
    }();
    return mode;
}

}  // namespace fnx

namespace fnx {

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): each row's recurrent core on its lane's state,
// its attention at its lane's position on its lane's pages (b.lane_rows[T])
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    const int T = (int) lanes.size();
    enqueue_layers(
        T, true,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            float** rec = b.lane_gdn[(size_t) T];
            const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
            gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, w.nv(), w.vmap, gl, T, m_.eps, b.st);
        },
        false);
}

std::vector<int> Engine::step(const std::vector<int>& lanes, const std::vector<int>& tokens) {
    const int T = (int) lanes.size();
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("step: 1..MAXT lanes, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("step: the current lane's window is not committed");
    for (int r = 0; r < T; ++r) {
        if (lanes[r] < 0 || lanes[r] >= (int) lanes_.size()) throw std::runtime_error("step: no lane " + std::to_string(lanes[r]));
        for (int q = 0; q < r; ++q)
            if (lanes[q] == lanes[r]) throw std::runtime_error("step: a lane twice in one window");
    }
    const auto t0 = std::chrono::steady_clock::now();
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * N);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    std::vector<int> lane_ctl((size_t) 2 * T);
    try {
        for (int r = 0; r < T; ++r) {
            LaneHost& L = lanes_[(size_t) lanes[r]];
            if (L.pos + 1 > max_ctx_) throw std::runtime_error("step: lane " + std::to_string(lanes[r]) + "'s context is full");
            ensure_pages_of(lanes[r], L.pos + 1);
            if (tokens[r] >= IMAGE_TOK) throw std::runtime_error("step: an image cell in a batched window");
            embed_token(tokens[r], L.pos, h_.R0 + (size_t) r * N);
            L.tag = L.tag + 1u;   // the lane's recurrent history half flips, as a window of its own would
            lane_ctl[2 * r] = L.pos;
            lane_ctl[2 * r + 1] = (int) L.tag;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int r = 0; r < T; ++r)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[r]].ctl, &lane_ctl[(size_t) 2 * r], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
    });
    const std::vector<int> bkey = graph_key(lanes);
    if (cb_[0]->batch_graphs[T].key != bkey || !cb_[0]->batch_graphs[T].exec) {
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) 2 * MAXT);
            std::vector<const int*> rows((size_t) 2 * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                rows[(size_t) r] = b.lanes[(size_t) lanes[r]].ctl;
                rows[(size_t) (MAXT + r)] = b.lanes[(size_t) lanes[r]].pt;
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            for (int l = 0; l < NL; ++l)
                for (int r = 0; r < T; ++r) {
                    gdn[(size_t) l * MAXT + r] = b.lanes[(size_t) lanes[r]].S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + r] = b.lanes[(size_t) lanes[r]].chist[(size_t) l];
                }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        enqueue_batch(lanes);
        each([&](int, CardBuf& b, const CardW&) {
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b.st, &g), "end capture");
            install_graph(b.batch_graphs[T], g, bkey);
        });
    }
    each([&](int, CardBuf& b, const CardW&) { ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch"); });
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], b.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "step sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    if (const char* e = window_error(err)) throw std::runtime_error(std::string("step: ") + e);
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
        LaneHost& L = lanes_[(size_t) lanes[r]];
        L.history.push_back(tokens[r]);
        ++L.pos;
    }
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

namespace {
// a verify_rows() window's layout: each lane's first row, and its lanes as segments ordered by row count (stable), so
// that the lanes with the same count are one launch of the recurrent core (a group: its count, first segment, size)
struct RowGroup {
    int count, s0, n;
};
struct RowSegs {
    int T = 0;
    std::vector<int> r0;
    std::vector<int> order;
    std::vector<RowGroup> groups;
    std::vector<int> row_lane;
};
RowSegs row_segs(const std::vector<int>& counts) {
    RowSegs sg;
    const int nl = (int) counts.size();
    for (int i = 0; i < nl; ++i) {
        sg.r0.push_back(sg.T);
        for (int j = 0; j < counts[(size_t) i]; ++j) sg.row_lane.push_back(i);
        sg.T += counts[(size_t) i];
        sg.order.push_back(i);
    }
    std::stable_sort(sg.order.begin(), sg.order.end(), [&](int a, int b) { return counts[(size_t) a] < counts[(size_t) b]; });
    for (int s = 0; s < nl; ++s) {
        const int k = counts[(size_t) sg.order[(size_t) s]];
        if (sg.groups.empty() || sg.groups.back().count != k) sg.groups.push_back({k, s, 0});
        ++sg.groups.back().n;
    }
    return sg;
}
}  // namespace

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): each lane's recurrent rows chained through
// its state as a window of its own (snapshots kept for commit_rows), every row's attention at its own position
// (b.row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    const RowSegs sg = row_segs(counts);
    const int T = sg.T;
    (void) lanes;
    enqueue_layers(
        T, true,
        [&](CardBuf& b, const CardW& w, const LayerW& L, int l) {
            float** rec = b.lane_gdn[(size_t) T];
            for (const RowGroup& g : sg.groups) {
                const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                  b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, w.nv(), w.vmap, gl, g.n,
                              g.count, m_.eps, b.st);
            }
        },
        false);
}

std::vector<int> Engine::verify_rows(const std::vector<int>& lanes, const std::vector<int>& counts, const std::vector<int>& tokens) {
    const int nl = (int) lanes.size();
    if (nl < 1 || (int) counts.size() != nl) throw std::runtime_error("verify_rows: one lane or more, a row count each");
    int T = 0;
    for (int k : counts) {
        if (k < 1) throw std::runtime_error("verify_rows: a lane with no rows");
        T += k;
    }
    if (T > MAXT || (int) tokens.size() != T) throw std::runtime_error("verify_rows: at most MAXT rows in all, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify_rows: the last window is not committed");
    for (int i = 0; i < nl; ++i) {
        if (lanes[i] < 0 || lanes[i] >= (int) lanes_.size()) throw std::runtime_error("verify_rows: no lane " + std::to_string(lanes[i]));
        for (int q = 0; q < i; ++q)
            if (lanes[q] == lanes[i]) throw std::runtime_error("verify_rows: a lane twice in one window");
    }
    const auto t0 = std::chrono::steady_clock::now();
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * N);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    for (CardBuf* b : cb_)
        if (!b->row_pos) {
            ck(cudaSetDevice(b->dev), "set");
            b->row_pos = dalloc<int>(MAXT);
        }
    const RowSegs sg = row_segs(counts);
    std::vector<int> lane_ctl((size_t) 2 * nl), row_pos((size_t) T);
    std::vector<unsigned> tags((size_t) nl);
    try {
        for (int i = 0; i < nl; ++i) {
            const LaneHost& L = lanes_[(size_t) lanes[i]];
            const int k = counts[(size_t) i], r = sg.r0[(size_t) i];
            if (L.pos + k > max_ctx_) throw std::runtime_error("verify_rows: lane " + std::to_string(lanes[i]) + "'s context is full");
            ensure_pages_of(lanes[i], L.pos + k);
            for (int j = 0; j < k; ++j) {
                const int tok = tokens[(size_t) (r + j)];
                if (tok >= IMAGE_TOK) throw std::runtime_error("verify_rows: an image cell in a batched window");
                embed_token(tok, L.pos + j, h_.R0 + (size_t) (r + j) * N);
            }
            tags[(size_t) i] = L.tag + 1u;   // the lane's recurrent history half flips, as a window of its own would
            lane_ctl[(size_t) 2 * i] = L.pos;
            lane_ctl[(size_t) 2 * i + 1] = (int) tags[(size_t) i];
            for (int j = 0; j < k; ++j) row_pos[(size_t) (r + j)] = L.pos + j;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    for (int i = 0; i < nl; ++i) lanes_[(size_t) lanes[i]].tag = tags[(size_t) i];
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int i = 0; i < nl; ++i)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[i]].ctl, &lane_ctl[(size_t) 2 * i], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
        ck(cudaMemcpyAsync(b.row_pos, row_pos.data(), (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "row positions");
    });
    std::vector<int> key = graph_key(lanes);
    key.push_back(-1);
    key.insert(key.end(), counts.begin(), counts.end());
    last_capture_ms = 0.0;
    if (cb_[0]->batch_graphs[T].key != key || !cb_[0]->batch_graphs[T].exec) {
        const auto tc = std::chrono::steady_clock::now();
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) 2 * MAXT);
            std::vector<const int*> rows((size_t) 2 * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                rows[(size_t) r] = b.row_pos + r;
                rows[(size_t) (MAXT + r)] = b.lanes[(size_t) lanes[(size_t) sg.row_lane[(size_t) r]]].pt;
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            if (!b.seg_ctl[(size_t) T]) b.seg_ctl[(size_t) T] = dalloc<const int*>(MAXT);
            if (!b.seg_row0[(size_t) T]) b.seg_row0[(size_t) T] = dalloc<int>(MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            std::vector<const int*> sctl((size_t) MAXT, nullptr);
            std::vector<int> srow0((size_t) MAXT, 0);
            for (int s = 0; s < nl; ++s) {
                const int i = sg.order[(size_t) s];
                const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                for (int l = 0; l < NL; ++l) {
                    gdn[(size_t) l * MAXT + s] = LD.S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + s] = LD.chist[(size_t) l];
                }
                sctl[(size_t) s] = LD.ctl;
                srow0[(size_t) s] = sg.r0[(size_t) i];
            }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaMemcpy(b.seg_ctl[(size_t) T], sctl.data(), sctl.size() * sizeof(const int*), cudaMemcpyHostToDevice), "segment ctl");
            ck(cudaMemcpy(b.seg_row0[(size_t) T], srow0.data(), srow0.size() * sizeof(int), cudaMemcpyHostToDevice), "segment rows");
            ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        enqueue_rows(lanes, counts);
        each([&](int, CardBuf& b, const CardW&) {
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b.st, &g), "end capture");
            install_graph(b.batch_graphs[T], g, key);
        });
        last_capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    each([&](int, CardBuf& b, const CardW&) { ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch"); });
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], b.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "verify_rows sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    rows_pending_ = RowsWindow{lanes, counts, tokens};
    if (const char* e = window_error(err)) throw std::runtime_error(std::string("verify_rows: ") + e);
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
    }
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

std::vector<std::vector<int>> Engine::held_layouts(const std::vector<int>& lanes) const {
    std::vector<int> pre = graph_key(lanes);
    pre.push_back(-1);
    std::vector<std::vector<int>> out;
    for (int T = (int) lanes.size(); T <= MAXT; ++T) {
        const CardBuf::GraphSlot& s = cb_[0]->batch_graphs[(size_t) T];
        if (s.exec && s.key.size() == pre.size() + lanes.size() && std::equal(pre.begin(), pre.end(), s.key.begin()))
            out.emplace_back(s.key.begin() + (long) pre.size(), s.key.end());
    }
    return out;
}

void Engine::commit_rows(const std::vector<int>& keep) {
    const RowsWindow& w = rows_pending_;
    const int nl = (int) w.lanes.size();
    if (nl == 0) throw std::runtime_error("commit_rows: no window to commit");
    if ((int) keep.size() != nl) throw std::runtime_error("commit_rows: a kept count per lane");
    for (int i = 0; i < nl; ++i)
        if (keep[i] < 1 || keep[i] > w.counts[(size_t) i]) throw std::runtime_error("commit_rows: keep 1 .. its rows of each lane");
    const RowSegs sg = row_segs(w.counts);
    bool cut = false;
    for (int i = 0; i < nl; ++i) cut = cut || keep[i] < w.counts[(size_t) i];
    if (cut) {
        constexpr int CJ = 2 * NL;
        if (!h_.rows_commit_jobs) h_.rows_commit_jobs = halloc<CopyJob>((size_t) NCARD * MAXT * CJ);
        for (CardBuf* b : cb_)
            if (!b->rows_commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->rows_commit_jobs = dalloc<CopyJob>((size_t) MAXT * CJ);
                ck(cudaEventCreateWithFlags(&b->rows_commit_ev, cudaEventDisableTiming), "commit_rows event");
                own_event(b->rows_commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& cw = m_.cards[c];
            const size_t sfloats = (size_t) cw.nv() * HS * HS, hfloats = (size_t) 3 * cw.conv_ch();
            ck(cudaEventSynchronize(b.rows_commit_ev), "commit_rows table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.rows_commit_jobs) + (size_t) c * MAXT * CJ;
            int k = 0;
            for (int i = 0; i < nl; ++i) {
                if (keep[i] == w.counts[(size_t) i]) continue;
                const int lane = w.lanes[(size_t) i];
                const LaneDev& LD = b.lanes[(size_t) lane];
                const unsigned wrote = ((lane == cur_ ? tag_ : lanes_[(size_t) lane].tag) & 1u) ^ 1u;
                const size_t at = (size_t) (sg.r0[(size_t) i] + keep[i] - 1);
                for (int l = 0; l < NL; ++l) {
                    if (is_attn(l)) continue;
                    jobs[k++] = {LD.S[(size_t) l], b.snap[l] + at * sfloats, sfloats * 4};
                    jobs[k++] = {LD.chist[(size_t) l] + wrote * hfloats, b.hsnap[l] + at * hfloats, hfloats * 4};
                }
            }
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit_rows: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.rows_commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit_rows jobs");
            ck(cudaEventRecord(b.rows_commit_ev, b.st), "commit_rows event");
            copy_many(b.rows_commit_jobs, k, b.st);
        });
    }
    for (int i = 0; i < nl; ++i) {
        const int lane = w.lanes[(size_t) i];
        const auto from = w.tokens.begin() + sg.r0[(size_t) i];
        std::vector<int>& h = lane == cur_ ? history_ : lanes_[(size_t) lane].history;
        h.insert(h.end(), from, from + keep[i]);
        (lane == cur_ ? pos_ : lanes_[(size_t) lane].pos) += keep[i];
    }
    rows_pending_ = RowsWindow{};
}

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.me = dalloc<float>((size_t) MAXT * N);
        b.mh = dalloc<float>((size_t) MAXT * N);
        b.mcat = dalloc<float>((size_t) MAXT * 2 * N);
        b.mR = dalloc<float>((size_t) MAXT * N);
        b.mout = dalloc<float>((size_t) MAXT * N);
        b.mlogits = dalloc<float>((size_t) std::max(1, mc.dv1 - mc.dv0));
        b.mbv = dalloc<float>(1);
        b.mbi = dalloc<int>(1);
        b.mctl = dalloc<int>(2);
        b.mpen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mpen_par = dalloc<float>(3);
        b.mtk_v = dalloc<float>(TOPK_N);
        b.mtk_i = dalloc<int>(TOPK_N);
        b.mtk_err = dalloc<unsigned>(1);
        b.mzero = dalloc<int>(1);
    }
    if (!h_.mtp_emb) {
        h_.mtp_emb = halloc<float>((size_t) MAXT * N);
        h_.mtp_ctl = halloc<int>(2);
        h_.mtp_pen = halloc<int>((size_t) NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par = halloc<float>(3);
        h_.mtp_tk_v = halloc<float>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_i = halloc<int>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_err = halloc<unsigned>(NCARD);
    }
    draft_index_.assign((size_t) m_.vocab, -1);
    for (size_t i = 0; i < mtp.vocab.size(); ++i) draft_index_[(size_t) mtp.vocab[i]] = (int) i;
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
}

// one draft-layer window of T rows of (token embedding me, source mh) at cells mctl[0]..: the nextn projection of
// [enorm(e) | hnorm(h)], an attention layer and an FFN as the trunk's (the draft layer's slot of the pages), the head
// norm (every row into mout: a chained pass reads its own row there), then the last row through this card's rows of
// the draft head, its penalties, its argmax
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        k_norm_cat<<<T, 1024, 0, b.st>>>(b.me, b.mh, w.mtp_enorm, w.mtp_hnorm, b.mcat, eps);
        ck(cudaGetLastError(), "norm_cat");
        quantize_q8_1(b.mcat, b.xq, 2 * N, T, b.st);
        mv(b, w.mtp_eh, b.xq, b.mR, T);
        attn_half(b, w.mtp, b.mR, false, S27_SLOTS - 1, Rows{b.mctl, LaneRows{}}, w.q0, w.kv0, 0, T, eps, rope_of(b.mrope));
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn27(b, w, w.mtp, b.mR, b.mctl, 1, T, eps);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0;
        ck(cudaSetDevice(b.dev), "set");
        add_norm(b, b.mR, b.bo, w.mtp_head_norm, T, eps, false);
        ck(cudaMemcpyAsync(b.mout, b.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "draft out");
        quantize_q8_1(b.mout + (size_t) (T - 1) * N, b.xq, N, 1, b.st);
        mv(b, mc.head, b.xq, b.mlogits, 1);
        penalize_row(b.mlogits, nv, b.mpen, b.mpen_par, b.st);
        argmax(b.mlogits, nv, 1, mc.dv0, b.mbv, b.mbi, b.am_keys, b.err, b.st);
    }
}

int Engine::draft(int T, bool from_main, int src_row, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen, bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (!h_.R0) throw std::runtime_error("draft: no window has run (it drafts from a window's rows)");
    if (src_row < 0 || src_row + T > MAXT) throw std::runtime_error("draft: source rows out of range");
    std::vector<const float*> src;
    for (CardBuf* b : cb_) src.push_back((from_main ? b->hfin : b->mout) + (size_t) src_row * N);
    return draft_rows(src, T, tokens, pos0, pen, cands);
}

int Engine::draft_rows(const std::vector<const float*>& src, int T, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen,
                       bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("draft: 1..MAXT rows, one token each");
    if (pos0 < 0 || pos0 + T > max_ctx_) throw std::runtime_error("draft: cells out of the context");
    ensure_pages(pos0 + T);
    if (!h_.mtp_emb) throw std::runtime_error("draft: mtp_attach first");
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < T; ++t) embed_token(tokens[t], pos0 + t + 1, h_.mtp_emb + (size_t) t * N);
    h_.mtp_ctl[0] = pos0;
    h_.mtp_ctl[1] = (int) ring_tag();
    int pen_k[NCARD] = {};
    if (pen) {
        if ((int) pen->counts.size() > PEN_CAP) throw std::runtime_error("draft: more penalized tokens than " + std::to_string(PEN_CAP));
        h_.mtp_pen_par[0] = pen->repeat, h_.mtp_pen_par[1] = pen->freq, h_.mtp_pen_par[2] = pen->present;
        for (int c = 0; c < NCARD; ++c) {
            const MtpCard& mc = mtp_->cards[c];
            int* hp = h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP);
            int k = 0;
            for (const auto& [tok, cnt] : pen->counts) {
                const int di = tok >= 0 && tok < (int) draft_index_.size() ? draft_index_[(size_t) tok] : -1;
                if (di >= mc.dv0 && di < mc.dv1) hp[1 + k] = di - mc.dv0, hp[1 + PEN_CAP + k] = cnt, ++k;
            }
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);
            hp[0] = pen_k[c] = k;
        }
    } else {
        for (int c = 0; c < NCARD; ++c) h_.mtp_pen[(size_t) c * (1 + 2 * PEN_CAP)] = 0;
    }
    const std::vector<int> key = graph_key({cur_});
    if (cb_[0]->mtp_graphs[T].key != key || !cb_[0]->mtp_graphs[T].exec) {
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            ck(cudaStreamBeginCapture(b->st, cudaStreamCaptureModeRelaxed), "begin capture");
        }
        mtp_enqueue(T);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b->st, &g), "end capture");
            install_graph(b->mtp_graphs[T], g, key);
        }
    }
    on_cards([&](int c) {
        CardBuf& b = *cb_[c];
        ck(cudaMemcpyAsync(b.mh, src[c], (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        ck(cudaMemcpyAsync(b.me, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
        ck(cudaMemcpyAsync(b.mctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
        ck(cudaMemcpyAsync(b.mpen, h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP), (size_t) (1 + 2 * pen_k[c]) * 4, cudaMemcpyHostToDevice, b.st),
           "draft penalties");
        if (pen_k[c]) ck(cudaMemcpyAsync(b.mpen_par, h_.mtp_pen_par, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaGraphLaunch(b.mtp_graphs[T].exec, b.st), "graph launch");
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0;
        if (cands && nv > 0) {   // the logits are penalized already
            ck(cudaMemsetAsync(b.mtk_err, 0, 4, b.st), "draft top err");
            fnx::top_tokens(b.mlogits, nv, 1, mc.dv0, b.mzero, b.mzero, b.mzero, 0, b.mpen_par, b.mtk_v, b.mtk_i, b.mtk_err, b.st);
        }
        ck(cudaEventRecord(b.ev1, b.st), "event");
        ck(cudaMemcpyAsync(&h_.bv[c], b.mbv, 4, cudaMemcpyDeviceToHost, b.st), "draft v");
        ck(cudaMemcpyAsync(&h_.bi[c], b.mbi, 4, cudaMemcpyDeviceToHost, b.st), "draft i");
        ck(cudaMemcpyAsync(&h_.err[c], b.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
        if (cands && nv > 0) {
            ck(cudaMemcpyAsync(h_.mtp_tk_v + (size_t) c * TOPK_N, b.mtk_v, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
            ck(cudaMemcpyAsync(h_.mtp_tk_i + (size_t) c * TOPK_N, b.mtk_i, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
            ck(cudaMemcpyAsync(h_.mtp_tk_err + c, b.mtk_err, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
        }
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c]->ev0, cb_[c]->ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    }
    last_draft_gpu_ms = gpu_ms;
    if (const char* e = window_error(err)) throw std::runtime_error(std::string("draft: ") + e);
    int best = 0;
    for (int c = 1; c < NCARD; ++c)
        if (mtp_->cards[c].dv1 > mtp_->cards[c].dv0 && h_.bv[c] > h_.bv[best]) best = c;
    if (cands) {
        last_draft_cands.clear();
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            if (h_.mtp_tk_err[c] & 16u) throw std::runtime_error("draft: the draft layer's logits hold a NaN or inf");
            for (int k = 0; k < TOPK_N; ++k)
                last_draft_cands.emplace_back(h_.mtp_tk_v[(size_t) c * TOPK_N + k], mtp_->vocab[h_.mtp_tk_i[(size_t) c * TOPK_N + k]]);
        }
        std::sort(last_draft_cands.begin(), last_draft_cands.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        last_draft_cands.resize(TOPK_N);
    }
    last_draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return mtp_->vocab[h_.bi[best]];
}

}  // namespace fnx
