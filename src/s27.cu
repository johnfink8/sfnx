#include "s27.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "s27_attn.cuh"
#include "s27_ring3.cuh"

namespace fnx {

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <typename T> T* dalloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    return (T*) p;
}
template <typename T> T* halloc(size_t n) {
    void* p = nullptr;
    ck(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable), "cudaHostAlloc");
    return (T*) p;
}

// the widest q8_1 row a window quantizes: the draft layer's [e | h], the FFN's slice, the attention's heads
constexpr int XQ_COLS = std::max({2 * N, FFD_CARD_MAX, NQH / NCARD * HD, NVH / NCARD * HS});

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
void add_norm(float* R, const float* pend, const float* w, float* y, int T, float eps, cudaStream_t st, int8_t* tq = nullptr,
              float* td = nullptr) {
    k_add_norm<<<T, 1024, 0, st>>>(R, pend, w, y, tq, td, eps);
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

// S27_SKIP=ring,gdn,attn: leave those kernels out of the window (wrong tokens; a timing breakdown only)
bool skip(const char* what) {
    const char* s = std::getenv("S27_SKIP");
    return s && std::strstr(s, what);
}

// x summed across the cards (every card ends with the sum). On the box GPU 0 sits on the other socket from 1 and 2:
// storing straight to both peers (2 hops, not the ring's 4) measured slower at every window size, 28.9 against 19.4 ms at 4
// rows, from the traffic it sends across the sockets
// S27_RING3=0: kernels.cu's ring (two values a word) instead of s27_ring3.cuh's three: 4-row windows 19.4 ms against
// 18.5, 8 rows 25.9 against 23.8
bool ring3_env() {
    static const bool on = [] {
        const char* s = std::getenv("S27_RING3");
        if (s && std::strcmp(s, "0") && std::strcmp(s, "1")) throw std::runtime_error(std::string("S27_RING3: 0 or 1, not '") + s + "'");
        return !s || !std::strcmp(s, "1");
    }();
    return on;
}
void sum_cards(S27Card& b, float* x, int T, const int* ctl, int call) {
    if (skip("ring")) return;
    const unsigned* tag = (const unsigned*) (ctl + 1);
    if (ring3_env()) {
        ring_call_check(call, "sum_cards");
        k_ring3<<<R3_BLOCKS, 256, 0, b.st>>>(b.r3, x, T, tag, call);
        ck(cudaGetLastError(), "ring3");
    } else {
        ring_sum(b.ring, x, T, tag, call, b.st);
    }
}

// *s = sum of exp(logits[i] - *m) over n: one card's share of the draft head's softmax denominator, about its own max
__global__ void k_sumexp(const float* logits, int n, const float* m, float* s) {
    __shared__ float sh[32];
    float a = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a += __expf(logits[i] - *m);
    a = block_sum1024(a, sh);
    if (threadIdx.x == 0) *s = a;
}

bool ring_f32_env() {
    const char* s = std::getenv("FNX_RING_F32");
    if (s && std::strcmp(s, "0") && std::strcmp(s, "1")) throw std::runtime_error(std::string("FNX_RING_F32: 0 or 1, not '") + s + "'");
    return s && !std::strcmp(s, "1");
}

// out[t][r] = w[r] . y[t] for a Q8_0 matrix of rows x cols. q8s beat three kernels written for these shapes (activations
// in registers with split columns; a lane on 64 columns; activations in shared memory, whole rows a warp): at 4 rows
// 13.8 ms of matrices a window against 14.9, 21.9 and ~15.2
// The rest of the gap to the same bytes in a few big matrices (11.5 ms at 4 rows) is at the boundaries between the window's
// 257 dependent matrices, and stayed there as one persistent launch with a grid barrier between matrices (13.8 ms), with
// each block prefetching its next rows into L2 before the barrier (15.1), with an L2 prefetch on a second stream
// (14.8 to 15.7) and with capped persistent grids (13.2 to 16.8)
void mv(S27Card& b, const Q8S& w, int rows, int cols, const void* y, float* out, int stride, int T) {
    q8s_gemv(w, rows, cols, y, out, stride, T, b.st);
}

// an attention layer's mixer half on one card: the last half's sum folded into R (pend) and the norm, the card's q heads and
// their two KV heads projected, the window's K/V stored, attention over every position (s27_attn.cuh), the card's heads
// through wo, summed across the cards into b.bo (folded into R by the next half)
void attn_half(S27Card& b, const LayerW& L, float* R, bool pend, const int* ctl, __half* kc, __half* vc, int q0, int kv0, int call,
               int T, float eps, const Rope& rope) {
    add_norm(R, pend ? b.bo : nullptr, L.attn_norm, b.mixed, T, eps, b.st, b.xtq, b.xtd);
    if (L.wq.n_out != S27_QKV) throw std::runtime_error("s27: an attention layer's q|k|v is not S27_QKV rows");
    mv(b, L.wq.s, S27_QKV, N, b.xq, b.qkv, S27_QKV, T);
    if (!skip("attn")) {
        k_s27_prep<<<dim3(S27_NQ + 2 * S27_NKV, T), HD, 0, b.st>>>(b.qkv, L.q_norm, L.k_norm, kc, vc, ctl, eps, rope);
        k_s27_attn<<<dim3(S27_ASPLIT, T, S27_NQ), 128, 0, b.st>>>(b.qkv, kc, vc, ctl, kv0, q0, b.apart);
        k_s27_merge<<<dim3(T, S27_NQ), HD, 0, b.st>>>(b.qkv, b.apart, b.att);
        ck(cudaGetLastError(), "attention");
    }
    quantize_q8_1(b.att, b.xq, S27_NQ * HD, T, b.st);
    mv(b, L.wo.s, L.wo.n_out, L.wo.n_in, b.xq, b.bo, N, T);
    sum_cards(b, b.bo, T, ctl, call);
}

void ffn_half(S27Card& b, const CardW& w, const LayerW& L, float* R, const int* ctl, int call, int T, float eps) {
    const int ffs = w.fd1 - w.fd0;
    add_norm(R, b.bo, L.ffn_norm, b.mixed, T, eps, b.st, b.xtq, b.xtd);
    mv(b, L.ffn_gu.s, 2 * ffs, N, b.xq, b.gu, 2 * ffs, T);
    swiglu_stacked_q8_1(b.gu, b.hq, ffs, T, b.st);
    mv(b, L.ffn_down.s, N, ffs, b.hq, b.bo, N, T);
    sum_cards(b, b.bo, T, ctl, call);
}
}  // namespace

S27::S27(Model& m, int max_ctx) : m_(m), max_ctx_(max_ctx) {
    if (max_ctx > m.ctx_train) throw std::runtime_error("s27: context past the trained length (no rope scaling here yet)");
    rope_.base = m.rope_base;
    cb_.resize(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        S27Card& b = cb_[c];
        b.dev = w.dev;
        ck(cudaSetDevice(b.dev), "set");
        for (int p = 0; p < NCARD; ++p)
            if (p != c) {
                const cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "peer access");
                cudaGetLastError();
            }
        ck(cudaStreamCreateWithFlags(&b.st, cudaStreamNonBlocking), "stream");
        ck(cudaEventCreate(&b.ev0), "event");
        ck(cudaEventCreate(&b.ev1), "event");
        const int ffs = w.fd1 - w.fd0;
        b.R = dalloc<float>((size_t) MAXT * N);
        b.mixed = dalloc<float>((size_t) MAXT * N);
        b.bo = dalloc<float>((size_t) MAXT * N);
        b.hfin = dalloc<float>((size_t) MAXT * N);
        b.xq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(XQ_COLS));
        b.hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        b.xtq = dalloc<int8_t>((size_t) MAXT * XQ_COLS);
        b.xtd = dalloc<float>((size_t) MAXT * XQ_COLS / 32);
        q8_twin(b.xq, (size_t) MAXT * q8_1_bytes(XQ_COLS), b.xtq, b.xtd);
        b.htq = dalloc<int8_t>((size_t) MAXT * ffs);
        b.htd = dalloc<float>((size_t) MAXT * ffs / 32);
        q8_twin(b.hq, (size_t) MAXT * q8_1_bytes(ffs), b.htq, b.htd);
        b.qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b.ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b.gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        if (w.nq != S27_NQ || w.nkv != S27_NKV || w.q0 != c * S27_NQ || w.kv0 != w.q0 / (NQH / NKVH))
            throw std::runtime_error("s27: card " + std::to_string(c) + "'s attention heads are not the split s27_attn.cuh runs");
        b.qkv = dalloc<float>((size_t) MAXT * S27_QKV);
        b.att = dalloc<float>((size_t) MAXT * S27_NQ * HD);
        b.apart = dalloc<float>((size_t) MAXT * S27_NQ * S27_ASPLIT * (HD + 2));
        b.gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b.head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b.best_v = dalloc<float>(MAXT);
        b.best_i = dalloc<int>(MAXT);
        b.am_keys = dalloc<unsigned long long>(MAXT);
        b.me = dalloc<float>((size_t) MAXT * N);
        b.mh = dalloc<float>((size_t) MAXT * N);
        b.mcat = dalloc<float>((size_t) MAXT * 2 * N);
        b.mR = dalloc<float>((size_t) MAXT * N);
        b.mout = dalloc<float>(N);
        b.mlogits = dalloc<float>((size_t) std::max(1, w.dv1 - w.dv0));
        b.msum = dalloc<float>(1);
        b.mctl = dalloc<int>(2);
        b.S.assign(NL, nullptr);
        b.chist.assign(NL, nullptr);
        b.snap.assign(NL, nullptr);
        b.hsnap.assign(NL, nullptr);
        b.kc.assign(NL, nullptr);
        b.vc.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l)
            if (!is_attn(l)) {
                b.S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b.chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b.snap[l] = dalloc<float>((size_t) (MAXT - 1) * w.nv() * HS * HS);
                b.hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        b.ctl = dalloc<int>(2);
        b.ring.card = c;
        b.ring.in = dalloc<unsigned long long>(ring_in_words(MAXT));
        b.ring.ag = dalloc<unsigned long long>(ring_ag_words(MAXT));
        b.ring.f32 = ring_f32_env();
        b.ring.err = dalloc<unsigned>(1);
        b.r3.card = c;
        b.r3.in = dalloc<unsigned long long>(r3_in_words());
        b.r3.ag = dalloc<unsigned long long>(r3_ag_words());
        b.r3.err = b.ring.err;
        // K/V: every position of the card's two KV heads, f16, a pair of buffers per attention layer and the draft layer
        const size_t kv = (size_t) max_ctx * S27_NKV * HD;
        for (int l = 0; l < NL; ++l)
            if (is_attn(l)) b.kc[l] = dalloc<__half>(kv), b.vc[l] = dalloc<__half>(kv);
        b.mkc = dalloc<__half>(kv);
        b.mkv = dalloc<__half>(kv);
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "meminfo");
        std::fprintf(stderr, "s27: card %d: %zu MiB free after the engine's buffers\n", c, fr >> 20);
    }
    for (int c = 0; c < NCARD; ++c) {
        cb_[c].ring.next_in = cb_[(c + 1) % NCARD].ring.in;
        cb_[c].ring.next_ag = cb_[(c + 1) % NCARD].ring.ag;
        cb_[c].r3.next_in = cb_[(c + 1) % NCARD].r3.in;
        cb_[c].r3.next_ag = cb_[(c + 1) % NCARD].r3.ag;
    }
    h_emb_ = halloc<float>((size_t) MAXT * N);
    h_memb_ = halloc<float>((size_t) MAXT * N);
    h_ctl_ = halloc<int>(2);
    h_mctl_ = halloc<int>(2);
    h_bv_ = halloc<float>(NCARD * MAXT);
    h_bi_ = halloc<int>(NCARD * MAXT);
    h_err_ = halloc<unsigned>(NCARD);
    h_ms_ = halloc<float>(NCARD);
    h_jobs_ = halloc<CopyJob>((size_t) NCARD * 2 * NL);
    for (S27Card& b : cb_) {
        ck(cudaSetDevice(b.dev), "set");
        commit_jobs_.push_back(dalloc<CopyJob>(2 * NL));
        ck(cudaDeviceSynchronize(), "init");
    }
}

// a trunk window's tag continues the history parity (it reads half tag & 1, which the last trunk window wrote); a
// draft pass's needs only to be new
unsigned S27::next_tag(bool trunk) {
    unsigned t = tag_max_ + 1u;
    if (trunk && (t & 1u) != ((tag_ + 1u) & 1u)) ++t;
    if (t >= (1u << 24)) throw std::runtime_error("s27: ring tags ran out (no clearing here yet)");
    if (t / 128 != tag_max_ / 128)   // s27_ring3.cuh's 16-bit tags: no word in its buffers 256 tags old (every card idle here)
        for (S27Card& b : cb_) {
            ck(cudaSetDevice(b.dev), "set");
            ck(cudaMemsetAsync(b.r3.in, 0, r3_in_words() * 8, b.st), "ring3 clear");
            ck(cudaMemsetAsync(b.r3.ag, 0, r3_ag_words() * 8, b.st), "ring3 clear");
        }
    tag_max_ = t;
    if (trunk) tag_ = t;
    return t;
}

void S27::embed(const std::vector<int>& tokens, float* out) const {
    const Tensor& e = *m_.tok_embd;
    const size_t rb = e.row_bytes();
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (tokens[t] < 0 || tokens[t] >= m_.vocab) throw std::runtime_error("embed: token outside the vocabulary");
        const uint8_t* row = e.data + (size_t) tokens[t] * rb;
        float* o = out + t * N;
        for (int bl = 0; bl < N / 32; ++bl) {
            __half_raw hr;
            std::memcpy(&hr.x, row + bl * 34, 2);
            const float d = __half2float(__half(hr));
            for (int j = 0; j < 32; ++j) o[bl * 32 + j] = d * (float) (int8_t) row[bl * 34 + 2 + j];
        }
    }
}

void S27::enqueue(int T, std::FILE* dump) {
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c].dev), "set");
            f(cb_[c], m_.cards[c]);
        }
    };
    // a debugging dump: card 0's rows of p (n columns) as one record, after everything queued so far
    auto dd = [&](const std::string& name, const float* p, int n) {
        if (!dump) return;
        ck(cudaSetDevice(cb_[0].dev), "set");
        ck(cudaStreamSynchronize(cb_[0].st), "dump sync");
        std::vector<float> h((size_t) T * n);
        ck(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost), "dump copy");
        const uint32_t nl = (uint32_t) name.size(), nd = 2;
        const int64_t ne[2] = {n, T};
        std::fwrite(&nl, 4, 1, dump);
        std::fwrite(name.data(), 1, nl, dump);
        std::fwrite(&nd, 4, 1, dump);
        std::fwrite(ne, 8, 2, dump);
        std::fwrite(h.data(), 4, h.size(), dump);
    };
    dd("model.input_embed", cb_[0].R, N);
    for (int l = 0; l < NL; ++l) {
        each([&](S27Card& b, const CardW& w) {
            const LayerW& L = w.L[l];
            if (!is_attn(l)) {
                const int nv = w.nv();
                add_norm(b.R, l > 0 ? b.bo : nullptr, L.attn_norm, b.mixed, T, eps, b.st, b.xtq, b.xtd);
                q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                if (!skip("gdn")) gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                         b.hsnap[l], w.nk, nv, w.vmap, T, eps, b.st);
                mv(b, L.ssm_out.s, L.ssm_out.n_out, L.ssm_out.n_in, b.xq, b.bo, N, T);
                sum_cards(b, b.bo, T, b.ctl, call);
            } else {
                attn_half(b, L, b.R, l > 0, b.ctl, b.kc[l], b.vc[l], w.q0, w.kv0, call, T, eps, rope_);
            }
        });
        ++call;
        dd("attn_residual-" + std::to_string(l), cb_[0].R, N);
        each([&](S27Card& b, const CardW& w) { ffn_half(b, w, w.L[l], b.R, b.ctl, call, T, eps); });
        ++call;
        dd("l_out-" + std::to_string(l), cb_[0].R, N);
    }
    each([&](S27Card& b, const CardW& w) {
        add_norm(b.R, b.bo, w.out_norm, b.mixed, T, eps, b.st, b.xtq, b.xtd);
        ck(cudaMemcpyAsync(b.hfin, b.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "hfin");
        mv(b, w.out.s, w.v1 - w.v0, N, b.xq, b.head_logits, w.v1 - w.v0, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
}

// the draft layer over T rows of (token embedding me, h mh): its projection, attention and FFN halves (the trunk's
// split, its own K/V slot), then the last row through the head norm (kept in mout for a chained pass) and the draft head
void S27::enqueue_draft(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        k_norm_cat<<<T, 1024, 0, b.st>>>(b.me, b.mh, w.mtp_enorm, w.mtp_hnorm, b.mcat, eps);
        ck(cudaGetLastError(), "norm_cat");
        quantize_q8_1(b.mcat, b.xq, 2 * N, T, b.st);
        mv(b, w.mtp_eh.s, N, 2 * N, b.xq, b.mR, N, T);
        attn_half(b, w.mtp, b.mR, false, b.mctl, b.mkc, b.mkv, w.q0, w.kv0, 0, T, eps, rope_);
    }
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(b, w, w.mtp, b.mR, b.mctl, 1, T, eps);
        add_norm(b.mR, b.bo, w.mtp_head_norm, b.mixed, T, eps, b.st);
        ck(cudaMemcpyAsync(b.mout, b.mixed + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, b.st), "draft out");
        quantize_q8_1(b.mout, b.xq, N, 1, b.st);
        mv(b, w.mtp_head.s, w.dv1 - w.dv0, N, b.xq, b.mlogits, w.dv1 - w.dv0, 1);
        argmax(b.mlogits, w.dv1 - w.dv0, 1, w.dv0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
        k_sumexp<<<1, 1024, 0, b.st>>>(b.mlogits, w.dv1 - w.dv0, b.best_v, b.msum);
        ck(cudaGetLastError(), "sumexp");
    }
}

std::vector<int> S27::verify(const std::vector<int>& tokens) {
    const int T = (int) tokens.size();
    if (pending_) throw std::runtime_error("verify: the last window is not committed");
    if (T < 1 || T > MAXT) throw std::runtime_error("verify: 1..MAXT tokens");
    if (pos_ + T > max_ctx_) throw std::runtime_error("verify: context full");
    embed(tokens, h_emb_);
    h_ctl_[0] = pos_;
    h_ctl_[1] = (int) next_tag(true);
    std::FILE* dump = nullptr;
    if (const char* dp = std::getenv("S27_DUMP"); dp && pos_ == 0) {
        dump = std::fopen(dp, "wb");
        if (!dump) throw std::runtime_error(std::string("S27_DUMP: cannot write ") + dp);
    }
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaMemcpyAsync(b.R, h_emb_, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "emb");
        ck(cudaMemcpyAsync(b.ctl, h_ctl_, 8, cudaMemcpyHostToDevice, b.st), "ctl");
    }
    if (dump) {
        enqueue(T, dump);
        std::fclose(dump);
        std::fprintf(stderr, "s27: dumped the first window\n");
    } else {
        if (!cb_[0].graphs[T]) {
            for (S27Card& b : cb_) {
                ck(cudaSetDevice(b.dev), "set");
                ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "capture");
            }
            enqueue(T, nullptr);
            for (S27Card& b : cb_) {
                ck(cudaSetDevice(b.dev), "set");
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                ck(cudaGraphInstantiate(&b.graphs[T], g, 0), "instantiate");
            }
        }
        for (S27Card& b : cb_) {
            ck(cudaSetDevice(b.dev), "set");
            ck(cudaGraphLaunch(b.graphs[T], b.st), "launch");
        }
    }
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemcpyAsync(h_bv_ + c * T, b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "bv");
        ck(cudaMemcpyAsync(h_bi_ + c * T, b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "bi");
        ck(cudaMemcpyAsync(h_err_ + c, b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    }
    unsigned err = 0;
    last_gpu_ms = 0.f;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c].dev), "set");
        ck(cudaStreamSynchronize(cb_[c].st), "window");
        err |= h_err_[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c].ev0, cb_[c].ev1), "elapsed");
        last_card_ms[c] = ms;
        last_gpu_ms = std::max(last_gpu_ms, ms);
    }
    if (err) throw std::runtime_error("window: ring/attention error bits " + std::to_string(err));
    std::vector<int> out(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_bv_[c * T + t] > h_bv_[best * T + t]) best = c;
        out[t] = h_bi_[best * T + t];
    }
    pending_ = T;
    return out;
}

int S27::draft(const std::vector<int>& tokens, int pos, bool from_trunk) {
    const int T = (int) tokens.size();
    if (T < 1 || T > MAXT) throw std::runtime_error("draft: 1..MAXT tokens");
    if (!from_trunk && T != 1) throw std::runtime_error("draft: a chained pass is one row");
    if (pos < 0 || pos + T > max_ctx_) throw std::runtime_error("draft: positions outside the context");
    embed(tokens, h_memb_);
    h_mctl_[0] = pos;
    h_mctl_[1] = (int) next_tag(false);
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemcpyAsync(b.me, h_memb_, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft emb");
        ck(cudaMemcpyAsync(b.mctl, h_mctl_, 8, cudaMemcpyHostToDevice, b.st), "draft ctl");
        ck(cudaMemcpyAsync(b.mh, from_trunk ? b.hfin : b.mout, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, b.st), "draft h");
    }
    if (!cb_[0].mgraphs[T]) {
        for (S27Card& b : cb_) {
            ck(cudaSetDevice(b.dev), "set");
            ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "capture");
        }
        enqueue_draft(T);
        for (S27Card& b : cb_) {
            ck(cudaSetDevice(b.dev), "set");
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b.st, &g), "end capture");
            ck(cudaGraphInstantiate(&b.mgraphs[T], g, 0), "instantiate");
        }
    }
    for (int c = 0; c < NCARD; ++c) {
        S27Card& b = cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaGraphLaunch(b.mgraphs[T], b.st), "launch");
        ck(cudaMemcpyAsync(h_bv_ + c, b.best_v, 4, cudaMemcpyDeviceToHost, b.st), "bv");
        ck(cudaMemcpyAsync(h_ms_ + c, b.msum, 4, cudaMemcpyDeviceToHost, b.st), "msum");
        ck(cudaMemcpyAsync(h_bi_ + c, b.best_i, 4, cudaMemcpyDeviceToHost, b.st), "bi");
        ck(cudaMemcpyAsync(h_err_ + c, b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "err");
    }
    unsigned err = 0;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c].dev), "set");
        ck(cudaStreamSynchronize(cb_[c].st), "draft");
        err |= h_err_[c];
    }
    if (err) throw std::runtime_error("draft: ring/attention error bits " + std::to_string(err));
    int best = 0;
    for (int c = 1; c < NCARD; ++c)
        if (h_bv_[c] > h_bv_[best]) best = c;
    // the draft's probability under the draft head: exp(max) over the cards' sums, each about its own max
    double s = 0;
    for (int c = 0; c < NCARD; ++c) s += (double) h_ms_[c] * std::exp((double) h_bv_[c] - (double) h_bv_[best]);
    last_draft_p = (float) (1.0 / s);
    return m_.dvocab[(size_t) h_bi_[best]];
}

void S27::commit(int n) {
    const int T = pending_;
    if (!T) throw std::runtime_error("commit: no window");
    if (n < 1 || n > T) throw std::runtime_error("commit: keep 1..T rows");
    if (n < T) {
        // the window read history half tag & 1 and wrote the other: the kept row's state and history replace them
        const unsigned wrote = (tag_ & 1u) ^ 1u;
        for (int c = 0; c < NCARD; ++c) {
            S27Card& b = cb_[c];
            const CardW& w = m_.cards[c];
            const size_t sbytes = (size_t) w.nv() * HS * HS * 4, hfloats = (size_t) 3 * w.conv_ch();
            CopyJob* jobs = h_jobs_ + (size_t) c * 2 * NL;
            int k = 0;
            for (int l = 0; l < NL; ++l) {
                if (is_attn(l)) continue;
                jobs[k++] = {b.S[l], b.snap[l] + (size_t) (n - 1) * w.nv() * HS * HS, sbytes};
                jobs[k++] = {b.chist[l] + wrote * hfloats, b.hsnap[l] + (size_t) (n - 1) * hfloats, hfloats * 4};
            }
            ck(cudaSetDevice(b.dev), "set");
            ck(cudaMemcpyAsync(commit_jobs_[c], jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit jobs");
            copy_many(commit_jobs_[c], k, b.st);
        }
        for (S27Card& b : cb_) {   // the host table is written again by the next commit
            ck(cudaSetDevice(b.dev), "set");
            ck(cudaStreamSynchronize(b.st), "commit");
        }
    }
    pos_ += n;
    pending_ = 0;
}

}  // namespace fnx
