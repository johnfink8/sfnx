#include <cstdlib>
#include "engine.hpp"
#include "engine_impl.hpp"
#include "env.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fnx {

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// a graph just captured into its slot: the slot's executable updated in place when it has one (the same kernels with
// other pointers: no new memory), else instantiated; key: what it was captured for
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

// MemAvailable from /proc/meminfo, in MiB (-1 if unreadable): what a failed page-locked allocation had to work with
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
// the ring stores straight into the next card's memory: with translated IOMMU domains every such write faults
// (dmesg: "DMAR: ... PTE Write access is not set") and the first window ends in a ring timeout that names nothing.
// Each card's IOMMU group must be identity (boot with iommu=pt); a card without a group has no IOMMU in the way
void require_identity_iommu(int dev) {
    char bus[32];
    ck(cudaDeviceGetPCIBusId(bus, sizeof bus, dev), "cudaDeviceGetPCIBusId");
    // CUDA's spelling ("0000:3B:00.0") in sysfs's (lower case, a 4-digit domain)
    unsigned dom = 0, b = 0, d = 0, fn = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &fn) != 4)
        throw std::runtime_error("card " + std::to_string(dev) + ": cannot read its PCI address '" + bus + "'");
    char bdf_s[32];
    std::snprintf(bdf_s, sizeof bdf_s, "%04x:%02x:%02x.%x", dom, b, d, fn);
    const std::string bdf(bdf_s);
    const std::string dir = "/sys/bus/pci/devices/" + bdf;
    if (access(dir.c_str(), F_OK) != 0)
        throw std::runtime_error("card " + std::to_string(dev) + " (" + bdf + "): no " + dir + " to read its IOMMU group from");
    const std::string path = dir + "/iommu_group/type";
    if (access((dir + "/iommu_group").c_str(), F_OK) != 0) return;
    std::FILE* f = std::fopen(path.c_str(), "r");
    char type[64] = {0};
    if (!f || !std::fgets(type, sizeof type, f)) {
        if (f) std::fclose(f);
        throw std::runtime_error("card " + std::to_string(dev) + " (" + bdf + "): cannot read " + path);
    }
    std::fclose(f);
    std::string t(type);
    while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
    if (t != "identity")
        throw std::runtime_error("card " + std::to_string(dev) + " (" + bdf + ") is in an IOMMU group of type '" + t + "' (" + path +
                                 "): the ring's writes between cards need identity domains; boot with iommu=pt");
}
}  // namespace


Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    // the rotary's scaling (choose_rope, src/rope.hpp): the model's own YaRN, FNX_ROPE_SCALE / FNX_YARN_ORIG_CTX, or
    // both when they agree; set once here, before any graph capture. Positions past what the rotary covers have angles
    // the model never saw, so a context past that is refused here rather than served
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
        std::fprintf(stderr, "fnx: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
        if (!rc.from_model && max_ctx <= m_.ctx_train)
            std::fprintf(stderr, "fnx: note: context %d is within the trained %d: the scaling changes every position's rotary "
                         "and no length needs it\n", max_ctx, m_.ctx_train);
    }
    const long long pf_rows = env_int("FNX_PF_ROWS", pf_c_);
    if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
        throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " +
                                 std::to_string(pf_rows));
    pf_c_ = (int) pf_rows;
    hc1_rows();   // FNX_HC1 checked now, not at the first hc read (inside a graph capture)
    pf_jobs_ = pf_c_ * TOPK / PF_JOB + NEXP;   // jobs of up to PF_JOB pairs per expert, at most
    set_prompt_rows(pf_c_);
    for (int c = 0; c < NCARD; ++c) require_identity_iommu(m.cards[c].dev);
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    // the paged K/V pool: pages for the whole context (one more for the block a position past the last one needs)
    // and the extra lanes' blocks. Int8 rows: the context's pages as many as fit in their f16 bytes (the smallest ratio
    // of the cards': every card has the same pages)
    int ctx_pages = max_ctx / KVP_POS + 2;
    if (kv_q8()) {
        double ratio = 1e9;
        for (int cc = 0; cc < NCARD; ++cc) ratio = std::min(ratio, (double) kvp_page(cc, false) / kvp_page(cc, true));
        ctx_pages = (int) (ctx_pages * ratio);
    }
    int n_pages = ctx_pages + extra_pages;
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
        ck(cudaStreamCreateWithFlags(&b->st2, cudaStreamNonBlocking), "stream");
        own_stream(b->st2);
        ck(cudaEventCreateWithFlags(&b->fork, cudaEventDisableTiming), "event");
        own_event(b->fork);
        ck(cudaEventCreateWithFlags(&b->join, cudaEventDisableTiming), "event");
        own_event(b->join);
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        const int P = MAXT * TOPK, ffs = w.ff1 - w.ff0;
        b->R = dalloc<float>((size_t) MAXT * D);
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo_a = dalloc<float>((size_t) MAXT * N);
        b->bo_f = dalloc<float>((size_t) MAXT * N);
        b->inj_a = dalloc<float>(MAXT * HC);
        b->inj_f = dalloc<float>(MAXT * HC);
        b->hcs = dalloc<float>(hc_scratch_floats(MAXT));
        if (std::getenv("FNX_HC_BAR_WRAP")) {
            const long long n = env_int("FNX_HC_BAR_WRAP", 0);
            if (n <= 0 || n > 0xffffffffLL) throw std::runtime_error("FNX_HC_BAR_WRAP: want a barrier count > 0, got " + std::to_string(n));
            hc_preset_barrier(b->hcs, (unsigned) n);
        }
        b->xq = dalloc<uint8_t>((size_t) P * q8_1_bytes(N));
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qg = dalloc<float>((size_t) MAXT * NQH * 2 * HD);
        b->k = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->v = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->att = dalloc<float>((size_t) MAXT * w.nq * HD);
        b->att_part = dalloc<float>(attn_scratch_floats(w.nq));
        b->logits = dalloc<float>((size_t) MAXT * (NEXP + 1));
        b->topw = dalloc<float>(P);
        b->topi = dalloc<int>(P);
        b->njobs = dalloc<int>(2);
        b->jobs_gu = dalloc<Job>(P + 1);
        b->jobs_dn = dalloc<Job>(P + 1);
        for (int T = 1; T <= MAXT; ++T) {
            Job j{};
            j.w = 0;
            j.n = T;
            for (int t = 0; t < T; ++t) { j.y[t] = t; j.out[t] = t; }
            b->dense[T] = dalloc<Job>(1);
            ck(cudaMemcpy(b->dense[T], &j, sizeof j, cudaMemcpyHostToDevice), "dense job");
        }
        if (w.ec.n_slots) b->cache_list = dalloc<int>(1 + P);
        b->gu = dalloc<float>((size_t) (P + MAXT) * 2 * ffs);
        b->part = dalloc<float>((size_t) (P + MAXT) * N);
        b->sh_gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->sh_hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        if (!std::getenv("FNX_NO_TWIN")) {
            // the split twins: xq's rows of n <= N columns sit at r q8_1_bytes(n), so r n < P N
            q8_twin(b->xq, (size_t) P * q8_1_bytes(N), dalloc<int8_t>((size_t) P * N), dalloc<float>((size_t) P * N / 32));
            q8_twin(b->sh_hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
            for (const void* q : {(const void*) b->xq, (const void*) b->sh_hq}) owned().push_back([q] { q8_untwin(q); });
        }
        b->ple_emb = dalloc<float>((size_t) MAXT * N);
        b->ple_key = dalloc<float>((size_t) 3 * MAXT * D);   // key, then room for gated and normalized rows
        b->ple_val = dalloc<float>((size_t) MAXT * N);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        if (c == 0 && env_flag("FNX_PREFETCH_PROBE")) {
            b->probe_logits = dalloc<float>((size_t) MAXT * NEXP);
            b->probe_guess = dalloc<int>((size_t) NL * MAXT * TOPK);
            b->probe_cnt = dalloc<unsigned long long>(7);
        }
        if (c == 0 && env_flag("FNX_ROUTE_COUNT")) {
            b->route_cnt = dalloc<unsigned>((size_t) 2 * NL * NEXP);
            ck(cudaMemset(b->route_cnt, 0, (size_t) 2 * NL * NEXP * sizeof(unsigned)), "route counts");
        }
        if (const long long tr = env_int("FNX_ROUTE_TRACE", 0); c == 0 && tr > 0) {
            b->route_trace_cap = (unsigned long long) tr * 1000000ull;
            void* hp = nullptr;   // in mapped host memory: no VRAM for it
            ck(cudaHostAlloc(&hp, b->route_trace_cap * sizeof(unsigned), cudaHostAllocMapped), "route trace");
            own_pinned(hp);
            b->route_trace = (unsigned*) hp;
            b->route_trace_len = dalloc<unsigned long long>(1);
            ck(cudaMemset(b->route_trace_len, 0, sizeof(unsigned long long)), "route trace");
        }
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        b->kc.assign(NL, nullptr);
        b->vc.assign(NL, nullptr);
        b->qsa_ring.assign(NL, nullptr);
        b->qsa_pooled.assign(NL, nullptr);
        b->qsa_y = dalloc<float>((size_t) MAXT * qsa_rows());
        b->qsa_q = dalloc<float>((size_t) MAXT * 512);
        b->qsa_scores = dalloc<float>((size_t) MAXT * (max_ctx / 4));
        b->qsa_cells = dalloc<int>((size_t) MAXT * qsa_maxsel());
        b->qsa_ncells = dalloc<int>(MAXT);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l) {
            if (is_attn(l)) {
                b->qsa_ring[l] = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);
            } else {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        }
        b->ple_hist = dalloc<float>((size_t) 9 * D);
        b->ple_hsnap = dalloc<float>((size_t) MAXT * 9 * D);
        b->ctl = dalloc<int>(2);
        b->ring.card = c;
        // sized for fp32 words (two per pair); fp16 pairs use the first half of each slot row's room
        b->ring.in = dalloc<unsigned long long>(ring_in_words(MAXT));
        b->ring.ag = dalloc<unsigned long long>(ring_ag_words(MAXT));
        b->ring.f32 = ring_f32();
        b->ring.err = dalloc<unsigned>(1);
        b->att_ring.card = c;
        b->att_ring.in = dalloc<unsigned long long>(att_ring_words());
        b->att_ring.err = b->ring.err;
        cb_.push_back(b);
    }
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->ring.next_in = cb_[(c + 1) % NCARD]->ring.in;
        cb_[c]->ring.next_ag = cb_[(c + 1) % NCARD]->ring.ag;
        cb_[c]->att_ring.next_in = cb_[(c + 1) % NCARD]->att_ring.in;
    }
    // the prompt path's buffers now, so the arena below is sized from what is left after them
    prefill_alloc();
    // FNX_ARENA: the pool's own pages are those, or as many as every card's free VRAM holds beside the arena pool's
    // slots and FNX_ARENA_RESERVE_MIB, or FNX_ARENA_BASE_PAGES; the slots follow the own pages in the same allocation,
    // and the pool grows into them (lent coldest first) up to where every card keeps 2 MAXT TOPK slots. The buffers
    // above and the draft layer (an Mtp is made before its Engine) are already allocated, so the reserve is only what
    // comes later: the window graphs captured at first use (with batched drafts a verify graph per row total), cuBLAS's
    // workspaces and the lanes' page tables. Default 512: they took ~270 MiB a card on three cards and ~340 on two,
    // through one to six lanes, batched drafts and a 114k prompt
    if (m.arena) {
        const long long reserve_mib = env_int("FNX_ARENA_RESERVE_MIB", 512);
        if (reserve_mib < 0) throw std::runtime_error("FNX_ARENA_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
        const size_t reserve = (size_t) reserve_mib << 20;
        // a card with less free VRAM than the slots, the reserve and the smallest own pool (two cards, another process
        // on it, a leak): the coldest experts that many slots' worth go to RAM, placed again, and the engine goes on.
        // Too few slots left for a window is still an error (Model::place_experts says so)
        for (int round = 0;; ++round) {
            int more = 0;
            size_t short_mib = 0;
            for (int c = 0; c < NCARD; ++c) {
                const ExpertCache& ec = m.cards[c].ec;
                if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                const size_t sb = ec.slot_bytes[ec.arena_pool],
                             need = (size_t) ec.n[ec.arena_pool] * sb + reserve + (size_t) (extra_pages + 1) * kvp_page(c) * sizeof(uint16_t) +
                                    2 * pf_stage_bytes(c);
                if (fr < need) {   // 256 MiB more than short, so the next engine's few MiB of drift place nothing again
                    more = std::max(more, (int) ((need - fr + (256u << 20) + sb - 1) / sb));
                    short_mib = std::max(short_mib, (need - fr) >> 20);
                }
            }
            if (!more) break;
            if (round == 8) throw std::runtime_error("FNX_ARENA: VRAM still short after placing the experts again 8 times");
            // two cards are short by design; on three the experts fit unless something else holds VRAM
            std::fprintf(stderr, "fnx: VRAM short by %zu MiB on a card%s: %d more of the coldest experts to RAM\n", short_mib,
                         NCARD == 3 ? " (another process on it?)" : "", more);
            m.free_experts();
            m.ram_extra += more;
            m.place_experts();
        }
        prefill_stages();
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
            ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            const size_t slots = (size_t) ec.n[ec.arena_pool] * ec.slot_bytes[ec.arena_pool], page = kvp_page(c) * sizeof(uint16_t);
            const int fit = fr > slots + reserve ? (int) ((fr - slots - reserve) / page) : 0;
            if (fit <= extra_pages)
                throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: the arena's " +
                                         std::to_string(slots >> 20) + " MiB of slots, " + std::to_string(reserve >> 20) +
                                         " MiB in reserve (FNX_ARENA_RESERVE_MIB) and the lanes' blocks do not fit");
            n_pages = std::min(n_pages, fit);
        }
        if (std::getenv("FNX_ARENA_BASE_PAGES")) {
            const long long want = env_int("FNX_ARENA_BASE_PAGES", 0);
            if (want <= extra_pages || want > n_pages)
                throw std::runtime_error("FNX_ARENA_BASE_PAGES: want " + std::to_string(extra_pages + 1) + " to " + std::to_string(n_pages) +
                                         " (the lanes' blocks take " + std::to_string(extra_pages) + ")");
            n_pages = (int) want;
        }
        arena_base_ = arena_w_ = n_pages;
        arena_cap_ = INT32_MAX;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool, keep = 2 * MAXT * TOPK;
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            const size_t room = (size_t) (ec.n[p] - keep) * ec.slot_bytes[p] + (arena_slot_off(c) - (size_t) n_pages * page);
            arena_cap_ = std::min(arena_cap_, n_pages + (int) (room / page));
        }
    } else {
        prefill_stages();
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        // the K/V pool and lane 0's page table (pages taken as it grows); with an arena the pool is its start and
        // n_pages counts the pages it can grow to
        if (m.arena) {
            ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            b->n_pages = arena_cap_;
            char* region = (char*) dalloc<uint8_t>(arena_slot_off(c) + (size_t) ec.n[p] * ec.slot_bytes[p]);
            b->pool = (uint16_t*) region;
            ec.pool[p] = region + arena_slot_off(c);
            m.cards[c].bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
            arena_fill(c, b->st);
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            std::fprintf(stderr, "fnx: arena: card %d: %zu MiB free after its arena\n", c, fr >> 20);
        } else {
            b->n_pages = n_pages;
            b->pool = dalloc<uint16_t>((size_t) b->n_pages * kvp_page(c));
        }
        b->pt = dalloc<int>((size_t) b->n_pages);
        b->pg.pt = b->pt;
        b->pg.page = kvp_page(c);
        b->pg.rows = kvp_rows(c);
        b->pg.q8 = kv_q8();
        const size_t kv_halves = kvp_rows_halves(c, kv_q8());
        for (int l = 0, slot = 0; l < NL; ++l)
            if (is_attn(l)) {
                b->kc[l] = b->pool + (size_t) slot * kvp_slot(c);
                b->vc[l] = b->kc[l] + kv_halves;
                b->qsa_pooled[l] = b->kc[l] + 2 * kv_halves;
                ++slot;
            }
        {   // lane 0: the buffers above
            LaneDev L;
            L.pt = b->pt;
            L.S = b->S;
            L.chist = b->chist;
            L.ple_hist = b->ple_hist;
            L.ring = b->qsa_ring;
            L.ctl = dalloc<int>(2);
            b->lanes.push_back(L);
        }
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    for (int i = m.arena ? arena_base_ : cb_[0]->n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
    if (m.arena) {
        std::string per;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            char buf[96];
            std::snprintf(buf, sizeof buf, "%s%d x %.2f MiB (%.1f GiB)", c ? ", " : "", ec.n[p], ec.slot_bytes[p] / 1048576.0,
                          (double) ec.n[p] * ec.slot_bytes[p] / (1ull << 30));
            per += buf;
        }
        std::fprintf(stderr, "fnx: arena: the K/V pool's own %d pages (%d positions), up to %d (%d positions) by lending the slots of %s\n",
                     arena_base_, arena_base_ * KVP_POS, arena_cap_, arena_cap_ * KVP_POS, per.c_str());
    }
    if (std::getenv("FNX_RING_TAG_START")) {
        const long long v = env_int("FNX_RING_TAG_START", 0);
        if (v < 1 || v >= (long long) TAG_END)
            throw std::runtime_error("FNX_RING_TAG_START: want 1.." + std::to_string(TAG_END - 1) + ", got " + std::to_string(v));
        tag_max_ = (unsigned) v;
    }
}

void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemset(b.ring.in, 0, ring_in_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.ring.ag, 0, ring_ag_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");
        if (!pf_.empty()) pf_clear_ring(c);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    std::fprintf(stderr, "fnx: ring tags reached %u: the rings' buffers cleared, tags start again\n", tag_max_);
    tag_max_ = 1;
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
    if (tag_max_ + n + 1u >= TAG_END) clear_rings();
}

unsigned Engine::ring_tag() {
    tags_room(1);
    return ++tag_max_;
}

unsigned Engine::take_tags(int n) {
    tags_room((unsigned) n);
    unsigned base = tag_max_ + 1u;
    if ((base & 1u) != ((tag_ + 1u) & 1u)) ++base;
    tag_ = base + (unsigned) n - 1u;
    tag_max_ = tag_;
    return base;
}

// an arena's K/V pool grows by at least this many pages a lend (64: ~12k positions f16), so lends stay rare
constexpr int ARENA_STEP = 64;

size_t Engine::arena_slot_off(int c) const {
    return ((size_t) arena_base_ * kvp_page(c) * sizeof(uint16_t) + 255) & ~(size_t) 255;
}

// the arena pool as loaded: the slot tables back to the model's first assignment (an earlier engine in this process
// lent, reclaimed and replaced slots), every expert of the pool's layers at its host copy, then each slot's expert
// copied in from it
void Engine::arena_fill(int c, cudaStream_t st) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool;
    // only the arena pool's part: the other pools' slots keep what an earlier engine copied into them, and their
    // tables say so
    const int s0 = ec.s0[p], n = ec.n[p];
    ck(cudaMemcpy(ec.owner + s0, w.ec_owner0.data() + s0, (size_t) n * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) slot_of[(size_t) l * NEXP + e] = w.ec_slot0[(size_t) l * NEXP + e];
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemset(ec.ref + s0, 0, (size_t) n), "arena: refs");
    ck(cudaMemcpy(ec.hand + p, &ec.s0[p], sizeof(int), cudaMemcpyHostToDevice), "arena: hand");
    ck(cudaMemset(ec.lent + p, 0, sizeof(int)), "arena: lent");
    const std::vector<int> owner(w.ec_owner0.begin() + s0, w.ec_owner0.begin() + s0 + n);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                epg[i] = w.ec_host_gu[i];
                epd[i] = w.ec_host_dn[i];
            }
    // the slots' experts in one kernel reading their host copies (a copy call each ran the cards one after another)
    std::vector<int> list{n};
    for (int s = 0; s < ec.n[p]; ++s) {
        const int i = owner[(size_t) s];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        list.push_back(i);
    }
    int* dlist = dalloc<int>(list.size());
    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: fill list");
    cache_copy(ec, p, dlist, st);
    ck(cudaMemcpyAsync(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
    // no wait here: the cards fill at once, and the constructor syncs every card before the engine is used (the tables
    // come from pageable vectors, copied out before cudaMemcpyAsync returns)
    ck(cudaMemcpyAsync(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
}

int Engine::arena_slots_for(int c, int w) const {
    const ExpertCache& ec = m_.cards[c].ec;
    const long need = (long) w * (long) (kvp_page(c) * sizeof(uint16_t)) - (long) arena_slot_off(c);
    if (need <= 0) return 0;
    const long sb = (long) ec.slot_bytes[ec.arena_pool];
    return (int) ((need + sb - 1) / sb);
}

std::vector<int> Engine::arena_lent() const {
    std::vector<int> v;
    if (arena_cap_)
        for (int c = 0; c < NCARD; ++c) v.push_back(arena_slots_for(c, arena_w_));
    return v;
}

// slots [to, from) of card c's arena pool, just given back empty: each takes one of the pool's experts not in a slot,
// the hottest first (the profile's score), copied in from its host copy (the card idle). Nothing lent then means every
// expert is in a slot, which the plan's early return relies on (cache_plan, moe.cuh)
void Engine::arena_refill(int c, int from, int to) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool, s0 = ec.s0[p];
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    std::vector<int> out;   // the pool's experts not in a slot
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e)
                if (slot_of[(size_t) l * NEXP + e] < 0) out.push_back(l * NEXP + e);
    std::stable_sort(out.begin(), out.end(), [&](int a, int b) { return m_.score[(size_t) a] > m_.score[(size_t) b]; });
    if (ec.arena_full && to == 0 && (int) out.size() > from)
        throw std::runtime_error("arena: " + std::to_string(out.size()) + " experts out of a slot, " + std::to_string(from) + " slots to take them");
    const int n = std::min((int) out.size(), from - to);
    std::vector<int> owner((size_t) (from - to), -1);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int k = 0; k < n; ++k) {
        const int i = out[(size_t) k], s = to + k;
        const LayerW& L = w.L[(size_t) (i / NEXP)];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "arena: refill");
        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "arena: refill");
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        slot_of[(size_t) i] = s0 + s;
        owner[(size_t) k] = i;
    }
    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemcpy(ec.owner + s0 + to, owner.data(), owner.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
}

// lending: the slots' experts to RAM (their tables first), then K/V may write there; giving back: the pages are free,
// the slots come back and are filled again at once (arena_refill). Every card idle first: a window or a prompt pass in
// flight may read a slot being lent, or a page being given back
void Engine::arena_set(int w) {
    if (w == arena_w_) return;
    if (w < arena_base_ || w > arena_cap_) throw std::runtime_error("arena: top " + std::to_string(w) + " outside its range");
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const ExpertCache& ec = m_.cards[c].ec;
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaDeviceSynchronize(), "arena: cards idle");
        const int from = arena_slots_for(c, arena_w_), to = arena_slots_for(c, w);
        if (to > ec.n[ec.arena_pool] - 2 * MAXT * TOPK) throw std::runtime_error("arena: card " + std::to_string(c) + " would keep too few slots");
        arena_slots(ec, from, to, b.st);
        ck(cudaStreamSynchronize(b.st), "arena: slots");
        if (to < from) arena_refill(c, from, to);
    }
    if (w > arena_w_) {
        for (int pg = arena_w_; pg < w; ++pg) free_pages_.push_back(pg);
        // zeroed: they held expert weights, and the draft layer reads K/V it never wrote (batched windows skip it), which
        // as int8 rows and scales could be inf or NaN (the own pages only ever held finite K/V or zeros)
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            ck(cudaMemset((char*) cb_[c]->pool + (size_t) arena_w_ * page, 0, (size_t) (w - arena_w_) * page), "arena: zero");
        }
        arena_last_lend_ = std::chrono::steady_clock::now();
    } else {
        free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int pg) { return pg >= w; }), free_pages_.end());
    }
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    const int old = arena_w_;
    arena_w_ = w;
    const std::vector<int> now = arena_lent();
    std::string per;
    double gib = 0;
    for (int c = 0; c < NCARD; ++c) {
        const ExpertCache& ec = m_.cards[c].ec;
        per += (c ? " / " : "") + std::to_string(now[(size_t) c]);
        gib = std::max(gib, (double) now[(size_t) c] * ec.slot_bytes[ec.arena_pool] / (1ull << 30));
    }
    std::fprintf(stderr, "fnx: arena: K/V pool %d -> %d pages (own %d, at most %d); slots lent %s (up to %.2f GiB of a card's experts in RAM), in %.1f ms\n",
                 old, w, arena_base_, arena_cap_, per.c_str(), gib,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Engine::arena_reclaim(double hold_s) {
    if (!arena_cap_ || arena_w_ == arena_base_) return;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - arena_last_lend_).count() < hold_s) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    if (top < arena_w_) arena_set(top);
}

// FNX_ARENA_CHURN=1 (a test): before every window, ARENA_STEP more pages lent and the pool's top then given back to the
// highest page in use (those slots refilled), so lending and giving back run beside every kind of window
void Engine::arena_churn() {
    static const bool on = env_flag("FNX_ARENA_CHURN");
    if (!on || !arena_cap_) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    arena_set(std::min(arena_cap_, std::max(arena_w_, top) + ARENA_STEP));
    arena_set(top);
}

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / KVP_POS + 1;   // the page of position n_pos too: a block's pooled key can sit one page on
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    // with an arena, slots lent for what the free pages lack: at least ARENA_STEP pages at a time
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && arena_w_ < arena_cap_)
        arena_set(std::min(arena_cap_, arena_w_ + std::max(short_by, ARENA_STEP)));
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
    // all the pages or none: a lane holding pages its device page table lacks would write K/V through stale entries
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
        b->pg.pt = L.pt;
        b->S = L.S;
        b->chist = L.chist;
        b->ple_hist = L.ple_hist;
        for (int l = 0; l < NL; ++l)
            if (is_attn(l)) b->qsa_ring[l] = L.ring[l];
        b->mtp_qsa_ring = L.mtp_ring;
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
    // the prompt path's buffers (and cuBLAS's) with a read of two chunks, every lane's rotary table
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
    // batched windows with drafts (they share step()'s graph slots, one per row count): 2 .. MAXT rows over as many lanes
    // as there are, up to the rows, dealt out evenly
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

std::vector<unsigned long long> Engine::probe_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.probe_cnt) return {};
    std::vector<unsigned long long> v(7);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.probe_cnt, 7 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "probe counts");
    return v;
}

std::vector<unsigned> Engine::route_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.route_cnt) return {};
    std::vector<unsigned> v((size_t) 2 * NL * NEXP);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.route_cnt, v.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), "route counts");
    return v;
}

std::vector<unsigned> Engine::route_trace(size_t from) const {
    const CardBuf& b = *cb_[0];
    if (!b.route_trace) return {};
    unsigned long long n = 0;
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(&n, b.route_trace_len, sizeof n, cudaMemcpyDeviceToHost), "route trace length");
    if (n > b.route_trace_cap) throw std::runtime_error("FNX_ROUTE_TRACE: " + std::to_string(n) + " picks, room for " +
                                                         std::to_string(b.route_trace_cap) + ": raise it");
    if (from >= n) return {};
    std::vector<unsigned> v((size_t) (n - from));
    std::memcpy(v.data(), b.route_trace + from, v.size() * sizeof(unsigned));
    return v;
}

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    k.push_back(cache_live());
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves), the
// PLE history, the QSA raw-key rings (every attention layer's and the draft layer's)
int Engine::lane_block_pages() const {
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    int n = 0;   // the most any card needs (pages differ in size between cards)
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = (size_t) 9 * D * 4 + ring_bytes + 256 * 64;   // the PLE history, the draft layer's ring, alignment
        for (int l = 0; l < NL; ++l)
            bytes += is_attn(l) ? ring_bytes : (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4;
        const size_t page = kvp_page(c) * sizeof(uint16_t);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    auto page_bytes = [](int c) { return kvp_page(c) * sizeof(uint16_t); };
    const int n = lane_block_pages();
    // a run of n free pages, highest first
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    // an arena lends the run at its top (a lane's block stays as long as the engine: those slots stay lent)
    if (start < 0 && arena_w_ + n <= arena_cap_) {
        start = arena_w_;
        arena_set(arena_w_ + n);
        std::fprintf(stderr, "fnx: arena: lane %zu's block at pages %d..%d, lent for good (no run of %d free pages below)\n", lanes_.size(),
                     start, start + n - 1, n);
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
        char* at = (char*) (b.pool + (size_t) start * kvp_page(c));
        ck(cudaMemset(at, 0, (size_t) n * page_bytes(c)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        L.ring.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) {
                L.ring[l] = (uint16_t*) take(ring_bytes);
            } else {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        }
        L.ple_hist = (float*) take((size_t) 9 * D * 4);
        L.mtp_ring = (uint16_t*) take(ring_bytes);
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

namespace {
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
}  // namespace

std::vector<std::function<void()>>& owned() {
    static std::vector<std::function<void()>> list;
    return list;
}

// every card idle, the window graphs destroyed, then what the engine made released, newest first (errors ignored: a
// destructor must not throw, and the process may be on its way out)
Engine::~Engine() {
    for (CardBuf* b : cb_) {
        cudaSetDevice(b->dev);
        cudaDeviceSynchronize();
        for (auto* slots : {&b->win_graphs, &b->mtp_graphs, &b->batch_graphs})
            for (CardBuf::GraphSlot& s : *slots)
                if (s.exec) cudaGraphExecDestroy(s.exec);
    }
    std::vector<std::function<void()>>& o = owned();
    for (auto it = o.rbegin(); it != o.rend(); ++it) (*it)();
    o.clear();
    cudaGetLastError();
    for (CardBuf* b : cb_) delete b;
}

void Engine::embed(const std::vector<int>& tokens, std::vector<float>& R0, std::vector<float>& ple) const {
    const int T = (int) tokens.size();
    std::vector<float> emb((size_t) T * N);
    ple.assign((size_t) T * N, 0.f);
    const int h0 = (int) history_.size();
    embed_rows(h0, T, [&](int q) -> int { return q < h0 ? history_[q] : tokens[q - h0]; }, emb.data(), ple.data());
    R0.assign((size_t) T * D, 0.f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < HC; ++c) std::memcpy(&R0[(size_t) t * D + c * N], &emb[(size_t) t * N], N * sizeof(float));
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

void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    std::memset(ple, 0, (size_t) T * N * sizeof(float));
    static const int8_t kv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const Tensor& pt = *m_.ple_table;
    const size_t prb = pt.row_bytes();
    // the PLE n-grams read an image cell as the model's image token
    const auto tok_at = [&](int q) -> int64_t {
        const int t = tok_at_cell(q);
        return t >= IMAGE_TOK ? m_.ple_image : t;
    };
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t;
        embed_token(tok_at_cell(p), p, emb + (size_t) t * N);
        // PLE rows: n-grams of this token and the two before it (EOS / sequence start cut the window)
        int64_t ctx[3];
        ctx[0] = tok_at(p);
        bool cut = false;
        for (int s = 1; s < 3; ++s) {
            const int q = p - s;
            const int64_t tk = q >= 0 ? tok_at(q) : -1;
            cut = cut || tk < 0 || tk == m_.ple_eos;
            ctx[s] = cut ? m_.ple_eos : tk;
        }
        for (int n = 2; n <= 3; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * m_.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * m_.ple_mult[j];
            for (int gi = 0; gi < 8; ++gi) {
                const int h = (n - 2) * 8 + gi;
                const uint64_t r = mixed % m_.ple_vocab[h] + m_.ple_off[h];
                const uint8_t* src = pt.data + r * prb;
                for (int b = 0; b < PLE_DIM / 32; ++b) {
                    float* dst = &ple[(size_t) t * N + h * PLE_DIM + b * 32];
                    uint16_t dh;
                    if (pt.type == T_Q8_0) {
                        std::memcpy(&dh, src + b * 34, 2);
                        const float d = h2f(dh);
                        for (int j = 0; j < 32; ++j) dst[j] = d * (float) (int8_t) src[b * 34 + 2 + j];
                        continue;
                    }
                    std::memcpy(&dh, src + b * 18, 2);   // IQ4_NL
                    const float d = h2f(dh);
                    const uint8_t* qs = src + b * 18 + 2;
                    for (int j = 0; j < 16; ++j) {
                        dst[j] = d * kv4[qs[j] & 15];
                        dst[j + 16] = d * kv4[qs[j] >> 4];
                    }
                }
            }
        }
    }
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
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    for (int l = 0; l < NL; ++l) {
        if (is_attn(l)) {
            v.emplace_back((char*) b.qsa_ring[l], ring_bytes);
        } else {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    }
    v.emplace_back((char*) b.ple_hist, (size_t) 9 * D * sizeof(float));
    if (mtp) v.emplace_back((char*) b.mtp_qsa_ring, ring_bytes);
    // the pages holding positions 0 .. pos (every layer's K/V rows and pooled keys, the draft layer's too)
    if (kv)
        for (int i = 0; i <= pos / KVP_POS && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * b.pg.page), b.pg.page * sizeof(uint16_t));
    return v;
}

// a state's pieces between a card and pageable host memory (laid end to end there), through the card's two
// page-locked halves in blocks of STAGE_BYTES: the copy engine fills (or empties) one half while the host copies the
// other. Pageable, because page-locked memory is mapped into every card: 2 MiB of each card's VRAM per GiB (64 MiB for
// 32 GiB, tools/pin_probe), 256 MiB for the 128 GiB cache, about what cards 0 and 1 have left
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
    size_t pi = 0, poff = 0;   // the next piece byte to move
    // the DMAs of block k between the device pieces and half k % 2
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
            if (k > 1) ck(cudaEventSynchronize(b.stage_ev[k % 2]), "state copy");   // the half's last DMA is done
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

// f(card) on a thread per card, the first failure rethrown after all of them end
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
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos_, kv, half, lanes_[(size_t) cur_].pages);
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
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos, s.kv, half, lanes_[(size_t) cur_].pages);
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
    // pages past the state's length go back to the pool
    LaneHost& L = lanes_[(size_t) cur_];
    const size_t keep = (size_t) (pos / KVP_POS + 1);
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
        ck(cudaMemsetAsync(b.ple_hist, 0, (size_t) 9 * D * 4, b.st), "reset PLE history");
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
        // the window read history parity tag & 1 and wrote the other: that one becomes the kept count's history. The
        // kept row's states, conv and PLE histories in one copy_many per card (as cudaMemcpyAsync calls: 73 a card,
        // ~0.5 ms of host time a commit)
        const unsigned wrote = (tag_ & 1u) ^ 1u;
        constexpr int CJ = 2 * NL + 1;
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
            // the page-locked table again only once the last commit's upload from it is done
            ck(cudaEventSynchronize(b.commit_ev), "commit table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.commit_jobs) + (size_t) c * CJ;
            int k = 0;
            for (int l = 0; l < NL; ++l) {
                if (is_attn(l)) continue;
                jobs[k++] = {b.S[l], b.snap[l] + (size_t) (n - 1) * w.nv() * HS * HS, sbytes};
                jobs[k++] = {b.chist[l] + wrote * hfloats, b.hsnap[l] + (size_t) (n - 1) * hfloats, hfloats * 4};
            }
            jobs[k++] = {b.ple_hist, b.ple_hsnap + (size_t) (n - 1) * 9 * D, (size_t) 9 * D * 4};
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
    // the rows' penalty lists at the stride of the longest (the copies carry only what is used)
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
        fnx::top_tokens(b.head_logits, w.v1 - w.v0, T, w.v0, b.tk_pen_tok, b.tk_pen_cnt, b.tk_pen_n, stride, b.tk_par,
                        b.tk_v, b.tk_i, b.tk_err, b.st);
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

std::vector<int> Engine::verify(const std::vector<int>& tokens) {
    const int T = (int) tokens.size();
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify: the last window is not committed");
    if (T < 1 || T > MAXT) throw std::runtime_error("window: 1..MAXT tokens");
    if (pos_ + T > max_ctx_) throw std::runtime_error("window: context full");
    arena_churn();
    ensure_pages(pos_ + T);
    const auto t0 = std::chrono::steady_clock::now();
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * D);
        h_.ple = halloc<float>((size_t) MAXT * N);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {   // shared with the draft layer, which can run first (over a prompt read by prefill)
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    std::vector<float> R0, ple;
    embed(tokens, R0, ple);
    std::memcpy(h_.R0, R0.data(), R0.size() * 4);
    std::memcpy(h_.ple, ple.data(), ple.size() * 4);
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
        ck(cudaMemcpyAsync(b.R, h_.R0, R0.size() * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ple_emb, h_.ple, ple.size() * 4, cudaMemcpyHostToDevice, b.st), "ple");
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
    };
    auto download = [&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "ring err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    if (dump) {
        each(upload);
        dump_dev("R0", -1, cb_[0]->R, T, D);
        enqueue(T, true);
        each(download);
    } else {
        const std::vector<int> key = graph_key({cur_});
        if (cb_[0]->win_graphs[T].key != key || !cb_[0]->win_graphs[T].exec) {   // capturing queues no work
            each([&](int, CardBuf& b, const CardW&) {
                ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
            });
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
    if (err & 1u) throw std::runtime_error("window: a ring sum timed out");
    if (err & 2u) throw std::runtime_error("window: a ring sum met a NaN or inf (or, on the fp16 ring, a value past its range)");
    if (err & 4u) throw std::runtime_error("window: an hc read's grid barrier timed out");
    if (err & 16u) throw std::runtime_error("window: the head's logits hold a NaN or inf");
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

// out[t][r] = W[r] . y[t]: the split Q8_0 kernel, or mmvq for other types
void Engine::dense(const DW& W, const void* y, float* out, CardBuf& b, int T) {
    if (W.s.q) q8s_gemv(W.s, W.n_out, W.n_in, y, out, W.n_out, T, b.st);
    else mmvq(W, 0, b.dense[T], nullptr, 1, y, out, W.n_out, b.st);
}

// an attention layer's mixer after its hc read (mixed in b.xq): q/k/v, the window's K/V into kc/vc at ctl[0], the
// attention, the output projection into b.bo_a
void Engine::attention(int T, CardBuf& b, const CardW& w, const LayerW& L, uint16_t* kc, uint16_t* vc, const int* ctl,
                       uint16_t* qsa_ring, uint16_t* qsa_pooled, int call) {
    // the indexer first (it reads mixed / its q8 copy, which the projections below leave as they are)
    if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
    qsa(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, L.idx_qn, L.idx_kn, qsa_ring, qsa_pooled, b.pg, b.qsa_q, b.qsa_scores, b.qsa_cells,
        b.qsa_ncells, max_ctx_ / 4, ctl, T, m_.eps, rope_of(b.mrope), b.att_ring.card, b.st);
    if (L.wq.s.q && L.wk.s.q && L.wv.s.q && L.wq.n_in == L.wk.n_in && L.wq.n_in == L.wv.n_in &&
        L.wq.n_out + L.wk.n_out + L.wv.n_out >= 8192) {   // q, k and v in one launch (compiled for 8192 rows and more)
        const Q8S ws[3] = {L.wq.s, L.wk.s, L.wv.s};
        const int rows[3] = {L.wq.n_out, L.wk.n_out, L.wv.n_out};
        float* const outs[3] = {b.qg, b.k, b.v};
        q8s_gemv_segs(ws, rows, outs, rows, 3, L.wq.n_in, b.xq, T, b.st);
    } else {
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
    }
    attn_prep_append(b.qg, b.k, b.v, L.q_norm, L.k_norm, kc, vc, b.pg, b.att_ring.card, T, ctl, m_.eps, rope_of(b.mrope), b.st);
    attn(b.qg, kc, vc, b.pg, b.att, b.att_ring, (const unsigned*) (ctl + 1), call, ctl, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells);
    quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
    dense(L.wo, b.xq, b.bo_a, b, T);
}

// the FFN half: the hc read with the attention half's write folded in, the shared expert (in the router's and the
// routed gate|up's launches on Q4_K layers, else on the second stream beside the router), the routing and the routed
// experts, the weighted sum across the cards into b.bo_f
void Engine::ffn_half(int T, int call, CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl) {
    const int P = T * TOPK, ffs = w.ff1 - w.ff0;
    hc_read(R, b.bo_a, b.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], m_.eps, b.mixed, b.inj_f, b.hcs, b.ring.err, T, b.st, b.xq);
    // the shared expert (rows P + t of part): its gate|up in the router's launch and its swiglu and down in the routed
    // gate|up's (Q4_K experts), else on the second stream (a fork and a join cost ~5 us more than the launches' blocks)
    const bool sh_in = L.gu_e.type == T_Q4_K && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;
    if (!sh_in) {
        ck(cudaEventRecord(b.fork, b.st), "fork");
        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);
        swiglu_stacked_q8_1(b.sh_gu, b.sh_hq, ffs, T, b.st2);
        q8s_gemv(L.sh_down.s, N, ffs, b.sh_hq, b.part + (size_t) P * N, N, T, b.st2);
        ck(cudaEventRecord(b.join, b.st2), "join");
    }
    const int lyr = (int) (&L - w.L.data());
    // experts in RAM: the router's last block plans this window's slots (the prefetch probe scores the picks against
    // the slots as they were, so with it the plan runs after, in moe_cache)
    const bool live = cache_live() && lyr >= 0 && lyr < NL;
    const bool plan_in_router = live && !b.probe_cnt;
    router_route(L.router, b.mixed, b.logits, T, b.topi, b.topw, b.jobs_gu, b.jobs_dn, b.njobs, b.st, plan_in_router ? &w.ec : nullptr,
                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu);
    if (const int l = lyr; l >= 0 && l < NL) {
        if (b.route_cnt) route_count(b.topi, P, b.route_cnt + (size_t) l * NEXP, b.st);
        if (b.route_trace) fnx::route_trace(b.topi, P, l, b.route_trace, b.route_trace_len, b.route_trace_cap, b.st);
        if (b.probe_cnt) {
            if (l > 0)
                probe_score(b.topi, b.probe_guess + (size_t) l * MAXT * TOPK, T, w.ec.n_slots ? w.ec.slot_of + (size_t) l * NEXP : nullptr,
                            b.probe_cnt, b.st);
            if (l + 1 < NL) {
                gemv_f32(w.L[(size_t) l + 1].router, b.mixed, b.probe_logits, NEXP, N, T, b.st);
                probe_guess(b.probe_logits, T, b.probe_guess + (size_t) (l + 1) * MAXT * TOPK, b.st);
            }
        }
        // the planned experts copied into their slots before the MoE kernels read the layer's tables
        if (live) moe_cache(l, L.gu_e_stride, L.down_e_stride, w.ec, b.cache_list, b.st, plan_in_router ? nullptr : b.jobs_gu, b.njobs);
    }
    if (sh_in) moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st, &L.sh_down.s, ffs, b.sh_gu, b.part + (size_t) P * N);
    else moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st);
    moe_down(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, b.gu, b.part, b.st);
    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
    RingMoe m;
    m.part = b.part;
    m.topw = b.topw;
    m.logits = b.logits;
    ring_sum(b.ring, b.bo_f, T, (const unsigned*) (ctl + 1), call, b.st, m);
}

// every card's work for one window of T rows, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
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
    for (int l = 0; l < NL; ++l) {
        const bool pending = l > 0;
        // ---- attention half
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = pending;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                ple_post(b.R, b.ple_key, b.ple_val, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv, w.ple_conv,
                         b.ple_hist, b.ple_hsnap, T, eps, b.st, b.ple_key + (size_t) MAXT * D);
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv(), ch = w.conv_ch();
                // ab beside qkvz in one launch (a fork and a join: ~5 us more; compiled for 4096..8191 rows)
                if (L.qkvz.s.q && 2 * nv <= 160 && L.qkvz.n_out >= 4096 && L.qkvz.n_out < 8192) {
                    q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                } else {
                    ck(cudaEventRecord(b.fork, b.st), "fork");
                    ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
                    gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st2);
                    ck(cudaEventRecord(b.join, b.st2), "join");
                    dense(L.qkvz, b.xq, b.qkvz, b, T);
                    ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
                }
                gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                         b.hsnap[l], w.nk, nv, w.vmap, T, eps, b.st);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                attention(T, b, w, L, b.kc[l], b.vc[l], b.ctl, b.qsa_ring[l], b.qsa_pooled[l], 128 + l);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        dd("hc_mixed", l, cb_[0]->mixed, N);
        dd("mixer_out", l, cb_[0]->bo_a, N);
        // ---- FFN half
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
        dd("ffn_mixed", l, cb_[0]->mixed, N);
        dd("ffn_out", l, cb_[0]->bo_f, N);
        dd("R_pending", l, cb_[0]->R, D);
        dd("inj_ffn", l, cb_[0]->inj_f, HC);
    }
    // ---- head: the final mixer (no injection) with the last write folded in, then this card's vocabulary
    each([&](int c, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
    dd("result_norm", -1, cb_[0]->mixed, N);
}

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): the shared work over all rows, the
// sequence's own work (the PLE block, the recurrent core, QSA, the rotation, the K/V append, the attention partials)
// per row with its lane's state and ctl
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    const int T = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int r = 0; r < T; ++r)   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    ple_post(b.R + (size_t) r * D, b.ple_key + (size_t) r * D, b.ple_val + (size_t) r * N, w.ple_norm_key, w.ple_norm_query,
                             w.ple_norm_conv, w.ple_conv, b.lanes[(size_t) lanes[r]].ple_hist, b.ple_hsnap, 1, eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
                gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, T, eps, b.st);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int r = 0; r < T; ++r) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    qsa_index(b.qsa_y + (size_t) r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + (size_t) r * 512,
                              LD.ctl, 1, eps, rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + (size_t) r * NQH * 2 * HD, b.k + (size_t) r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, 1, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + (size_t) r * NKVH * HD, b.v + (size_t) r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, 1, LD.ctl,
                              b.st);
                }
                // every row's selection and attention partials at once, each on its lane's position and pages
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
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
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * D);
        h_.ple = halloc<float>((size_t) MAXT * N);
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
            const std::vector<int>& h = L.history;
            const int tok = tokens[r], h0 = (int) h.size();
            std::vector<float> emb(N);
            embed_rows(L.pos, 1, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tok; }, emb.data(), h_.ple + (size_t) r * N);
            for (int c = 0; c < HC; ++c) std::memcpy(h_.R0 + (size_t) r * D + c * N, emb.data(), N * sizeof(float));
            // the lane's recurrent history half flips, as a window of its own would
            L.tag = L.tag + 1u;
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
        ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * D * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ple_emb, h_.ple, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "ple");
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
        ck(cudaMemcpyAsync(&h_.err[c], b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "ring err");
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
    if (err & 1u) throw std::runtime_error("step: a ring sum timed out");
    if (err & 2u) throw std::runtime_error("step: a ring sum met a NaN or inf (or, on the fp16 ring, a value past its range)");
    if (err & 4u) throw std::runtime_error("step: an hc read's grid barrier timed out");
    if (err & 16u) throw std::runtime_error("step: the head's logits hold a NaN or inf");
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

double rope_scale() {
    const char* e = std::getenv("FNX_ROPE_SCALE");
    if (!e) return 0.0;
    // 1 is off, so a profile's setting can be overridden without unsetting it
    const double S = env_num("FNX_ROPE_SCALE", 0.0);
    if (S == 1.0) return 0.0;
    if (!(S > 1.0)) throw std::runtime_error(std::string("FNX_ROPE_SCALE: want 1 (off) or a scale above 1, got ") + e);
    return S;
}

bool ring_f32() {
    // fp32 by default on two cards: with fp16 pairs there the code and edit prompts' KL against llama.cpp is 1.25x the
    // exactness bar (code 7.48e-4, fp32 5.73e-4 against a bar of 6.06e-4)
    const long long v = env_int("FNX_RING_F32", NCARD == 2 ? 1 : 0);
    if (v != 0 && v != 1) throw std::runtime_error("FNX_RING_F32: 0 or 1, got " + std::to_string(v));
    return v == 1;
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

namespace {
// a verify_rows() window's layout: each lane's first row, and its lanes as segments ordered by row count (stable), so
// that the lanes with the same count are one launch of the recurrent core (a group: its count, first segment, size)
struct RowGroup {
    int count, s0, n;
};
struct RowSegs {
    int T = 0;
    std::vector<int> r0;          // per lane
    std::vector<int> order;       // per segment: its lane
    std::vector<RowGroup> groups;
    std::vector<int> row_lane;    // per row: its lane
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

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): the shared work over all rows as there;
// each lane's own work over its rows as a window of its own does it (the rows chained through its recurrent state, the
// state and histories after each row kept for commit_rows), and every row's selection and attention partials at once,
// each at its own position (row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    const RowSegs sg = row_segs(counts);
    const int T = sg.T, nl = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int i = 0; i < nl; ++i) {   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    ple_post(b.R + r * D, b.ple_key + r * D, b.ple_val + r * N, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv,
                             w.ple_conv, b.lanes[(size_t) lanes[i]].ple_hist, b.ple_hsnap + r * 9 * D, counts[(size_t) i], eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
                }
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                for (const RowGroup& g : sg.groups) {
                    const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                      b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                    gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, g.n,
                                  g.count, eps, b.st);
                }
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int i = 0; i < nl; ++i) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    const int k = counts[(size_t) i];
                    qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + r * 512, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + r * NQH * 2 * HD, b.k + r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + r * NKVH * HD, b.v + r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, k, LD.ctl, b.st);
                }
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
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
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * D);
        h_.ple = halloc<float>((size_t) MAXT * N);
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
        std::vector<float> emb((size_t) MAXT * N);
        for (int i = 0; i < nl; ++i) {
            const LaneHost& L = lanes_[(size_t) lanes[i]];
            const int k = counts[(size_t) i], r = sg.r0[(size_t) i];
            if (L.pos + k > max_ctx_) throw std::runtime_error("verify_rows: lane " + std::to_string(lanes[i]) + "'s context is full");
            ensure_pages_of(lanes[i], L.pos + k);
            const std::vector<int>& h = L.history;
            const int h0 = (int) h.size();
            embed_rows(L.pos, k, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tokens[(size_t) (r + q - h0)]; }, emb.data(),
                       h_.ple + (size_t) r * N);
            for (int j = 0; j < k; ++j)
                for (int c = 0; c < HC; ++c)
                    std::memcpy(h_.R0 + (size_t) (r + j) * D + c * N, emb.data() + (size_t) j * N, N * sizeof(float));
            // the lane's recurrent history half flips, as a window of its own would
            tags[(size_t) i] = L.tag + 1u;
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
        ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * D * 4, cudaMemcpyHostToDevice, b.st), "R0");
        ck(cudaMemcpyAsync(b.ple_emb, h_.ple, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "ple");
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int i = 0; i < nl; ++i)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[i]].ctl, &lane_ctl[(size_t) 2 * i], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
        ck(cudaMemcpyAsync(b.row_pos, row_pos.data(), (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "row positions");
    });
    // the graph slot step() uses for T rows; the layout (each lane's count) is part of the key, and -1 tells it from step()'s
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
        ck(cudaMemcpyAsync(&h_.err[c], b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "ring err");
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
    // the window ran: its rows wait for commit_rows even when an error below ends the caller (the lanes' states moved)
    rows_pending_ = RowsWindow{lanes, counts, tokens};
    if (err & 1u) throw std::runtime_error("verify_rows: a ring sum timed out");
    if (err & 2u) throw std::runtime_error("verify_rows: a ring sum met a NaN or inf (or, on the fp16 ring, a value past its range)");
    if (err & 4u) throw std::runtime_error("verify_rows: an hc read's grid barrier timed out");
    if (err & 16u) throw std::runtime_error("verify_rows: the head's logits hold a NaN or inf");
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
        // a lane cut back takes its kept row's states, conv and PLE histories from the window's snapshots at its rows, as
        // commit() does (the history half its window wrote: the parity after its tag's)
        constexpr int CJ = 2 * NL + 1;
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
            // the page-locked table again only once the last commit_rows' upload from it is done
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
                jobs[k++] = {LD.ple_hist, b.ple_hsnap + at * 9 * D, (size_t) 9 * D * 4};
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

}  // namespace fnx

namespace fnx {

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.mtp_emb = dalloc<float>((size_t) MAXT * N);
        b.mtp_en = dalloc<float>((size_t) MAXT * N);
        b.mtp_e2 = dalloc<float>((size_t) MAXT * N);
        b.mtp_src = dalloc<float>((size_t) MAXT * D);
        b.mtp_hn = dalloc<float>((size_t) MAXT * D);
        b.mtp_h2 = dalloc<float>((size_t) MAXT * D);
        b.mtp_R = dalloc<float>((size_t) MAXT * D);
        b.mtp_kc = b.pool + (size_t) (KVP_SLOTS - 1) * kvp_slot(c);   // the draft layer: the pool's last slot
        b.mtp_vc = b.mtp_kc + kvp_rows_halves(c, kv_q8());
        b.lanes[0].mtp_ring = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);   // lanes after 0 carry theirs in their blocks
        b.mtp_qsa_ring = b.lanes[(size_t) cur_].mtp_ring;
        b.mtp_qsa_pooled = b.mtp_kc + 2 * kvp_rows_halves(c, kv_q8());
        b.mtp_logits = dalloc<float>((size_t) MAXT * std::max(1, mc.dv1 - mc.dv0));
        b.mtp_bv = dalloc<float>(MAXT);
        b.mtp_bi = dalloc<int>(MAXT);
        b.mtp_ctl = dalloc<int>(2);
        b.mtp_pen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mtp_pen_par = dalloc<float>(3);
        b.mtp_tk_v = dalloc<float>(TOPK_N);
        b.mtp_tk_i = dalloc<int>(TOPK_N);
        b.mtp_tk_err = dalloc<unsigned>(1);
        b.mtp_zero = dalloc<int>(1);
        ck(cudaMemset(b.mtp_zero, 0, 4), "mtp zero");
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

// one draft-layer window of T rows from b.mtp_src (rows of D) and b.mtp_emb (the tokens' embeddings), cells at
// mtp_ctl[0]: R = fc_hidden(rms(src) w) per stream + fc_embedding(rms(emb) w), then the layer as a main attention
// layer, then the final mixer and this card's draft head rows
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)   // every stream of every row through fc_hidden
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N,
                     std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T,
                b.st, b.xq);
        attention(T, b, w, L, b.mtp_kc, b.mtp_vc, b.mtp_ctl, b.mtp_qsa_ring, b.mtp_qsa_pooled, 128);
        ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0, b.st);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        penalize_row(b.mtp_logits + (size_t) (T - 1) * (mc.dv1 - mc.dv0), mc.dv1 - mc.dv0, b.mtp_pen, b.mtp_pen_par, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
    }
}

int Engine::draft(int T, bool from_main, int src_row, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen,
                  bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (!h_.R0) throw std::runtime_error("draft: no window has run (it drafts from a window's rows)");
    if (src_row < 0 || src_row + T > MAXT) throw std::runtime_error("draft: source rows out of range");
    std::vector<const float*> src;
    for (CardBuf* b : cb_) src.push_back((from_main ? b->R : b->mtp_R) + (size_t) src_row * D);
    return draft_rows(src, T, tokens, pos0, pen, cands);
}

int Engine::draft_rows(const std::vector<const float*>& src, int T, const std::vector<int>& tokens, int pos0,
                       const Penalties::Row* pen, bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("draft: 1..MAXT rows, one token each");
    if (pos0 < 0 || pos0 + T > max_ctx_) throw std::runtime_error("draft: cells out of the context");
    ensure_pages(pos0 + T);
    if (!h_.mtp_emb) throw std::runtime_error("draft: mtp_attach first");
    const auto t0 = std::chrono::steady_clock::now();
    // the tokens' embeddings (Q8_0 rows of the main table; the token after row t sits at cell pos0 + t + 1)
    for (int t = 0; t < T; ++t) embed_token(tokens[t], pos0 + t + 1, h_.mtp_emb + (size_t) t * N);
    h_.mtp_ctl[0] = pos0;
    h_.mtp_ctl[1] = (int) ring_tag();
    // the penalized tokens in each card's slice of the draft vocabulary: {k, indexes, counts}
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
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);   // counts right after the indexes
            hp[0] = pen_k[c] = k;
        }
    } else {
        for (int c = 0; c < NCARD; ++c) h_.mtp_pen[(size_t) c * (1 + 2 * PEN_CAP)] = 0;
    }
    const std::vector<int> key = graph_key({cur_});
    if (cb_[0]->mtp_graphs[T].key != key || !cb_[0]->mtp_graphs[T].exec) {   // capturing queues no work
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
        ck(cudaMemcpyAsync(b.mtp_src, src[c], (size_t) T * D * 4, cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        ck(cudaMemcpyAsync(b.mtp_emb, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
        ck(cudaMemcpyAsync(b.mtp_ctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
        ck(cudaMemcpyAsync(b.mtp_pen, h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP), (size_t) (1 + 2 * pen_k[c]) * 4,
                           cudaMemcpyHostToDevice, b.st), "draft penalties");
        if (pen_k[c]) ck(cudaMemcpyAsync(b.mtp_pen_par, h_.mtp_pen_par, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaGraphLaunch(b.mtp_graphs[T].exec, b.st), "graph launch");
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0;
        if (cands && nv > 0) {   // the last row's logits are penalized already
            ck(cudaMemsetAsync(b.mtp_tk_err, 0, 4, b.st), "draft top err");
            fnx::top_tokens(b.mtp_logits + (size_t) (T - 1) * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0, b.mtp_pen_par,
                       b.mtp_tk_v, b.mtp_tk_i, b.mtp_tk_err, b.st);
        }
        ck(cudaEventRecord(b.ev1, b.st), "event");
        ck(cudaMemcpyAsync(&h_.bv[c], b.mtp_bv + T - 1, 4, cudaMemcpyDeviceToHost, b.st), "draft v");
        ck(cudaMemcpyAsync(&h_.bi[c], b.mtp_bi + T - 1, 4, cudaMemcpyDeviceToHost, b.st), "draft i");
        ck(cudaMemcpyAsync(&h_.err[c], b.ring.err, 4, cudaMemcpyDeviceToHost, b.st), "ring err");
        if (cands && nv > 0) {
            ck(cudaMemcpyAsync(h_.mtp_tk_v + (size_t) c * TOPK_N, b.mtp_tk_v, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
            ck(cudaMemcpyAsync(h_.mtp_tk_i + (size_t) c * TOPK_N, b.mtp_tk_i, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
            ck(cudaMemcpyAsync(h_.mtp_tk_err + c, b.mtp_tk_err, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
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
    if (err & 1u) throw std::runtime_error("draft: a ring sum timed out");
    if (err & 2u) throw std::runtime_error("draft: a ring sum met a NaN or inf (or, on the fp16 ring, a value past its range)");
    if (err & 4u) throw std::runtime_error("draft: an hc read's grid barrier timed out");
    if (err & 16u) throw std::runtime_error("draft: the draft layer's logits hold a NaN or inf");
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
