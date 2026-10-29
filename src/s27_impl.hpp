// The 27B engine's per-card buffers (s27_engine.cu, s27_prefill.cu): the Engine of engine.hpp built for FNX_MODEL=27b.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <functional>
#include <vector>

#include "kernels.hpp"
#include "model.hpp"
#include "s27_attn.cuh"
#include "s27_ring3.cuh"

namespace fnx {

// a lane's state on one card: its page table, recurrent states and conv histories; the engine's current-lane pointers
// in CardBuf are a copy of the lane's (Engine::use_lane)
struct LaneDev {
    int* pt = nullptr;
    std::vector<float*> S, chist;
    int* ctl = nullptr;               // [2]: the lane's position and tag in a batched window
    int* mrope_buf = nullptr;         // the lane's rotary table ([max_ctx][3]), made at its first image
    int* mrope = nullptr;             // mrope_buf while its sequence holds an image, else null (rotary position = cell)
};

// what the engine made on the cards and in page-locked memory, each with how to release it (newest first, ~Engine)
std::vector<std::function<void()>>& owned();
inline int current_card() {
    int d = -1;
    cudaGetDevice(&d);
    return d;
}
inline void own_dev(void* p) {
    const int d = current_card();
    owned().push_back([d, p] { cudaSetDevice(d); cudaFree(p); });
}
inline void own_pinned(void* p) { owned().push_back([p] { cudaFreeHost(p); }); }
inline void own_stream(cudaStream_t s) {
    const int d = current_card();
    owned().push_back([d, s] { cudaSetDevice(d); cudaStreamDestroy(s); });
}
inline void own_event(cudaEvent_t e) {
    const int d = current_card();
    owned().push_back([d, e] { cudaSetDevice(d); cudaEventDestroy(e); });
}

// the widest q8_1 row a window quantizes: the draft layer's [e | h], the FFN's slice, the attention's heads
constexpr int S27_XQ_COLS = 2 * N > FFD_CARD_MAX ? (2 * N > S27_NQ * HD ? 2 * N : S27_NQ * HD) : FFD_CARD_MAX;
static_assert(S27_XQ_COLS >= NVH / NCARD * HS && S27_XQ_COLS >= S27_NQ * HD, "xq holds every row a window quantizes");

struct CardBuf {
    int dev = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float *R = nullptr, *mixed = nullptr, *bo = nullptr;   // the residual, a half's normed input, its sum across the cards
    float* hfin = nullptr;  // the last window's rows after the output norm: the draft layer's source (draft from_main)
    void* xq = nullptr;     // q8_1 rows of up to S27_XQ_COLS, with a split twin (q8_twin) add_norm writes directly
    int8_t* xtq = nullptr;
    float* xtd = nullptr;
    void* hq = nullptr;     // the FFN's swiglu as q8_1 rows of the card's FF slice, with its twin
    float *qkvz = nullptr, *ab = nullptr, *gdn_scratch = nullptr;
    float* qkv = nullptr;   // [T][S27_QKV]: the card's q | gate heads, K heads, V heads
    float *att = nullptr, *apart = nullptr;
    float* gu = nullptr;
    float* head_logits = nullptr;
    int *tk_pen_tok = nullptr, *tk_pen_cnt = nullptr, *tk_pen_n = nullptr, *tk_i = nullptr;   // top_tokens()
    float* tk_v = nullptr;
    unsigned* tk_err = nullptr;
    float* tk_par = nullptr;
    float* best_v = nullptr;
    int* best_i = nullptr;
    unsigned long long* am_keys = nullptr;
    unsigned* err = nullptr;          // the window's error bits (ring timeouts and range, NaN logits)
    // state: the current lane's
    std::vector<float*> S, chist;
    __half* pool = nullptr;           // the paged K/V pool (s27_attn.cuh): n_pages pages of S27_PAGE halves
    int n_pages = 0;
    int* pt = nullptr;                // the current lane's page table (device)
    // what commit() cuts a window back to: the state after each row, each kept count's histories
    std::vector<float*> snap, hsnap;
    float* snap_block = nullptr;      // every recurrent layer's snap (the prompt path borrows it between windows)
    size_t snap_block_bytes = 0;
    R3 r3;
    Ring ring;                        // FNX_RING_F32=1: the sums as fp32 words (kernels.cu's ring) instead of r3's
    bool ring32 = false;
    int* ctl = nullptr;               // [0] window position, [1] ring tag (as unsigned)
    int* mrope = nullptr;             // the current lane's rotary table
    char* stage[2] = {nullptr, nullptr};   // a parked state's copies pass through these page-locked halves
    cudaEvent_t stage_ev[2] = {nullptr, nullptr};
    std::vector<LaneDev> lanes;
    struct GraphSlot {
        cudaGraphExec_t exec = nullptr;
        std::vector<int> key;
    };
    std::vector<GraphSlot> win_graphs = std::vector<GraphSlot>(MAXT + 1), mtp_graphs = std::vector<GraphSlot>(MAXT + 1),
                           batch_graphs = std::vector<GraphSlot>(MAXT + 1);
    // batch_graphs[T]'s rows (LaneRows): [MAXT] each row's position word, then [MAXT] its page table; and their
    // recurrent state then conv history per layer (GdnLanes): [NL][MAXT] S, then [NL][MAXT] hist
    std::vector<const int**> lane_rows = std::vector<const int**>(MAXT + 1, nullptr);
    std::vector<float**> lane_gdn = std::vector<float**>(MAXT + 1, nullptr);
    int* row_pos = nullptr;
    std::vector<const int**> seg_ctl = std::vector<const int**>(MAXT + 1, nullptr);
    std::vector<int*> seg_row0 = std::vector<int*>(MAXT + 1, nullptr);
    CopyJob* rows_commit_jobs = nullptr;
    CopyJob* commit_jobs = nullptr;
    cudaEvent_t commit_ev = nullptr, rows_commit_ev = nullptr;
    // the draft layer: its token rows, source rows, [e | h] normed, residual, outputs after the head norm, head logits
    float *me = nullptr, *mh = nullptr, *mcat = nullptr, *mR = nullptr, *mout = nullptr, *mlogits = nullptr, *mbv = nullptr;
    int *mbi = nullptr, *mctl = nullptr;
    int* mpen = nullptr;              // the last row's penalties (penalize_row's pen, this card's draft-vocabulary indexes)
    float* mpen_par = nullptr;
    float* mtk_v = nullptr;
    int *mtk_i = nullptr, *mzero = nullptr;
    unsigned* mtk_err = nullptr;
};

}  // namespace fnx
