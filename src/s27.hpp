// s27: Qwen3.8-27B (dense) split across the cards, one sequence. Every layer on every card: the recurrent heads,
// the FFN by FF width, the attention by heads (s27_attn.cuh), the vocabulary; the residual and the norms copied. Two sums a layer.
// The draft (nextn) layer runs the same way, on the draft vocabulary's head.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "kernels.hpp"
#include "model.hpp"
#include "rope.hpp"

namespace fnx {

// the three-values-a-word ring's buffers (s27_ring3.cuh)
struct R3 {
    int card = 0;
    unsigned long long *in = nullptr, *ag = nullptr, *next_in = nullptr, *next_ag = nullptr;
    unsigned* err = nullptr;
};

struct S27Card {
    int dev = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float *R = nullptr, *mixed = nullptr, *bo = nullptr;
    float* hfin = nullptr;  // the last window's rows after the output norm: the draft layer's h
    void* xq = nullptr;     // q8_1 rows of up to XQ_COLS
    int8_t* xtq = nullptr;  // its split twin (q8_twin), which add_norm writes directly
    float* xtd = nullptr;
    void* hq = nullptr;     // the FFN's swiglu as q8_1 rows of the card's FF slice
    int8_t* htq = nullptr;  // its split twin
    float* htd = nullptr;
    float *qkvz = nullptr, *ab = nullptr, *gdn_scratch = nullptr;
    float* qkv = nullptr;   // [T][S27_QKV]: the card's q | gate heads, K heads, V heads
    float *att = nullptr, *apart = nullptr;
    float* gu = nullptr;
    float* head_logits = nullptr;
    float* best_v = nullptr;
    int* best_i = nullptr;
    unsigned long long* am_keys = nullptr;
    std::vector<float*> S, chist, snap, hsnap;
    std::vector<__half*> kc, vc;      // per attention layer: [ctx][2 KV heads][HD]
    __half *mkc = nullptr, *mkv = nullptr;   // the draft layer's
    Ring ring;
    R3 r3;
    int* ctl = nullptr;     // [0] the window's first position, [1] its tag
    // the draft pass: its token rows, h rows, [e | h] normed, residual, its output after the head norm, ctl
    float *me = nullptr, *mh = nullptr, *mcat = nullptr, *mR = nullptr, *mout = nullptr, *mlogits = nullptr;
    int* mctl = nullptr;
    float* msum = nullptr;   // the draft head's softmax denominator about this card's max
    cudaGraphExec_t graphs[MAXT + 1] = {};
    cudaGraphExec_t mgraphs[MAXT + 1] = {};
};

class S27 {
public:
    S27(Model& m, int max_ctx);
    // a window of T = tokens.size() rows at pos(): each row's argmax
    std::vector<int> verify(const std::vector<int>& tokens);
    // keep the window's first n rows
    void commit(int n);
    // the draft layer over T = tokens.size() rows at positions pos..: row r's h is the last window's row r
    // (from_trunk) or the last draft pass's output (T = 1). Returns the last row's draft token
    int draft(const std::vector<int>& tokens, int pos, bool from_trunk);
    int pos() const { return pos_; }
    float last_gpu_ms = 0.f;
    float last_draft_p = 0.f;   // the last draft's probability under the draft head (its vocabulary's softmax)
    float last_card_ms[NCARD] = {};   // each card's, of the last window

private:
    void enqueue(int T, std::FILE* dump);
    void enqueue_draft(int T);
    void embed(const std::vector<int>& tokens, float* out) const;
    unsigned next_tag(bool trunk);
    Model& m_;
    int max_ctx_;
    int pos_ = 0, pending_ = 0;
    unsigned tag_ = 1;        // the last trunk window's tag (its parity: the recurrent history half it wrote)
    unsigned tag_max_ = 1;    // the largest tag used: ring words tell exchanges apart by it
    Rope rope_;
    std::vector<S27Card> cb_;
    float* h_emb_ = nullptr;   // page-locked: the window's embedded rows
    float* h_memb_ = nullptr;  // the draft pass's
    int* h_ctl_ = nullptr;
    int* h_mctl_ = nullptr;
    float* h_bv_ = nullptr;
    int* h_bi_ = nullptr;
    unsigned* h_err_ = nullptr;
    float* h_ms_ = nullptr;
    std::vector<CopyJob*> commit_jobs_;
    CopyJob* h_jobs_ = nullptr;
};

}  // namespace fnx
