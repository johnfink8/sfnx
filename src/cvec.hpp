// A control vector: one direction per model layer, from a GGUF whose tensors are named direction.N (llama.cpp's
// controlvector format; its index N names the output of block N, 1..NL-1: block 0 and the draft layer carry none).
// The engine projects the model's hidden at each block boundary: at the read that folds block N's finished output
// into the residual, every hyper-connection stream of every row loses its component along direction.N,
// r -= scale * (r . v) * v, the dot taken on the folded row itself (the value the read computes, measured before
// the read stores it). scale 1 removes the component, past 1 it inverts it (a published abliterated release of this
// model applies 1.78, measured from its weights). A row whose request asked the vector off has scale 0: at every
// boundary its bits stay exactly what they were.
//
// Why the boundary and not the block's own writers: the component being removed is largely one that has already
// accumulated in the hidden (from the embeddings, from earlier blocks, from the routed experts, which no direction
// file names). An engine that edits only what a block writes leaves all of that untouched: this one measured it
// that way first (11 of 24 refusal-probe prompts still refused with the vector at 1.0, against ~1 of 50 for the
// published llama.cpp port of the same method). The boundary reading is that port's: a published patch projects
// each stream after the block completes (build_cvec runs after build_hc_combine; its ggml_mul_mat broadcasts over
// the stream dimension), and a block's output joins the residual exactly at the next read's fold - here, rprime.
#pragma once

#include <string>
#include <vector>

namespace fnx {

// What FNX_CVEC names. Nothing set: the engine loads no vector and makes no cvec call anywhere.
struct CvecSpec {
    std::string path;
    double scale = 0.0;      // 0..4
    int range_lo = -1, range_hi = -1;   // FNX_CVEC_RANGE=START:END, when given
};

// reads FNX_CVEC=FILE:SCALE and FNX_CVEC_RANGE=START:END. No FNX_CVEC: returns false. A bad value throws naming it,
// at engine start, like every other setting.
bool cvec_from_env(CvecSpec& out);

// The directions, host side: rows[l] is direction.l (a block outside the applied range is zero and not carried).
// The removal above is exact for unit directions only, so each is checked to 1e-3 at load.
struct Cvec {
    CvecSpec spec;
    int n = 0, nl = 0;
    std::vector<float> rows;      // [nl][n]
    std::vector<char> carried;    // [nl]
    int lo = 0, hi = -1;          // the blocks whose output a direction names (the file's, cut by range)
    bool on() const { return !rows.empty(); }
    bool edits(int produced) const { return produced >= lo && produced <= hi && carried[(size_t) produced] != 0; }
};

// loads the file and checks it against this model: every direction 1-D F32 named direction.N with 1 <= N < NL and n
// wide, each unit-norm; the file must carry at least one; when it names a model hint or an architecture, they must
// name this engine's. Throws, naming the first violation, on anything else.
Cvec load_cvec(const CvecSpec& spec, const std::string& hint, int n_embd, int nl);

}  // namespace fnx
