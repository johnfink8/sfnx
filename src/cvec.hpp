// A control vector: one direction per model layer, from a GGUF whose tensors are named direction.N (llama.cpp's
// controlvector format; its index N names the output of block N, 1..NL-1: block 0 and the draft layer carry none).
// The engine applies it to a finished block output before that output joins the residual: each row x loses its
// component along the block's unit direction, x -= scale * (x . v) * v, the component read from x itself. scale 1
// removes it (a refusal-direction ablation); past 1 it inverts it (a published abliterated release of this model
// applies 1.78, measured from its weights). A row whose request asked the vector off gets scale 0, and its bits
// stay exactly what they were.
//
// Two outputs are edited, matching what the file's writers make: a block's attention half (its o_proj / ssm_out),
// and its shared expert's down_proj rows (before the ring folds the routed experts in - a weight-edited release
// leaves the routed experts whole, and so does this). Editing the summed half, once, is the same as editing each
// writer and letting the model sum them: removals along one direction commute with summation, since
// (a - (a.v)v) + (b - (b.v)v) = (a + b) - ((a + b).v)v. What it is not, and no runtime reading of a direction
// file can be, is the weight edit itself: an abliterated W' = W - s r u^T sets each writer's coefficient from
// that writer's input (u . x), not from its output. The two agree exactly when the edited weights' effect is a
// component removal read from the output; whether they agree in the behaviour that matters is a measured question
// this engine's exactness bar and a refusal probe are the checks for.
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
