// FNX_ROPE_SCALE / FNX_YARN_ORIG_CTX on a llama.cpp context for the reference tools, read by the engine's own rules
// (src/env.hpp; src/engine.cu's rope_scale), so a teacher is never made with a setting the engine would refuse
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "../src/env.hpp"
#include "llama.h"

// sets YaRN on cp when FNX_ROPE_SCALE is above 1 (1 is off), its parameters spelled out as the ones src/rope.hpp
// follows rather than left to llama.cpp's defaults, and says on stderr what it applied
inline void apply_yarn(llama_context_params& cp) {
    const char* rs = std::getenv("FNX_ROPE_SCALE");
    const double S = rs ? fnx::env_num("FNX_ROPE_SCALE", 0.0) : 0.0;
    if (rs && S != 1.0 && !(S > 1.0)) throw std::runtime_error(std::string("FNX_ROPE_SCALE: want 1 (off) or a scale above 1, got ") + rs);
    const bool on = rs && S > 1.0;
    if (!on) {
        if (std::getenv("FNX_YARN_ORIG_CTX")) throw std::runtime_error("FNX_YARN_ORIG_CTX is set without FNX_ROPE_SCALE, which it is for");
        return;
    }
    cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    cp.rope_freq_scale = (float) (1.0 / S);
    cp.yarn_ext_factor = 1.0f;
    cp.yarn_beta_fast = 32.0f;
    cp.yarn_beta_slow = 1.0f;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        const long long orig = fnx::env_int("FNX_YARN_ORIG_CTX", 0);
        if (orig <= 0 || orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(orig));
        cp.yarn_orig_ctx = (uint32_t) orig;
    }
    std::fprintf(stderr, "logref: YaRN x%.9g from n_ctx_orig %s\n", S,
                 std::getenv("FNX_YARN_ORIG_CTX") ? std::getenv("FNX_YARN_ORIG_CTX") : "(the model's)");
}
