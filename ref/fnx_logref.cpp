// Reference logits from upstream llama.cpp for the fnx comparison: reads a prompt (make_prompt.py's token file) in
// one batch, then either generates greedily one token at a time (llama.cpp's own decode path: its text is the
// reference) or feeds a given continuation (teacher-forced, e.g. for another backend on the same tokens). At every
// step it saves the top-K logits and the log-sum-exp of the whole row.
//
//   fnx_logref MODEL.gguf PROMPT.bin N OUT.lgr [CONT.lgr]      (N steps; CONT: take its tokens instead of greedy)
//   env FNX_LOGREF_GPU=0: CPU only; FNX_LOGREF_UBATCH=n: the prompt in batches of n (default 512); FNX_LOGREF_KV=q8_0:
//   a q8_0 K/V cache
//
// OUT.lgr: "LGR1", u32 K, i32 n_prompt, i32 prompt[n_prompt], i32 n_steps, then per step i (the distribution of the
// token after prompt + steps 0..i-1): i32 token taken, f32 lse, K x (i32 id, f32 logit) by logit descending.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t K = 64;

static std::vector<int> read_prompt(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("open ") + path);
    uint32_t nl = 0, nd = 0;
    int64_t n = 0;
    if (std::fread(&nl, 4, 1, f) != 1) throw std::runtime_error("prompt: header");
    std::string name(nl, '\0');
    if (std::fread(name.data(), 1, nl, f) != nl || std::fread(&nd, 4, 1, f) != 1 || nd != 1 || std::fread(&n, 8, 1, f) != 1)
        throw std::runtime_error("prompt: header");
    std::vector<float> v((size_t) n);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) throw std::runtime_error("prompt: tokens");
    std::fclose(f);
    return std::vector<int>(v.begin(), v.end());
}

static std::vector<int> read_lgr_tokens(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("open ") + path);
    char magic[4];
    uint32_t k = 0;
    int32_t np = 0, ns = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "LGR1", 4) || std::fread(&k, 4, 1, f) != 1 ||
        std::fread(&np, 4, 1, f) != 1)
        throw std::runtime_error("cont: header");
    std::fseek(f, (long) np * 4, SEEK_CUR);
    if (std::fread(&ns, 4, 1, f) != 1) throw std::runtime_error("cont: header");
    std::vector<int> t((size_t) ns);
    for (int i = 0; i < ns; ++i) {
        int32_t tok;
        if (std::fread(&tok, 4, 1, f) != 1) throw std::runtime_error("cont: step");
        t[(size_t) i] = tok;
        std::fseek(f, 4 + (long) k * 8, SEEK_CUR);
    }
    std::fclose(f);
    return t;
}

// the step record for one row of logits; returns the argmax (the lower id on a tie)
static int put_step(std::FILE* f, const float* lg, int n_vocab, int taken_or_neg) {
    float mx = -INFINITY;
    int best = 0;
    for (int i = 0; i < n_vocab; ++i)
        if (lg[i] > mx) { mx = lg[i]; best = i; }
    double s = 0;
    for (int i = 0; i < n_vocab; ++i) s += std::exp((double) lg[i] - mx);
    const float lse = (float) (mx + std::log(s));
    std::vector<int> idx((size_t) n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    const int32_t taken = taken_or_neg >= 0 ? taken_or_neg : best;
    std::fwrite(&taken, 4, 1, f);
    std::fwrite(&lse, 4, 1, f);
    for (uint32_t k = 0; k < K; ++k) {
        const int32_t id = idx[k];
        std::fwrite(&id, 4, 1, f);
        std::fwrite(&lg[id], 4, 1, f);
    }
    return taken;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: fnx_logref MODEL.gguf PROMPT.bin N OUT.lgr [CONT.lgr]\n");
        return 2;
    }
    try {
        const std::vector<int> prompt = read_prompt(argv[2]);
        const int N = std::atoi(argv[3]);
        std::vector<int> cont;
        if (argc > 5) {
            cont = read_lgr_tokens(argv[5]);
            if ((int) cont.size() < N) throw std::runtime_error("cont: fewer steps than N");
        }
        const bool gpu = !(std::getenv("FNX_LOGREF_GPU") && std::atoi(std::getenv("FNX_LOGREF_GPU")) == 0);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = gpu ? 999 : 0;
        llama_model* model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 1;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) (prompt.size() + N + 16);
        // FNX_ROPE_SCALE=S + FNX_YARN_ORIG_CTX: YaRN over the trained length, for scaled-backend comparisons
        // (llama.cpp's own defaults: ext_factor 1, betas 32/1; its net attention magnitude matches sfnx's)
        if (const char* rs = std::getenv("FNX_ROPE_SCALE")) {
            cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
            cp.rope_freq_scale = 1.0f / std::stof(rs);
            if (const char* oc = std::getenv("FNX_YARN_ORIG_CTX")) cp.yarn_orig_ctx = (uint32_t) std::stoul(oc);
        }
        // FNX_LOGREF_UBATCH: the prompt's batch size (1: the prompt through the decode path, token by token)
        cp.n_batch = cp.n_ubatch = std::getenv("FNX_LOGREF_UBATCH") ? std::atoi(std::getenv("FNX_LOGREF_UBATCH")) : 512;
        // FNX_LOGREF_KV=q8_0: the K/V cache as q8_0 (a quantized V cache needs flash attention)
        if (const char* kv = std::getenv("FNX_LOGREF_KV")) {
            if (std::strcmp(kv, "q8_0") != 0) throw std::runtime_error("FNX_LOGREF_KV: only q8_0");
            cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;
        std::FILE* out = std::fopen(argv[4], "wb");
        if (!out) { std::perror(argv[4]); return 1; }
        const int32_t np = (int32_t) prompt.size(), ns = N;
        std::fwrite("LGR1", 1, 4, out);
        std::fwrite(&K, 4, 1, out);
        std::fwrite(&np, 4, 1, out);
        std::fwrite(prompt.data(), 4, prompt.size(), out);
        std::fwrite(&ns, 4, 1, out);
        // the prompt in batches of n_batch, logits for its last row only
        llama_batch b = llama_batch_init((int) cp.n_batch, 0, 1);
        for (int p0 = 0; p0 < np; p0 += (int) cp.n_batch) {
            b.n_tokens = std::min((int) cp.n_batch, np - p0);
            for (int i = 0; i < b.n_tokens; ++i) {
                b.token[i] = prompt[(size_t) (p0 + i)];
                b.pos[i] = p0 + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = p0 + i == np - 1;
            }
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode (prompt) failed");
        }
        // then one token per decode
        for (int s = 0; s < N; ++s) {
            const float* lg = llama_get_logits_ith(ctx, -1);
            const int tok = put_step(out, lg, n_vocab, cont.empty() ? -1 : cont[(size_t) s]);
            if (s + 1 == N) break;
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = np + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
        }
        std::fclose(out);
        std::fprintf(stderr, "fnx_logref: %d prompt tokens, %d steps (%s, %s)\n", np, N, cont.empty() ? "greedy" : "forced",
                     gpu ? "CUDA" : "CPU");
        llama_batch_free(b);
        llama_free(ctx);
        llama_model_free(model);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx_logref: %s\n", e.what());
        return 1;
    }
    return 0;
}
