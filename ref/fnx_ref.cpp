// Reference dump for the fnx engine: runs a prompt through upstream llama.cpp (Flash-Next GGUF, on the GPUs) in one
// batch with every row's logits, and saves the named graph tensors that match a regex as float32.
//
//   fnx_ref MODEL.gguf OUT.bin REGEX "prompt text"
//
// OUT.bin: repeated records of  u32 name_len, name, u32 n_dims, i64 ne[n_dims], f32 data (ne0 fastest, contiguous).
// The first record is "tokens" (the prompt's token ids as floats).
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

struct Ctx {
    std::regex re;
    std::FILE* out;
    int saved = 0;
};

static void put(std::FILE* f, const std::string& name, const std::vector<int64_t>& ne, const std::vector<float>& v) {
    const uint32_t nl = (uint32_t) name.size(), nd = (uint32_t) ne.size();
    std::fwrite(&nl, 4, 1, f);
    std::fwrite(name.data(), 1, nl, f);
    std::fwrite(&nd, 4, 1, f);
    std::fwrite(ne.data(), 8, nd, f);
    std::fwrite(v.data(), 4, v.size(), f);
}

static float half_to_f(uint16_t h) {
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

static bool on_tensor(ggml_tensor* t, bool ask, void* user) {
    auto* c = (Ctx*) user;
    const std::string name = ggml_get_name(t);
    const bool want = std::regex_match(name, c->re);
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32) return true;
    // read the bytes the view spans, then walk ne/nb into a contiguous float array
    const size_t span = (size_t) ((t->ne[3] - 1) * t->nb[3] + (t->ne[2] - 1) * t->nb[2] + (t->ne[1] - 1) * t->nb[1] +
                                  (t->ne[0] - 1) * t->nb[0]) + ggml_type_size(t->type);
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, span);
    std::vector<float> v;
    v.reserve((size_t) ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const uint8_t* p = raw.data() + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    float x;
                    if (t->type == GGML_TYPE_F32) std::memcpy(&x, p, 4);
                    else if (t->type == GGML_TYPE_I32) { int32_t q; std::memcpy(&q, p, 4); x = (float) q; }
                    else { uint16_t h; std::memcpy(&h, p, 2); x = half_to_f(h); }
                    v.push_back(x);
                }
    std::vector<int64_t> ne(t->ne, t->ne + ggml_n_dims(t));
    put(c->out, name, ne, v);
    ++c->saved;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: fnx_ref MODEL.gguf OUT.bin REGEX \"prompt\"\n");
        return 2;
    }
    Ctx c{std::regex(argv[3]), std::fopen(argv[2], "wb")};
    if (!c.out) { std::perror(argv[2]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const std::string prompt = argv[4];
    std::vector<llama_token> toks(prompt.size() + 8);
    const int n = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true);
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize((size_t) n);
    std::vector<float> tf(toks.begin(), toks.end());
    put(c.out, "tokens", {(int64_t) n}, tf);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = std::getenv("REF_UBATCH") ? std::atoi(std::getenv("REF_UBATCH")) : 512;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &c;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    // in chunks of n_batch rows (REF_UBATCH): dumps then come once per chunk, in order
    const int chunk = (int) cp.n_batch;
    llama_batch b = llama_batch_init(chunk, 0, 1);
    for (int p0 = 0; p0 < n; p0 += chunk) {
        b.n_tokens = std::min(chunk, n - p0);
        for (int i = 0; i < b.n_tokens; ++i) {
            b.token[i] = toks[(size_t) (p0 + i)];
            b.pos[i] = p0 + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    }
    llama_synchronize(ctx);
    std::fclose(c.out);
    std::fprintf(stderr, "fnx_ref: %d tokens, %d tensors saved\n", n, c.saved);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
