// Reference logits from upstream llama.cpp (mtmd) for a prompt with a picture: the chat-formatted user message
// "<picture>\n<question>" with thinking off, read through mtmd (its image encoder, its M-RoPE positions), then N greedy
// steps. Writes what sFNX needs to read the same prompt: the cells' token ids (an image's cells as the image-pad token),
// the encoder's rows (strata-vision records), llama.cpp's own (t, h, w) per image cell, and the steps' logits.
//
//   img_logref MODEL.gguf MMPROJ.gguf IMAGE QUESTION N OUT_PREFIX [CONT.lgr]   (CONT: take its tokens instead of greedy)
//   env IMG_LOGREF_GPU=0: the language model on the CPU (the encoder stays on the GPU: the same rows)
//     OUT_PREFIX.bin (the cells' tokens, ref .bin), .sve (the image rows), .pos (t h w per image cell, text), .lgr
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

static constexpr uint32_t K = 64;
static constexpr int32_t IMAGE_PAD = 248056;

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
    if (argc < 7) {
        std::fprintf(stderr, "usage: img_logref MODEL.gguf MMPROJ.gguf IMAGE QUESTION N OUT_PREFIX\n");
        return 2;
    }
    try {
        const int N = std::atoi(argv[5]);
        const std::string out = argv[6];
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        const bool gpu = !(std::getenv("IMG_LOGREF_GPU") && std::atoi(std::getenv("IMG_LOGREF_GPU")) == 0);
        mp.n_gpu_layers = gpu ? 999 : 0;
        const std::vector<int> cont = argc > 7 ? read_lgr_tokens(argv[7]) : std::vector<int>();
        if (!cont.empty() && (int) cont.size() < N) throw std::runtime_error("cont: fewer steps than N");
        llama_model* model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 1;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const int n_embd = llama_model_n_embd(model);
        mtmd_context_params mcp = mtmd_context_params_default();
        mcp.use_gpu = true;
        mcp.n_threads = 8;
        mtmd_context* mctx = mtmd_init_from_file(argv[2], model, mcp);
        if (!mctx) throw std::runtime_error("mtmd_init_from_file failed");
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) throw std::runtime_error(std::string("cannot read ") + argv[3]);
        const std::string text = std::string("<|im_start|>user\n") + mtmd_default_marker() + "\n" + argv[4] +
                                 "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        mtmd_input_text it{text.c_str(), text.size(), false, true};
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const mtmd_bitmap* bms[1] = {bw.bitmap};
        if (mtmd_tokenize(mctx, chunks, &it, bms, 1) != 0) throw std::runtime_error("mtmd_tokenize failed");

        // the cells, the image rows and positions
        std::vector<int32_t> cells;
        std::FILE* sve = std::fopen((out + ".sve").c_str(), "wb");
        std::FILE* pos = std::fopen((out + ".pos").c_str(), "w");
        if (!sve || !pos) throw std::runtime_error("cannot write " + out + ".sve/.pos");
        llama_pos p = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token* t = mtmd_input_chunk_get_tokens_text(ch, &n);
                cells.insert(cells.end(), t, t + n);
                p += (llama_pos) n;
                continue;
            }
            const mtmd_image_tokens* im = mtmd_input_chunk_get_tokens_image(ch);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ch);
            if (mtmd_encode_chunk(mctx, ch) != 0) throw std::runtime_error("mtmd_encode_chunk failed");
            const float* e = mtmd_get_output_embd(mctx);
            int nx = 0, ny = 0;
            for (int j = 0; j < n; ++j) {
                const mtmd_decoder_pos dp = mtmd_image_tokens_get_decoder_pos(im, p, (size_t) j);
                std::fprintf(pos, "%d %u %u %u\n", (int) cells.size() + j, dp.t, dp.y, dp.x);
                nx = std::max(nx, (int) dp.x - (int) p + 1);
                ny = std::max(ny, (int) dp.y - (int) p + 1);
            }
            if (nx * ny != n) throw std::runtime_error("the image's positions are not an nx x ny grid");
            const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
            std::fwrite(hdr, 4, 5, sve);
            std::fwrite(e, 4, (size_t) n * n_embd, sve);
            cells.insert(cells.end(), (size_t) n, IMAGE_PAD);
            p += mtmd_input_chunk_get_n_pos(ch);
            std::fprintf(stderr, "img_logref: image of %d cells (%d x %d), %d positions\n", n, nx, ny, (int) mtmd_input_chunk_get_n_pos(ch));
        }
        std::fclose(sve);
        std::fclose(pos);
        {   // the cells as a ref .bin ("tokens", one dimension, floats)
            std::FILE* f = std::fopen((out + ".bin").c_str(), "wb");
            const std::string name = "tokens";
            const uint32_t nl = (uint32_t) name.size(), nd = 1;
            const int64_t ne = (int64_t) cells.size();
            std::fwrite(&nl, 4, 1, f);
            std::fwrite(name.data(), 1, nl, f);
            std::fwrite(&nd, 4, 1, f);
            std::fwrite(&ne, 8, 1, f);
            for (int32_t c : cells) {
                const float v = (float) c;
                std::fwrite(&v, 4, 1, f);
            }
            std::fclose(f);
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) (cells.size() + N + 64);
        cp.n_batch = cp.n_ubatch = 512;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(mctx, ctx, chunks, 0, 0, (int) cp.n_batch, true, &n_past) != 0)
            throw std::runtime_error("mtmd_helper_eval_chunks failed");
        std::FILE* lg = std::fopen((out + ".lgr").c_str(), "wb");
        const int32_t np = (int32_t) cells.size(), ns = N;
        std::fwrite("LGR1", 1, 4, lg);
        std::fwrite(&K, 4, 1, lg);
        std::fwrite(&np, 4, 1, lg);
        std::fwrite(cells.data(), 4, cells.size(), lg);
        std::fwrite(&ns, 4, 1, lg);
        llama_batch b = llama_batch_init(1, 0, 1);
        for (int s = 0; s < N; ++s) {
            const int tok = put_step(lg, llama_get_logits_ith(ctx, -1), n_vocab, cont.empty() ? -1 : cont[(size_t) s]);
            if (s + 1 == N) break;
            b.n_tokens = 1;
            b.token[0] = tok;
            b.pos[0] = n_past + s;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
        }
        std::fclose(lg);
        std::fprintf(stderr, "img_logref: %d cells, %d positions, %d steps\n", np, (int) n_past, N);
        llama_batch_free(b);
        llama_free(ctx);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        mtmd_free(mctx);
        llama_model_free(model);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "img_logref: %s\n", e.what());
        return 1;
    }
    return 0;
}
