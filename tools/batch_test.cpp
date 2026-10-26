// Batched windows against each conversation alone (tools/batch_check.hpp); the lab runs the same check as "batch".
//
//   batch_test MODEL.gguf RT_DIR VOCAB.bin GEN A.bin [B.bin ...]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "batch_check.hpp"
#include "ref_tokens.hpp"
#include "../src/env.hpp"

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "batch_test MODEL.gguf RT_DIR VOCAB.bin GEN A.bin [B.bin ...]\n"); return 2; }
    fnx::Model m(argv[1]);
    fnx::Engine e(m, (int) fnx::env_int("FNX_MAX_CTX", 32768));
    std::vector<std::vector<int>> prompts;
    for (int i = 5; i < argc; ++i) prompts.push_back(ref_tokens(argv[i]));
    return batch_check(m, e, argv[2], argv[3], std::atoi(argv[4]), prompts);
}
