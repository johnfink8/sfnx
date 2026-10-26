"""front/engine_limits.py - bin/sfnx's limits that the front end checks or fills in before it sends a request.

Each value mirrors a constant in the engine's C++ sources, named beside it.  test_engine_limits.py reads those sources
and fails when one side changes without the other.
"""
# tools/sfnx.cpp refuses a GEN / BGEN whose prompt + max_new + 8 exceed the context: max_new is fitted to the same margin
CTX_SLACK = 8
# tools/sfnx_input.hpp MAX_SLOTS: --batch takes 1..8 slots
PARALLEL_MAX = 8
# tools/sfnx.cpp EOS: <|endoftext|>, <|im_end|>
EOS_IDS = {248044, 248046}
# src/engine.hpp Engine::PEN_CAP: the longest penalty window
PENALTY_LAST_N_MAX = 4096
# src/session.hpp Sampling::pen_last_n: the window a penalty counts over when the request names none
PENALTY_LAST_N_DEFAULT = 64
# src/kernels.hpp TOPK_N: the candidates per row the sampled path keeps, so the widest top_k
TOP_K_MAX = 64
# src/session.hpp Sampling::seed is a uint64_t
SEED_MAX = 2**64 - 1
