#!/bin/sh
# The prompt path on the Q8_0 model as sfnx-q8.json runs it (int8 K/V, 52 GiB of experts in RAM; GIB= another amount):
# per-chunk timing and the logits after the prompt (paths from fnx/). BIN=path picks another fnx_run. Its exit status
# is fnx_run's.
#   tools/pf_q8.sh ref/code1.bin ref/lgr/llama_cpu_q8w_code1.lgr OUT.lgr
. "$(dirname "$0")/labenv.sh"
BIN=${BIN:-bin/fnx_run}
# shellcheck disable=SC2086
unit -E FNX_PREFILL=1 -E FNX_PF_TIMING=1 -E FNX_MAX_CTX="${CTX:-65536}" -E FNX_KV8=1 -E FNX_ARENA=0 \
    -E FNX_EXPERT_RAM_GIB="${GIB:-52}" -E FNX_EXPERT_PROFILE="$FNX/data/expert-routes.bin" ${ENVS:-} \
    -- "$BIN" "$Q8" logits "$1" "$2" 5 "${3:-/dev/null}"
