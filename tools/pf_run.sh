#!/bin/sh
# The prompt path on one prompt, with per-chunk timing: prompt, reference .lgr, output .lgr (default /dev/null; paths
# from fnx/). TRACE=dir also records a cupti kernel trace there (dir/fnxtr.<pid>.csv). Extra settings:
# ENVS="-E VAR=value ...". Its exit status is fnx_run's.
#   tools/pf_run.sh ref/long2.bin ref/lgr/llama_cuda_long2.lgr [OUT.lgr]
. "$(dirname "$0")/labenv.sh"
CTX=${CTX:-16384}
tr=""
if [ -n "${TRACE:-}" ]; then
    mkdir -p "$TRACE" || exit 1
    tr="-E CUPTI_TRACE_OUT=$TRACE/fnxtr -E CUDA_INJECTION64_PATH=$P/libcupti_trace.so -E LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64"
fi
# shellcheck disable=SC2086
unit -E FNX_PREFILL=1 -E FNX_PF_TIMING=1 -E FNX_MAX_CTX="$CTX" $tr ${ENVS:-} -- bin/fnx_run "$G" logits "$1" "$2" 5 "${3:-/dev/null}"
