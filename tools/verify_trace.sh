#!/bin/sh
# Where a solo verify window's time goes: a cupti trace of W verify windows of T rows on one lane at a context of C
# tokens, Q4 and Q8 (as sfnx-q8.json runs it), summed by kernel group per card (tools/trace/kernel_groups.py). Exits 1
# when a run fails or writes no trace.
#   tools/verify_trace.sh [T] [C] [W]      traces in a new directory, named at the end
T=${1:-5}
C=${2:-4096}
W=${3:-200}
. "$(dirname "$0")/labenv.sh"
lab_tmp
KEEP_O=1
vtrace() {   # label gguf envs...
    label=$1 g=$2
    shift 2
    echo "== $label, $T rows, context $C"
    run "$label" -E FNX_MAX_CTX=$((C + 4096)) -E FNX_BENCH_CTX="$C" -E CUPTI_TRACE_OUT="$O/fnxtr-vt-$label" \
        -E CUDA_INJECTION64_PATH="$P/libcupti_trace.so" -E LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64 "$@" \
        -- bin/fnx_run "$g" bench ref/long64.bin "$T" "$W" || return
    grep -E "^bench" "$O/$label.log"
    t=$(ls "$O"/fnxtr-vt-"$label".*.csv 2> /dev/null)
    [ -n "$t" ] || { failed "$label: no trace written"; return 1; }
    # shellcheck disable=SC2086
    python3 "$P/kernel_groups.py" $t || failed "kernel_groups.py on $label: exit $?"
}
vtrace Q4 "$G"
vtrace Q8 "$Q8" -E FNX_KV8=1 -E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=52 -E FNX_EXPERT_PROFILE="$FNX/data/expert-routes.bin"
finish
