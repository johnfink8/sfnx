#!/bin/sh
# Where a batched window's time goes: cupti kernel traces of W verify windows of L rows on one lane (A) and W batched
# windows of L lanes (B), Q4 (G= another GGUF, ENVS="-E VAR=value ..." for it), every lane at a context of C tokens;
# then tools/trace/graph_kernels.py per kernel on each card. Exits 1 when a run fails or writes no trace.
#   tools/step_trace.sh [L] [C] [W]        traces in a new directory, named at the end
L=${1:-6}
C=${2:-16384}
W=${3:-200}
. "$(dirname "$0")/labenv.sh"
lab_tmp
KEEP_O=1
trace() {   # label mode-args...
    label=$1
    shift
    # shellcheck disable=SC2086
    run "$label" -E FNX_MAX_CTX=$((L * (C + 1024))) -E FNX_LANES="$L" -E FNX_BENCH_CTX="$C" \
        -E CUPTI_TRACE_OUT="$O/fnxtr-$label" -E CUDA_INJECTION64_PATH="$P/libcupti_trace.so" \
        -E LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64 ${ENVS:-} -- bin/fnx_run "$G" "$@" || return
    grep -E "^(bench|step|verify)" "$O/$label.log"
}
trace verify bench ref/long64.bin "$L" "$W"
trace step bench_step ref/long64.bin "$L" "$W" "$C" step
a=$(ls "$O"/fnxtr-verify.*.csv 2> /dev/null) b=$(ls "$O"/fnxtr-step.*.csv 2> /dev/null)
if [ -z "$a" ] || [ -z "$b" ]; then
    failed "no trace written (verify: ${a:-none}, step: ${b:-none})"
    finish
fi
for d in $(seq 0 $(($(cat build/ncard) - 1))); do
    echo "== card $d: A = verify, $L rows on one lane; B = step, $L lanes"
    python3 "$P/graph_kernels.py" "$a" "$b" "$W" $d 18 || failed "graph_kernels.py on card $d: exit $?"
done
finish
