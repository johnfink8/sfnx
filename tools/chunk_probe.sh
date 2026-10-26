#!/bin/sh
# Prompt read speed by chunk size: fnx_run builds with different FNX_PF_C (rows per chunk), each reading prompts of
# several lengths from ref/long64.bin through the prompt path (bench_step's context read, FNX_PF_TIMING=1: its passes,
# not warm_graphs'), Q4 at prod's settings (the engine's defaults), at a context of 40,960 (room for the larger
# chunks' buffers); the GPUs sampled per run. (tools/lab.sh read is the quicker way to time one build.) Exits 1 when a
# run fails or shows no prefill passes.
#   tools/chunk_probe.sh "LABEL=FNX_RUN ..." "4096 8192 16384 32768"     (ENVS="-E VAR=value ..." for every run)
. "$(dirname "$0")/labenv.sh"
for spec in $1; do
    label=${spec%%=*} bin=${spec#*=}
    for n in $2; do
        out=/tmp/chunk-$label-$n.txt
        # shellcheck disable=SC2086
        sampled "$out" unit -E FNX_MAX_CTX=40960 -E FNX_PF_TIMING=1 ${ENVS:-} -- "$bin" "$G" bench_step ref/long64.bin 1 2 "$n" step ||
            { failed "$label $n: exit $? ($(grep -m1 -iE 'error|fnx_run:|cudaMalloc' "$out"))"; continue; }
        # the read's passes: from the last one at position 0 (warm_graphs' own read comes first, in one or more passes)
        ms=$(awk '/^prefill pass at 0:/ {s = 0} /^prefill pass/ {match($0, /rows in [0-9.]+ ms/); s += substr($0, RSTART + 8, RLENGTH - 11)} END {print s}' "$out")
        if [ -z "$ms" ]; then failed "$label $n: no prefill passes in $out"; continue; fi
        echo "$label $n tokens: $ms ms, $(awk -v n="$n" -v m="$ms" 'BEGIN {printf "%.0f", n / m * 1000}') tok/s"
    done
done
finish
