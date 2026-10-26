#!/bin/sh
# The prompt path on chunks of 1, 2 and 3 rows (a read cut at a checkpoint can leave one): fnx_run bench after a
# prefill of N tokens, Q4, for N in 1 2 3 1026. BIN=path picks another fnx_run. Exits 1 when a run fails.
#   tools/tiny_chunk.sh
. "$(dirname "$0")/labenv.sh"
BIN=${BIN:-bin/fnx_run}
lab_tmp
for n in 1 2 3 1026; do
    if run "prefill-$n" -E FNX_MAX_CTX=8192 -E FNX_BENCH_CTX="$n" -- "$BIN" "$G" bench ref/long64.bin 1 5; then
        echo "prefill of $n tokens: $(grep -E "^bench" "$O/prefill-$n.log" | head -1)"
    else
        echo "prefill of $n tokens: FAILED"
    fi
done
finish
