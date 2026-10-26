#!/bin/sh
# The prompt path's chunk size (FNX_PF_ROWS, default 4096) on the served config: the load at 262k (each card's free VRAM
# and the arena's own pool) and an ~8k-token random-word read through the API (serve/prompt_read.py); logits on
# ref/long64 (a 62k-token prompt through the prompt path, its read speed in the same load) against llama.cpp CUDA with
# int8 K/V, held to long64's bar (README: llama.cpp's own q8_0 K/V cache, KL mean 4.98e-3; top-1 97.5% less 1.5, as
# fnx_run exact). Exits 1 when sfnx does not come up, a run fails or the logits are outside the bar.
#   tools/chunk_check.sh [ROWS]        (default: the engine's, 4096)
. "$(dirname "$0")/labenv.sh"
need_py
lab_tmp
ROWS=${1:+-E FNX_PF_ROWS=$1}
URL=http://127.0.0.1:8095/v1
ENVS="$ROWS" serve/run.sh serve/sfnx.json > "$O/run.log" 2>&1 || { failed "serve/run.sh: exit $?"; tail -5 "$O/run.log"; finish; }
for _ in $(seq 120); do curl -sf -m 2 $URL/models > /dev/null && break; sleep 5; done
if ! curl -sf -m 2 $URL/models > /dev/null; then
    journalctl -u sfnx-test -n 20 --no-pager | grep -iE "error|fail|cuda"
    failed "sfnx did not come up"
    finish
fi
grep -E "arena: the K/V|sfnx: loaded" serve/sfnx.log | tail -2 | cut -c1-200
sampled "$O/read.txt" python3 serve/prompt_read.py 1 7000 $URL 41 || failed "serve/prompt_read.py: exit $? (log $O/read.txt)"
tail -1 "$O/read.txt"
sudo systemctl stop sfnx-test
echo "== logits, ref/long64, int8 K/V"
# shellcheck disable=SC2086
if run logits $ROWS -E FNX_KV8=1 -E FNX_PREFILL=1 -E FNX_MAX_CTX=65536 -- bin/fnx_run "$G" logits ref/long64.bin \
    ref/lgr/llama_cuda_long64.lgr 5 "$O/long64.lgr"; then
    grep -E "prefill:" "$O/logits.log" | tail -1
    "$PY" ref/lgrcmp.py ref/lgr/llama_cuda_long64.lgr "$O/long64.lgr" 4.98e-3 96.0 > "$O/cmp.txt" 2>&1 ||
        failed "long64 logits: outside the bar or not compared ($(tail -1 "$O/cmp.txt"))"
    head -n 2 "$O/cmp.txt"
fi
finish
