#!/bin/sh
# The arena (FNX_ARENA) against the experts all in VRAM, Q4 with int8 K/V: logits byte for byte on ref/long64 (a 62k-token
# prompt through the prompt path, then decode windows of 5) with nothing lent, and with a small own pool
# (FNX_ARENA_BASE_PAGES) so the pool lends slots during the read and the decode (the lent experts copied back in on
# their misses); then batched windows against each conversation alone (bin/batch_test) with lending. Exits 1 when a
# run fails, the logits differ or a lane differs from its conversation alone.
#   tools/arena_check.sh [BASE_PAGES]          (default 64)
BASE=${1:-64}
. "$(dirname "$0")/labenv.sh"
lab_tmp
ARENA="-E FNX_ARENA=1 -E FNX_EXPERT_RAM_GIB=0"
VRAM="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=off"   # every expert in VRAM, no cache (the defaults are prod's: arena, 3 GiB in RAM)
logits() {   # name envs...
    name=$1
    shift
    run "$name" -E FNX_KV8=1 -E FNX_PREFILL=1 -E FNX_MAX_CTX=65536 "$@" -- bin/fnx_run "$G" logits ref/long64.bin \
        ref/lgr/llama_cuda_long64.lgr 5 "$O/$name.lgr" || return
    echo "== $name: $(grep -cE '^fnx: arena: K/V pool' "$O/$name.log") lends"
    grep -E "^fnx: arena" "$O/$name.log" | head -3
}
same_as() {   # a b
    if same "$O/$1.lgr" "$O/$2.lgr"; then echo "$2 logits: identical to $1's"; else failed "$2 logits differ from $1's (or one is missing)"; fi
}
# shellcheck disable=SC2086
logits all-vram $VRAM
# shellcheck disable=SC2086
logits arena-none $ARENA
# shellcheck disable=SC2086
logits arena-lend $ARENA -E FNX_ARENA_BASE_PAGES="$BASE"
same_as all-vram arena-none
same_as all-vram arena-lend
# the own pool at most the context's pages (58 at 100,000 positions): 24 makes the lanes lend ~750 slots
echo "== batch_test, lending (own pool 24 pages)"
# shellcheck disable=SC2086
run batch-lend -E FNX_KV8=1 -E FNX_MAX_CTX=100000 $ARENA -E FNX_ARENA_BASE_PAGES=24 -- bin/batch_test "$G" "$FNX/data/mtp-rt" \
    data/draft_vocab_en.bin 64 ref/long2.bin ref/code1.bin ref/long2.bin
grep -vE "^\[|loaded" "$O/batch-lend.log"
finish
