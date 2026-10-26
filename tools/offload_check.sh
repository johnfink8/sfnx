#!/bin/sh
# Routed experts in RAM (FNX_EXPERT_RAM_GIB, ranked by FNX_EXPERT_PROFILE): the logits on code1 against all in VRAM
# (byte for byte: the same weights read from another place), then decode speed (fnx_run mtp, K=4, lookup 7) on code1
# and edit1 at each amount. The all-in-VRAM logits are made by this run first, whatever the amounts asked for. Exits 1
# when a run fails or the logits differ.
#   tools/offload_check.sh [GIB ...]        (default: 0 8 12 16 24; G=model's first shard, PROFILE=tallies)
. "$(dirname "$0")/labenv.sh"
lab_tmp
PROFILE=${PROFILE:-data/expert-routes.bin}
[ $# -gt 0 ] || set -- 0 8 12 16 24
envs() {   # GIB: the settings for that many GiB of experts in RAM (0: every expert in VRAM, no cache)
    if [ "$1" = 0 ]; then echo "-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=off"
    else echo "-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=$1 -E FNX_EXPERT_PROFILE=$FNX/$PROFILE"; fi
}
logits() {   # GIB
    # shellcheck disable=SC2046
    run "logits-$1" -E FNX_PREFILL=1 -E FNX_MAX_CTX=16384 $(envs "$1") -- bin/fnx_run "$G" logits ref/code1.bin \
        ref/lgr/llama_cuda_code1.lgr 5 "$O/$1-code1.lgr"
}
echo "== every expert in VRAM: the logits to compare with"
logits 0 || finish
for gib in "$@"; do
    echo "== $gib GiB of experts in RAM"
    if [ "$gib" != 0 ] && logits "$gib"; then
        grep -E "in RAM" "$O/logits-$gib.log"
        if same "$O/0-code1.lgr" "$O/$gib-code1.lgr"; then echo "  logits: identical to every expert in VRAM"
        else failed "$gib GiB: logits differ from every expert in VRAM"; fi
    fi
    for p in code1 edit1; do
        # shellcheck disable=SC2046
        run "mtp-$gib-$p" -E FNX_PREFILL=1 -E FNX_MAX_CTX=16384 $(envs "$gib") -- bin/fnx_run "$G" mtp data/mtp-rt "ref/$p.bin" \
            600 4 data/draft_vocab_en.bin 7 && grep -E "tok/s" "$O/mtp-$gib-$p.log" | head -n 1 | sed "s/^/  $p: /"
    done
done
finish
