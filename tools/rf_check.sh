#!/bin/sh
# GPU checks for a build against master's: the prompt path and a few windows' logits byte for byte (ref/long2, Q4), the
# same with the hc barrier counter preset just before its wrap (FNX_HC_BAR_WRAP) and the ring tags just before 2^24
# (FNX_RING_TAG_START), then batched windows against each conversation alone, also near both wraps. Exits 1 when a run
# fails, a logits file differs from master's (or is missing) or a lane differs from its conversation alone.
#   tools/rf_check.sh NEW_BIN_DIR [BASE_BIN_DIR]      (bin dirs from fnx/, holding fnx_run and batch_test)
NEW=${1:?usage: tools/rf_check.sh NEW_BIN_DIR [BASE_BIN_DIR]}
BASE=${2:-bin}
. "$(dirname "$0")/labenv.sh"
lab_tmp
logits() {   # bin_dir name envs...
    b=$1 name=$2
    shift 2
    run "$name" -E FNX_PREFILL=1 -E FNX_MAX_CTX=16384 "$@" -- "$b/fnx_run" "$G" logits ref/long2.bin ref/lgr/llama_cuda_long2.lgr 5 \
        "$O/$name.lgr"
    grep -E "ring tags|barrier" "$O/$name.log"
}
logits "$BASE" base
logits "$NEW" new
logits "$NEW" bar -E FNX_HC_BAR_WRAP=3
logits "$NEW" tag -E FNX_RING_TAG_START=16777000
for f in new bar tag; do
    if same "$O/base.lgr" "$O/$f.lgr"; then echo "logits $f: identical to master's"; else failed "logits $f: differ from master's (or missing)"; fi
done
i=0
for e in "" "-E FNX_HC_BAR_WRAP=3" "-E FNX_RING_TAG_START=16777000"; do
    i=$((i + 1))
    echo "== batch_test ${e:-plain}"
    # shellcheck disable=SC2086
    run "batch-$i" -E FNX_MAX_CTX=100000 $e -- "$NEW/batch_test" "$G" "$FNX/data/mtp-rt" data/draft_vocab_en.bin 48 \
        ref/long2.bin ref/code1.bin ref/long2.bin
    grep -E "lanes:|batched|MISMATCH|ring tags|barrier" "$O/batch-$i.log"
done
finish
