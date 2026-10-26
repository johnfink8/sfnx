#!/bin/sh
# A build against master's on the Q8 decode path: logits byte for byte (tools/pf_q8.sh on ref/code1: the prompt path,
# then windows of 5 through the expert cache), then solo rounds (tools/solo_rounds.sh, Q8) for master's build and for
# the new one at several FNX_CACHE_COPY_BLOCKS. Exits 1 when a run fails or the logits differ.
#   tools/ov_check.sh NEW_FNX_RUN [BLOCKS ...]       (default blocks: 160 80 40)
NEW=${1:?usage: tools/ov_check.sh NEW_FNX_RUN [BLOCKS ...]}
shift
[ $# -gt 0 ] || set -- 160 80 40
. "$(dirname "$0")/labenv.sh"
lab_tmp
q8_logits() {   # name [BIN]
    CTX=16384 BIN=${2:-bin/fnx_run} tools/pf_q8.sh ref/code1.bin ref/lgr/llama_cpu_q8w_code1.lgr "$O/$1.lgr" > "$O/$1.log" 2>&1 ||
        { failed "$1 logits: exit $? (log $O/$1.log)"; tail -3 "$O/$1.log"; }
}
q8_logits base
q8_logits new "$NEW"
if same "$O/base.lgr" "$O/new.lgr"; then echo "Q8 logits: identical to master's"; else failed "Q8 logits differ from master's (or one is missing)"; fi
solo() {   # label envs-for-solo_rounds...
    label=$1
    shift
    echo "== $label"
    env "$@" tools/solo_rounds.sh 512 > "$O/$label.txt" 2>&1 || failed "$label: exit $? (log $O/$label.txt)"
    grep -E "^(==|mtp K|per round|FAILED)" "$O/$label.txt"
}
solo master ONLY=Q8
for b in "$@"; do
    solo "new-blocks-$b" ONLY=Q8 BIN="$NEW" ENVS="-E FNX_CACHE_COPY_BLOCKS=$b"
done
finish
