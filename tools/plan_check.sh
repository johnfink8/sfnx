#!/bin/sh
# The expert cache's plan in the router's last block against master's separate launch: Q8 logits byte for byte on the
# decode path through the cache (tools/pf_q8.sh on ref/code1, experts copied in every window), then the cost with
# every expert resident (FNX_EXPERT_RAM_GIB=0) against the cache off: the 6-lane window (bench_step) and solo rounds
# (Q4), the GPUs sampled per run. Exits 1 when a run fails or the logits differ.
#   tools/plan_check.sh MASTER_FNX_RUN LABEL         logs in /tmp/LABEL-*.txt; the new build is bin/fnx_run
MASTER=${1:?usage: tools/plan_check.sh MASTER_FNX_RUN LABEL}
LABEL=${2:?usage: tools/plan_check.sh MASTER_FNX_RUN LABEL}
. "$(dirname "$0")/labenv.sh"
lab_tmp
CACHE="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=0"
VRAM="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=off"   # the cache off: every expert in VRAM (the defaults are prod's)
q8_logits() {   # name BIN
    CTX=16384 BIN=$2 tools/pf_q8.sh ref/code1.bin ref/lgr/llama_cpu_q8w_code1.lgr "$O/$1.lgr" > "$O/$1.log" 2>&1 ||
        { failed "$1 logits: exit $? (log $O/$1.log)"; tail -3 "$O/$1.log"; }
}
q8_logits master "$MASTER"
q8_logits new bin/fnx_run
if same "$O/master.lgr" "$O/new.lgr"; then echo "Q8 logits: identical to master's ($(stat -c %s "$O/new.lgr") bytes)"
else failed "Q8 logits differ from master's (or one is missing)"; fi
measure() {   # name command...
    out=/tmp/$LABEL-$1.txt
    shift
    sampled "$out" "$@" || failed "$(basename "$out" .txt): exit $? (log $out)"
    grep -E "^(verify, 1 lane, 5|step, (1|6) lanes|== Q4|mtp K|FAILED)" "$out"
}
step() {   # binary envs...
    bin=$1
    shift
    unit -E FNX_MAX_CTX=104448 -E FNX_LANES=6 "$@" -- "$bin" "$G" bench_step ref/long64.bin 6 200 16384
}
for round in a b; do
    # shellcheck disable=SC2086
    measure "off-step-$round" step bin/fnx_run $VRAM
    # shellcheck disable=SC2086
    measure "on-step-$round" step bin/fnx_run $CACHE
    measure "off-solo-$round" env ONLY=Q4 ENVS="$VRAM" tools/solo_rounds.sh
    measure "on-solo-$round" env ONLY=Q4 ENVS="$CACHE" tools/solo_rounds.sh
done
finish
