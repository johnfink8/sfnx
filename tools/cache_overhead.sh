#!/bin/sh
# What the expert cache's machinery costs with every expert resident (FNX_EXPERT_RAM_GIB=0 with a profile: the plan
# and copy kernels run each layer, nothing is in RAM) against the cache off: solo rounds (tools/solo_rounds.sh, Q4)
# and the 6-lane window (fnx_run bench_step, 200 windows at 16k), the GPUs sampled per run. The dynamic arena's
# measurement 2: over ~2% means the plan goes into the router kernel first. Exits 1 when a
# run fails.
#   tools/cache_overhead.sh LABEL           logs in /tmp/LABEL-*.txt
set -u
LABEL=${1:?usage: tools/cache_overhead.sh LABEL}
. "$(dirname "$0")/labenv.sh"
CACHE="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=0"
VRAM="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=off"   # the cache off: every expert in VRAM (the defaults are prod's)
measure() {   # name pattern command...: the output's lines matching pattern
    out=/tmp/$LABEL-$1.txt pat=$2
    shift 2
    sampled "$out" "$@" || failed "$(basename "$out" .txt): exit $? (log $out)"
    grep -E "$pat" "$out"
}
step() {   # envs...
    unit -E FNX_MAX_CTX=104448 -E FNX_LANES=6 "$@" -- bin/fnx_run "$G" bench_step ref/long64.bin 6 200 16384
}
STEP_LINES="^(verify|step|fnx_run|fnx: [0-9]+ of)"
for round in a b; do
    measure "off-solo-$round" . env ONLY=Q4 ENVS="$VRAM" tools/solo_rounds.sh
    measure "on-solo-$round" . env ONLY=Q4 ENVS="$CACHE" tools/solo_rounds.sh
    # shellcheck disable=SC2086
    measure "off-step-$round" "$STEP_LINES" step $VRAM
    # shellcheck disable=SC2086
    measure "on-step-$round" "$STEP_LINES" step $CACHE
done
finish
