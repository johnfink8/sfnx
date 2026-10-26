#!/bin/sh
# The arena's cost with nothing lent (FNX_ARENA=1) against every expert in VRAM, Q4 with int8 K/V: the 6-lane window
# (fnx_run bench_step, 200 windows at 16k) and solo rounds (tools/solo_rounds.sh), the GPUs sampled; BASELINE=1 also
# runs every expert in VRAM (otherwise compare with the numbers already measured that day). Exits 1 when a run fails.
#   tools/arena_speed.sh LABEL           logs in /tmp/LABEL-*.txt
LABEL=${1:?usage: tools/arena_speed.sh LABEL}
. "$(dirname "$0")/labenv.sh"
ARENA="-E FNX_ARENA=1 -E FNX_EXPERT_RAM_GIB=0"
VRAM="-E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=off"   # every expert in VRAM, no cache (the defaults are prod's: arena, 3 GiB in RAM)
measure() {   # name command...
    out=/tmp/$LABEL-$1.txt
    shift
    sampled "$out" "$@" || failed "$(basename "$out" .txt): exit $? (log $out)"
    grep -E "^(verify, 1 lane, 5|step, (1|6) lanes|== Q4|mtp K|per round|FAILED)" "$out"
}
step() {   # envs...
    unit -E FNX_KV8=1 -E FNX_MAX_CTX=104448 -E FNX_LANES=6 "$@" -- bin/fnx_run "$G" bench_step ref/long64.bin 6 200 16384
}
for round in a; do
    # shellcheck disable=SC2086
    [ -n "${BASELINE:-}" ] && measure "off-step-$round" step $VRAM
    # shellcheck disable=SC2086
    [ -z "${ONLY_BASELINE:-}" ] && measure "arena-step-$round" step $ARENA
    [ -n "${BASELINE:-}" ] && measure "off-solo-$round" env ONLY=Q4 ENVS="-E FNX_KV8=1 $VRAM" tools/solo_rounds.sh
    [ -z "${ONLY_BASELINE:-}" ] && measure "arena-solo-$round" env ONLY=Q4 ENVS="-E FNX_KV8=1 $ARENA" tools/solo_rounds.sh
done
finish
