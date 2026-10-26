#!/bin/sh
# What a decode window costs by its rows, Q4 (UD-Q4_K_XL, all experts in VRAM) and Q8 (int8 K/V, 52 GiB of experts in
# RAM): fnx_run bench_step at a context of C tokens on six lanes (verify windows of 1..16 rows on one lane, batched
# windows of 1..6 lanes). Exits 1 when a run fails.
#   tools/step_cost.sh [C]
C=${1:-16384}
. "$(dirname "$0")/labenv.sh"
lab_tmp
cost() {   # label gguf envs...
    label=$1 g=$2
    shift 2
    echo "== $label, context $C"
    run "$label" -E FNX_MAX_CTX=$((6 * (C + 1024))) -E FNX_LANES=6 "$@" -- bin/fnx_run "$g" bench_step ref/long64.bin 6 40 "$C"
    grep -E "^(verify|step)" "$O/$label.log"
}
cost Q4 "$G"
cost Q8 "$Q8" -E FNX_KV8=1 -E FNX_EXPERT_RAM_GIB=52 -E FNX_EXPERT_PROFILE="$FNX/data/expert-routes.bin"
finish
