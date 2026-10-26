#!/bin/sh
# Where a solo decode round's time goes (fnx_run mtp: K=4 MTP drafts, lookup drafts up to 7), Q4 and Q8 (as
# sfnx-q8.json runs it), on ref/code1 (new code) and ref/edit1 (an edit), 512 tokens each. Exits 1 when a run fails.
#   tools/solo_rounds.sh [N]
N=${1:-512}
. "$(dirname "$0")/labenv.sh"
BIN=${BIN:-bin/fnx_run}   # another fnx_run; ONLY=Q4 or ONLY=Q8 runs one model; ENVS="-E VAR=value ..." for every run
lab_tmp
solo() {   # label gguf envs...
    label=$1 g=$2
    shift 2
    for p in code1 edit1; do
        echo "== $label $p"
        # shellcheck disable=SC2086
        run "$label-$p" -E FNX_MAX_CTX=16384 "$@" ${ENVS:-} -- "$BIN" "$g" mtp data/mtp-rt "ref/$p.bin" "$N" 4 \
            data/draft_vocab_en.bin 7
        grep -E "^(mtp K|draft step|per round|lookup drafts|per step|prefetch probe)" "$O/$label-$p.log"
    done
}
[ "${ONLY:-Q4}" = Q4 ] && solo Q4 "$G"
[ "${ONLY:-Q8}" = Q8 ] && solo Q8 "$Q8" -E FNX_KV8=1 -E FNX_ARENA=0 -E FNX_EXPERT_RAM_GIB=52 \
    -E FNX_EXPERT_PROFILE="$FNX/data/expert-routes.bin"
finish
