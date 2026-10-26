#!/bin/sh
# A/B timing of engine variants in one binary: each argument is a set of VAR=value settings (comma-separated, "-" for
# none); every variant runs ROUNDS times in turn (interleaved, so drift hits all alike), MODE is passed to fnx_run.
#   tools/ab.sh "rows ref/ref1.bin 5 60" 3 - FNX_ROUTE_IN_HC=0 FNX_ROUTE_IN_HC=0,FNX_SHARED_IN_MOE=0
mode=$1
rounds=$2
shift 2
. "$(dirname "$0")/labenv.sh"
lab_tmp
r=0 i=0
while [ $r -lt "$rounds" ]; do
    for v in "$@"; do
        envs=""
        bin=bin/fnx_run
        if [ "$v" != "-" ]; then
            for kv in $(echo "$v" | tr ',' ' '); do
                case "$kv" in
                    bin=*) bin=${kv#bin=} ;;   # another build's fnx_run (run from fnx/: the same ref/)
                    *) envs="$envs -E $kv" ;;
                esac
            done
        fi
        i=$((i + 1))
        # shellcheck disable=SC2086
        if run "ab-$i" $envs -- "$bin" "$G" $mode; then echo "$v | $(tail -1 "$O/ab-$i.log")"; else echo "$v | FAILED"; fi
    done
    r=$((r + 1))
done
finish
