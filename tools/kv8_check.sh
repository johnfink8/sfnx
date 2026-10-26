#!/bin/sh
# The int8 K/V cache's drift: sFNX with f16 and with int8 K/V (FNX_KV8=1), teacher-forced on llama.cpp's tokens, each
# against llama.cpp CUDA (f16 cache); with LLAMA_Q8=1 also llama.cpp's own q8_0 cache against its f16 one. The numbers
# are measured, not judged (fnx_run exact holds prod's int8 K/V to the bar); a run that fails is an error, and its
# comparison is not made. Each result replaces its ref/lgr file only once its run succeeded.
#   tools/kv8_check.sh [PROMPT ...]       (default: code1 edit1 long64; results in ref/lgr/fnx_{kv16f,kv8}_NAME.lgr)
. "$(dirname "$0")/labenv.sh"
need_py
lab_tmp
[ $# -gt 0 ] || set -- code1 edit1 long64
for n in "$@"; do
    ref=ref/lgr/llama_cuda_$n.lgr
    [ -s "$ref" ] || { failed "$n: no reference $ref"; continue; }
    CTX=$((($(stat -c %s "ref/$n.bin") / 4 + 4096 + 191) / 192 * 192))
    [ "$CTX" -lt 16384 ] && CTX=16384
    for kv in 0 1; do
        name=fnx_$([ $kv = 1 ] && echo kv8 || echo kv16f)_$n
        CTX=$CTX ENVS="-E FNX_KV8=$kv" tools/pf_run.sh "ref/$n.bin" "$ref" "$O/$name.lgr" > "$O/$name.log" 2>&1
        rc=$?
        if [ $rc != 0 ]; then
            failed "$n kv8=$kv: fnx_run exit $rc (log $O/$name.log)"
            tail -3 "$O/$name.log"
            continue
        fi
        mv -f "$O/$name.lgr" "ref/lgr/$name.lgr" || { failed "$n kv8=$kv: cannot move the result into ref/lgr"; continue; }
        echo "== $n, sFNX $([ $kv = 1 ] && echo int8 || echo f16) K/V vs llama.cpp CUDA f16"
        "$PY" ref/lgrcmp.py "$ref" "ref/lgr/$name.lgr" > "$O/$name.cmp" 2>&1 || failed "$n kv8=$kv: not compared ($(tail -1 "$O/$name.cmp"))"
        head -n 2 "$O/$name.cmp"
    done
    if [ "${LLAMA_Q8:-0}" = 1 ]; then
        name=llama_cuda_q8_$n
        N=$("$PY" ref/lgr_steps.py "$ref") || { failed "$n: cannot count $ref's steps"; continue; }
        run "$name" -E FNX_LOGREF_KV=q8_0 -- ref/fnx_logref "$G" "ref/$n.bin" "$N" "$O/$name.lgr" "$ref" || continue
        mv -f "$O/$name.lgr" "ref/lgr/$name.lgr" || { failed "$n q8_0: cannot move the result into ref/lgr"; continue; }
        echo "== $n, llama.cpp CUDA q8_0 K/V vs its f16"
        "$PY" ref/lgrcmp.py "$ref" "ref/lgr/$name.lgr" > "$O/$name.cmp" 2>&1 || failed "$n q8_0: not compared ($(tail -1 "$O/$name.cmp"))"
        head -n 2 "$O/$name.cmp"
    fi
done
finish
