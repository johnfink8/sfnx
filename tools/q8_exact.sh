#!/bin/sh
# Q8_0 exactness: llama.cpp's CPU backend (the Q8 model does not fit the cards for llama.cpp) generates N greedy steps
# on the Q8_0 GGUF (its text and logits are the reference), then sFNX's Q8 path (int8 K/V, 52 GiB of experts in RAM,
# the prompt through the prompt path) is teacher-forced on those tokens, compared by ref/lgrcmp.py and held to the Q4
# bar as fnx_run exact holds it (README: llama.cpp's CPU backend against its CUDA one, KL mean x1.25, top-1 less 1.5
# points); prompts other than code1 and edit1 have no bar and are only measured. A reference or result replaces its
# ref/lgr file only once its run succeeded. Exits 1 when a run fails or a result is outside the bar.
#   tools/q8_exact.sh [N] [PROMPT ...]     (default 300 steps, code1 edit1; refs in ref/lgr/llama_cpu_q8w_NAME.lgr)
N=${1:-300}
[ $# -gt 0 ] && shift   # (dash exits on a shift past the arguments)
[ $# -gt 0 ] || set -- code1 edit1
. "$(dirname "$0")/labenv.sh"
need_py
lab_tmp
for n in "$@"; do
    ref=ref/lgr/llama_cpu_q8w_$n.lgr out=ref/lgr/fnx_q8w_$n.lgr
    if [ ! -s "$ref" ]; then
        # llama.cpp's CPU backend on 176 GiB of weights: 300 steps take over an hour
        UNIT_TIMEOUT=7200 run "ref-$n" -E FNX_LOGREF_GPU=0 -- ref/fnx_logref "$Q8" "ref/$n.bin" "$N" "$O/ref-$n.lgr" || continue
        mv -f "$O/ref-$n.lgr" "$ref" || { failed "$n: cannot move the reference into ref/lgr"; continue; }
    fi
    CTX=16384 tools/pf_q8.sh "ref/$n.bin" "$ref" "$O/fnx-$n.lgr" > "$O/fnx-$n.log" 2>&1
    rc=$?
    if [ $rc != 0 ]; then
        failed "$n: fnx_run exit $rc (log $O/fnx-$n.log)"
        tail -3 "$O/fnx-$n.log"
        continue
    fi
    mv -f "$O/fnx-$n.lgr" "$out" || { failed "$n: cannot move the result into ref/lgr"; continue; }
    case "$n" in
    code1) bar="6.06e-4 96.83" ;;
    edit1) bar="5.15e-6 98.5" ;;
    *) bar="" ;;
    esac
    echo "== $n, $N steps: sFNX Q8_0 vs llama.cpp CPU Q8_0${bar:+ (bar: KL mean, top-1 % = $bar)}"
    # shellcheck disable=SC2086
    "$PY" ref/lgrcmp.py "$ref" "$out" $bar > "$O/cmp-$n.txt" 2>&1 || failed "$n: $(tail -1 "$O/cmp-$n.txt")"
    head -n 2 "$O/cmp-$n.txt"
    [ -n "$bar" ] && tail -1 "$O/cmp-$n.txt"
done
finish
