#!/bin/bash
# One request at a time through the front end: a code edit (ref/edit1.bin, its reply ends at 636 tokens) and new code
# (ref/code1.bin, 1,024 tokens), temperature 0, and prose reasoning in the style of thinking (ref/prose2.bin, 1,024
# tokens) with the server's own sampling (prod: temperature 0.6, top_p 0.95, presence 1.0) as real thinking is
# decoded; thinking off, three runs each (prose five: it is sampled) with a random tag in front, so nothing is reused;
# then the engine's own line for each request (tokens a round, ms a round) and the GPUs' mean use while they ran.
#   serve/solo_decode.sh [BASE_URL] [ENGINE_LOG]      (default: prod, http://127.0.0.1:8080/v1, serve/sfnx.log)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F=$(dirname "$HERE")
PY=$F/.venv/bin/python
URL=${1:-http://127.0.0.1:8080/v1}
L=${2:-$HERE/sfnx.log}
G=$F/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
curl -sf -m 5 "$URL/models" > /dev/null || { echo "no server at $URL"; exit 1; }
stream() {   # prompt tokens [greedy|served]
    PYTHONPATH=$F/llama.cpp/gguf-py $PY "$F/tools/strata_stream.py" "$G" "$F/ref/$1.bin" "$URL/chat/completions" "$2" "${3:-greedy}"
}
stream edit1 64 > /dev/null   # warm-up: graphs, the first request's costs
from=$(stat -c %s "$L")
gpu=$(mktemp)
nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > "$gpu" &
smi=$!
for p in edit1 code1; do
    for i in 1 2 3; do
        printf '%s run %s: ' $p $i
        stream $p 1024 | tail -1
    done
done
for i in 1 2 3 4 5; do
    printf 'prose2 run %s: ' $i
    stream prose2 1024 served | tail -1
done
kill $smi
tail -c +$((from + 1)) "$L" | grep -E "^sfnx: slot [0-9]+: prompt" | sed 's/^sfnx: //'
awk -F', ' '{u[$1] += $2; w[$1] += $3; n[$1]++} END {s = "GPUs:"; for (c = 0; c < 3; c++) if (n[c]) s = s sprintf(" %d: %.0f%% %.0f W", c, u[c] / n[c], w[c] / n[c]); print s}' "$gpu"
rm -f "$gpu"
