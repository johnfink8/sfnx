#!/bin/bash
# A short real pi session against the server: a copy of bench/pi-task (a stock ledger whose sell() takes the newest
# lot instead of the oldest), the task to fix it, write tests and run them, pi 1.0 in print mode with its defaults
# (thinking on, the server's sampling). Then the engine's line for each of pi's requests and their sums: tokens,
# decode time, tokens a round, ms a round. Sampled, so runs differ: compare sums over a few.
#   serve/pi_session.sh [BASE_URL] [ENGINE_LOG]      (default: prod, http://127.0.0.1:8080/v1, serve/sfnx.log)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F=$(dirname "$HERE")
URL=${1:-http://127.0.0.1:8080/v1}
L=${2:-$HERE/sfnx.log}
curl -sf -m 5 "$URL/models" > /dev/null || { echo "no server at $URL"; exit 1; }
W=$(mktemp -d)
cp -r "$F/bench/pi-task/." "$W/"
TASK="inventory.py's Ledger.sell is meant to sell from the oldest lots first (FIFO), but it does not. Fix it. Then write \
test_inventory.py with pytest tests for receive (including its errors), FIFO sells across several lots with the cost \
of goods sold, selling more than is on hand, and report. Run them with $F/.venv/bin/python -m pytest -q and make them pass."
from=$(stat -c %s "$L")
gpu=$(mktemp)
nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > "$gpu" &
smi=$!
t0=$(date +%s.%N)
(cd "$W" && timeout 900 pi -p --provider sfnx --model qwen3.8-flash-next-unsloth-ud-iq4_xs "$TASK" > "$W/.pi-out.txt" 2>&1)
rc=$?
t1=$(date +%s.%N)
kill $smi
echo "pi exited $rc after $(echo "$t1 - $t0" | bc) s; tests now: $(cd "$W" && "$F/.venv/bin/python" -m pytest -q 2>&1 | tail -1)"
tail -c +$((from + 1)) "$L" | grep -E "^sfnx: slot [0-9]+: prompt" | sed 's/^sfnx: //' > "$W/.lines"
cat "$W/.lines"
awk '{
    for (i = 1; i <= NF; ++i) {
        if ($i == "generated") { g += $(i - 1); ms += substr($(i + 2), 1); }
        if ($i == "rounds") r += $(i - 1);
    }
} END { if (g) printf "%d requests: %d tokens in %.1f s of decode: %.1f tok/s, %.2f tokens a round, %.2f ms a round\n", NR, g, ms / 1000, g * 1000 / ms, g / r, ms / r }' "$W/.lines"
awk -F', ' '{u[$1] += $2; w[$1] += $3; n[$1]++} END {s = "GPUs over the session:"; for (c = 0; c < 3; c++) if (n[c]) s = s sprintf(" %d: %.0f%% %.0f W", c, u[c] / n[c], w[c] / n[c]); print s}' "$gpu"
rm -f "$gpu"
echo "(work dir: $W)"
