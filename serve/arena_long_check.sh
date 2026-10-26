#!/bin/sh
# Six long conversations at once on the served arena config (serve/sfnx.json, the test unit on port 8095):
# serve/arena_test.py (6 x 150k words), the GPUs sampled, then the engine's arena / parking / pause lines; stopped at
# the end.
#   serve/arena_long_check.sh [N WORDS GEN]
HERE=$(cd "$(dirname "$0")" && pwd)
URL=http://127.0.0.1:8095/v1
"$HERE/run.sh" "$HERE/sfnx.json" > /dev/null 2>&1
for _ in $(seq 120); do curl -sf -m 2 $URL/models > /dev/null && break; sleep 5; done
curl -sf -m 2 $URL/models > /dev/null || { echo "sfnx did not come up"; journalctl -u sfnx-test -n 20 --no-pager; exit 1; }
from=$(stat -c %s "$HERE/sfnx.log")
grep -E "arena: the K/V|sfnx: loaded" "$HERE/sfnx.log" | tail -2
nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 1000 > /tmp/arena-long.gpu &
smi=$!
python3 "$HERE/arena_test.py" $URL "$@"
kill $smi
echo "GPUs over the run: $(awk -F', ' '{u+=$2; p+=$3; n++} END {printf "util %.0f%% power %.0f W", u/n, p/n}' /tmp/arena-long.gpu)"
echo "== the engine's events"
tail -c +$((from + 1)) "$HERE/sfnx.log" | grep -E "arena|paused|resumed|pool full|pool short|waits for a page|parked|idle" | cut -c1-170 | tail -25
sudo systemctl stop sfnx-test
