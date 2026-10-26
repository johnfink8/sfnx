#!/bin/sh
# sFNX's front end (front/) driving the engine, as a transient systemd unit with the memlock
# limit raised (sFNX locks its host-side tables). Stops a previous run of the unit first.
#   serve/run.sh [CONFIG.json]            (default serve/sfnx.json; port from the config)
# Env: ENVS="-E VAR=value ..." for the unit (the engine inherits it).
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
cfg=$(realpath "${1:-$here/sfnx.json}")
sudo systemctl stop sfnx-test 2>/dev/null || true
sudo systemctl reset-failed sfnx-test 2>/dev/null || true
# shellcheck disable=SC2086
exec sudo systemd-run --unit=sfnx-test $ENVS -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" \
    -p WorkingDirectory="$root" -E HOME="$HOME" -E PATH="$PATH" \
    -- "$root/.venv/bin/python" -m front.server --engine sfnx --config "$cfg"
