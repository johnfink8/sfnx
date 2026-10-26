#!/bin/sh
# Run an fnx tool with RLIMIT_MEMLOCK raised (the loader locks the token and PLE tables in RAM), as the calling user,
# in a transient systemd service with its output piped here:  ./run.sh bin/fnx_run MODEL gen REF N
exec sudo systemd-run --quiet --wait --pipe --collect -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" \
    -p WorkingDirectory="$(pwd)" -- "$@"
