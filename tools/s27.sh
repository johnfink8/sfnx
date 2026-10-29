#!/bin/sh
# runs a program of this checkout with the memlock limit the loader needs (the embedding table locked in RAM), and the
# engine's settings (FNX_*, S27_*, CUDA_*) of this shell, which systemd-run does not pass on its own
set -- $(env | grep -E '^(FNX|S27|CUDA)_' | sed 's/^/--setenv=/') "$@"
exec sudo -n systemd-run --quiet --pipe --wait --collect -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" \
    -p WorkingDirectory="$(pwd)" "$@"
