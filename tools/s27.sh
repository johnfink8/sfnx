#!/bin/sh
# runs a program of this checkout with the memlock limit the loader needs (the embedding table locked in RAM)
exec sudo -n systemd-run --quiet --pipe --wait --collect -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" \
    -p WorkingDirectory="$(pwd)" "$@"
