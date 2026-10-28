"""make install's check of the config the unit will run (front.server.check_config); exits non-zero naming the problem.
    python -m front.check_config CONFIG.json"""
import sys

from front.server import check_config

if len(sys.argv) != 2:
    raise SystemExit("usage: python -m front.check_config CONFIG.json")
check_config(sys.argv[1])
print(f"[sfnx] {sys.argv[1]}: the front end takes it (the engine checks the context against the model when it starts)")
