"""The number of steps in an .lgr file.

    lgr_steps.py FILE.lgr"""
import sys

from refio import load_lgr

print(len(load_lgr(sys.argv[1])[1]))
