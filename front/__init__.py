"""front - sFNX's front end: the API (text in, token ids to the engine, text deltas out) and the web app. Taken from
Strata's serve/ (MIT, LICENSE here); the helper modules it imports from Strata are in vendor/, on the path from here."""
import sys
from pathlib import Path

_vendor = str(Path(__file__).resolve().parent / "vendor")
if _vendor not in sys.path:
    sys.path.insert(0, _vendor)
