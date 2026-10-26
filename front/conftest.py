"""pytest: the vendored helpers write under the user's config and state folders (lifeline's PID records, saved
settings); every test session gets its own."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="sfnx-pytest-home-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home
os.environ["XDG_STATE_HOME"] = _home
