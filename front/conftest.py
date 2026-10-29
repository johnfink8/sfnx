"""pytest: the vendored helpers write under the user's config and state folders (lifeline's PID records, saved
settings); every test session gets its own."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="sfnx-pytest-home-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home
os.environ["XDG_STATE_HOME"] = _home


def pytest_sessionfinish(session, exitstatus):
    """Every skip here is for a missing package or tool (anthropic, node, Pillow, jsonschema). CI has them all and sets
    FNX_NO_SKIPS=1, so a skip there is a test that never ran: it fails the run."""
    if os.environ.get("FNX_NO_SKIPS") != "1" or exitstatus != 0:
        return
    stats = session.config.pluginmanager.getplugin("terminalreporter").stats
    skipped = stats.get("skipped", [])
    if skipped:
        for r in skipped:
            print(f"FNX_NO_SKIPS: skipped {r.nodeid}: {r.longrepr[-1] if isinstance(r.longrepr, tuple) else r.longrepr}")
        session.exitstatus = 1
