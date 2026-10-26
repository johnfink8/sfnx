"""front/lifeline.py: a process the server starts ends when the server ends, however that ends.

    python -m unittest front.test_lifeline

Each case runs a small "server" process (a fresh interpreter) that starts a sleeping child through
`lifeline.popen`, then ends that server the hard way (kill -9), and checks that the child is gone (PR_SET_PDEATHSIG).
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVER = textwrap.dedent("""
    import os, sys, time, threading, subprocess
    sys.path.insert(0, {root!r})
    from front import lifeline
    mode = sys.argv[1]
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    if mode == "thread":                       # started on a request thread that ends at once
        box = []
        t = threading.Thread(target=lambda: box.append(lifeline.popen(sleeper, "engine")))
        t.start(); t.join()
        child = box[0]
    else:
        child = lifeline.popen(sleeper, "engine")
    print(__import__("json").dumps({{"child": child.pid, "server": os.getpid()}}), flush=True)
    if mode == "exit":                         # ends normally, the child still running
        sys.exit(0)
    time.sleep(600)
""")


def alive(pid: int) -> bool:
    """The process runs (a zombie, ended but not yet reaped by its parent, does not count)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


class Lifeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lifeline-")
        self.addCleanup(self.temp.cleanup)
        self.env = {**os.environ, "PYTHONPATH": str(ROOT)}
        self.script = Path(self.temp.name) / "server.py"
        self.script.write_text(SERVER.format(root=str(ROOT)), encoding="utf-8")
        self.spawned = []
        self.addCleanup(self._reap)

    def _reap(self):
        for pid in self.spawned:
            if alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

    def launch(self, mode):
        proc = subprocess.Popen([sys.executable, str(self.script), mode], stdout=subprocess.PIPE, text=True,
                                env=self.env)
        info = json.loads(proc.stdout.readline())
        self.spawned.append(info["child"])
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        return proc, info

    def wait_gone(self, pid, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not alive(pid):
                return True
            time.sleep(0.1)
        return not alive(pid)

    def hard_kill(self, proc):
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

    def test_a_child_ends_when_its_server_is_killed(self):
        proc, info = self.launch("sleep")
        self.assertTrue(alive(info["child"]))
        self.hard_kill(proc)
        self.assertTrue(self.wait_gone(info["child"]), "the child outlived a killed server")

    def test_a_child_started_on_a_request_thread_outlives_that_thread_but_not_the_server(self):
        proc, info = self.launch("thread")
        time.sleep(1.0)
        self.assertTrue(alive(info["child"]), "the request thread ended and took the child with it")
        self.hard_kill(proc)
        self.assertTrue(self.wait_gone(info["child"]))

    def test_a_server_that_exits_normally_leaves_nothing_behind(self):
        proc, info = self.launch("exit")
        proc.wait(timeout=15)
        self.assertTrue(self.wait_gone(info["child"]), "end_children at exit must stop what is still running")


STUBBORN = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('READY 4096 stop', flush=True)\n"
    "time.sleep(600)\n"          # never reads QUIT, never answers STOP, ignores SIGTERM
)


class HardFallback(unittest.TestCase):
    """The clean stop has a deadline: an engine that ignores STOP, QUIT and SIGTERM is killed, and close() returns."""

    def test_an_engine_that_ignores_everything_is_killed(self):
        from unittest import mock
        from front import server
        engine = server.StrataEngine("missing-executable", [], lazy=True)
        engine.proc = subprocess.Popen([sys.executable, "-c", STUBBORN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True)
        engine.proc.stdout.readline()
        victim = engine.proc
        t0 = time.monotonic()
        with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", 1), mock.patch.object(server, "ENGINE_TERM_WAIT_S", 1), \
                mock.patch.object(server, "ENGINE_KILL_WAIT_S", 10):
            engine.close()
        self.assertLess(time.monotonic() - t0, 12)
        self.assertIsNotNone(victim.poll(), "the engine is still running after close()")
        self.assertIsNone(engine.proc)
        self.assertEqual(victim.returncode, -signal.SIGKILL, "it ignored SIGTERM, so only the kill ended it")


if __name__ == "__main__":
    unittest.main()
