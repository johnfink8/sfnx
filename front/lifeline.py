"""front/lifeline.py - a process the server starts ends when the server ends, however that ends.

The graceful path is the server's own (QUIT to the engine, drain, exit). This module is the guarantee behind it
for the cases where nothing gets to run: kill -9, a crash. ``PR_SET_PDEATHSIG`` = SIGKILL is set in the child before
it execs. The kernel ties the signal to the THREAD that forked the child, not to the process, so every start goes
through one long-lived "anchor" thread; a request thread that starts an engine and then finishes would otherwise take
the engine with it.

``popen`` is ``subprocess.Popen`` plus the guarantee.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import ctypes
import os
import queue
import signal
import subprocess
import threading
import time

_REAL_POPEN = subprocess.Popen                  # tests replace subprocess.Popen; isinstance needs the class
_libc = ctypes.CDLL(None, use_errno=True)
_libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
PR_SET_PDEATHSIG = 1


class _Anchor:
    """One thread that lives as long as the process and performs every Popen, so PDEATHSIG's "parent" is it."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="lifeline-anchor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, future = self._jobs.get()
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)

    def call(self, fn):
        if threading.current_thread() is self._thread:
            return fn()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, future))
        return future.result()


_anchor: _Anchor | None = None
_anchor_lock = threading.Lock()


def _the_anchor() -> _Anchor:
    global _anchor
    with _anchor_lock:
        if _anchor is None:
            _anchor = _Anchor()
        return _anchor


def _die_with_parent(parent: int, then=None):
    """The preexec function of a child: SIGKILL when the starting thread ends; and if it ended already, now."""
    def run() -> None:
        _libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        if os.getppid() != parent:                 # the parent died between fork and prctl
            os.kill(os.getpid(), signal.SIGKILL)
        if then is not None:
            then()
    return run


_children: set = set()
_children_lock = threading.Lock()


def end_children(grace: float = 3.0) -> None:
    """Whatever this process started and has not ended yet: terminate, then kill. Runs at interpreter exit, so an
    exception or a signal that unwinds main() before its own cleanup still leaves nothing behind (the engine and
    the others have already been closed on every normal path, so this finds nothing then)."""
    with _children_lock:
        alive = [c for c in _children if c.poll() is None]
    for child in alive:
        try:
            child.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + grace
    for child in alive:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError):
            try:
                child.kill()
            except OSError:
                pass


atexit.register(end_children)


def _after_exit(child: subprocess.Popen) -> None:
    child.wait()
    with _children_lock:
        _children.discard(child)


ISOLATED_ROLES = ("engine", "vision")      # speak a stdin protocol with us: the server alone decides when they stop


def _own_group(then=None):
    def run() -> None:
        os.setpgid(0, 0)
        if then is not None:
            then()
    return run


def popen(args, role: str = "engine", **kw) -> subprocess.Popen:
    """``subprocess.Popen(args, **kw)`` for a process that must not outlive us.

    An engine or image encoder is also taken out of the terminal's reach (its own process group): Ctrl+C or a closing
    terminal go to the server only, which then ends the engine the clean way (STOP, QUIT, wait) instead of the engine
    dying under a request that could have been finished or cancelled. What ties the engine's life to ours is
    PDEATHSIG, not the shared terminal."""
    if role in ISOLATED_ROLES and "start_new_session" not in kw:
        kw["preexec_fn"] = _own_group(kw.pop("preexec_fn", None))
    parent, then = os.getpid(), kw.pop("preexec_fn", None)
    kw["preexec_fn"] = _die_with_parent(parent, then)
    child = _the_anchor().call(lambda: subprocess.Popen(args, **kw))
    if isinstance(child, _REAL_POPEN):
        with _children_lock:
            _children.add(child)
        threading.Thread(target=_after_exit, args=(child,), name="lifeline-reaper", daemon=True).start()
    return child


def install_stop_handlers(request_stop) -> None:
    """Make every ordinary way a server is asked to end take the clean path (`request_stop()`): SIGTERM (systemd,
    `kill`) and SIGHUP (the terminal closed, the ssh connection dropped; not when it was ignored at start, which is
    what `nohup` does on purpose). SIGINT keeps Python's own handler (Ctrl+C, and a second Ctrl+C to stop waiting).
    Must be called from the main thread."""
    def handler(signum, frame):
        request_stop()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if sig == signal.SIGHUP and signal.getsignal(sig) == signal.SIG_IGN:
            continue
        try:
            signal.signal(sig, handler)
        except ValueError:                                  # not the main thread
            pass
