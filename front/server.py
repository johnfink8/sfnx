"""front/server.py - sFNX's OpenAI, Anthropic and Responses endpoints over an engine that maps token ids to tokens.

    python -m front.server --engine mock --port 8095                       (a scripted engine, for clients and tests)
    python -m front.server --engine sfnx --config serve/sfnx.json          (sFNX, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), POST /v1/messages/count_tokens, POST /v1/responses, GET /v1/models, GET /models, GET /props, GET /slots,
GET /health, GET /v1/status, GET /metrics, and the web app.  With --batch N in the engine's args up to N requests
decode at once in its batch slots; more wait their turn.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `bin/sfnx-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
A request whose prompt alone does not fit the engine's context is REJECTED with 400, never truncated.  When the prompt
fits but prompt plus max tokens does not (Claude Code asks for 32000, the Harness for 16384), max tokens is shortened to
the room left (less CTX_SLACK) and the thinking budget follows it; `--no-fit-max-tokens` / `"fit_max_tokens": false`
brings back the strict 400 of llama.cpp.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `bin/sfnx --serve` process resident (weights and expert arena load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import collections
import base64
import hashlib
import hmac
import codecs
import json
import math
import os
import queue
import re
import select
import signal
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "front" / "vendor"))
sys.path.insert(0, str(ROOT))   # run as a script as well as a module
from front.frontend import (LATE_NOTE, THINK_END, ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            forced_call, images_of, literal_tags, mark_think_literals, openai_to_messages,
                            tool_choice_of, unmark_think_literals)


from front.bounded_text import BoundedText  # noqa: E402
from front.engine_limits import (CTX_SLACK, EOS_IDS, PARALLEL_MAX, PENALTY_LAST_N_DEFAULT,  # noqa: E402
                                 PENALTY_LAST_N_MAX, SEED_MAX, TOP_K_MAX)


from front import lifeline  # noqa: E402
from front.lifeline import popen  # noqa: E402
from front.structured import StructuredOutputError, prepare_format, validated_json  # noqa: E402
from front import responses as responses_api  # noqa: E402
from front.responses import ResponsesError, error_body as responses_error_body  # noqa: E402


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


# How long close() gives the engine at each step of ending it (seconds). Module values so that a test can shorten them.
ENGINE_QUIT_WAIT_S, ENGINE_TERM_WAIT_S, ENGINE_KILL_WAIT_S = 20, 5, 20


def listen_problem(host: str, port: int, e: OSError) -> str:
    """#769: why the server cannot listen.  Only an address that is taken says "already in use"; anything else (an
    address this machine does not have, no permission) says what the OS said."""
    import errno
    code = e.errno
    if code == errno.EADDRINUSE:
        return (f"port {port} is already in use - is sfnx.service (or another server) already running? "
                f"Stop it, or start this one with a different --port")
    what = e.strerror or str(e)
    if code == errno.EACCES:
        hint = "no permission to listen there: start this one with a different --port"
    elif code == errno.EADDRNOTAVAIL:
        hint = f"{host} is not an address of this machine: check \"host\" in the config or --host"
    else:
        hint = "check --host and --port"
    return f"cannot listen on {host}:{port}: {what} [{code}]. {hint}"


def combined_embeddings_path(vision_dir: Path, nbytes: int) -> Path:
    """#874: where one request's combined image-embeddings file goes.  Every request with images rewrites it (it is
    deleted after the request), which on a disk is steady heavy writing: it goes to /dev/shm when that has room (the
    engine reads it back at once), else to the encoder's own directory."""
    name = f"req-{uuid.uuid4().hex[:12]}.sve"
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free > 2 * nbytes + (16 << 20):
                return shm / f"sfnx-vision-{os.getpid()}-{name}"
        except OSError:
            pass
    return Path(vision_dir) / name


def write_temporary(path: Path, parts: list) -> None:
    """#874: write `parts` (files) one after the other into `path`, readable by this user only."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        for part in parts:
            f.write(Path(part).read_bytes())


VISION_START = "<|vision_start|>"
# #606: a reply that repeats one token this many times in a row is ended there ("length"): a model in a loop, or a
# broken state that answers one token forever (an issue saw 36,689 tokens of "!"). The config's "repeat_stop_tokens"
# sets it; 0 turns it off.
REPEAT_STOP_TOKENS = 256
ANSWER_ROOM_MIN_CAP = 1024    # 0.1.45: from this max_tokens on, thinking without a budget is closed early enough to leave room to answer
ANSWER_RESERVE_MIN = 512      # #984: the tokens kept for the answer when a thinking budget is cut down to fit max_tokens
# #123: what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
# #1053 (opt-in "reasoning_close_retry": true): a reply that ends on its stop token still inside <think>, with no answer
# and no call, is continued once with the thinking closed (the same way the budget's wrap-up closes it)
REASONING_CLOSE = "\n</think>\n\n"
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# A stop token written inside an open code span or fence is the model quoting the marker (`<|im_end|>`), not ending
# its turn: it is replaced by the marker's text and the reply goes on, at most this many times per reply.
# STRATA_LITERAL_STOP=0 turns it off.
LITERAL_STOPS_PER_REPLY = 4


def in_open_code(text: str) -> bool:
    """True when `text` ends inside a Markdown code fence or an inline code span on its last line."""
    if text.count("```") % 2:
        return True
    line = text.rsplit("\n", 1)[-1].replace("```", "")
    return line.count("`") % 2 == 1


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often (#1619)
KEEPALIVE_S = 10.0          # a held-back tool argument leaves the stream quiet: ping at least this often
# Anthropic clients add /v1/messages to their base URL themselves. A base URL that already ends in /v1 (copied from an OpenAI-style
# setup) makes them post /v1/v1/messages; these two paths mean the same as the real ones. Nothing else is aliased.
V1_V1_ALIASES = {"/v1/v1/messages": "/v1/messages", "/v1/v1/messages/count_tokens": "/v1/messages/count_tokens"}
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# #481: a running request whose engine prints nothing (no T, PP or any other line) for this long has lost step with the
# server (the engine's main thread waits, untimed, for its next command): the engine is ended and the request fails;
# the next request starts it again.  The config's "engine_silence_s" sets it (0: wait forever, as before).
ENGINE_SILENCE_S = 300.0
# ... except while a prompt is read: a PP line comes once per chunk (up to 32768 tokens with --prefill auto, issue
# #282), and the slowest PCs read ~100 tok/s, so a first chunk can take minutes before the first line.  Until the first
# PP the wait adds the chunk's tokens at PP_FLOOR_TOK_S; after one, a chunk may take PP_SLACK x the last one's time.
PP_CHUNK_MAX = 32768
PP_FLOOR_TOK_S = 50.0
PP_SLACK = 3.0
# Reads that had no timeout (#1317).  A blocking read of the image encoder's pipe or of the engine's READY line that
# never returned held the request turn for good: the engine then sat idle in its command loop, so no watchdog could
# fire and every later request queued behind it (the "lost step" class of #481).  Generous on purpose: a cold model
# load, a slow disk or a slow CPU encode is not a hang.  Only a process that says nothing at all for this long is
# ended; the next request starts it again.
# STRATA_ENGINE_READY_S / STRATA_VISION_READY_S / STRATA_VISION_ENCODE_S override them (0 = wait for ever, as before).


def _timeout_env(name: str, default: float) -> float | None:
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return None if v <= 0 else v


# #1317 part 2: the quick watchdog for a FROZEN engine.  engine_silence_s (above) is generous on purpose: a slow PC reads a
# prompt chunk for minutes without a line.  But an engine that prints nothing for ENGINE_STALL_S and also uses no CPU time and
# moves no bytes of I/O in that time is not slow, it is idle or frozen (a deadlock in a CUDA call, a driver stall, a lost
# step), and waiting longer cannot help: it is ended, and the next request starts it again.  An engine that is silent but
# still working is never ended by this, only reported once (the recommendation: engine_silence_s, or this variable).
# STRATA_ENGINE_STALL_S overrides it (0 = off).
ENGINE_STALL_S = _timeout_env("STRATA_ENGINE_STALL_S", 90.0)
STALL_CPU_EPS_S = 0.5       # CPU seconds the engine may use in the whole stall window and still count as frozen
STALL_IO_EPS_B = 1 << 20    # bytes read or written in that window


def engine_frozen(base: tuple[float, int], now: tuple[float, int]) -> bool:
    """#1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them."""
    return (now[0] - base[0]) <= STALL_CPU_EPS_S and (now[1] - base[1]) <= STALL_IO_EPS_B


VISION_READY_S = _timeout_env("STRATA_VISION_READY_S", 300.0)
VISION_ENCODE_S = _timeout_env("STRATA_VISION_ENCODE_S", 300.0)
ENGINE_READY_S = _timeout_env("STRATA_ENGINE_READY_S", 900.0)


# ------------------------------------------------------------------------------------------------ engines
class Prompt(list):
    """A prompt's token ids, and `keep`: the position where its closing run of LATE_NOTE messages starts (None
    without one), which the engine checkpoints.  The prompt of a later pass (this one plus generated ids) is a
    plain list: the notes were already read."""

    def __init__(self, ids, keep):
        super().__init__(ids)
        self.keep = keep


class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0,
                 cycle: bool = False):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        # `cycle` picks the script by request position instead of advancing to the last one and staying
        # there. A harness adds requests the script author did not plan for -- a session title, a re-ask
        # after a tool result -- and with the default behaviour one extra request shifts every later
        # reply, so a run looks like a model that simply chose differently.
        self.cycle = bool(cycle)
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            if self.cycle:
                self.script = self.scripts[self.turns % len(self.scripts)]
            else:
                self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet (#344): no context size to plan a request with - a 503,
    not a 400 about the prompt."""


class ClientGone(Exception):
    """#1999: the client hung up while its request was being prepared (an image URL downloading, a picture encoding or
    waiting for its turn): nothing is left to answer."""


class QueueCancelled(RuntimeError):
    """The client left before its first serial engine turn."""


class EngineSilent(EngineDied):
    """#481: the engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


# the finish words the engine's DONE and BDONE lines carry (fnx::finish_name)
FINISHES = ("stop", "length", "cancel")


def _done_field(line: str, name: str, text: str, kind):
    """One field of a DONE / BDONE line as `kind` (int: digits only; float: a finite number >= 0; str: a FINISHES
    word); ValueError naming the field and the line otherwise."""
    if kind is int and text.isdigit():
        return int(text)
    if kind is float:
        try:
            value = float(text)
        except ValueError:
            value = -1.0
        if math.isfinite(value) and value >= 0:
            return value
    if kind is str and text in FINISHES:
        return text
    raise ValueError(f"the engine's {line.split()[0]} line has {name}={text!r}: {line.strip()[:200]!r}")


@dataclass(frozen=True)
class EngineDone:
    """bin/sfnx's end of a request (tools/sfnx.cpp): `DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>
    <drafts_accepted> <drafts_offered> <reused>`.  prompt_tokens: the prompt's tokens the lane holds, reused ones
    included (on a cancelled read as far as it got)."""
    generated: int
    prompt_tokens: int
    prompt_ms: float
    decode_ms: float
    finish: str
    drafts_accepted: int
    drafts_offered: int
    reused: int

    @classmethod
    def parse(cls, line: str) -> EngineDone:
        """ValueError, naming the field, for a line of another shape."""
        f = line.split()
        if not f or f[0] != "DONE" or len(f) != 9:
            raise ValueError(f"the engine's DONE line has {len(f) - 1} fields, not 8: {line.strip()[:200]!r}")
        return cls(*(_done_field(line, name, text, kind) for (name, kind), text in zip(
            (("generated", int), ("prompt_tokens", int), ("prompt_ms", float), ("decode_ms", float),
             ("finish", str), ("drafts_accepted", int), ("drafts_offered", int), ("reused", int)), f[1:])))

    def figures(self) -> dict:
        """The request's figures as StrataEngine.last holds them."""
        return asdict(self)


@dataclass(frozen=True)
class BatchDone:
    """bin/sfnx's end of a batch slot's decode (tools/sfnx.cpp): `BDONE <slot> <produced> <finish> <ms>
    <drafts_accepted> <drafts_offered> <rounds>` (rounds: the batched windows after its first token; the drafts 0
    with FNX_BATCH_DRAFTS=0)."""
    slot: int
    produced: int
    finish: str
    ms: float
    drafts_accepted: int
    drafts_offered: int
    rounds: int

    @classmethod
    def parse(cls, line: str) -> BatchDone:
        """ValueError, naming the field, for a line of another shape."""
        f = line.split()
        if not f or f[0] != "BDONE" or len(f) != 8:
            raise ValueError(f"the engine's BDONE line has {len(f) - 1} fields, not 7: {line.strip()[:200]!r}")
        return cls(*(_done_field(line, name, text, kind) for (name, kind), text in zip(
            (("slot", int), ("produced", int), ("finish", str), ("ms", float), ("drafts_accepted", int),
             ("drafts_offered", int), ("rounds", int)), f[1:])))


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill: the server keeps it (and says so) rather than
    reporting its GPU and RAM as given back."""


class GpuBusy(RuntimeError):
    """#1994: the engine's batch slots stayed refused: a 503 worth retrying.  `retry_after`: seconds for the
    Retry-After header."""

    def __init__(self, message="", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def start_log_tail(log: str | None, offset: int, n: int = 20) -> str:
    """#496: the last n lines this start wrote to the engine log, for the error when the engine ended before READY -
    whatever the failure, the engine's own reason is in them (people posted the traceback without the log).  "" when
    there is no log or nothing in it from this start."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # enough for 20 lines, never an earlier start's
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    lines = [x.rstrip() for x in lines if x.strip()][-n:]
    if not lines:
        return ""
    return "\nthe engine log's last lines:\n" + "\n".join("  " + x for x in lines)


def engine_exit_words(rc: int | None) -> str:
    """What a process's exit status says, in words: "killed by signal 9 (SIGKILL, Killed)" (a negative status) or
    "exit code N".  "" when there is no status."""
    if rc is None:
        return ""
    if rc < 0:
        try:
            sig = signal.Signals(-rc)
            return f"killed by signal {-rc} ({sig.name}, {signal.strsignal(sig) or sig.name})"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}"


def start_exit_note(rc: int | None) -> str:
    """The sentence the start error carries about how the engine process ended (the engine log is silent when the
    system ends it): "" without a status."""
    words = engine_exit_words(rc)
    return f"\nthe engine process ended: {words}" if words else ""


def narrate_start(done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts (~45 s with the model in the page cache, minutes from a cold disk), say so in the
    journal now and then, so a quiet start does not read as a hang."""
    t0 = time.time()
    print("[sfnx] starting the engine: reading the model's weights ...", flush=True)
    while not done.wait(heartbeat):
        print(f"[sfnx] still starting ({time.time() - t0:.0f} s) ...", flush=True)


class ConvCacheLog:
    """#596: the engine's conversation cache as its log tells it, read on from where it was last read, from the start
    of the engine's current run.  bin/sfnx (src/session.cpp) writes to stderr, which is the log:
      conversation cache: parked N tokens in T ms, parked=P bytes=B
      conversation cache: restored N tokens in T ms, parked=P bytes=B
      conversation cache: the first message (N tokens) parked on its own with C checkpoints inside, in T ms
                                                                                     (one more parked, no totals)
      conversation cache: evicted N tokens                                           (one fewer, no totals)
    The totals on the first two set the count; the other two move it by one until the next line with totals."""
    EVENT = re.compile(r"conversation cache: (parked|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)")
    ROOT = re.compile(r"conversation cache: the first message \((\d+) tokens\) parked on its own")
    EVICTED = re.compile(r"conversation cache: evicted (\d+) tokens")
    READ_MAX = 32 << 20                                 # at most this much of new lines per read: bin/sfnx logs a line
                                                        # per arena step, so a monitor opened after hours has MiBs to read

    def __init__(self):
        self.key, self.pos = None, 0
        self.reset()

    def reset(self):
        self.state = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None,
                      "last_tokens": None, "last_at": None}

    def poll(self, path, start) -> dict:
        """The state after the log's new lines; `start` is where the engine's current run began in it."""
        if not path or start is None:
            return dict(self.state)
        if self.key != (path, start):                   # another start of the engine: its cache starts empty
            self.key, self.pos = (path, start), start
            self.reset()
        try:
            size = os.path.getsize(path)
            if size < self.pos:                         # the log was emptied or replaced
                self.pos = 0
            if size - self.pos > self.READ_MAX:
                self.pos = size - self.READ_MAX
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return dict(self.state)
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        now = time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if "conversation cache:" not in line:
                continue
            m = self.EVENT.search(line)
            if m:
                st = self.state
                st["parked"], st["bytes"] = int(m.group(3)), int(m.group(4))
                st["parks" if m.group(1) == "parked" else "restores"] += 1
                st["last_event"], st["last_tokens"], st["last_at"] = m.group(1), int(m.group(2)), now
                continue
            m = self.ROOT.search(line)
            if m:
                st = self.state
                st["parked"] += 1
                st["parks"] += 1
                st["last_event"], st["last_tokens"], st["last_at"] = "parked", int(m.group(1)), now
                continue
            if self.EVICTED.search(line):
                self.state["parked"] = max(0, self.state["parked"] - 1)
                self.state["evictions"] += 1
        return dict(self.state)


class ReadMeter:
    """A prompt read's rate for the Monitor, from when its PP lines arrive: bin/sfnx's `PP <done> <total>` says how
    far the read got and nothing about time.  The first PP marks the start (the reused prefix and the first chunk are
    behind it), so the rate is known from the second chunk on and counts only tokens read since; decode windows that
    run between chunks are in it, as they are in the read's real speed."""

    MIN_SPAN_S = 0.2                                    # two PP lines closer than this give no meaningful rate

    def __init__(self):
        self.start()

    def start(self) -> None:
        """A new read was sent (GEN / BGEN)."""
        self.first = self.last = None

    def note(self, done: int, now: float | None = None) -> None:
        mark = (time.monotonic() if now is None else now, int(done))
        if self.first is None:
            self.first = mark
        self.last = mark

    def rate(self) -> float | None:
        first, last = self.first, self.last             # one read of each: another thread may note meanwhile
        if first is None or last is None or last[0] - first[0] < self.MIN_SPAN_S:
            return None
        return round((last[1] - first[1]) / (last[0] - first[0]), 1)


READ_RATE_MIN = 1000     # new prompt tokens a read needs for its time to be a reading rate (not a turn's setup)


def last_read(hist: list) -> dict | None:
    """The Monitor's Prefill when nothing is read: the newest finished request that read at least READ_RATE_MIN new
    tokens (agents' turns mostly reuse all but a few hundred, so the newest request is rarely one)."""
    for r in reversed(hist):
        read = (r.get("prompt_tokens") or 0) - (r.get("reused") or 0)
        if read >= READ_RATE_MIN and (r.get("prompt_ms") or 0) > 0:
            return {"tokens": read, "ms": r["prompt_ms"], "tok_s": round(read / (r["prompt_ms"] / 1000), 1),
                    "time": r.get("time")}
    return None


def conversation_cache_view(info: dict, hist: list, totals: dict, parked: dict) -> dict:
    """#596: the Monitor's Conversation cache card: the parked conversations (the engine's opt-in
    --conversation-cache-mib: budget, slots, what its log says) and how much of the prompts the cache gave back."""
    mib = info.get("conversation_cache_mib")
    last = hist[-1] if hist else None
    return {"enabled": isinstance(mib, int) and mib > 0, "budget_mib": mib if isinstance(mib, int) else None,
            "slots": info.get("conversation_cache_slots"), **parked,
            "requests": len(hist), "requests_reused": sum(1 for r in hist if (r.get("reused") or 0) > 0),
            "reused_tokens": totals.get("reused", 0), "prompt_tokens": totals.get("prompt_tokens", 0),
            "last_reused": last.get("reused") if last else None,
            "last_prompt": last.get("prompt_tokens") if last else None}


_BTRACE = bool(os.environ.get("STRATA_BATCH_TRACE"))


def btrace(*a):
    if _BTRACE:
        print("[batch-trace]", threading.get_ident() % 10000, *a, file=sys.stderr, flush=True)


class StrataEngine:
    """The resident engine: `bin/sfnx --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """
    silence_s = ENGINE_SILENCE_S         # #481: main() sets the config's engine_silence_s (survives restart())
    silent_note = None                   # #481: why the server ended a silent engine (death_note says it)
    gpu_busy = None                      # #1317: () -> bool, "the GPU is working" (the telemetry's reading); a GPU at work is not frozen
    last_err = None                      # #997: an ERR that was its last stdout line (death_note says it)
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)

    # `last`: the figures of the engine's last DONE line.  With batch slots several requests run at once, each on its
    # own thread, so each one reads its own request's figures (Service.run records them); one at a time, the one dict.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False):
        """`lazy`: the object only, the process not started (tests; restart() starts it)."""
        self.spawn = (exe, list(args), cwd, log, env, False)   # to start it again after it died (issue #27)
        self.model_path = args[args.index("--native") + 1] if "--native" in args[:-1] else None
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        self.ended = True
        self.max_context = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 4096
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.last = {}
        self.info = {}                   # the INFO line's key=value facts: engine, batch slots, ... (Monitor tab)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        self.progress_ms = 0             # PP's own milliseconds since the prompt started (its third field)
        self.reused = 0                  # RESUME: prompt tokens not read again (a client takes them out of the work)
        self.silent_note = None
        self.last_err = None
        if lazy:
            return
        # `ended` stays True until READY (below): not alive while starting (#344)
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0  # where this start's lines begin (start_log_tail)
        self.log_start = log_start                      # #596: the Monitor's conversation cache reads from here
        if log:
            threading.Thread(target=narrate_start, args=(loading,), daemon=True).start()                                          # #779: when an engine started (a crash report has the time)
            try:
                self.log.write(f"[sfnx] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {exe} --serve "
                               f"{' '.join(args)}\n")
                self.log.flush()
            except OSError:
                pass
        self.max_context = 0                             # before the new process is visible: never its predecessor's
        try:
            self.proc = popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        except OSError:                                  # nothing started: the narrator stops, the log handle goes back
            loading.set()
            if log:
                self.log.close()
            raise
        # the READY read has a timeout (#1317): the lines come from a thread, up to READY, so the rest of the stream
        # stays for _pump (one reader at a time)
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q), daemon=True).start()
        ready_deadline = None if ENGINE_READY_S is None else time.monotonic() + ENGINE_READY_S
        timed_out = False
        while True:
            left = None if ready_deadline is None else ready_deadline - time.monotonic()
            if left is not None and left <= 0:
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                break
        loading.set()
        if self.max_context <= 0:
            if timed_out:                               # still loading, or hung: end it, so the turn is not held for good
                try:
                    self.proc.kill()
                except OSError:
                    pass
            try:                                        # its pipes and our handle on its log (the log stays)
                self.proc.wait(timeout=5)
                self.proc.stdin.close()
                self.proc.stdout.close()
                if log:
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            why = (f"the engine did not report READY within {ENGINE_READY_S:.0f} s" if timed_out else
                   "the engine exited before it was ready")
            # EngineDied, not bare RuntimeError: a request waiting on this load must end with a 503
            # through the request path's handler - a RuntimeError there left the turn with no answer (#1527)
            note = "" if timed_out else start_exit_note(self.proc.returncode)
            tail = start_log_tail(log, log_start)
            lines = [x for x in tail.split("\n")[2:] if x.strip()]
            if tail and lines and all(x.strip().startswith("[sfnx]") for x in lines):
                tail += "\n  (those are the launcher's own lines: the engine itself wrote nothing)"
            raise EngineDied(why + (f" (see {log})" if log else "") + note + tail)
        self.known_ctx = self.max_context   # survives a failed restart: requests keep their limit and restart it
        self.ended = False                              # READY: alive from here (restart() set it True, #344)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        # --batch: concurrent requests in the engine's batch slots (see generate_batched).  The engine says how many
        # it runs (INFO batch_slots=N: it may have fewer than asked, or none, when they do not fit)
        asked = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
        self.batch = int(self.info.get("batch_slots") or 0)
        if asked and self.batch != asked:
            print(f"[sfnx] parallel requests: {asked} asked, the engine runs {self.batch or 'one at a time'} "
                  "(its log says why)", flush=True)
        self.slot_q = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # #1603: who owns a busy flag.  slot_claims: slots a request of this server picked or reserved and has not
        # given back; slot_releasing: slots handed to the background BDONE wait.  A busy slot in neither has no owner
        # (its flag leaked) and would keep every later request waiting for a slot that nothing will free.
        self.slot_claims: set[int] = set()
        self.slot_releasing: set[int] = set()
        self._orphan_seen: dict[int, float] = {}
        # what each slot's sessions hold (prompt + every token a window fed), so a conversation's next turn goes to
        # the slot that has its start (the engine checks it again); when the slot was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        self.slot_live: list[dict | None] = [None] * self.batch   # /metrics: the request in each slot
        # #1012: the admission state belongs to the server, not to one engine process.  restart() runs this again
        # while requests still wait on it (they hold these very objects), so it is made once and kept: new ones
        # would leave the waiters on a lock and a condition nobody notifies, with their counts missing from the
        # new lists.
        if "slot_cv" not in self.__dict__:
            self.slot_cv = threading.Condition()
            self.waiting = 0                            # requests waiting for the control lines (ctl)
            self.admitting = 0                          # #1603: ... and those that gave the lines back to wait for a slot
            self.wait_lens: list[list[int]] = []        # ... their prompt lengths (a long read gives way to short ones)
            self.ctl_epoch = 0                          # how often the control lines were taken
            self.ctl = threading.Lock()                 # one admission or solo request on the control lines at a time
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is (a request notes its own)
        self._ctl_erred = False                         # #1059: the control lines ended on an ERR
        self._yielded = None                            # (slot, tokens read): the last request on them gave way
        self.wlock = threading.Lock()                   # stdin writes from several request threads
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        proc, lines = self.proc, self.lines             # this process's: a restart replaces both (#344)
        slot_q = self.slot_q
        line = None
        for line in proc.stdout:
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # --batch: a batch slot's own lines
                try:
                    slot_q[int(line.split()[1])].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            lines.put(line)
        if self.proc is proc:                           # a killed engine's pump must not mark its successor dead
            self.ended = True                           # its output closed: it is gone, even before the OS says so
            if line and line.startswith("ERR"):         # #997 #890: why it exited, though no request may read it
                self.last_err = line[4:].strip()        # (a failed batch window: every request reads its slot)
        lines.put(None)
        for q in slot_q:
            q.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        if getattr(self, "silent_note", None):          # #481: the server ended it, not the OS or the engine itself
            return self.silent_note
        tail = wide = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                wide = f.read().decode("utf-8", "replace")
            tail = wide[-4096:]
        except (OSError, TypeError):
            pass
        # #1705: STRATA_VERIFY_TRACE dumps are ~30 KB and push the watchdog's line out of a 4 KB tail
        for line in reversed(wide.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - keep the engine log for the report.")
        proc = self.proc                                 # None while a restart has not started the next one yet
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc >= 0 and self.last_err:  # its own last words on stdout: they say why
            return (f"The engine exited (code {rc}) after it reported: {self.last_err} - keep the engine log for the "
                    "report.")
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("sfnx", "fnx", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, keep the "
                    "engine log for the report.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'). Keep the end of the engine log for the report.")

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart() / close() may set it to None meanwhile
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    RESTART_RETRY_S = 15.0   # between the tries of restart(): a dying engine's VRAM may take a while to come back

    @staticmethod
    def _ready_pump(proc, out: queue.Queue) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def restart(self, tries: int = 3):
        """Start the engine again (the same command) after it died; the new process has its own line queue.
        close() ends and waits for the old process first (EngineStuck when it cannot be ended).  A start that still
        exits before READY - a dead engine's VRAM can take a while to come back - is retried (PR #637)."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 "starting" meanwhile (#344)
        try:
            self.close()
            for i in range(tries):
                # #344: not alive until READY - __init__ sets max_context to 0 and blocks until the engine says
                # READY, and a request that saw alive() in that window skipped load() and failed with "context
                # (0)".  __init__ clears `ended` itself once READY (before its pump thread can set it again).
                self.ended = True
                try:
                    self.__init__(*self.spawn)
                    break
                except RuntimeError:
                    try:
                        self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[sfnx] the engine did not start (try {i + 1} of {tries}); again in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
        finally:
            self.starting = False
            # a start that failed never reached the post-READY __init__ where slot_cv is made (#1527)
            if "slot_cv" in self.__dict__:
                with self.slot_cv:               # #1012: wake the requests that waited through it: they go on with the
                    self.slot_cv.notify_all()    # new engine, or (it did not start) end with a clean EngineDied
        self.info = {**info, **self.info}

    def _out_of_step(self, what: str) -> EngineDied:
        """The engine said something the protocol does not allow: the two sides lost step, so it is ended now and the
        next request starts it again."""
        self.ended = True                               # not alive from now: the next request restarts it
        proc = self.proc
        try:
            proc.kill()
            proc.wait(timeout=20)
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineDied(f"{what}; the server ended the engine")

    def _parse_done(self, line: str) -> EngineDone:
        """A DONE line into `last` (its figures); a malformed one ends the engine (EngineDied)."""
        try:
            done = EngineDone.parse(line)
        except ValueError as e:
            raise self._out_of_step(str(e)) from None
        self.last = done.figures()
        return done

    def _parse_bdone(self, line: str) -> BatchDone:
        """A BDONE line; a malformed one ends the engine (EngineDied)."""
        try:
            return BatchDone.parse(line)
        except ValueError as e:
            raise self._out_of_step(str(e)) from None

    @staticmethod
    def keep_key(ids) -> str:
        """`keep=N`: a checkpoint where the prompt's closing notes start (see Prompt)."""
        return f" keep={ids.keep}" if isinstance(ids, Prompt) and ids.keep is not None else ""

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most TOP_K_MAX candidates: 0 ("off") and wider lists get all of them
            keys += f" top_k={tk if 1 <= tk <= TOP_K_MAX else TOP_K_MAX}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default window
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += f" penalty_last_n={PENALTY_LAST_N_DEFAULT}"
        # 0 is a seed like any other; a negative one (llama.cpp's -1: "random") is no seed, which the engine draws
        seed = sampling.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
            keys += f" seed={seed}"
        # "strata_checkpoint": false - a one-shot call (a classification, a probe) whose turn no later request
        # extends: no conversation checkpoint for it (#830).  It still reuses a cached prefix.  Absent = as before.
        if sampling.get("strata_checkpoint") is False:
            keys += " ckpt=0"
        # #2228: the pictures' keys (hex, in prompt order): the engine matches a picture by them, not by its embeddings
        ik = sampling.get("strata_img_keys")
        if isinstance(ik, list) and ik and all(isinstance(k, str) and k and all(c in "0123456789abcdef" for c in k)
                                               for k in ik):
            keys += " img_keys=" + ",".join(ik)
        return keys

    @property
    def read_meter(self) -> ReadMeter:
        """The prompt read's rate, from when its PP lines arrive (the Monitor).  Made on first use: the control-line
        tests build engines with __new__ and set only the attributes they use."""
        meter = self.__dict__.get("_read_meter")
        if meter is None:
            meter = self.__dict__["_read_meter"] = ReadMeter()
        return meter

    def _send(self, text: str):
        btrace("send>", text[:60])
        if text.startswith(("GEN", "BGEN")):           # a new read: the last one's progress and rate are not its own
            self.progress = None
            self.read_meter.start()
        try:
            with self.wlock:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
        except (OSError, AttributeError):                # the pipe is gone (or a restart's close() took the process):
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None

    def _control(self, cancel, on_token, stop_when=None, on_quiet=None):
        """Reads the control lines of the request on them (GEN / BGEN), yielding None heartbeats.  Calls on_token(id)
        for each `T`; returns ("done", None) at DONE, or ("badm", continues) at BADM (after DONE).  `stop_when()`
        true sends STOP once (the request is then read to its DONE).  `on_quiet()` runs whenever the engine has been
        quiet for half a second: between two PP lines of a read whose chunks take longer than that."""
        stopped = False
        beat = time.monotonic()
        while True:
            try:
                # A client can leave before the first PP/token.  Poll its Event as often as _take_control does;
                # the ten-second HTTP heartbeat is independent of how promptly STOP reaches the engine.
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if cancel.is_set() and not stopped:
                    self._send("STOP")
                    stopped = True
                if on_quiet is not None:
                    on_quiet()
                if time.monotonic() - beat >= 10.0:
                    beat = time.monotonic()
                    yield None
                continue
            beat = time.monotonic()
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if not line.startswith(("PP ", "INFO")):
                btrace("ctl<", self._ctl_mode, line.strip()[:60])
            if line.startswith("T "):
                on_token(int(line[2:]))
                if not stopped and (cancel.is_set() or (stop_when is not None and stop_when())):
                    self._send("STOP")
                    stopped = True
                yield False                              # a token is pending (not a heartbeat)
            elif line.startswith("PP "):
                f = line.split()
                if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                    self.progress = (int(f[1]), int(f[2]))
                    self.read_meter.note(int(f[1]))
                    self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                    self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                if not stopped and cancel.is_set():
                    # #879: a Stop during the prompt read went out only after the first token (or after 10 s of
                    # silence, which a stream of PP lines never allows): the engine read the whole prompt first
                    self._send("STOP")
                    stopped = True
                yield None
            elif line.startswith("DONE"):
                self._last_done = self._parse_done(line)
            elif line.startswith("BADM "):
                f = line.split()
                self._ctl_result = ("badm", len(f) >= 3 and f[2] == "1")
                return
            elif line.startswith("YIELDED "):            # the read gave way (BYIELD): <slot> <tokens read>
                f = line.split()
                if len(f) >= 3 and f[1].lstrip("-").isdigit() and f[2].isdigit():
                    self._yielded = (int(f[1]), int(f[2]))
            elif line.startswith("ERR"):
                self._ctl_erred = True                   # #1059: the engine refused it and is idle: no DONE follows
                raise ValueError(line[4:].strip())
            if line.startswith("DONE") and self._ctl_mode == "solo":
                self._ctl_result = ("done", None)
                return

    def _done_finish(self) -> str | None:
        """The finish field of the DONE line the control lines read last (None: none since it was cleared)."""
        done = getattr(self, "_last_done", None)
        return done.finish if done is not None else None

    def _drain_control(self, until: str, timeout: float = 300.0, born: int | None = None):
        """After a consumer left early: read the control lines up to the next `until` line (DONE or BADM) so the next
        request does not read this one's leftovers.  Returns that line (None: the engine ended or never answered).
        `born`: the engine process (self.gen) the request started on; once that one is gone there is nothing to
        drain - its end line was read already (by the request's own EngineDied), and waiting for another one held
        a request that had failed for the whole 300 s (#1012), or read the lines of the engine that replaced it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if born is not None and (self.gen != born or not self.alive()):
                return None
            try:
                line = self.lines.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith("DONE"):
                self._parse_done(line)
            if line.startswith(until) or line.startswith("ERR"):
                return line
        return None

    def _apply_bdone(self, done: BatchDone) -> None:
        """A slot's BDONE into this request's figures."""
        self.last = {**self.last, "finish": done.finish, "decode_ms": done.ms, "generated": done.produced,
                     "drafts_accepted": done.drafts_accepted, "drafts_offered": done.drafts_offered, "rounds": done.rounds}

    def _bdone_after_eos(self, slot: int, wait_s: float = 2.0):
        """#1869: a request in a slot whose consumer left at the stop token has its BDONE right behind it (the engine prints it
        in the same window).  Read it here, so the request's figures are the slot leg's (the background wait would eat it,
        and the request kept the admission's single token).  None when it does not come in `wait_s`: then the background
        wait takes over."""
        if slot >= len(self.slot_q):
            return None
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            try:
                line = self.slot_q[slot].get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                return None
            if line is None:
                return None
            if line.startswith("BDONE "):
                return line
        return None

    def _release_slot_when_done(self, slot: int, stream: list[int] | None = None):
        """A slot whose consumer left (a stop token, a stop string, a disconnect): BSTOP it and free it once the engine
        says BDONE (in the background).  `stream`: the prompt and every token of it so far - with the tokens still to
        come before BDONE, all but the last are what the slot holds then (the next turn of its conversation)."""
        if slot >= len(self.slot_q):                    # the engine restarted with fewer slots: this one is gone
            return
        try:
            self._send(f"BSTOP {slot}")
        except EngineDied:
            pass
        q, busy, held = self.slot_q[slot], self.slot_busy, self.slot_held   # this process's: restart() replaces them
        releasing = self.slot_releasing
        releasing.add(slot)

        def wait():
            end = time.monotonic() + 600.0
            tail = list(stream or [])
            while time.monotonic() < end:
                try:
                    line = q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if line is None:
                    tail = []
                    break
                if line.startswith("BT "):
                    try:
                        tail.append(int(line.split()[2]))
                    except (IndexError, ValueError):
                        tail = []
                if line.startswith("BDONE "):
                    try:
                        self._parse_bdone(line)
                    except EngineDied as e:                  # the engine is ended: the slot holds nothing
                        print(f"[sfnx] {e}", flush=True)
                        tail = []
                    break
            held[slot] = tail[:-1] if stream and tail else []
            with self.slot_cv:
                busy[slot] = False
                releasing.discard(slot)
                self.slot_cv.notify_all()
        threading.Thread(target=wait, daemon=True).start()

    BGEN_REFUSALS_MAX = 6    # #1994: how often one request tries another slot after the engine refused a BGEN
    SLOT_PARK_S = 2.0        # ... and how long a refused slot stays out of the picks

    def _park_slot(self, slot: int) -> None:
        """#1994: the engine refused a BGEN for `slot`.  Keep it busy here for SLOT_PARK_S (no request owns it), then
        give it back: the engine's view has caught up, or the next refusal parks it again."""
        with self.slot_cv:
            self.slot_claims.discard(slot)
            self.slot_busy[slot] = True
        busy = self.slot_busy

        def free():
            with self.slot_cv:
                if slot not in self.slot_claims and slot not in self.slot_releasing:
                    busy[slot] = False
                self.slot_cv.notify_all()
        t = threading.Timer(self.SLOT_PARK_S, free)
        t.daemon = True
        t.start()

    ORPHAN_SLOT_S = 3.0   # #1603: a busy flag nobody owns this long is a leak and is given back

    def _reap_orphan_slots(self) -> list[int]:
        """#1603: free the slots whose busy flag has no owner (no request holds or reserves them and no BDONE wait is
        running for them).  The caller holds slot_cv.  A flag that leaked kept every later request waiting for a slot
        that nothing would free - with nothing running, nothing queued and nothing for the frozen-engine check to see.
        A slot is only given back after it has looked ownerless for ORPHAN_SLOT_S, so a flag set a moment before its
        owner registers is never taken."""
        now = time.monotonic()
        freed = []
        for b, busy in enumerate(self.slot_busy):
            if busy and b not in self.slot_claims and b not in self.slot_releasing:
                first = self._orphan_seen.setdefault(b, now)
                if now - first >= self.ORPHAN_SLOT_S:
                    self.slot_busy[b] = False
                    self._orphan_seen.pop(b, None)
                    freed.append(b)
            else:
                self._orphan_seen.pop(b, None)
        if freed:
            print(f"[sfnx] parallel: slot(s) {', '.join(map(str, freed))} were marked busy with no request using them "
                  "(issue #1603); freed, the waiting request goes on", flush=True)
            self.slot_cv.notify_all()
        return freed

    YIELDS_MAX = 2   # #656: how often one request's prompt read gives way to a shorter waiting one

    def _take_control(self, cancel, plen: int, after_epoch: int | None = None):
        """Waits for the control lines (one prompt read at a time), yielding None heartbeats; False when cancelled.
        `after_epoch`: a request whose read gave way lets the requests waiting then go first."""
        entry = [plen]                                  # its prompt length (_shorter_waiting)
        with self.slot_cv:
            self.waiting += 1
            self.wait_lens.append(entry)
        beat = time.monotonic()
        try:
            while True:
                if not self.alive() and not getattr(self, "starting", False):
                    # #1012: the engine is gone and nobody is starting it again: this request was not sent, so it
                    # ends now (a clean 503) instead of waiting for control lines nothing will answer on.  While a
                    # restart is under way it keeps waiting and goes on with the new engine.
                    raise EngineDied("the engine stopped while this request waited; it was not sent")
                with self.slot_cv:
                    turn = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                if turn and self.ctl.acquire(timeout=0.5):
                    if self.alive():
                        break
                    self.ctl.release()                  # it died (or is being restarted) while this one was getting
                    turn = False                        # the lines: not the engine this request was meant for
                if not turn:
                    with self.slot_cv:
                        self.slot_cv.wait(timeout=0.5)
                if cancel.is_set():
                    return False
                if time.monotonic() - beat >= 10.0:     # a heartbeat every 10 s, as the engine's own wait
                    beat = time.monotonic()
                    yield None
        finally:
            with self.slot_cv:
                self.waiting = max(0, self.waiting - 1)
                if entry in self.wait_lens:
                    self.wait_lens.remove(entry)
        with self.slot_cv:
            self.ctl_epoch += 1
            self.slot_cv.notify_all()
        return True

    SOLO_AGAIN_MAX = 2      # how often a request left alone in a slot goes back to the solo path
    SOLO_AGAIN_MIN_LEFT = 32  # ... only with at least this many tokens still allowed (max_tokens)

    def _may_go_solo(self, left: int, times: int, embeddings) -> bool:
        """A request decoding in a slot that is alone now (no other slot busy, nobody waiting) goes back to the solo
        path with its MTP drafts: the engine continues it from the slot's sessions (its slot cache; INFO
        slot_cache=1).  STRATA_PARALLEL_SOLO=0 keeps it in the slot."""
        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or
                not (self.info or {}).get("slot_cache") or os.environ.get("STRATA_PARALLEL_SOLO") == "0"):
            return False
        with self.slot_cv:
            return sum(1 for b in self.slot_busy if b) == 1 and self.waiting == 0

    def _shorter_waiting(self, plen: int) -> bool:
        """#656: someone waits for the control lines with a prompt under half this one's: worth giving way to."""
        with self.slot_cv:
            return any(e[0] * 2 <= plen for e in self.wait_lens)

    def generate_batched(self, ids, max_new, sampling, cancel, embeddings=None):
        """--batch.  Alone (no slot busy, nobody waiting): the solo path (GEN, with drafts: the fastest stream)
        - and when another request arrives meanwhile, this one is STOPped and continues in a batch slot (BGEN with
        its prompt + what it generated: the engine reuses that prefix).  Otherwise: BGEN into a free slot, then the
        slot's own BT lines until BDONE.  Several requests run at once; the control lines (prompt reading, admission)
        are taken one request at a time - and a long prompt read gives way at a chunk boundary to a waiting request
        with a much shorter prompt (#656: `BYIELD <slot>`; the part read waits in a slot and the read goes on after).
        A consumer that stops early leaves the engine in step: the solo request is STOPped and read to its DONE, an
        admission to its BADM, a slot is BSTOPped and freed at its BDONE."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.progress, self.progress_ms, self.reused = None, 0, 0
        born = self.gen                                 # the engine process this request is sent to
        keys = self.sampling_keys(sampling or {}) + self.keep_key(ids)
        out: list[int] = []
        pending: list[int] = []
        prompt, left = list(ids), int(max_new)
        reused0 = None      # the first read's reused tokens: a continued leg's count includes this request's own output
        carry = {}          # #1869: the figures of the legs that ended in a hand-back to the solo path (the final DONE adds to them)
        ok = yield from self._take_control(cancel, len(prompt))
        if not ok:
            return
        holding, born = True, self.gen                  # (the engine it now has the control lines of)
        btrace("ctl acquired")
        slot, gen0, reserved = None, None, None
        phase = "none"            # solo -> (admit -> slot) ; "done" once the engine has finished with this request
        stop_sent = False
        yields, solo_again, refused = 0, 0, 0
        admit_wait, admit_since, admit_said = False, 0.0, False   # #1603: waiting for a slot, since when, said
        try:
            while True:   # a request in a slot that is left alone goes back to the solo path
                if not holding:
                    ok = yield from self._take_control(cancel, len(prompt))
                    if not ok:
                        return
                    holding, born = True, self.gen                  # (the engine it now has the control lines of)
                with self.slot_cv:
                    alone = not any(self.slot_busy) and self.waiting == 0
                if alone and left > 1:
                    head = f"GENI {left}{keys} {embeddings}" if embeddings else f"GEN {left}{keys}"
                    self._last_done = None                  # (#879: this request's DONE decides below)
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    phase = "solo"
                    self._ctl_mode, self._ctl_result, self._yielded = "solo", None, None
                    self._ctl_erred = False
                    def others():
                        with self.slot_cv:
                            return self.waiting > 0

                    def give_way():   # at a PP line, and between two far apart (the engine takes it at its next boundary)
                        nonlocal reserved
                        if (reserved is None and not out and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            with self.slot_cv:          # the slot the part read will wait in
                                reserved = self.pick_slot(prompt)
                                if reserved is not None:
                                    self.slot_busy[reserved] = True
                                    self.slot_claims.add(reserved)
                            if reserved is not None:
                                self._send(f"BYIELD {reserved}")
                    for x in self._control(cancel, pending.append, stop_when=others, on_quiet=give_way):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:                       # a heartbeat (False: a token, flushed above)
                            give_way()
                            yield None
                    phase = "none"                          # its DONE is read
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if self._yielded is not None and reserved is not None and self._yielded[0] == reserved:
                        slot, reserved = reserved, None     # gave way: the read goes on in that slot (below)
                    elif reserved is not None:              # it did not give way: the slot is free again
                        with self.slot_cv:
                            self.slot_busy[reserved] = False
                            self.slot_claims.discard(reserved)
                            self.slot_cv.notify_all()
                        reserved = None
                    if slot is None:
                        finish = (self.last or {}).get("finish") if isinstance(self.last, dict) else None
                        left = int(max_new) - len(out)
                        if cancel.is_set() or left <= 0 or finish in ("stop", "length") or (out and out[-1] in EOS_IDS):
                            return
                        if reused0 is None:
                            reused0 = (self.last or {}).get("reused")
                        carry_leg(carry, self.last)         # #1869: the solo leg's figures; the slot's admission (a copy of
                                                            # the held prefix) and BDONE replace them
                        prompt = list(ids) + out            # promoted: it continues in a batch slot from here
                while True:
                    if slot is None:
                        # a free slot (they free themselves at BDONE, which needs no control lines): the one that holds
                        # the start of this prompt (its conversation's last turn), else the one used longest ago.
                        # With none free the control lines are given back while waiting: a slot can be held by a read
                        # that gave way (BYIELD) and needs them to go on, so waiting with them is a deadlock - long,
                        # long, short at 0.25 s steps under parallel: 2 (ENGINE_REVIEW finding 1, reproduced in 0.1.41).
                        while True:
                            with self.slot_cv:
                                slot = self.pick_slot(prompt)
                                if slot is not None:
                                    self.slot_busy[slot] = True
                                    self.slot_claims.add(slot)
                                    if admit_wait:
                                        admit_wait = False
                                        self.admitting = max(0, self.admitting - 1)
                                    break
                                if not admit_wait:          # #1603: counted while it waits for a slot (it holds no
                                    admit_wait, admit_since = True, time.monotonic()   # control lines then)
                                    self.admitting += 1
                            if holding:
                                self.ctl.release()
                                holding = False
                            with self.slot_cv:
                                if self.pick_slot(prompt) is None:
                                    if (ENGINE_STALL_S or 0) > 0 and time.monotonic() - admit_since >= ENGINE_STALL_S:
                                        # the stall check of #1317 only watches a request on the control lines; one
                                        # waiting here is covered by looking for a slot nobody owns, and by saying so
                                        if not admit_said:
                                            admit_said = True
                                            busy = [b for b, v in enumerate(self.slot_busy) if v]
                                            print(f"[sfnx] parallel: a request has waited {time.monotonic() - admit_since:.0f} s "
                                                  f"for a free slot (busy: {busy}, in use: {sorted(self.slot_claims)}, "
                                                  f"ending: {sorted(self.slot_releasing)}) (issue #1603)", flush=True)
                                    self._reap_orphan_slots()
                                if self.pick_slot(prompt) is None:
                                    self.slot_cv.wait(timeout=1.0)
                            if cancel.is_set():
                                return
                            if not holding:
                                ok = yield from self._take_control(cancel, len(prompt))
                                if not ok:
                                    return
                                holding, born = True, self.gen
                    if self._yielded is not None:           # it gave way: the others waiting then go first
                        self.slot_held[slot] = list(prompt[:self._yielded[1]])
                        self._yielded = None
                        yields += 1
                        self.ctl.release()
                        holding = False
                        with self.slot_cv:
                            epoch = self.ctl_epoch
                        ok = yield from self._take_control(cancel, len(prompt), after_epoch=epoch)
                        if not ok:
                            return
                        holding, born = True, self.gen                  # (the engine it now has the control lines of)
                    while not self.slot_q[slot].empty():
                        self.slot_q[slot].get_nowait()
                    head = f"BGENI {slot} {left}{keys} {embeddings}" if embeddings else f"BGEN {slot} {left}{keys}"
                    live = {"slot": slot, "request_id": getattr(cancel, "_strata_request_id", None),
                            "state": "reading", "prompt_tokens": len(prompt), "prompt_read": 0, "generated": 0,
                            "started": time.time(), "first_token": None}
                    self.slot_live[slot] = live
                    self._last_done = None                  # (#879: this admission's DONE decides below)
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    self.slot_held[slot] = []               # the admission overwrites what the slot held
                    phase = "admit"
                    self._ctl_mode, self._ctl_result, self._yielded = "batch", None, None
                    self._ctl_erred = False
                    asked = False

                    def ask():        # as give_way above, for the slot this admission reads into
                        nonlocal asked
                        if (not asked and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            self._send(f"BYIELD {slot}")
                            asked = True

                    def note_progress():
                        # the terminal layer's live prompt counter (the batch admission path)
                        if self.progress and live["state"] == "reading":
                            live["prompt_read"] = self.progress[0]
                            live["read_tok_s"] = self.read_meter.rate()
                    try:
                        for x in self._control(cancel, pending.append, on_quiet=lambda: (ask(), note_progress())):
                            while pending:
                                t = pending.pop(0)
                                out.append(t)
                                yield t
                            if x is None:
                                ask()
                                note_progress()
                                yield None
                    except ValueError as e:
                        # #1994: the engine says this slot is not free (its view and ours differ for a moment, or a
                        # slot it still holds).  That is not the request's fault and nothing was read: park the slot
                        # for a moment (busy here too), and wait for another one - never a 400 for the conversation.
                        if not (self._ctl_erred and str(e).startswith("BGEN:") and left >= 1 and not cancel.is_set()):
                            raise
                        refused += 1
                        print(f"[sfnx] parallel: the engine refused slot {slot} ({e}); trying another slot "
                              f"({refused}/{self.BGEN_REFUSALS_MAX}) (issue #1994)", flush=True)
                        if refused > self.BGEN_REFUSALS_MAX:
                            raise GpuBusy("the engine's batch slots are not free right now; try again in a moment",
                                      retry_after=5) from e
                        self.slot_live[slot] = None
                        self.slot_held[slot] = []
                        phase = "none"
                        self._ctl_erred = False
                        self._park_slot(slot)
                        slot = None
                        continue
                    cont = bool(self._ctl_result and self._ctl_result[1])
                    phase = "slot" if cont else "none"
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if not cont and self._yielded is not None and self._yielded[0] == slot and not cancel.is_set():
                        continue                            # gave way: again once the shorter request is in
                    break
                self.ctl.release()
                holding = False
                if not cont:
                    return
                live.update(state="decoding", first_token=time.time(), generated=len(out))
                if self._last_done is not None:         # the admission's read, for the Monitor's slot view
                    live.update(reused=self._last_done.reused, read_ms=self._last_done.prompt_ms)
                gen0 = len(out) - 1                         # the admission's own token: the slot feeds it first
                going_solo = False
                while True:
                    try:
                        line = self.slot_q[slot].get(timeout=10.0)
                    except queue.Empty:
                        if cancel.is_set() and not stop_sent:
                            self._send(f"BSTOP {slot}")
                            stop_sent = True
                        yield None
                        continue
                    if line is None:
                        phase = "none"
                        raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                    if line.startswith("BT "):
                        t = int(line.split()[2])
                        out.append(t)
                        live["generated"] = len(out)
                        if cancel.is_set():
                            if not stop_sent:
                                self._send(f"BSTOP {slot}")
                                stop_sent = True
                            continue
                        yield t
                        if not going_solo and not stop_sent and self._may_go_solo(int(max_new) - len(out), solo_again,
                                                                                  embeddings):
                            self._send(f"BSTOP {slot}")         # alone now: on with the drafts (below)
                            going_solo = True
                    elif line.startswith("BDONE "):
                        phase = "none"
                        bd = self._parse_bdone(line)
                        self._apply_bdone(bd)
                        # what the slot's sessions hold now: the prompt and every token fed (all but the last one)
                        self.slot_held[slot] = list(prompt) + out[gen0:-1] if len(out) > gen0 else []
                        if (going_solo and bd.finish == "cancel" and not cancel.is_set() and len(out) < int(max_new)
                                and not (out and out[-1] in EOS_IDS)):
                            # the solo path continues it: the engine copies the slot's sessions back (all but the
                            # last token are in them) and decodes with MTP drafts again
                            carry_leg(carry, self.last)         # #1869: the solo path's DONE replaces these figures
                            self.slot_live[slot] = None
                            self.slot_used[slot] = time.time()
                            with self.slot_cv:
                                self.slot_busy[slot] = False
                                self.slot_claims.discard(slot)
                                self.slot_cv.notify_all()
                            slot, gen0 = None, None
                            if reused0 is None:
                                reused0 = (self.last or {}).get("reused")
                            prompt, left = list(ids) + out, int(max_new) - len(out)
                            solo_again += 1
                            btrace("back to the solo path")
                            break
                        return
        finally:
            # a consumer that left early (or an error): keep the engine and this server in step
            btrace("finally phase", phase, "slot", slot, "holding", holding)
            try:
                if phase in ("solo", "admit") and (self.gen != born or not self.alive()):
                    phase = "none"                      # #1012: its engine is gone: nothing to stop or drain
                if phase in ("solo", "admit") and self._ctl_erred:
                    phase = "none"                      # #1059: it ended on the engine's ERR: no DONE to wait 300 s for
                if phase == "solo":
                    self._send("STOP")
                    self._drain_control("DONE", born=born)
                elif phase == "admit":
                    line = self._drain_control("BADM", born=born)
                    if line and line.startswith("BADM ") and line.split()[2:3] == ["1"]:
                        phase = "slot"
            except EngineDied:
                pass
            bdone_in = False
            if phase == "slot" and slot is not None and out and out[-1] in EOS_IDS and self.gen == born:
                try:
                    bd = self._bdone_after_eos(slot)        # (#1869) the slot's figures, not the admission's
                    if bd is not None:
                        self._apply_bdone(self._parse_bdone(bd))
                        bdone_in = True
                except EngineDied:
                    pass                                    # the engine is marked ended; the slot release below still runs
            if reused0 is not None and isinstance(self.last, dict):
                self.last = {**self.last, "reused": reused0}
            if carry and isinstance(self.last, dict):
                self.last = add_carry(carry, self.last)
            if holding:
                self.ctl.release()
            if admit_wait:
                with self.slot_cv:
                    self.admitting = max(0, self.admitting - 1)
            if reserved is not None:
                with self.slot_cv:
                    self.slot_busy[reserved] = False
                    self.slot_claims.discard(reserved)
                    self.slot_cv.notify_all()
            if slot is not None:
                self.slot_live[slot] = None
                self.slot_used[slot] = time.time()
                if phase == "slot" and bdone_in:
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self.slot_held[slot] = stream[:-1] if stream else []     # as the BDONE wait leaves it
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()
                elif phase == "slot":
                    self.slot_held[slot] = []
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self._release_slot_when_done(slot, stream)    # freed at its BDONE
                    self.slot_claims.discard(slot)                # (the BDONE wait owns it now)
                else:
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()

    TURN_ID = 248045        # <|im_start|>: the engine's default --turn-token (its turn checkpoint sits at the last one)

    def pick_slot(self, prompt: list[int]) -> int | None:
        """A free slot for `prompt` (the caller holds slot_cv): the one whose held tokens are the longest start of the
        prompt (the engine then reads only the rest), else an empty one, else the one used longest ago - so the
        conversations other slots hold stay for their next turns.  None: every slot is busy."""
        free = [b for b in range(self.batch) if not self.slot_busy[b]]
        if not free:
            return None

        def held_prefix(b):
            # what of `prompt` the slot's cache serves: the slot's whole held stream (the prompt and what it decoded)
            # when this prompt continues it, else its turn checkpoint - the engine takes one at the last <|im_start|> of
            # a prompt (the next request of the chat renders the history again, but not the reply it decoded or the
            # assistant header), so a prompt that agrees with the held tokens up to there is that chat's next turn
            # too (#2518: it was only found by the engine's own source search, after pick_slot had sent it to another
            # slot and the request in that slot had overwritten the chat whose cache the next one needed)
            h = self.slot_held[b]
            if not h:
                return 0
            if len(h) < len(prompt) and prompt[:len(h)] == h:
                return len(h)
            k = len(h) - 1 - h[::-1].index(self.TURN_ID) if self.TURN_ID in h else 0
            return k if 0 < k < len(prompt) and prompt[:k] == h[:k] else 0
        best = max(free, key=held_prefix)
        if held_prefix(best) >= 1:
            return best
        return min(free, key=lambda b: (bool(self.slot_held[b]), self.slot_used[b]))

    def slots_view(self) -> list[dict]:
        """/metrics: each batch slot - idle (with the tokens it holds for a next turn), reading (how far, how fast)
        or decoding (its rate, and the read it was admitted with)."""
        now, view = time.time(), []
        for b in range(self.batch):
            r = self.slot_live[b]
            if r is None:
                view.append({"slot": b, "state": "idle", "held_tokens": len(self.slot_held[b])})
                continue
            ft = r.get("first_token")
            reading = r["state"] == "reading"
            view.append({"slot": b, "state": r["state"], "request_id": r.get("request_id"),
                         "prompt_tokens": r["prompt_tokens"],
                         "prompt_read": r.get("prompt_read") if reading else None,
                         "read_tok_s": r.get("read_tok_s") if reading else None,
                         "reused": r.get("reused"), "read_ms": r.get("read_ms"),
                         "generated": r["generated"], "context_tokens": r["prompt_tokens"] + r["generated"],
                         "elapsed_s": round(now - r["started"], 1),
                         "tok_s": round(r["generated"] / max(1e-6, now - ft), 1) if ft else None})
        return view

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")

        if getattr(self, "batch", 0):
            yield from self.generate_batched(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress, self.progress_ms, self.reused = None, 0, 0
        self.prefill_tok_s_mean = None
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        keys = self.sampling_keys(sampling or {}) + self.keep_key(ids)
        head = f"GENI {int(max_new)}{keys} {embeddings}" if embeddings else f"GEN {int(max_new)}{keys}"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        # #481: how long the engine may stay silent from here.  Until the first line: the request's first prompt
        # chunk at the slowest prompt reading on top of silence_s; a PP line resets it to its own chunk's time.
        silence = float(self.silence_s or 0)
        allow = silence + min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
        heard, read_to = time.monotonic(), 0
        beat = heard
        stall_s = ENGINE_STALL_S or 0.0
        stall_state: dict = {}
        try:
            while True:
                wait = 10.0
                if allow > 0:
                    wait = min(wait, allow - (time.monotonic() - heard))
                    if wait <= 0:
                        done = True                       # no STOP and no drain: nothing is listening
                        raise self._silent(f"the engine said nothing for {time.monotonic() - heard:.0f} s during "
                                           "the request")
                try:
                    line = self.lines.get(timeout=min(wait, 0.5))
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if stall_s > 0:
                        try:
                            self._stall_check(heard, stall_state, stall_s)
                        except EngineSilent:
                            done = True                   # as above: nothing is listening
                            raise
                    if allow > 0 and time.monotonic() - heard >= allow:
                        continue                          # enforce the expired deadline before another heartbeat
                    if time.monotonic() - beat >= 10.0:
                        beat = time.monotonic()
                        yield None                        # polling cancellation does not add HTTP heartbeats
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()                  # any line is output: T, PP, RESUME, INFO ...
                beat = heard
                if stall_state:
                    stall_state.clear()
                if line.startswith("T "):
                    allow = silence
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))             # prompt progress, one per chunk: also a heartbeat (the
                        self.read_meter.note(int(f[1]))
                        self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        rate, chunk = self.prefill_tok_s_mean or 0.0, int(f[1]) - read_to
                        read_to = int(f[1])
                        if silence > 0 and rate > 0 and chunk > 0:   # #481: the next chunk, as long as this one
                            allow = max(silence, PP_SLACK * chunk / rate)
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("RESUME "):          # the reused tokens: the first chunk starts after them
                    try:
                        read_to = self.reused = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # #481: never an untimed wait here - it holds the request FIFO, and an engine that lost step never
                # answers.  An engine that honours STOP gets the current allowance in all (a STOP during a prompt
                # chunk is seen after it); an older one runs on to max_new, so each line only has to come in time.
                heard = time.monotonic()
                while True:
                    left = allow - (time.monotonic() - heard) if allow > 0 else None
                    try:
                        if left is not None and left <= 0:
                            raise queue.Empty
                        line = self.lines.get(timeout=left)
                    except queue.Empty:
                        raise self._silent("the engine did not finish the request after it was stopped (STOP) "
                                           f"within {allow:.0f} s") from None
                    if line is None or line.startswith("ERR"):
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break
                    if not self.can_stop:
                        heard = time.monotonic()

    def _activity(self) -> tuple[float, int] | None:
        """#1317: (CPU seconds, I/O bytes) the engine process has used so far, or None when they cannot be read."""
        try:
            pr = psutil.Process(self.proc.pid)
            t = pr.cpu_times()
            io = pr.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied):   # the process gone, or not ours to read
            return None
        return (t.user + t.system, io.read_bytes + io.write_bytes)

    def _stall_check(self, heard: float, state: dict, stall_s: float) -> None:
        """#1317: called while the engine is silent.  Takes a baseline soon after the last line, and at stall_s of silence
        compares: no CPU and no I/O in all that time -> the engine is ended (EngineSilent); work going on -> one warning."""
        age = time.monotonic() - heard
        if state.get("base") is None:
            state["base"] = self._activity()
            return
        if age < state.get("next", stall_s) or state["base"] is None:
            return
        now = self._activity()
        try:
            gpu_working = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu_working = False
        if now is not None and not gpu_working and engine_frozen(state["base"], now):
            raise self._silent(f"the engine said nothing for {age:.0f} s and used no CPU or disk in that time (frozen; "
                               "STRATA_ENGINE_STALL_S sets this, 0 = off)")
        if now is not None:   # working: look again after another window from here, and say so once
            state["base"], state["next"] = now, age + stall_s
            if not state.get("warned"):
                state["warned"] = True
                print(f"[sfnx] the engine has said nothing for {age:.0f} s but is still working; it is not ended (the "
                      "limit is engine_silence_s). If it is stuck, \"engine_silence_s\" can be set lower.", flush=True)
        else:
            state["next"] = float("inf")   # no reading (the process gone): nothing to decide on

    def _silent(self, what: str) -> EngineSilent:
        """#481: end an engine that lost step with the server (its main thread waits for a command the server never
        sends), so the next request starts it again: killed now, its GPU and RAM go back with the process."""
        self.silent_note = ("The engine and the server lost step (issue #481; a very slow PC can raise "
                            "\"engine_silence_s\" in the config, 0 = wait forever). If it happens again, keep the end "
                            "of the engine log for the report.")
        self.ended = True                               # not alive from now: the next request restarts it
        proc = self.proc
        try:
            proc.kill()
            proc.wait(timeout=20)                       # restart() -> close() handles one that is still exiting
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineSilent(f"{what}; the server ended the engine")

    def close(self):
        """End the engine process: STOP + QUIT first (the engine frees its memory itself - unpinning tens of GB can take
        a while; ENGINE_QUIT_WAIT_S), then terminate (ENGINE_TERM_WAIT_S), then kill (ENGINE_KILL_WAIT_S).  Raises
        EngineStuck when it still runs after all three."""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    # STOP first: a request still running ends at its next step instead of being waited for (the
                    # engine reads stdin on its own thread; QUIT behind a long answer would sit in its queue)
                    self.proc.stdin.write("STOP\nQUIT\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                    self.proc.wait(timeout=ENGINE_QUIT_WAIT_S)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.proc.terminate()
                    self.proc.wait(timeout=ENGINE_TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=ENGINE_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                raise EngineStuck("the engine is still releasing its GPU memory and RAM; it did not exit after "
                                  "STOP, QUIT, terminate and kill") from None
        except OSError:
            pass
        finally:
            if self.proc.poll() is not None:
                if self.pump is not None and self.pump.ident is not None:   # a restart can stop before starting it
                    self.pump.join(timeout=2)
                try:
                    self.proc.stdin.close()             # buffered data may flush again after the engine has exited
                except OSError:
                    pass
                self.proc.stdout.close()
                if self.log not in (None, subprocess.DEVNULL):
                    self.log.close()
                self.proc = None
                self.ended = True
                self.progress, self.last = None, {}
                self.progress_ms, self.reused = 0, 0


IMAGE_URL_MAX = 32 << 20        # an image URL is read up to this (the web app attaches pictures of up to 20 MB)


def vision_start_error(line: str, log=None) -> str:
    """The message when the image encoder answered something other than READY.  #1445: a card in the Exclusive_Process
    compute mode (or one another process holds) refuses the encoder's CUDA context with "busy or unavailable"; that
    cause is named instead of leaving the user the first line of a traceback.  `log` is the encoder's stderr file, whose
    tail is searched too (the CUDA error is printed there)."""
    text = line.strip()
    tail = ""
    try:
        name = getattr(log, "name", None)
        if name:
            with open(name, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        pass
    if "busy or unavailable" in (text + tail).lower():
        return ("the vision encoder did not start: the GPU refused a second CUDA context (\"CUDA-capable device(s) "
                "is/are busy or unavailable\"). This is usually the compute mode Exclusive_Process: only one process "
                "may use the card, and the engine already holds it. Check with: nvidia-smi --query-gpu=compute_mode "
                "--format=csv; set it back with: sudo nvidia-smi -c DEFAULT (or run the encoder on another card). "
                "First line from the encoder: " + (text or "(none)"))
    return "the vision encoder did not start: " + text


class Vision:
    """The resident image encoder: `bin/sfnx-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    @staticmethod
    def work_dir() -> Path:
        """#480: the directory for the encoder's files, in the system temp directory.  The engine reads the
        embeddings file named in a `GENI` line, and both lines are split at spaces, so a temp directory whose path has
        a space (TMPDIR) is refused here rather than failing every picture later."""
        d = Path(tempfile.mkdtemp(prefix="sfnx-vision-"))
        if " " in str(d):
            d.rmdir()
            raise ValueError(f"the picture encoder's work directory {d} has a space in its path; the engine's GENI "
                             "line cannot name a file there: set TMPDIR to a path without spaces")
        return d

    def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = False):
        args = self.reader_args(cfg)
        self.dir = self.work_dir()
        self.spawn = (args, log, env)                   # to start it again after it died (restart)
        # the config's "cpu_exe": the CPU reader the GPU one falls back on when it does not start (the card is full or
        # busy); _start switches to it by itself
        if cfg.get("gpu") and cfg.get("cpu_exe"):
            cpu = {k: v for k, v in cfg.items() if k != "cpu_exe"} | {"exe": cfg["cpu_exe"], "gpu": False}
            self.fallback = (self.reader_args(cpu), log, env)
        else:
            self.fallback = None
        self.proc = None
        self.stopped = True
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}
        if not lazy:
            self._start()

    @staticmethod
    def reader_args(cfg: dict) -> list:
        absolute = lambda p: os.path.abspath(p) if os.path.dirname(str(p)) else p   # the encoder runs in its own dir (#480)  # noqa: E731
        args = [absolute(cfg["exe"]), "--mmproj", absolute(cfg["mmproj"]), "--model", absolute(cfg["model"])]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        if cfg.get("min_tokens"):                       # #767: mtmd's image_min_tokens (a hand-edited key)
            args += ["--min-tokens", str(cfg["min_tokens"])]
        return args

    def _ensure_dir(self):
        """#3370: a temp cleaner (systemd-tmpfiles ages out /tmp) can remove the work folder while the server runs; every
        later encode then failed for as long as the server lived.  Make it again; pictures cached in it are gone."""
        if not self.dir.is_dir():
            self.dir.mkdir(parents=True, exist_ok=True)
            self.cache.clear()

    def _start(self):
        """Start the reader.  When a GPU reader that has a CPU reader to fall back on does not start (the GPU runtime does not
        load, the card is not in its build, the card is busy), the CPU reader is started instead, for good, with one log line
        and no question."""
        try:
            self._start_reader()
        except (RuntimeError, OSError) as e:
            fallback = getattr(self, "fallback", None)
            if fallback is None:
                raise
            why = " ".join(str(e).split())[:240]
            print(f"[sfnx] the GPU picture reader did not start ({why}); pictures are read on the CPU", flush=True)
            self.spawn, self.fallback = fallback, None
            self._start_reader()

    def _start_reader(self):
        args, log, env = self.spawn
        self._ensure_dir()
        # cwd is the encoder's directory: the ENC lines name the files relative to it (#480: no space in the line)
        self.proc = popen(args, role="vision", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=log or subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, env=env,
                          cwd=self.dir)
        try:
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
        except RuntimeError as e:                       # silent for VISION_READY_S: it was ended, and it is not left half-started
            self.proc, self.stopped = None, True
            raise RuntimeError(str(e)) from None
        if not line.startswith("READY"):
            proc, self.proc = self.proc, None
            self.stopped = True
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise RuntimeError(vision_start_error(line, self.spawn[1]))
        self.stopped = False

    def _readline(self, timeout: float, what: str) -> str:
        """One line from the encoder, waiting at most `timeout` s (#1317).  On a timeout the encoder is killed, so the
        next request starts a fresh one, and the read raises RuntimeError instead of holding the request turn."""
        proc = self.proc
        got: queue.Queue = queue.Queue()

        def read():
            try:
                got.put(proc.stdout.readline())
            except (OSError, ValueError):               # killed under the read
                got.put("")

        threading.Thread(target=read, daemon=True).start()
        try:
            return got.get(timeout=timeout)
        except queue.Empty:
            try:
                proc.kill()
            except OSError:
                pass
            raise RuntimeError(f"{what}: it said nothing for {timeout:.0f} s") from None

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder process (its VRAM or RAM goes back); the encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again after an unload (or if it died); the cache of encoded images is kept."""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
        self.proc = None
        self.stopped = True
        self._start()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            try:
                return base64.b64decode(source.split(",", 1)[1])
            except (IndexError, ValueError) as e:                # no comma, or not base64
                raise ValueError(f"the image's data: URL could not be read ({e})") from None
        if source.startswith(("http://", "https://")):
            return Vision.download(source)
        path = source[7:] if source.startswith("file://") else source
        # file://host/x.png names another computer's file: refused before anything looks at the path
        host = path.split("/", 1)[0] if source.startswith("file://") else ""
        if host and host.lower() != "localhost":
            raise ValueError("an image file must be on this computer: file://host/... is not read")
        if path and os.path.isfile(path):
            try:
                return Path(path).read_bytes()
            except OSError as e:
                raise ValueError(f"the image could not be read from {path} ({e})") from None
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def download(url: str) -> bytes:
        """An image URL's bytes, at most IMAGE_URL_MAX of them: the whole response was read, so a huge or endless one
        filled the memory.  One that cannot be read is a ValueError (a 400 that says so), not a dropped connection."""
        too_big = f"the image URL's file is over {IMAGE_URL_MAX >> 20} MiB"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sfnx"}), timeout=60) as r:
                size = r.headers.get("Content-Length") or ""
                if size.isdigit() and int(size) > IMAGE_URL_MAX:
                    raise ValueError(too_big)
                data = r.read(IMAGE_URL_MAX + 1)
        except (OSError, HTTPException) as e:
            raise ValueError(f"the image URL could not be read: {e}") from None
        if len(data) > IMAGE_URL_MAX:
            raise ValueError(too_big)
        return data

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        (stb_image) ignores the tag.  False without Pillow, or when the tag is missing, 1 or unreadable (#1229)."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats bin/sfnx-vision's decoder (stb_image, through llama.cpp's mtmd helper) reads pass through; anything else is converted to PNG.
        A JPEG with an EXIF Orientation other than 1 is turned upright first (needs Pillow)."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied (#1229)
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode_all(self, sources: list, cancel=None) -> list[tuple[Path, int]]:
        """encode() for every image of one request: the files of this request's earlier images are never evicted to
        make room for a later one (#1072: with more than 64 unique images the first was deleted before the request
        had read it).  `cancel` (#1999): a request whose client left does not encode its next pictures."""
        out: list[tuple[Path, int]] = []
        for src in sources:
            if cancel is not None and cancel.is_set():
                raise ClientGone()
            out.append(self.encode(src, keep=[p for p, _ in out]))
        return out

    def encode(self, source: str | bytes, keep=()) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens).  `source`: what load() reads, or the image's bytes.
        `keep`: files the caller still needs (never evicted by this call)."""
        data = self.normalize(source if isinstance(source, bytes) else self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self._ensure_dir()
            if key in self.cache:
                self.cache[key] = self.cache.pop(key)                  # most recently used last
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            try:
                self.proc.stdin.write(f"ENC {img.name} {out.name}\n")      # relative to the encoder's cwd (#480)
                self.proc.stdin.flush()
                try:
                    line = self._readline(VISION_ENCODE_S, "the image could not be read").strip()
                except RuntimeError as e:                 # silent for VISION_ENCODE_S: the encoder was ended (#1317)
                    self.stopped = True                   # alive() is False, so the next request starts it again
                    raise ValueError(str(e)) from None
            finally:                                                   # #352: also when the encoder's pipe is gone
                img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):                               # oldest first, but never one still in use
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            self.stopped = True
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.stopped = True

    def shutdown(self):
        """#914: close() for good: the server ends, so the encoder's directory (one ~10 MB .sve per image, on a tmpfs
        /tmp that is RAM) goes with it instead of piling up with every start.  unload() keeps it for the restart."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": the cards the engine runs on ([0, 1, 2], "0,1,2" or one number), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def engine_silence_s(cfg: dict) -> float:
    """#481: the config's "engine_silence_s" - seconds an engine may print nothing during a request before the server
    ends it (default ENGINE_SILENCE_S; 0 = wait forever).  ValueError for anything but a number >= 0."""
    v = cfg.get("engine_silence_s")
    if v is None:
        return ENGINE_SILENCE_S
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(f'"engine_silence_s" must be a number of seconds >= 0 (0 = no limit), not {v!r}')
    return float(v)


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and --batch for "parallel" (--parallel).  A "vision" section needs
    --vision in the args (the engine reads pictures' embeddings only then): refused here, before the minutes of
    loading, rather than on the first picture."""
    args = list(cfg["args"])
    if isinstance(cfg.get("vision"), dict) and "--vision" not in args:
        raise SystemExit('[sfnx] config: a "vision" section needs "--vision" in "args" (the engine refuses every '
                         "picture without it)")
    return args + parallel_args(cfg, args)


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    """#465: "parallel": N (the --parallel option) - up to N requests decode together in the engine's batch slots
    (--batch N); more wait their turn.  1: one request at a time.  More than PARALLEL_MAX, or beside a --batch of the
    args' own, refuses to start."""
    n = cfg.get("parallel")
    if n is None:
        return []
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= PARALLEL_MAX:
        raise SystemExit(f'[sfnx] "parallel" must be a whole number of requests 1..{PARALLEL_MAX} (the engine\'s '
                         f"batch slots), not {n!r}")
    if "--batch" in args:
        raise SystemExit('[sfnx] "parallel" and the args\' own --batch both set the batch slots: keep one')
    return [] if n == 1 else ["--batch", str(n)]


def child_env(cfg: dict) -> dict:
    """The engine's environment: the config's cards as CUDA_VISIBLE_DEVICES (numbered as nvidia-smi numbers them:
    CUDA's own default order, fastest first, can number them otherwise), its "env", and its "lib_dirs" first on the
    library search path."""
    env = dict(os.environ)
    if gpu_list(cfg):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def vision_env(cfg: dict, env: dict) -> dict:
    """The image encoder's environment: the engine's, unless the config's vision section names its own "cuda_device"
    (numbered like nvidia-smi) - then the encoder runs on that card alone (#408)."""
    dev = (cfg.get("vision") or {}).get("cuda_device")
    if dev is None:
        return env
    return {**env, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(dev)}


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    ALWAYS = ()                                     # specials matched without parse_special (type 4, as <think>)

    @property
    def max_special_len(self):
        return max(len(s) for s in self.SPECIALS)

    @property
    def control_tokens(self):
        return [s for s in self.SPECIALS if s not in self.ALWAYS]

    def encode(self, text, parse_special=False, plain=()):
        return self.encode_marked(text, parse_special, plain)[0]

    def encode_marked(self, text, parse_special=False, plain=()):
        """encode() and its resume points (after each special), as strata_tokenizer's for PromptEncoder."""
        out, marks, i = [], [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(
                        a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    marks.append((i, len(out)))
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out, marks

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


OPENAI_MAX_STOP = 4                                  # OpenAI's limit on `stop`


def stop_strings(req: dict) -> list[str]:
    """The request's stop strings: OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences`
    (a list).  Empty strings are left out.  A bad value is a ValueError, so a 400 before anything is generated."""
    req = req or {}
    if req.get("stop") is not None:
        stop = req["stop"]
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > OPENAI_MAX_STOP:
            raise ValueError(f"stop takes at most {OPENAI_MAX_STOP} strings")
    elif req.get("stop_sequences") is not None:
        stop = req["stop_sequences"]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop_sequences must be a list of strings")
    else:
        return []
    return [x for x in stop if x]


class StopMatcher:
    """Cuts the answer's text at the first stop string.  Text that could still be the start of a stop string is held
    back (only that tail, never more), so a stop string split across tokens or chunks is still found and never sent."""

    def __init__(self, stops: list[str]):
        self.stops, self.held, self.hit = stops, "", None

    def push(self, text: str) -> str:
        """-> the text that is safe to send now; after a match, self.hit is the stop string and nothing more comes."""
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0                                      # the longest end of buf that a stop string starts with
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = True):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.literals = literal_tags(getattr(tokenizer, "control_tokens", ()))   # texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # default: clamp the output cap to the room left; False (--no-fit-max-tokens) = 400

        self.answer_room = True                       # 0.1.45: see ANSWER_ROOM_MIN_CAP (the config's "answer_room": false turns it off)
        self.reasoning_close_retry = False            # #1053 (opt-in): close the thinking once when a reply ends inside it

        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.model_arch = None                        # "qwen4exp" with the engine; None (the mock) = no model-card defaults
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.config_effort = None                     # #1641: the config's "reasoning_effort": the level for requests with none
        self.fifo = threading.Lock()

        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.conv_log = ConvCacheLog()                  # #596: the parked conversations, from the engine's log
        # DNS rebinding: extra Host names this server answers to (the config's allowed_hosts, $STRATA_ALLOWED_HOSTS;
        # "*" = any), and every name it answers to, which serve() works out from the address it listens on
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        # "parallel" (the engine's batch slots): every running request's own status and rate window; self.status
        # then says busy while any runs and shows the newest one
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        # #332: the API request monitor (/api-monitor and the Monitor tab's request details) keeps the last 100
        # requests' prompts and answers in memory, each text cut to its start and end (front/bounded_text.py). On by
        # default: this box serves its LAN without an API key by choice, and the journal holds raw replies already;
        # "api_monitor": false (or --no-api-monitor) keeps nothing
        self.api_monitor = True
        self.api_requests = collections.deque(maxlen=100)  # bounded I/O in memory; no headers or API keys
        self.request_trace = threading.local()
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0,
                       "drafts_offered": 0, "drafts_accepted": 0}   # #457: the MTP drafts, summed where reported
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)

        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()

        self.reasoning_budget_tokens = 0                 # #123: the config's default thinking budget (0: none)
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply (0: off)
        self.anthropic_think_unasked = True               # #278: "anthropic_thinking": "on_request" -> False

        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))
        # a stop token's own spelling, encoded as plain text (see LITERAL_STOPS_PER_REPLY)
        self.stop_text = {tokenizer.encode(m, parse_special=True)[0]: tokenizer.encode(m, parse_special=False)
                          for m in (IM_END, "<|endoftext|>")}
        # #567: a prompt re-encodes only what follows the last special token it shares with a recent prompt (the
        # same ids as a full encode: tools/strata_tokenizer.py PromptEncoder).  Tokenizers without resume points
        # encode in full.
        self.prompts = None
        if hasattr(tokenizer, "encode_marked") and hasattr(tokenizer, "max_special_len"):
            from strata_tokenizer import PromptEncoder
            self.prompts = PromptEncoder(tokenizer)

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def reported_ctx(self) -> int:
        """The context the endpoints report (/v1/status, /v1/models, /props, /health, /slots, /metrics): the engine's,
        or the last one it reported while it restarts (max_context is 0 until READY, and a client that sizes its
        prompt from it would send nothing, #351).  Reporting only: requests are checked in prepare()."""
        return int(getattr(self.engine, "max_context", 0) or getattr(self.engine, "known_ctx", 0) or 0)

    def reasoning_budget(self, req) -> int | None:
        """#123: the most tokens this request may think, or None: the request's `reasoning_budget_tokens` (or, as
        llama.cpp reads it, `thinking_budget_tokens`), else the config's.  0 (or less) means no budget, so a request can
        turn a configured one off.  ValueError (a 400) for anything that is not a whole number."""
        req = req if isinstance(req, dict) else {}
        value = req.get("reasoning_budget_tokens")
        if value is None:                                # a null does not hide the other name
            value = req.get("thinking_budget_tokens")
        if value is None:
            value = self.reasoning_budget_tokens
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens={value!r}: expected a whole number of tokens (0: no budget)")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        """The encoder should run and does not (the engine's load starts it again)."""
        return self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()

    def ensure_loaded(self):
        """Start the engine (and the picture encoder) again when it is not running: it died (issue #27), or a restart
        failed.  The caller holds self.fifo."""
        if self.loaded() and not self._vision_down():
            return
        vision_started = False
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[sfnx] starting the vision encoder ...", flush=True)   # the engine sizes its cache
            try:
                self.vision.restart()
            except Exception as e:
                raise EngineDied(f"the vision encoder failed to start: {e}") from e
            vision_started = True
        if self.loaded():
            return
        code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
        print(f"[sfnx] the engine had stopped (exit code {code}); starting it again (a minute or two) ...", flush=True)
        try:
            self.engine.restart()
        except Exception as e:
            if vision_started:
                self.vision.unload()
            # the program could not be started at all (missing, not executable - #735): the request path answers
            # EngineDied with a 503, but takes a bare OSError for a client that went away (#2496)
            if isinstance(e, OSError):
                raise EngineDied(f"the engine could not be started: {e}") from e
            raise
        print("[sfnx] the engine is running again", flush=True)

    def _say_died(self, e: Exception) -> None:
        """The journal's line for an engine that died (or was ended, #481) in the middle of a request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        print(f"[sfnx] {e}. {note} The next request starts the engine again."
              f"{' Its log: ' + log if log else ''}", flush=True)

    def load(self):
        """Every generation request, before its prompt is planned: start the engine again if it died."""
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        with self.status_lock:                          # #1603: waiting for the fifo to start the engine is queued, not nothing
            self.status["queued"] += 1
        counted = True
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                counted = False
                loading = time.perf_counter()
                if trace is not None:
                    with self.status_lock:
                        trace["queue_s"] += round(loading - waiting, 3)
                        trace["state"] = "loading"
                try:
                    self.ensure_loaded()
                finally:
                    if trace is not None:
                        with self.status_lock:
                            trace["load_s"] += round(time.perf_counter() - loading, 3)
                            trace["state"] = "queued"
        finally:
            if counted:
                with self.status_lock:
                    self.status["queued"] -= 1

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[sfnx] could not save the shared settings: {e}", flush=True)
        return self.shared

    def effort_default(self) -> str | None:
        """The thinking level of a request that names none: the Chat settings shared with apps first, then the config's
        "reasoning_effort" (#1641).  Every API reads it here (the Responses API did not: 0.1.44.1)."""
        return self.shared.get("reasoning_effort") or self.config_effort

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        effort = self.effort_default()
        if not s and not effort:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if req.get("reasoning_effort") in (None, "") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (front/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from front.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None))
            if getattr(self, "engine", None) is not None and hasattr(self.engine, "gpu_busy"):
                # #1317: the quick frozen-engine check does not end an engine whose GPU is busy (a long prompt chunk on a
                # slow card keeps the GPU at work while the host thread sleeps)
                self.engine.gpu_busy = lambda: (self.telemetry.snapshot()["now"].get("gpu_util") or 0) >= 10

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating.  With parallel requests:
        the sum of theirs."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """The prompt read's rate over newly read tokens (the cached prefix excluded) for the history series: the
        engine's own figure when it reports one, else the rate its PP lines arrive at; 0.0 while nothing is read."""
        if not self._reading_now():
            return 0.0
        rate = self._prefill_now()
        return 0.0 if rate is None else rate

    def _reading_now(self) -> bool:
        """Whether a prompt is being read: a batch slot reading, or (no slot busy) a request without its first token
        - the solo path.  One read at a time, so the engine's one read meter is its rate."""
        slots = self.engine.slots_view() if getattr(self.engine, "batch", 0) and hasattr(self.engine, "slots_view") \
            else []
        if any(x["state"] == "reading" for x in slots):
            return True
        if any(x["state"] != "idle" for x in slots):
            return False
        with self.status_lock:
            many = [st for st, _ in self.live_reqs.values()] or [self.status]
            return any(st.get("busy") and st.get("first_token") is None for st in many)

    def _prefill_now(self) -> float | None:
        """The read in progress' tok/s: the engine's PP figure when it sends one (`PP done total ms mean`), else its
        read meter's (bin/sfnx sends `PP done total` only); None before the read's second chunk."""
        reported = getattr(self.engine, "prefill_tok_s_mean", None)
        if reported is not None:
            return reported
        meter = getattr(self.engine, "read_meter", None)
        return meter.rate() if meter is not None else None

    def begin_request(self, path, req):
        """#332: a monitor record for this request, or None when the monitor is off (nothing is kept then)."""
        if not self.api_monitor:
            return None
        raw = json.dumps(req, ensure_ascii=False, indent=2)
        record = {"id": uuid.uuid4().hex[:12], "path": path, "model": req.get("model") or self.model,
                  "started_at": time.time(), "state": "queued", "stream": bool(req.get("stream")),
                  "response_format": (req.get("response_format") or {}).get("type")
                  if isinstance(req.get("response_format"), dict) else None,
                  "input": BoundedText(raw), "output": BoundedText(), "reasoning": BoundedText(),
                  "queue_s": 0.0, "load_s": 0.0, "first_token_s": None,
                  "_clock": time.perf_counter()}
        with self.status_lock:
            self.api_requests.append(record)
        self.request_trace.record = record
        return record

    def request_records(self, request_id=None):
        with self.status_lock:
            records = list(self.api_requests)
            if request_id:
                record = next((r for r in records if r["id"] == request_id), None)
                if record is None:
                    return None
                out = {}
                for k, v in record.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, BoundedText):          # the text, its full length, and whether its middle went
                        out.update({k: v.text(), k + "_chars": len(v), k + "_truncated": v.truncated})
                    else:
                        out[k] = v
                return out
            return [{k: v for k, v in r.items()
                     if k not in ("input", "output", "reasoning", "response") and not k.startswith("_")}
                    | {"wallclock_s": r.get("wallclock_s", round(time.perf_counter() - r["_clock"], 3)),
                       "input_chars": len(r["input"]), "output_chars": len(r["output"])}
                    for r in reversed(records)]

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel requests: the newest one's status
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
            totals = dict(self.totals)
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and s.get("first_token") is None:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        elif not self.loaded():
            state = "stopped"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0), "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": self._prefill_now() if self._reading_now() else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        par = int(getattr(self.engine, "batch", 0) or 0)
        if par:                                         # #465: the requests running together, slot by slot
            with self.status_lock:
                running = len(self.live_reqs)
            slots = self.engine.slots_view() if hasattr(self.engine, "slots_view") else []
            with self.status_lock:
                by_id = {st["request_id"]: (dict(st), list(rt)) for st, rt in self.live_reqs.values() if st.get("request_id")}
            for x in slots:                             # the request in the slot: its live rate and its details
                mine = by_id.get(x.get("request_id"))
                if mine is None:
                    continue
                x["record"] = mine[0].get("record")
                if x["state"] == "decoding":
                    x["tok_s_mean"], x["tok_s"] = x["tok_s"], round(self._rate_of(*mine), 1)
            # the requests in flight that are not in a slot: the one alone on the solo path, or waiting their turn
            in_slots = sum(1 for x in slots if x["state"] != "idle")
            live.update(parallel=par, running=running, slots=slots, outside_slots=max(0, running - in_slots),
                        waiting=int(getattr(self.engine, "waiting", 0) or 0),
                        admitting=int(getattr(self.engine, "admitting", 0) or 0))
            if running and state == "idle":
                live["state"] = "generating"

        engine = {"model": self.model, "max_context": self.reported_ctx(), "images": self.vision is not None,
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        parked = self.conv_log.poll(getattr(self.engine, "log_path", None), getattr(self.engine, "log_start", None))
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "conversation_cache": conversation_cache_view(engine, hist, totals, parked),
                "last_read": last_read(hist),
                "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now}

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
            running = len(self.live_reqs) if self.live_reqs else int(bool(s.get("busy")))
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        ctx = self.reported_ctx()
        images = self.vision is not None
        return {
            "service": "sfnx", "model": self.model,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            # one request at a time (more wait their turn), or "parallel": N batch slots (#465)
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": getattr(self, "requested_slots", max(1, int(getattr(self.engine, "batch", 0) or 0)))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},

            "activity": {"requests": totals["requests"] + running,
                         "in_flight": running + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def encode_prompt(self, messages, tools, kwargs, store: bool = True) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens.  `store=False` (a token count) reuses the kept
        prompts without adding or replacing one."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.template.render(marked, tools=marked_tools, **kwargs)
        plain = ()
        if changed:
            prompt, plain = unmark_think_literals(prompt, self.literals)
        if self.prompts is not None:            # #567: only what follows the last special shared with a recent prompt
            return self.prompts.encode(prompt, plain, store)
        return self.tok.encode(prompt, parse_special=True, plain=plain) if plain else \
            self.tok.encode(prompt, parse_special=True)

    @contextlib.contextmanager
    def _turn(self, cancel=None):
        """The engine's single turn (self.fifo) for an image encode.  #1999: a client that leaves while this request
        waits for it gives its place up (ClientGone), as a queued generation does (#1619)."""
        if cancel is None:
            self.fifo.acquire()
        else:
            while not self.fifo.acquire(timeout=0.25):
                if cancel.is_set():
                    raise ClientGone()
        try:
            yield
        finally:
            self.fifo.release()

    def _note_unreadable_tool_images(self, messages, cancel=None):
        """A picture a tool returned (Claude Code's Read of an image file) that this server cannot read - it has no
        image encoder, or the encoder refuses that picture - becomes a short note in its place instead of a 400.
        Clients resend the whole history every turn, so one unreadable tool picture would fail every later request of
        the conversation; a picture the user sends keeps the 400.  A picture that encodes stays in the cache, so the
        prompt's own encode below finds it."""
        fetched = {}                     # a URL's bytes, so prepare() does not download the same picture again
        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool" or not isinstance(content, list):
                continue
            for n, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                why = "this server has no image encoder" if self.vision is None else None
                if why is None:
                    src = item["source"]
                    try:
                        if cancel is not None and cancel.is_set():
                            raise ClientGone()
                        if src.startswith(("http://", "https://")):
                            src = fetched[item["source"]] = Vision.download(src)   # outside the FIFO, as in prepare()
                        with self._turn(cancel):  # the encoder takes its turn with the requests (see below)
                            self.vision.encode(src)
                    except (ValueError, OSError) as e:
                        why = str(e)
                if why is not None:
                    print(f"[sfnx] a picture a tool returned was left out of the prompt: {why}", flush=True)
                    content[n] = {"type": "text", "text": f"[image omitted: {why}]"}
        return fetched

    def forget_prompt(self, ids):
        if self.prompts is not None:
            self.prompts.forget(ids)

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, cancel=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        # `cancel` (#1999): set when the client hangs up; the downloads, the wait for the turn and the encodes stop
        # between pictures and raise ClientGone
        fetched = self._note_unreadable_tool_images(messages, cancel)
        ids = encoded_ids = self.encode_prompt(messages, tools, kwargs)
        if force and kwargs.get("enable_thinking", True) is False:
            ids = ids + self.tok.encode(force, parse_special=True)
        self.embeddings.path = None
        self.embeddings.keys = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (the config's \"vision\" section "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # An image URL is downloaded first, outside the FIFO: under it, a slow server held every other request
            # for as long as urlopen waited.
            def fetch(src):
                if cancel is not None and cancel.is_set():
                    raise ClientGone()
                return fetched[src] if src in fetched else Vision.download(src) \
                    if src.startswith(("http://", "https://")) else src
            images = [fetch(src) for src in images]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self._turn(cancel):
                encoded = (self.vision.encode_all(images, cancel) if hasattr(self.vision, "encode_all")
                           else [self.vision.encode(src) for src in images])
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more pairs than images: text parts that cut both markers apart (the literal marks keep whole ones text)
                    if k == len(encoded):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # #344: (re)starting, not a prompt that is too long
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: check against the last known context, so the request reaches
            # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                # A prompt refused for the context is not kept (#567): the client cannot resend it unchanged.  The
                # other refusals (engine starting, images) keep it: the same prompt comes back and its encode is reused.
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "server's config to shorten it to the room left (#545)")
            print(f"[sfnx] max_tokens {max_new} shortened to {max(1, room)}: the prompt ({len(ids)} tokens) leaves that much "
                  f"of the context ({ctx}); \"fit_max_tokens\": false answers 400 instead", flush=True)
            max_new = max(1, room)          # a shorter completion beats a 400 (the default)
        if images:
            # The request's images in one file for GENI (~10 MB a picture), written once nothing above refuses the
            # request: one refused after it (the engine starting, no room) left it in the vision directory for good,
            # one more for every retry of a 503.  run() deletes it; drop_embeddings() if run() never starts.
            combined = combined_embeddings_path(self.vision.dir, sum(Path(p).stat().st_size for p, _ in encoded))
            self.embeddings.path = combined             # first, so a half-written one is found as well
            # #2228: each picture's key is its normalized bytes' hash (the encoder's file name), so the engine's
            # conversation cache matches a picture by what it is, not by its embedding floats: a picture encoded again
            # (an encoder restart, the cache's eviction) with other bits still matches its checkpoints
            self.embeddings.keys = [Path(p).stem[:16] for p, _ in encoded]
            write_temporary(combined, [p for p, _ in encoded])

        return Prompt(ids, self.late_notes_start(messages, ids)), kwargs.get("enable_thinking", True) is not False, \
            max_new

    def late_notes_start(self, messages, ids):
        """-> where the closing run of LATE_NOTE messages starts in ids, or None without one."""
        notes = 0
        while notes < len(messages) and messages[-1 - notes].get(LATE_NOTE):
            notes += 1
        if not notes:
            return None
        im_start = self.tok.encode(IM_START, parse_special=True)[0]
        starts = [i for i, t in enumerate(ids) if t == im_start]
        # each note opens one turn, and the reply's header opens the last
        if len(starts) < notes + 1:
            raise RuntimeError(f"the prompt has {len(starts)} turn starts for {notes} closing notes and the reply")
        return starts[-1 - notes]

    def drop_embeddings(self) -> None:
        """Delete the combined image file prepare() wrote when no run() took it over (a run deletes its own as it
        ends): a request can end before its answer starts - its client gone, anything raised after prepare() - and
        then nothing else would.  The handler calls this when a request is done."""
        path, self.embeddings.path = getattr(self.embeddings, "path", None), None
        if path is not None:
            Path(path).unlink(missing_ok=True)

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the journal every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[sfnx] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])
            # #984: while a thinking budget applies, the thinking line shows it, not the request's max_tokens
            limit = (f"budget {s['thinking_budget']} (max {s.get('max_tokens')})"
                     if s["phase"] == "thinking" and s.get("thinking_budget") else f"max {s.get('max_tokens')}")
            print(f"[sfnx] {s['phase']}: {s['generated']} of {limit} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
            declared_only=False) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..}).
        `force` (forced_call): the opening of the call the reply must make - see prepare()."""
        # the model card's sampling for this thinking state, then the config's block, then the Chat settings shared with
        # apps; the request's own fields win over all of them
        defaults = {**model_card_sampling(self.model_arch, thinking), **self.sampling_defaults, **self.shared}
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        # #123: read after the merge, so a budget shared through POST /settings is seen like the other keys
        budget = self.reasoning_budget(sampling) if thinking else None
        if thinking and not budget and self.answer_room and max_new and max_new >= ANSWER_ROOM_MIN_CAP:
            # 0.1.45: max_tokens counts the thinking too.  A client that caps the reply at 8192 (the desktop agent's
            # harness sends max_completion_tokens: 8192 on every request) and a task the model thinks long about got
            # the cut-off in the middle of the thinking and NO answer.  So thinking is closed early enough to leave
            # the answer its quarter of the cap (the budget's wrap-up: a short closing sentence, </think>, the answer).
            # A request with a budget of its own, or max_tokens below ANSWER_ROOM_MIN_CAP, is left as it was.
            budget = max_new - min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if max_new <= 32768: print(f"[sfnx] thinking limited to {budget} of max_tokens {max_new} so the answer has room "
                  "(\"answer_room\": false in the config turns this off)", flush=True)
        if budget and max_new and max_new > 0:
            # #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking, so
            # it has no answer.  Close the thinking early enough to leave room for one.
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if budget > max_new - reserve:
                print(f"[sfnx] thinking capped at {max_new - reserve} of max_tokens {max_new} to leave room for "
                      f"the answer (budget {budget})", flush=True)
                budget = max(1, max_new - reserve)
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True,
                              declared_only=declared_only)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # #606: the current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        opening = []                                    # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            opening, force = parser.feed(force), None
        tail = ""                                       # the last characters written (the newlines before a call)
        answered, close_retried = False, False          # #1053: content or a call came out; the thinking was closed
        timings, before = None, None                    # this request's timings; the engine's `last` before it

        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        emb, self.embeddings.path = getattr(self.embeddings, "path", None), None   # this run's to delete now
        img_keys, self.embeddings.keys = getattr(self.embeddings, "keys", None), None
        if emb and img_keys:
            sampling = {**(sampling or {}), "strata_img_keys": img_keys}   # #2228: sent beside the embeddings file
        # Only a DONE line replaces engine.last, so a request that died, errored or was disconnected must not have
        # the PREVIOUS request's decode figures recorded as its own: each pass keeps its own DONE (a segment) if
        # one arrived.  A reasoning-budget continuation is another native generation, not another API request,
        # so the generated reasoning must not become this request's cached input (#615).
        segments = []
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        first_engine_token_s = None  # this request, from entering the engine queue; includes queue/prefill
        # #465: with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        request_id = uuid.uuid4().hex[:12]
        setattr(cancel, "_strata_request_id", request_id)
        with self.status_lock:
            self.status["queued"] += 1
        queued_counted = True
        try:
            # --batch: the engine runs several requests at once (StrataEngine.generate_batched orders them)
            turn = contextlib.ExitStack()
            if not par:
                # #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S, so a client with
                # a stream idle timeout does not abort a healthy queued request (and its retry read the prompt again);
                # a client that left while queued gives its place up instead of taking the turn only to cancel there
                got = False
                while not got:
                    if cancel.is_set():
                        raise QueueCancelled("the client disconnected while waiting for the engine")
                    got = self.fifo.acquire(timeout=QUEUE_BEAT_S)
                    if got:
                        if cancel.is_set():
                            self.fifo.release()
                            raise QueueCancelled("the client disconnected while waiting for the engine")
                        turn.callback(self.fifo.release)
                    else:
                        yield "ping", None
            with turn:
                try:
                    with self.status_lock:
                        if trace is not None:
                            trace["queue_s"] += round(time.perf_counter() - waiting, 3)
                            trace["state"] = "generating"
                        if not par:
                            self.status["queued"] -= 1
                            queued_counted = False
                    # issue #27: it died in an earlier request - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that).
                    # #1603: a parallel request stays counted as queued until it is registered as running below -
                    # waiting here for the fifo (an image encode) left it neither - and one on a
                    # loaded engine has nothing to start, so it does not wait for the fifo at all
                    if not par:
                        self.ensure_loaded()
                    elif not (self.loaded() and not self._vision_down()):
                        with self.fifo:
                            self.ensure_loaded()
                    with self.status_lock:
                        if par:
                            self.status["queued"] -= 1
                            queued_counted = False
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, thinking_budget=budget,
                                  request_id=request_id, record=trace["id"] if trace is not None else None)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # thought: the reasoning tokens so far (the budget's count)
                    literal_left = LITERAL_STOPS_PER_REPLY if os.environ.get("STRATA_LITERAL_STOP", "1") != "0" else 0
                    written = ""                        # the decoded reply so far (open code span or fence?)
                    for ev in opening:
                        yield "event", ev
                    while True:
                        segment_before = getattr(self.engine, "last", None)
                        gen = self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, sampling, cancel)

                        seg, wrap, leaving = [], False, False   # this pass's tokens; the budget is reached; closed
                        literal = None                  # a stop token that quoted the marker: its text, to go on
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # A value the frontend holds back (a tool call's array or object argument is sent
                            # whole, once complete) can keep the stream silent for minutes while tokens are
                            # generated; a client's idle timeout then ends the request.  Send the same keep-alive
                            # the engine's heartbeat sends when nothing has gone out for KEEPALIVE_S.
                            last_out = time.perf_counter()
                            for t in gen:
                                if t is None:               # heartbeat while the engine is quiet
                                    last_print = self._progress(last_print, st=st)
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                    continue
                                n += 1
                                if first_engine_token_s is None:
                                    first_engine_token_s = round(time.perf_counter() - waiting, 3)
                                if trace is not None and trace["first_token_s"] is None:
                                    with self.status_lock:
                                        trace["first_token_s"] = round(time.perf_counter() - trace["_clock"], 3)
                                if t in self.stop_ids:
                                    if force and parser.state == "reasoning" and parser.buf == THINK_END:
                                        opens = True    # #537: the held </think> was the end: the call opens there
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    quoting = (parser._in_code() if parser.state == "reasoning" else
                                               parser.state == "content" and in_open_code(written.rsplit("</think>", 1)[-1]))
                                    if literal_left and quoting and not parser.buf and not detok.pending() \
                                            and not parser.pending:
                                        literal = self.stop_text[t]
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += parser.state in ("reasoning", "rcall")
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # #606: a degenerate output, not an answer: end it here
                                    break
                                piece = detok.push(t)
                                tail = (tail + piece)[-2:]
                                written += piece
                                evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    if ev.kind in ("content", "tool_start", "tool_call"):
                                        answered = True
                                    yield "event", ev
                                if evs:
                                    last_out = time.perf_counter()
                                elif time.perf_counter() - last_out >= KEEPALIVE_S:
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                if stops is not None and stops.hit is not None:
                                    finish = "stop"         # gen.close() below STOPs the engine, as for a stop token
                                    break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # at a clean point: no tag held back, no character split across tokens
                                    if thought >= budget and not parser.buf and not detok.pending():
                                        wrap = True
                                        break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and not parser.buf and not detok.pending():
                                        opens = True
                                        break
                        except EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except ValueError as e:             # the engine's ERR line (it may have ended after it)
                            finish = "error"
                            print(f"[sfnx] the engine reported an error: {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away: an engine that never acknowledges
                            leaving = True                  # the STOP below is ended, but no error replaces this
                            raise
                        finally:
                            try:
                                gen.close()             # STOP+drain to THIS request's DONE while still holding the
                                #                         fifo, so a stop-token break can't leave the shared engine
                                #                         queue mid-drain for the next request to read as its own DONE
                            except EngineSilent as e:   # #481: the STOP was never acknowledged: the engine is ended
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                            segment_done = getattr(self.engine, "last", None)
                            if segment_done is not None and segment_done is not segment_before:
                                segments.append(dict(segment_done))
                        if literal is not None and not cancel.is_set() and max_new - n - len(literal) >= 1:
                            literal_left -= 1
                            print("[sfnx] a stop token inside an open code span: the model quoted the marker; "
                                  "writing it as text and going on", flush=True)
                            for t in literal:
                                n += 1
                                raw_ids.append(t)
                                thinking_n += parser.state in ("reasoning", "rcall")
                                piece = detok.push(t)
                                written += piece
                                evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                for ev in evs:
                                    yield "event", ev
                            prompt = prompt + seg + literal
                            finish = "length"           # overwritten by how the continued pass ends
                            continue
                        if (self.reasoning_close_retry and thinking and finish == "stop" and not close_retried
                                and not answered and not wrap and not opens and parser.state == "reasoning"
                                and parser.buf != THINK_END         # #537: not after a held </think>, the end
                                and (stops is None or stops.hit is None) and not cancel.is_set()):
                            # #1053: the model wrote its reasoning and stopped before </think>: the client would get
                            # an empty answer.  Close the thinking once and let it answer.
                            close_retried = True
                            extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                            if max_new - n - len(extra) >= 1:
                                print("[sfnx] the reply ended inside its thinking with no answer: closing the "
                                      "thinking once and continuing (reasoning_close_retry)", flush=True)
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    thinking_n += parser.state in ("reasoning", "rcall")
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                                prompt = prompt + seg + extra
                                finish = "length"
                                continue
                        if not (wrap or opens) or cancel.is_set():
                            break
                        # #123: the thinking reached reasoning_budget_tokens.  Close it the way the model would (a
                        # short wrap-up and </think>) and let it answer: the next pass's prompt is this one plus what
                        # was generated plus the wrap-up, so the engine continues from the prefix it already holds.
                        # A forced call (tool_choice) is opened the same way: after the wrap-up, or where the
                        # thinking ended, after the blank line the template puts before a call.
                        if wrap:
                            budget = st["thinking_budget"] = None
                            text = REASONING_WRAP_UP + (force or "")
                        else:
                            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
                        force = None
                        extra = self.tok.encode(text, parse_special=True)
                        if max_new - n - len(extra) < 1:
                            break                       # no room left to answer: "length", as without a budget
                        if wrap:
                            print(f"[sfnx] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state in ("reasoning", "rcall")
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = prompt + seg + extra
                    if cancel.is_set():
                        finish = "cancel"
                    elif repeated:
                        print(f"[sfnx] the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times in a row: ended as \"length\" (repeat_stop_tokens in the config; 0 "
                              "turns this off). If a new request with a short prompt does the same, restart the "
                              "service and report it (#606)", flush=True)
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # #266: settle this request's status, history and totals while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    with self.status_lock:
                        if st.get("busy"):
                            last = request_stats(segments)          # only this request's own DONEs count
                            started = st.get("started", time.time())
                            seen = prompt_tokens_seen(len(ids), last)   # #471: < len(ids) when cancelled mid-read
                            self.history.append({
                                "id": request_id,
                                # the monitor's record of this request (prompt and answer), when it keeps one
                                "record": trace["id"] if trace is not None else None,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "first_engine_token_s": first_engine_token_s,
                                "prompt_total": len(ids),           # the request's whole prompt
                                "engine_generated": last.get("generated"),
                                "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                # #457: the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            if trace is not None:               # the request details' timings, as the table has them
                                trace["engine"] = dict(self.history[-1])
                            t = self.totals
                            t["requests"] += 1
                            t["prompt_tokens"] += seen
                            t["reused"] += last.get("reused") or 0
                            t["output_tokens"] += n
                            t["prompt_ms"] += last.get("prompt_ms") or 0.0
                            t["decode_ms"] += last.get("decode_ms") or 0.0
                            t["drafts_offered"] += last.get("drafts_offered") or 0
                            t["drafts_accepted"] += last.get("drafts_accepted") or 0
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            rate = answer_speed(last, n, now - ft if ft else None)
                            rate_txt = f"{rate:.1f} tok/s" if rate is not None else "no speed"
                            print(f"[sfnx] done: {n} tokens in {el:.0f} s ({rate_txt}) "
                                  f"({finish}, cancel={cancel.is_set()})", flush=True)
                            if finish == "length" and parser.state in ("reasoning", "rcall"):   # #530
                                print("[sfnx] the reply reached max tokens while still thinking, so it has no "
                                      "answer: a thinking budget (reasoning_budget_tokens, in the request or in "
                                      "the config for every request) leaves room to answer", flush=True)
                            if os.environ.get("STRATA_DEBUG") and raw_ids:
                                print(f"[sfnx] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # #212: the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        except QueueCancelled:
            finish = "cancel"
            yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                           "reasoning_tokens": 0}
            return
        finally:
            with self.status_lock:
                if queued_counted:
                    self.status["queued"] -= 1
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish(finish)):
            yield "event", ev
        if parser.rescued or parser.refused:
            self.totals["tool_calls_from_reasoning"] = self.totals.get("tool_calls_from_reasoning", 0) + parser.rescued
            print(f"[sfnx] tool calls inside the thinking: {parser.rescued} read as calls, {parser.refused} kept as "
                  f"reasoning (quoted, or the turn did not end on a stop)", flush=True)
        if parser.unread:
            print(f"[sfnx] {parser.unread} <tool_call> block(s) in the answer did not read as a call of a declared "
                  "tool and went back as text (a name the request did not declare, or a form the template does "
                  "not use)", flush=True)
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string after all
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        yield "done", done


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """#471: the prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way: then its prompt_tokens field, which is as far as the read got (reused ones included, 0
    when it never started).  /metrics' history and totals count these, so a cancelled read is neither recorded as the
    whole prompt nor given a rate from tokens it never read.  No DONE (MockEngine): the whole prompt."""
    if last.get("finish") != "cancel":
        return prompt_tokens
    held = last.get("prompt_tokens")
    return prompt_tokens if held is None else min(prompt_tokens, int(held))


def carry_leg(carry: dict, leg: dict) -> None:
    """#1869: a request that decoded in a batch slot and went back to the solo path ends with the solo path's DONE, which
    replaces the engine's figures: keep the slot leg's decode time, tokens and prompt figures to add to it."""
    carry["decode_ms"] = carry.get("decode_ms", 0.0) + float(leg.get("decode_ms") or 0.0)
    carry["generated"] = carry.get("generated", 0) + int(leg.get("generated") or 0)
    carry.setdefault("prompt_ms", float(leg.get("prompt_ms") or 0.0))


def add_carry(carry: dict, last: dict) -> dict:
    """The request's totals: the earlier legs' decode time and tokens plus the final one's; the prompt figures of the leg
    that read the prompt (the solo continuation reads next to nothing)."""
    return {**last, "decode_ms": float(last.get("decode_ms") or 0.0) + carry.get("decode_ms", 0.0),
            "generated": int(last.get("generated") or 0) + carry.get("generated", 0),
            "prompt_ms": max(float(last.get("prompt_ms") or 0.0), carry.get("prompt_ms", 0.0))}


def answer_speed(last: dict, generated: int, wall_s: float | None = None) -> float | None:
    """The one definition of "answer speed": every token the model produced (thinking and answer, the tokens it kept - not
    the MTP drafts it threw away) over the decode time, first generated token to last.  The engine's DONE line holds both
    numbers; /metrics (`decode_tok_s`), the timings (`predicted_per_second`), the Terminal monitor and the app all use them.
    Only an engine that sent no DONE falls back to the server's own clock from the first token, and a clock younger than
    RATE_MIN_SPAN_S says nothing (it printed 1000000 tok/s for a one-token reply).  Prompt reading is never in it."""
    produced, decode_ms = last.get("generated"), last.get("decode_ms")
    if produced and decode_ms and decode_ms > 0:
        return round(produced / (decode_ms / 1000), 1)
    if generated and wall_s is not None and wall_s >= RATE_MIN_SPAN_S:
        return round(generated / wall_s, 1)
    return None


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def request_stats(segments: list[dict]) -> dict:
    """Keep input/cache accounting at the original API boundary and total native work across continuations."""
    if not segments:
        return {}
    result = dict(segments[-1])
    if "reused" in segments[0]:
        result["reused"] = segments[0]["reused"]
    else:
        result.pop("reused", None)
    for key in ("prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted"):
        if any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[sfnx] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None):
    """The chat completion's chunks from Service.run's events.  `force`: see Service.run."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came (#211)
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True):
        if kind == "ping":                         # nothing to say yet: an SSE comment keeps the stream alive
            yield None
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            # a streamed call without its final tool_call is one the output ended inside: not "tool_calls" (#211)
            whole = calls and streamed.keys() <= finished
            finish = "tool_calls" if whole and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            yield last


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last = [], [], {}, None
    for c in chunks:
        if c is None:                              # a heartbeat
            continue

        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls

    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


def structured_chunks(chunks, validator):
    """Buffer structured streams so a client never receives unvalidated content. The reasoning is not validated, so
    it streams as it comes: a long think shows progress instead of a silent wait."""
    buffered, started = [], False
    heartbeat = time.monotonic()
    try:
        for chunk in chunks:
            if chunk is not None:
                buffered.append(chunk)
                reasoning = chunk["choices"][0]["delta"].get("reasoning_content") if chunk.get("choices") else None
                if reasoning:
                    if not started:                 # the role chunk, held until there is something to send
                        started = True
                        yield buffered[0]
                    heartbeat = time.monotonic()
                    yield {**chunk, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning},
                                                 "finish_reason": None}]}
                    continue
            if chunk is None or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield None
        result = openai_collect(buffered)
        choice = result["choices"][0]
        content = validated_json(choice["message"]["content"], validator, choice["finish_reason"])
        if not started:
            yield buffered[0]
        yield {**buffered[0], "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
        yield buffered[-1]
    finally:
        chunks.close()


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came (#211)
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                "tool_use" if used_tool and streamed <= finished and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts (message_start could only say the whole prompt): what was read now stays for the next
            # request, a cache write; input_tokens keeps the last token, as 0 makes Claude Code count the prompt twice (#1013)
            reused = max(0, min(x.get("reused") or 0, len(ids) - 1))
            read_now = len(ids) - reused
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                       "stop_sequence": x.get("stop_sequence")},
                                    "usage": {"input_tokens": 1, "cache_creation_input_tokens": read_now - 1,
                                              "cache_read_input_tokens": reused,
                                              "output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                     # a call the output ended inside (#211): it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
CHUNKED_BODY_MAX = 256 << 20                # #893: the most a Transfer-Encoding: chunked body may hold (read into memory)


def body_limit() -> int:
    """The most a request body may hold, in bytes (read into memory): 256 MiB, which is a million-token conversation
    with room to spare; STRATA_MAX_BODY_MIB changes it (0 or an unreadable value: the default).  A larger one is
    answered 413 before it is read (a Content-Length of 10 TB used to be read until the client gave up)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return mib << 20 if mib > 0 else CHUNKED_BODY_MAX


class BadBody(Exception):
    """A request body that cannot be read (a malformed or oversized chunked body): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()

        def send_response(self, code, message=None):
            self.answer_started = True                 # a malformed-request answer can only replace one not yet begun
            super().send_response(code, message)

        def _chunked(self) -> bool:
            """#893: a body sent as Transfer-Encoding: chunked (a relay or proxy that does not buffer it).  By RFC 9112
            it wins over a Content-Length."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read_chunked(self, limit: int, keep: bool = True, deadline: float | None = None) -> bytes:
            """Decode a chunked body: at most `limit` bytes (BadBody 413 beyond it - the size is checked before a chunk
            is read, so a huge announced chunk is never allocated), BadBody 400 for a malformed one.  keep=False reads
            and drops it (the drain); `deadline` (monotonic) ends the read early."""
            parts, total = [], 0
            while True:
                if deadline is not None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        return b"".join(parts)
                    self.connection.settimeout(wait)
                line = self.rfile.readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, including the final blank line
                        trailer = self.rfile.readline(8193)
                        if trailer == b"\r\n":
                            return b"".join(parts)
                        if not trailer.endswith(b"\r\n"):
                            raise BadBody(400, "malformed or incomplete chunked request trailers")
                    raise BadBody(400, "too many chunked request trailers")
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {limit >> 20} MiB")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            self.body_read = True
            limit = body_limit()
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True               # the body is not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _drain_body(self):
            """An answer sent before the body was read (a 401, a 403, /load, a method with no handler) must not close
            the connection on unread bytes: the close then sends a reset, and a client that sends its body after the
            headers (http.client, urllib, requests) gets a connection error instead of the answer.  So the body is
            read and dropped here, once, after an answer, in pieces so that its size is never held in memory.  What
            has not arrived DRAIN_SECONDS later is left unread: the limit is time, so that a conversation of many
            megabytes, at any speed the client has, still gets its answer."""
            headers = getattr(self, "headers", None)
            if self.body_read or headers is None:
                return
            if self._chunked():
                try:
                    self._read_chunked(body_limit(), keep=False, deadline=time.monotonic() + self.DRAIN_SECONDS)
                except (BadBody, OSError):
                    self.close_connection = True
                return
            try:
                left = int(headers.get("Content-Length", 0))
            except ValueError:
                return
            if left <= 0:
                return
            deadline = time.monotonic() + self.DRAIN_SECONDS
            try:
                while left > 0 and (wait := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(wait)         # a client that never sends what it announced
                    piece = self.rfile.read1(min(left, 1 << 20))   # one socket read: read() would wait for it all
                    if not piece:
                        break
                    left -= len(piece)
            except OSError:
                pass

        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[sfnx] refused a request for Host {host!r} from {self.client_address[0]}: not a name this server "
                  f"answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching sFNX under this "
                             f"name on purpose? Add it to \"allowed_hosts\" in the config (serve/sfnx.json) or to "
                             f"the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which "
                             f"turns this check off"}})
            return False

        def _foreign_page(self) -> bool:
            """Without an API key, a /v1 POST from a browser page of another site (any site can POST text/plain
            there without a CORS preflight) would burn GPU time: an Origin header must name an allowed page, and
            then the body must be JSON.  No Origin (curl, the SDKs, other servers): any content type, as before."""
            origin = self.headers.get("Origin")
            if svc.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names):
                print(f"[sfnx] refused an API request from the web page {origin!r} (no API key; add its host to "
                      f"\"allowed_hosts\" in the config)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; set \"api_key\", "
                                 f"or add the page's host to \"allowed_hosts\" in the config"}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def end_headers(self):
            # HTTP/1.0 without keep-alive: the server closes the connection after every response.  Say so, or a
            # client that pools connections (.NET's SocketsHttpHandler) can put its next request on this socket
            # before the close reaches it, and that request fails with "response ended prematurely" without the
            # server ever reading it (a .NET agent app, 2026-10-03: 25 of ~1000 requests, 1-24 ms after the previous answer).
            self.send_header("Connection", "close")
            super().end_headers()

        def _watch_client(self, cancel: threading.Event) -> None:
            """#430 #431: cancel the request as soon as its client hangs up.  A non-streamed request writes nothing
            until it ends, and a streamed one only a keep-alive per prompt chunk (and the first write after a hang-up
            usually still succeeds), so a dropped request kept the engine busy until its answer or the whole prompt
            was done.  Every 0.5 s: the socket readable with nothing to read (EOF) means the client closed it.  A
            request is HTTP/1.0 and fully read here, so no later bytes are expected - data is not a hang-up."""
            done = self.watch_done = threading.Event()
            sock = self.connection

            def watch():
                while not done.wait(0.5) and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # the socket was closed here: the request has ended
                        return
                    if gone and not done.is_set():
                        self._note(outcome="disconnected")
                        cancel.set()
                        return

            threading.Thread(target=watch, daemon=True, name="client-watch").start()

        def _note(self, **values):
            """#332: what the monitor shows about this request (nothing when the monitor is off)."""
            if self.record is not None:
                with svc.status_lock:
                    self.record.update(values)

        def _json(self, code, obj, headers=None):
            if self.record is not None:
                with svc.status_lock:
                    self.record["http_status"] = code
                    self.record["response"] = BoundedText(json.dumps(obj, ensure_ascii=False, indent=2))
                    for key in ("error", "usage", "timings"):
                        if key in obj:
                            self.record[key] = obj[key]
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: front/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "front" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (front/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "front" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics":
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/api/requests" and svc.api_monitor:
                if self._authorized():
                    request_id = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    records = svc.request_records(request_id)
                    self._json(404 if records is None else 200,
                               {"error": {"message": "request no longer retained"}} if records is None else
                               records if request_id else {"requests": records, "retention": 100, "persistent": False,
                                                          "loaded": svc.loaded()})
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return

            if path == "" or (path == "/api-monitor" and svc.api_monitor):
                body = (ROOT / "front" / "web" / ("monitor.html" if path else "index.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": svc.reported_ctx(), "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key),

                                 "loaded": svc.loaded(), "service": "sfnx"})
            elif path == "/status":
                if not self._authorized():                  # #212: it shows the end of the last answer
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = svc.loaded()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.reported_ctx()},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    self._json(200, {"object": "list", "data": [model] if loaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                        # llama.cpp's name for what the slot's context holds: the running request's prompt size,
                        # kept after it ends (the conversation cache carries it on).  A front-end's context meter
                        # divides it by n_ctx; without the key it reads 0 % forever (#562).
                        in_use = int(svc.status.get("prompt_tokens") or 0)
                    if getattr(svc.engine, "batch", 0) and hasattr(svc.engine, "slots_view"):
                        # --batch: one entry per slot (the same view /metrics has), not one for the whole engine
                        n_ctx = svc.reported_ctx()
                        slots = [{"id": s["slot"], "n_ctx": n_ctx, "is_processing": s["state"] != "idle",
                                  "n_prompt_tokens": int(s.get("prompt_tokens") or s.get("held_tokens") or 0)}
                                 for s in svc.engine.slots_view()]
                    else:
                        slots = [{"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy,
                                  "n_prompt_tokens": in_use}]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                self._post()
            except BadBody as e:                              # #893: a malformed or oversized chunked body
                self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _post(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            path = V1_V1_ALIASES.get(path, path)         # an Anthropic client given ".../v1" as its base posts /v1/v1/messages
            if path.startswith("/v1/") and self._foreign_page():
                return
            if path == "/settings":
                self._settings()
                return
            try:
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # retrieve/delete/cancel/compact: nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    self.record = svc.begin_request(path, req)
                if path == "/v1/responses":
                    self._responses(req)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)

                else:
                    self._json(404, {"error": {"message": "not found"}})
            except ValueError as e:
                print(f"[sfnx] 400 invalid request: {e}", flush=True)
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except StructuredOutputError as e:
                self._json(502, {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                          "message": str(e)}})
            except (GpuBusy, EngineStarting) as e:
                ra = getattr(e, "retry_after", None)       # #1997 / #1994: a refusal that is worth retrying
                self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                           headers={"Retry-After": int(ra)} if ra else None)
            except (TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
                # a body that is JSON but the wrong shape ("messages": 5, a content part that is a number): a 400 that
                # says so, with the traceback in the log, instead of a dropped connection - unless the answer began
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[sfnx] 400 malformed request ({type(e).__name__}: {e}):\n" + traceback.format_exc(),
                      flush=True)
                body = {"error": {"type": "invalid_request_error",
                                  "message": f"the request is malformed ({type(e).__name__}: {e})"}}
                self._json(400, responses_error_body(body["error"]["message"]) if path == "/v1/responses" else body)
            except EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
            except EngineStuck as e:                         # an unload or restart that could not end the engine
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            except ClientGone:                               # #1999: the client left while the request was prepared
                self._note(outcome="disconnected")
            except OSError:
                self._note(outcome="disconnected")
                raise                                        # as before #332: the server's own handling
            finally:
                svc.drop_embeddings()                        # the images' file of a request that never got to run()
                if self.watch_done is not None:
                    self.watch_done.set()
                record = self.record
                if record is not None:
                    with svc.status_lock:
                        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                        record["finished_at"] = time.time()
                        record["state"] = "error" if record.get("error") else record.get("outcome", "completed")
                    svc.request_trace.record = None

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
                return
            if not svc.loaded():
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**model_card_sampling(svc.model_arch, True), **svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "batch", 0) or 0)),   # #1004: the batch slots
                     "model_alias": svc.model, "chat_template": svc.template.source,
                     "chat_template_caps": svc.template.caps,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "sfnx " + str(version)
            self._json(200, props)

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            if self._foreign_origin():
                self._json(403, {"error": {"message": f"{what} only from this server's own page"}})
                return False
            return True

        def _foreign_origin(self) -> bool:
            """A web page of another origin sent this.  The Origin must be this server's own address (host and port);
            no Origin is a client that is no browser.  Headers a proxy adds (X-Forwarded-*, CF-Ray, CF-Connecting-IP)
            prove nothing about the page that sent the request, so they open nothing."""
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return bool(origin) and origin.split("://", 1)[-1] != self.headers.get("Host", "")

        def _no_local_images(self, messages):
            """A web page of another origin (one on a name in allowed_hosts, or any page with the API key) must not have
            a file on this machine read as an image: data: and http(s) images are the page's own to send; files come
            from this server's own page or a client that is no browser.  ValueError (a 400)."""
            if self._foreign_origin() and any(not src.startswith(("data:", "http://", "https://"))
                                              for src in images_of(messages)):
                raise ValueError("a web page of another origin cannot have a file on this machine read as an "
                                 "image: send it as a data: URL")

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "defaults" not in req:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                              "message": 'the body must be {"defaults": {...}} to set them, '
                                                         'or {"defaults": null} to clear them'}})
                    return
                shared = svc.set_shared(req["defaults"])
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[sfnx] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[sfnx] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self._note(http_status=200)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # nginx and similar proxies pass each event at once
            self.end_headers()

        def _capture(self, items, api):
            """Retain bounded input/output for the monitor without changing the API response (the items as they
            are when the monitor is off)."""
            if self.record is None:
                return items
            return self._captured(items, api)

        def _captured(self, items, api):
            try:
                for item in items:
                    if item is not None:
                        if api == "openai":
                            delta = item["choices"][0]["delta"]
                            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
                            usage, timings = item.get("usage"), item.get("timings")
                        elif api == "responses":
                            kind = item["type"]
                            content = item["delta"] if kind == "response.output_text.delta" else ""
                            reasoning = item["delta"] if kind == "response.reasoning_text.delta" else ""
                            usage, timings = (item.get("response") or {}).get("usage"), None
                        else:
                            _, event = item
                            delta = event.get("delta", {})
                            content, reasoning = delta.get("text", ""), delta.get("thinking", "")
                            usage, timings = event.get("usage"), None
                        with svc.status_lock:
                            for name, value in (("output", content), ("reasoning", reasoning)):
                                if value:
                                    self.record[name].append(value)
                            if usage:
                                self.record["usage"] = usage
                            if timings:
                                self.record["timings"] = timings
                    yield item
            finally:
                items.close()

        def _openai(self, req):
            req = svc.with_shared(req, "openai")

            messages, tools, kw = openai_to_messages(req)
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the Responses route: no tools are offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools)      # a bad value is a 400 before anything is sent
            messages, validator = prepare_format(req.get("response_format"), messages)
            if validator is not None and tools:
                raise ValueError("structured response_format with tools is not supported")
            svc.load()
            max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest

            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, cancel=cancel)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, force=force)
            if validator is not None:
                chunks = structured_chunks(chunks, validator)
            chunks = self._capture(chunks, "openai")
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                             # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except StructuredOutputError as e:
                err = {"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _responses(self, req):
            """#451: POST /v1/responses - OpenAI's Responses API, stateless (front/responses.py), on the chat path.
            Errors use the Responses format; once the stream has started they arrive as a response.failed event."""
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            try:
                ids, thinking, tools, max_new, asm, validator, req = self._responses_prepare(req, cancel)
            except ClientGone:
                self._note(outcome="disconnected")
                return
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except (GpuBusy, EngineStarting, EngineStuck, EngineDied) as e:
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))

            def check(text, finish):
                return validated_json(text, validator, finish)

            def events():
                yield from asm.start()
                done = None
                for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done, check)
            items = self._capture(events(), "responses")
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except StructuredOutputError as e:
                    return self._json(502, responses_error_body(str(e), "server_error",
                                                                code="structured_output_failed"))
                except EngineDied as e:
                    return self._json(503, responses_error_body(f"{e}; the next request restarts it", "server_error",
                                                                code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                for e in items:
                    if e is None:
                        # Codex's idle timeout (5 minutes by default) counts events, not SSE comments: while a long
                        # prompt is read, a response.in_progress every 15 s tells it the server is still working
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(e)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                             # client went away: stop the engine
                items.close()
            except EngineDied as e:                          # mid-stream: response.failed, then the stream ends
                self._responses_failed(asm, f"{e}; the next request restarts it", "server_error", send)
            except StructuredOutputError as e:
                self._responses_failed(asm, str(e), "structured_output_failed", send)
            except ValueError as e:                          # the engine's ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)

        def _responses_failed(self, asm, message, code, send):
            e = asm.failed(message, code)
            self._note(error=e["response"]["error"])
            try:
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req, cancel=None):
            responses_api.check_request(req)
            messages = responses_api.input_messages(req)
            tools, names, skipped = responses_api.request_tools(req)
            kw = responses_api.template_kwargs(req, {"reasoning_effort": svc.effort_default()})
            try:
                self._no_local_images(messages)              # as on the chat route: no file read for a foreign page
            except ValueError as e:
                raise ResponsesError(str(e), "input") from None
            try:
                messages, validator = prepare_format(responses_api.text_format(req), messages, with_tools=bool(tools))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            try:
                call_validators = responses_api.call_validators(req, names)
            except ValueError as e:
                raise ResponsesError(str(e), "tools") from None
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # said once per tool, not per request
            if set(skipped) - noted:
                print(f"[sfnx] /v1/responses: left out the hosted tools {', '.join(sorted(set(skipped) - noted))} "
                      f"(this server cannot run them)", flush=True)
                noted.update(skipped)
            if req.get("max_output_tokens") is None and "max_tokens" in svc.shared:   # the shared Chat settings
                req = {**req, "max_output_tokens": svc.shared["max_tokens"]}
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before anything is sent
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            try:
                check_request_sampling(req)                   # the request is also the sampling dict (see below)
            except ValueError as e:                           # its message starts with the field's name: the param
                raise ResponsesError(str(e), str(e).partition(":")[0]) from None
            svc.load()
            try:
                ids, thinking, max_new = svc.prepare(messages, tools, kw, req.get("max_output_tokens") or 0,
                                                     cancel=cancel)
            except ValueError as e:                          # too long for the context, an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, svc.model, len(ids), names,
                                          "reasoning.encrypted_content" in include, validator is not None,
                                          call_validators)
            # the request is also the sampling dict, as on the chat path (temperature, top_p, top_k, seed, ...)
            return ids, thinking, tools, max_new, asm, validator, req

        def _count_tokens(self, req):
            """Anthropic's token count, which Claude Code asks for its context figures: the prompt this server would
            read for the same request, rendered and tokenized - the model does not run."""
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            stop_strings(req)                                 # the same 400 as the request itself would get
            check_request_sampling(req)                       # ... likewise a sampling field of the wrong type
            self._json(200, {"input_tokens": len(svc.encode_prompt(messages, tools, kw, store=False))})

        def _anthropic(self, req):
            svc.load()
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # Anthropic's {"type": "none"}: no tools offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools)        # "any" / {"type": "tool", "name": N}
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, cancel=cancel)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force)
            events = self._capture(events, "anthropic")
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    allow_reuse_address = True                          # a restart binds the port its predecessor left in TIME_WAIT
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)

    def server_close(self):
        # the hardware sampler serve() started for this server stops with it (test isolation: every Service a test
        # started used to leave its sampler running, 21 of them after test_responses, which slowed test_parallel's timing)
        tel = getattr(getattr(self, "svc", None), "telemetry", None)
        if tel is not None and hasattr(tel, "close"):
            tel.close()
        super().server_close()


def lan_addresses() -> list[str]:
    """This machine's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def host_name(value) -> str:
    """The name in a Host header (or an origin's host[:port]): "Example.com:8080" -> "example.com",
    "[::1]:8095" -> "::1"; "" when it is malformed."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, sep, rest = v[1:].partition("]")
        return name if sep and (not rest or (rest[:1] == ":" and rest[1:].isdigit())) else ""
    if v.count(":") == 1:
        v, port = v.split(":")
        if not port.isdigit():
            return ""
    elif ":" in v:                                       # a bare IPv6 address (a config entry)
        return v
    v = v.rstrip(".")
    return v if v and all(c.isalnum() or c in "-._" for c in v) else ""


def _is_ip(name: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name: str, names) -> bool:
    """`name` is one of `names`, or below an entry that starts with a dot (".example.com")."""
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def allowed_hosts_of(value, env: str = "") -> list[str]:
    """The config's allowed_hosts (a name or a list) plus $STRATA_ALLOWED_HOSTS (comma-separated): host names, "*" or
    ".example.com" (it and every name below it).  A scheme, port or path is dropped ("https://a.example.com:8443/"
    -> "a.example.com"); a wrong entry stops the start (ValueError)."""
    items = [] if value in (None, "") else [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("allowed_hosts: expected a host name or a list of them")
    out = []
    for raw in items + [x for x in env.split(",") if x.strip()]:
        x = raw.strip().lower()
        if x == "*":
            out.append(x)
            continue
        x = x.split("://", 1)[-1].split("/", 1)[0]
        dot = x.startswith(".")
        name = host_name(x[1:] if dot else x)
        if not name:
            raise ValueError(f"allowed_hosts: {raw!r} is not a host name like sfnx.example.com")
        out.append("." + name if dot else name)
    return list(dict.fromkeys(out))


def host_names_for(bind_host: str, allowed_hosts=()) -> set[str]:
    """Every name this server answers to besides an IP address: localhost, the address it listens on, the config's
    allowed_hosts, and - listening beyond this machine (0.0.0.0 or a LAN address) - its name and LAN addresses (the
    LAN addresses matter for the Origin check, which takes no IP on trust)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        names.update(("host.docker.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    return names


def host_allowed(host, names, any_host=False) -> bool:
    """DNS rebinding: a web page of another site whose name its DNS points at 127.0.0.1 reaches this server as the
    same origin, so the browser lets it read every answer.  Its requests carry that site's name in Host, so only
    the names this server answers to pass.  An IP address passes (a page served from an IP is that IP's own page;
    rebinding needs a name), as does "*.localhost" (browsers never ask DNS for it) and a request without a Host
    header (HTTP/1.0 clients; browsers always send one)."""
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    return bool(name) and (_is_ip(name) or name == "localhost" or name.endswith(".localhost")
                           or _name_in(name, names))


def key_list(value) -> list[str]:
    """#1344: the keys a server accepts.  --api-key / STRATA_API_KEY / "api_key" take llama.cpp's form, several keys
    separated by commas ("k1,k2"; spaces around each are dropped), or a list in the config ("api_key": ["k1", "k2"],
    the way to give a key that contains a comma).  A list is taken as it is."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def explicit_no_key(value) -> bool:
    """The owner chose to serve without a key: "api_key": "none" in the config, STRATA_API_KEY=none or --api-key none
    (or --no-api-key, handled by the caller).  Never a key called "none"."""
    return isinstance(value, str) and value.strip().lower() == "none"


def open_server_warning(port) -> str:
    """The one plain warning for a server on the network with no API key (recommend, never force)."""
    return f"no API key: anyone who can reach port {port} can use the model. Add an API key to keep others out."


def api_key_of(value):
    """The key as a client can send it (#725).  An HTTP header loses the spaces and line ends around its value, so a
    key kept with them (a config file or an environment file with CRLF line ends, a quoted " key ") matched no
    request: every client got 401 with the right key.  A value that is only such characters raises ValueError, it
    never means "no key" (#213).  #1344: a config list of keys returns the list (each key stripped, none empty); a
    string stays a string (its commas separate keys, see key_list)."""
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value]
        if not keys or any(not k for k in keys):
            raise ValueError("an API key was given but it is empty")
        return keys
    key = "" if value is None else str(value)
    if key and not key_list(key):
        raise ValueError("an API key was given but it is empty")
    return key.strip()


def key_matches(given: str, key) -> bool:
    """given: a header's value as http.server read it, each byte one character.  A key with characters outside ASCII
    arrives as UTF-8 from most clients and as Latin-1 from some; the right key passes in both forms (#725: the
    bytes were re-encoded before, so a UTF-8 key never matched).  Constant-time (#213).  #1344: `key` may name several
    keys (see key_list); the given one must match any, and every key is compared whatever the outcome."""
    raw = given.encode("latin-1", "replace")
    ok = False
    for k in key_list(key):
        ok |= hmac.compare_digest(raw, k.encode())
        try:
            ok |= hmac.compare_digest(raw, k.encode("latin-1"))
        except UnicodeEncodeError:
            pass
    return ok


def origin_allowed(origin: str, host, names) -> bool:
    """A browser page's Origin that may use the model without an API key: this server's own page (the Origin is the
    request's own Host) or a page on one of the names this server answers to (any port).  Unlike the Host check no IP
    passes on trust: a page served from any other IP is another site.  "null" (a sandboxed frame, a file:// page) does
    not pass: any site can send it.  An origin of another scheme (chrome-extension://, moz-extension://, an Electron
    app's app://) does: no web site can send one."""
    origin = (origin or "").strip().rstrip("/")
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False
    if scheme not in ("http", "https"):
        return True
    if host and rest == host.strip().lower():
        return True
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if number and not math.isfinite(value):
            raise ValueError(f"{key}: a finite number, not {value!r}")
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high"):
                raise ValueError("reasoning_effort: none, low, medium or high")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= TOP_K_MAX:
                raise ValueError(f"top_k: an integer 1..{TOP_K_MAX}")
            value = int(value)
        elif key == "seed":
            if not number or value != int(value) or not 0 <= value <= SEED_MAX:
                raise ValueError(f"seed: an integer 0..{SEED_MAX}")
            value = int(value)
        elif key == "max_tokens":
            if not number or value != int(value) or value <= 0:
                raise ValueError("max_tokens: a positive integer")
            value = int(value)
        elif key == "reasoning_budget_tokens":                 # #123: 0 means no budget, so 0 is allowed here
            if not number or value != int(value) or value < 0:
                raise ValueError("reasoning_budget_tokens: a whole number of tokens (0: no budget)")
            value = int(value)
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


def checkout_paths(cfg: dict) -> None:
    """sFNX's configs name their files relative to the checkout: a relative "cwd" is the checkout's (ROOT), and a
    relative "tokenizer" or "log" is the cwd's, as "exe", the vision paths and the engine's arguments already are."""
    cwd = ROOT / cfg.get("cwd", ".")
    cfg["cwd"] = str(cwd.resolve())
    for k in ("tokenizer", "log"):
        if isinstance(cfg.get(k), str):
            cfg[k] = str((cwd / cfg[k]).resolve())


# The run config's keys (serve/sfnx.json; docs/config.md says what each does).  Any other key stops the start: a typo,
# or a key of Strata's that sFNX does not have, would otherwise be ignored without a word.
CONFIG_KEYS = frozenset((
    # the engine: its program, arguments, environment, cards and log
    "exe", "args", "cwd", "env", "lib_dirs", "log", "gpu", "parallel", "engine_silence_s", "vision",
    # the server
    "tokenizer", "model_name", "host", "port", "api_key", "allowed_hosts", "api_monitor",
    # requests
    "sampling", "reasoning_effort", "reasoning_budget_tokens", "reasoning_close_retry", "answer_room",
    "fit_max_tokens", "repeat_stop_tokens", "anthropic_thinking"))
# the "vision" section's keys (Vision.reader_args, vision_env and the CPU fallback read them)
VISION_KEYS = frozenset(("exe", "cpu_exe", "mmproj", "model", "gpu", "threads", "max_tokens", "min_tokens",
                         "cuda_device"))


def check_config_keys(cfg: dict, path) -> None:
    """SystemExit naming every key of the run config (and of its "vision" section) that sFNX does not read."""
    unknown = sorted(set(cfg) - CONFIG_KEYS)
    vision = cfg.get("vision")
    if isinstance(vision, dict):
        unknown += [f"vision.{k}" for k in sorted(set(vision) - VISION_KEYS)]
    elif vision is not None:
        raise SystemExit(f'[sfnx] {path}: "vision" must be an object (the picture encoder\'s settings)')
    if unknown:
        raise SystemExit(f"[sfnx] {path}: unknown config key(s) {', '.join(unknown)}; sFNX reads "
                         f"{', '.join(sorted(CONFIG_KEYS))} (docs/config.md)")


def read_config(path) -> dict:
    """The run config's JSON.  A mistake in it (a missing comma between two args, a trailing comma) is said with its
    place in the file instead of a traceback (#2267: editing "args" by hand is where it happens)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))     # an editor may have put a BOM first
    except json.JSONDecodeError as e:
        raise SystemExit(f"[sfnx] {path} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}. Check the commas "
                         'between the entries of "args" (each option and each value is its own quoted entry, e.g. '
                         '"--ple-io", "mmap",) and that no entry is followed by a comma at the end of a list.') from None


# #1819: llama.cpp clients call the repetition penalty and its window repeat_penalty / repeat_last_n; they are the
# same settings as repetition_penalty / penalty_last_n (the /props answer already reports them under those names)
SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty (also written repeat_penalty), frequency_penalty, penalty_last_n (also
    repeat_last_n), seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value or an unknown key refuses to start the server (a typo'd config should not quietly change sampling)."""
    out = {}
    block = dict(cfg.get("sampling") or {})
    for alias, canon in SAMPLING_ALIASES.items():      # #1819: llama.cpp's names for the same two fields
        if block.get(alias) is not None:
            if block.get(canon) is not None:
                raise SystemExit(f"[sfnx] config sampling: both {alias!r} and {canon!r} are set; they are the same "
                                 f"setting, keep one")
            block[canon] = block.pop(alias)
    for key, value in block.items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if number and not math.isfinite(value):
            raise SystemExit(f"[sfnx] config sampling.{key}={value!r}: expected a finite number")
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[sfnx] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[sfnx] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[sfnx] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= TOP_K_MAX:
                raise SystemExit(f"[sfnx] config sampling.top_k={value!r}: the sampled path takes an integer "
                                 f"1..{TOP_K_MAX}")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[sfnx] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[sfnx] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[sfnx] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or not 0 <= value <= PENALTY_LAST_N_MAX:
                raise SystemExit(f"[sfnx] config sampling.penalty_last_n={value!r}: expected an integer "
                                 f"0..{PENALTY_LAST_N_MAX}")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or not 0 <= value <= SEED_MAX:
                raise SystemExit(f"[sfnx] config sampling.seed={value!r}: expected an integer 0..{SEED_MAX}")
            out[key] = int(value)
        else:
            raise SystemExit(f"[sfnx] config sampling.{key}: not a sampling setting (temperature, top_p, top_k, min_p, "
                             "presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed)")
    return out


def check_request_sampling(req) -> None:
    """A request's sampling field of the wrong JSON type ("temperature": "0.7", "top_k": 40.0, "seed": true) is a 400
    that names the field, as the same value is through POST /settings or the run config.  Unchecked, the value
    replaced the configured default in run()'s merge and sampling_keys dropped it, so the reply was greedy.  Only
    types are checked: a number (int or float, never bool) for temperature, top_p, min_p and the three penalties; an
    integer (never bool or float) for top_k, penalty_last_n and seed.  Ranges behave as before (a temperature <= 0 is
    greedy, top_k 0 or above TOP_K_MAX is clamped, a negative seed is unset), and an absent or null field is not checked.
    A seed past SEED_MAX, which the engine cannot take, is a 400 too; 0 is a seed like any other.  NaN and the
    infinities (JSON numbers Python accepts, or strings) are a 400: a NaN temperature was sent as greedy.  A
    `logit_bias` is a 400: the engine has none."""
    if isinstance(req, dict) and req.get("logit_bias"):
        raise ValueError("logit_bias: not supported by sfnx")
    if isinstance(req, dict):      # #1819: llama.cpp's names; the OpenAI-style field wins when both are sent
        if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
            req["repetition_penalty"] = req["repeat_penalty"]
        rl = req.get("repeat_last_n")
        if req.get("penalty_last_n") is None and isinstance(rl, int) and not isinstance(rl, bool) and rl > 0:
            req["penalty_last_n"] = rl      # 0 and -1 (off / the whole context) keep the engine's default window
    for key, integer in (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                         ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                         ("penalty_last_n", True), ("seed", True)):
        value = (req or {}).get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():      # some clients send numbers as strings: read and used
            try:
                num = float(value)
                num = int(num) if integer and num == int(num) else (num if not integer else None)
                if integer and num is None:
                    raise ValueError
                if not integer and re.fullmatch(r"\s*[+-]?\d+\s*", value):
                    num = int(value)
            except (ValueError, OverflowError):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}") from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
        if isinstance(value, float) and not math.isfinite(value):    # JSON's NaN / Infinity, which Python reads
            raise ValueError(f"{key}: a finite number, not {value!r}")
        if key == "seed" and value > SEED_MAX:
            raise ValueError(f"seed: at most {SEED_MAX}")


# The sampling each model card recommends ("Best Practices"), used when neither the request, the Chat settings nor the
# run config's `sampling` block says otherwise.  Greedy is only for a request or config that asks for temperature 0.
# min_p 0, repetition_penalty 1 and presence_penalty 0 are the engine's own "off", so they are not listed.
MODEL_CARD_SAMPLING = {
    # Qwen3.8-Flash-Next (and Gyro, the same base model)
    "qwen4exp": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                 "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
}


def model_card_sampling(arch, thinking) -> dict:
    """The model card's sampling for the family `arch` and the thinking state ({} for an unknown or unset family)."""
    sets = MODEL_CARD_SAMPLING.get(arch) if arch else None
    return dict(sets["thinking" if thinking else "instruct"]) if sets else {}


def default_model_name(cfg: dict) -> str:
    """The config's "model_name", else the model's own."""
    return cfg.get("model_name") or "qwen3.8-flash-next"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "sfnx"], default="mock")
    ap.add_argument("--config", help="the run config (JSON: exe, args, cwd, tokenizer, model_name, ...; serve/sfnx.json, "
                                     "docs/config.md)")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this machine only (the default), 0.0.0.0 = the network "
                         "too; also \"host\" in the config")
    ap.add_argument("--script-cycle", action="store_true",
                    help="with --script given more than once: repeat the scripts by position instead of advancing "
                         "to the last one and staying there, so extra requests a harness makes (a session title, "
                         "a re-ask after a tool result) cannot shift the sequence")
    ap.add_argument("--script-delay", type=float, default=0.0,
                    help="seconds to wait between the mock engine's tokens, so a turn stays live long enough for a "
                         "client to cancel it; the mock honours cancellation between tokens")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=None, help="the port to listen on (default 8095; also \"port\" in the "
                                                           "config)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="request this many simultaneous engine slots without changing the config")
    ap.add_argument("--gpu", help="the GPUs to run on, as nvidia-smi numbers them (\"0,1,2\"; also \"gpu\" in the "
                                  "config)")
    ap.add_argument("--tokenizer", default=None,
                    help="the tokenizer directory (vocab.json, merges.txt, token_type.json; also \"tokenizer\" in the "
                         "config, which --engine sfnx needs); without one the mock engine uses a byte tokenizer")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(the default; kept so older scripts keep working)")
    ap.add_argument("--no-fit-max-tokens", action="store_true",
                    help="answer 400 when prompt + max_tokens does not fit the context, like llama.cpp "
                         "(also \"fit_max_tokens\": false in the config). A prompt that alone does not fit is always 400")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY. "
                         "Several keys: separate them with commas (key1,key2), as llama.cpp does")
    ap.add_argument("--no-api-key", action="store_true",
                    help="serve on the network without an API key (anyone who can reach the port can use the model; "
                         "also \"api_key\": \"none\" in the config). Warns once, never refuses")

    ap.add_argument("--api-monitor", action=argparse.BooleanOptionalAction, default=None,
                    help="the API request monitor (/api-monitor and the Monitor tab's request details): keeps the last "
                         "100 requests' prompts and answers in memory, each cut to its start and end (on by default; "
                         "--no-api-monitor or \"api_monitor\": false in the config keeps nothing)")

    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    # Every ordinary way of being asked to stop takes Ctrl+C's path (QUIT to the engine, then exit): SIGTERM (systemd)
    # and SIGHUP (the terminal closed, ssh dropped). They are installed before anything slow starts, so one that
    # arrives while the model loads is not the default "die at once". SIGINT keeps Python's own handler (a second
    # Ctrl+C stops waiting).
    def request_stop():
        if threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        import _thread
        _thread.interrupt_main()
    lifeline.install_stop_handlers(request_stop)
    cfg = read_config(a.config) if a.config else {}
    check_config_keys(cfg, a.config)
    if a.config and a.engine == "sfnx":
        checkout_paths(cfg)
    if a.parallel is not None:
        if a.parallel < 1 or a.parallel > PARALLEL_MAX:
            ap.error(f"--parallel must be in 1..{PARALLEL_MAX} (the engine's batch slots)")
        if "--batch" in cfg.get("args", []):
            ap.error("--parallel conflicts with an explicit --batch in the config's engine args")
        cfg["parallel"] = a.parallel
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    a.port = a.port if a.port is not None else cfg.get("port", 8095)
    supplied_args = argv if argv is not None else sys.argv[1:]
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or \
            any((x == "--api-key" and i + 1 < len(supplied_args) and not supplied_args[i + 1].strip()) or
                x.strip() == "--api-key=" for i, x in enumerate(supplied_args)):
        print("[sfnx] an API key was given but it is empty: set a key, or leave it out", file=sys.stderr)
        return 2
    raw_key = a.api_key or cfg.get("api_key", "")
    no_key_chosen = bool(getattr(a, "no_api_key", False)) or explicit_no_key(raw_key)
    if explicit_no_key(raw_key):
        raw_key = ""
    try:
        preflight_api_key = api_key_of(raw_key)
    except ValueError as e:
        print(f"[sfnx] {e}: set a key, or leave it out", file=sys.stderr)
        return 2
    # A key is recommended on a network, never required (owner decision 2026-10-20): the banner below warns once.
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError as e:
        ap.error(listen_problem(a.host, a.port, e))
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer or cfg.get("tokenizer") or "")
    if a.engine == "sfnx" and not (tpath / "vocab.json").is_file():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}): \"tokenizer\" in the config names "
                 "its directory (data/tokenizer)")
    if a.tokenizer or cfg.get("tokenizer"):
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)

    if a.engine == "sfnx":
        if not cfg:
            ap.error("--engine sfnx needs --config")
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[sfnx] sampling defaults from the config: {pretty}", flush=True)
        if cfg.get("vision"):
            print("[sfnx] loading the vision encoder ...", flush=True)
            # relative paths are the config's cwd's, as for the engine below
            vcfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                        if k in ("exe", "cpu_exe", "mmproj", "model") and isinstance(v, str) and not os.path.isabs(v) else v)
                    for k, v in cfg["vision"].items()}
            vision = Vision(vcfg, log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=vision_env(cfg, env))
        print("[sfnx] loading the model ...", flush=True)
        exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
        try:
            silence = engine_silence_s(cfg)             # #481: checked before the (minutes-long) start
        except ValueError as e:
            raise SystemExit(f"[sfnx] config {e}")
        engine = StrataEngine(exe, engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
        engine.silence_s = silence                      # an attribute of its own: restart() keeps it
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."],
            cycle=getattr(a, "script_cycle", False),
            delay_s=getattr(a, "script_delay", 0.0) or 0.0), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja" if a.tokenizer or cfg.get("tokenizer") else None
    svc = Service(engine, tok, ChatTemplate(tpl if tpl is not None and tpl.exists() else ROOT / "front/chat_template.jinja"),
                  model_name=default_model_name(cfg), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or (not a.no_fit_max_tokens and cfg.get("fit_max_tokens") is not False))
    if a.engine == "sfnx":                            # the model card's sampling when nothing else is said (see MODEL_CARD_SAMPLING)
        svc.model_arch = "qwen4exp"
        requested = cfg.get("parallel")
        if requested is None:
            args = cfg.get("args", [])
            if "--batch" in args:
                requested = int(args[args.index("--batch") + 1])
        svc.requested_slots = requested if isinstance(requested, int) and not isinstance(requested, bool) \
            and requested > 0 else 1
    svc.answer_room = cfg.get("answer_room") is not False
    svc.reasoning_close_retry = cfg.get("reasoning_close_retry") is True    # #1053: opt-in, off by default
    if cfg.get("reasoning_effort") is not None:         # #1641: the thinking level of requests that name none
        try:
            svc.config_effort = clean_shared_defaults({"reasoning_effort": cfg["reasoning_effort"]}).get("reasoning_effort")
        except ValueError as e:
            raise SystemExit(f"[sfnx] config {e}")
        if svc.config_effort:
            print(f"[sfnx] thinking level: {svc.config_effort} (reasoning_effort in the config; a request, or the "
                  "Chat settings shared with other apps, can set its own)", flush=True)

    svc.api_key = preflight_api_key
    if not svc.api_key and "api_key" in cfg and not no_key_chosen:             # #569: written, but empty: the server has none (a warning, not a stop)
        print("[sfnx] the api_key in the config is empty: this server has no API key (anyone who can reach it can "
              "use it); set one, or leave api_key out", file=sys.stderr, flush=True)

    try:
        svc.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    except ValueError as e:
        raise SystemExit(f"[sfnx] config {e}")
    if svc.allowed_hosts:
        print("[sfnx] Host check off: any name reaches this server (allowed_hosts \"*\")" if "*" in svc.allowed_hosts
              else f"[sfnx] also answers to the host names {', '.join(svc.allowed_hosts)} (allowed_hosts)", flush=True)
    if a.api_monitor is not None:
        svc.api_monitor = a.api_monitor
    elif "api_monitor" in cfg:
        if not isinstance(cfg["api_monitor"], bool):
            raise SystemExit(f"[sfnx] config \"api_monitor\" must be true or false, not {cfg['api_monitor']!r}")
        svc.api_monitor = cfg["api_monitor"]
    if svc.api_monitor:
        print("[sfnx] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in "
              "memory" + ("" if svc.api_key else "; anyone who can reach this server can read them (no API key)"),
              flush=True)

    mode = str(cfg.get("anthropic_thinking") or "model")   # #278: "on_request" = only when the request asks
    if mode not in ("model", "on_request"):
        raise SystemExit(f"[sfnx] config anthropic_thinking must be \"model\" or \"on_request\", not {mode!r}")
    svc.anthropic_think_unasked = mode == "model"

    rs = cfg.get("repeat_stop_tokens", REPEAT_STOP_TOKENS)   # #606: opt-out with 0
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise SystemExit(f"[sfnx] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {rs!r}")
    svc.repeat_stop_tokens = rs

    if cfg.get("reasoning_budget_tokens") is not None:  # #123: a default thinking budget for every request
        try:
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"]
            budget = svc.reasoning_budget({})
        except ValueError as e:
            raise SystemExit(f"[sfnx] config {e}")
        if budget:
            print(f"[sfnx] thinking budget: {budget} tokens (reasoning_budget_tokens; a request can set its own)",
                  flush=True)

    svc.gpu_index = (gpu_list(cfg) or [0])[0]           # the Monitor reads the card the engine runs on (issue #51)
    svc.gpu_indices = gpu_list(cfg)                     # ... and every card it is split over (issue #112)
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            text = Path(svc.shared_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            text = None
        if text is not None:
            try:
                svc.shared = clean_shared_defaults(json.loads(text))
            except ValueError as e:
                raise SystemExit(f"[sfnx] {svc.shared_path} (the Chat settings shared with other apps): {e}; fix "
                                 "it or delete the file") from None
            if svc.shared:
                print("[sfnx] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)

    httpd = serve(svc, host=a.host, port=a.port)
    bound_port = httpd.server_address[1]               # --port 0: report the port the OS actually assigned
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{bound_port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{bound_port}/ in a browser to chat", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{bound_port}/   (API: http://{ip}:{bound_port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this machine's IP address>:" + str(bound_port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: " + open_server_warning(bound_port), flush=True)
    interrupted = False
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[sfnx] stopping (Ctrl+C again to end the engine at once) ...", flush=True)
    finally:
        closers = [httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # a second Ctrl+C: don't wait for the engine to free its memory
                if getattr(engine, "proc", None):
                    engine.proc.kill()
        if interrupted:
            print("[sfnx] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
