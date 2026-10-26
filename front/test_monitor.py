"""Request inspection through HTTP; no native model or GPU is required."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from front.frontend import ChatTemplate
from front.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    """An engine that can stop (close) and be started again (restart), as one that died is."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True

    def alive(self):
        return self.running

    def close(self):
        self.running = False

    def restart(self):
        self.running = True


class MonitorOff(unittest.TestCase):
    """#332 is on by default (the Monitor tab's request details); off ("api_monitor": false), no prompt or answer is
    kept and the monitor's endpoints do not exist."""

    def test_on_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertTrue(svc.api_monitor)

    def test_off_keeps_nothing(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.api_monitor = False
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            for path in ("/api/requests", "/api-monitor"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        # a request the server refuses is an error record: the strict mode (fit_max_tokens off) refuses a max_tokens
        # that overshoots the context, the default shortens it (front/test_server.py MaxTokens)
        self.svc.fit_max_tokens = False
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.svc.fit_max_tokens = True
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        from front.bounded_text import HEAD, TAIL
        # a long prompt: what it is about is at its end (the newest message), so the end is kept with the start
        huge = "start " + "x" * 262200 + " the question at the end"
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        detail = self.request("/api/requests?id=" + first["id"])[1]
        self.assertTrue(detail["input_truncated"])
        self.assertEqual(detail["input_chars"], len(json.dumps({"messages": [{"content": huge}]}, indent=2)))
        self.assertIn("start x", detail["input"])
        self.assertIn("the question at the end", detail["input"])
        self.assertIn("characters left out here", detail["input"])
        self.assertLess(len(detail["input"]), HEAD + TAIL + 100)
        self.assertEqual(self.request("/api/requests")[1]["requests"][0]["input_chars"], detail["input_chars"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk, chunk]), "openai"))
        detail = self.request("/api/requests?id=" + handler.record["id"])[1]
        for field in ("output", "reasoning"):
            self.assertEqual(detail[field + "_chars"], 2 * len(huge))
            self.assertTrue(detail[field + "_truncated"])
            self.assertTrue(detail[field].startswith("start x"))
            self.assertTrue(detail[field].endswith("the question at the end"))

    def test_a_request_row_links_its_record_and_the_record_has_the_engine_figures(self):
        code, _ = self.chat()
        self.assertEqual(code, 200)
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            row = json.loads(r.read())["requests"][0]
        record = self.latest()
        self.assertEqual(row["record"], record["id"])
        self.assertEqual(record["engine"]["id"], row["id"])
        self.assertEqual(record["engine"]["output_tokens"], row["output_tokens"])
        self.assertGreater(record["engine"]["prompt_tokens"], 0)

    def test_off_rows_link_nothing(self):
        self.svc.api_monitor = False
        self.assertEqual(self.chat()[0], 200)
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            self.assertIsNone(json.loads(r.read())["requests"][0]["record"])

    def test_queue_wait_and_an_engine_restart_are_attributed_separately(self):
        self.engine.close()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    # bin/sfnx's own lines (src/session.cpp), as serve/sfnx.log has them; arena and slot lines between them
    SFNX = [
        "conversation cache: parked 11305 tokens in 140 ms, parked=40 bytes=55713546240",
        "fnx: arena: K/V pool 3913 -> 3977 pages (own 1021, at most 19089); slots lent 2928 / 2928 / 3416 "
        "(up to 2.93 GiB of a card's experts in RAM), in 0.6 ms",
        "conversation cache: restored 83269 tokens in 237 ms, parked=39 bytes=56315904000",
        "conversation cache: the first message (9352 tokens) parked on its own in 141 ms",
        "sfnx: slot 3 (lane 2) admitted: prompt 80366 = 78361 reused + 2005 read in 1870 ms; 3 slots decoding",
        "conversation cache: evicted 246169 tokens",
        "conversation cache: evicted 8822 tokens",
    ]

    def test_the_sfnx_log(self):
        from front.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "sfnx.log"
            log.write_text("", encoding="utf-8")
            c = ConvCacheLog()
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(self.SFNX[:3]) + "\n")
            st = c.poll(str(log), 0)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (39, 56315904000, 1, 1))
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(self.SFNX[3:5]) + "\n")
            st = c.poll(str(log), 0)                   # the system prompt parked on its own: one more, no totals
            self.assertEqual((st["parked"], st["parks"], st["last_event"], st["last_tokens"]), (40, 2, "parked", 9352))
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(self.SFNX[5:]) + "\n")
            st = c.poll(str(log), 0)                   # each eviction one fewer, counted (no evictions= on sFNX lines)
            self.assertEqual((st["parked"], st["evictions"]), (38, 2))
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.SFNX[0] + "\n")           # the next line with totals sets the count again
            self.assertEqual(c.poll(str(log), 0)["parked"], 40)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-slots-progress", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


class PromptReadRate(unittest.TestCase):
    """bin/sfnx's `PP <done> <total>` carries no time: the Monitor's read rate comes from when the lines arrive."""

    def test_the_meter(self):
        from front.server import ReadMeter
        m = ReadMeter()
        self.assertIsNone(m.rate())
        m.note(4000, now=100.0)                        # the first chunk (after the reused prefix): the start mark
        self.assertIsNone(m.rate())
        m.note(4100, now=100.1)                        # too close to the start for a rate
        self.assertIsNone(m.rate())
        m.note(10000, now=102.0)
        self.assertEqual(m.rate(), 3000.0)
        m.start()                                      # the next read owes nothing to this one
        self.assertIsNone(m.rate())

    def engine(self):
        import io
        import queue
        from types import SimpleNamespace
        from front.server import StrataEngine
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.max_context = 262144
        engine.wlock = threading.Lock()
        return engine

    def test_pp_lines_feed_the_meter_and_metrics(self):
        engine = self.engine()
        engine.lines.put("PP 6000 12000")
        engine.lines.put("PP 10000 12000")
        engine.lines.put("DONE 1 12000 4000 10 stop 0 0 4000")
        gen = engine.generate([1], 1, {}, threading.Event())
        self.assertIsNone(next(gen))
        self.assertIsNone(next(gen))
        self.assertEqual((engine.read_meter.first[1], engine.read_meter.last[1]), (6000, 10000))
        self.assertIsNone(engine.prefill_tok_s_mean)   # bin/sfnx says no rate: the PP lines' own field stays empty
        tok = ByteTokenizer()
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        t = engine.read_meter.first[0]
        engine.read_meter.last = (t + 2.0, 10000)      # as if the second PP came 2 s after the first
        svc.status.update(busy=True, first_token=None)
        live = svc.metrics()["live"]
        self.assertEqual(live["prefill_tok_s_mean"], 2000.0)
        self.assertEqual(svc._prefill_tok_s_mean(), 2000.0)
        svc.status.update(first_token=time.time(), generated=1)   # decoding now: no read in progress
        self.assertIsNone(svc.metrics()["live"]["prefill_tok_s_mean"])
        self.assertEqual(svc._prefill_tok_s_mean(), 0.0)
        list(gen)
        engine._send("GEN 1 1")                         # a new read: the old one's progress and rate go
        self.assertIsNone(engine.progress)
        self.assertIsNone(engine.read_meter.rate())

    def test_last_read_skips_turns_that_read_little(self):
        from front.server import last_read
        hist = [{"prompt_tokens": 20000, "reused": 4000, "prompt_ms": 5000.0, "time": 1.0},
                {"prompt_tokens": 90000, "reused": 89900, "prompt_ms": 300.0, "time": 2.0}]   # newest: 100 read
        self.assertEqual(last_read(hist), {"tokens": 16000, "ms": 5000.0, "tok_s": 3200.0, "time": 1.0})
        self.assertIsNone(last_read(hist[1:]))
        self.assertIsNone(last_read([]))

    def test_slots_view(self):
        engine = self.engine()
        engine.batch = 3
        now = time.time()
        engine.slot_held = [[1] * 500, [], []]
        engine.slot_live = [
            None,
            {"state": "reading", "request_id": "r1", "prompt_tokens": 90000, "prompt_read": 45000,
             "read_tok_s": 3100.5, "generated": 0, "started": now - 15, "first_token": None},
            {"state": "decoding", "request_id": "r2", "prompt_tokens": 70000, "prompt_read": 70000, "reused": 68000,
             "read_ms": 900.0, "generated": 200, "started": now - 10, "first_token": now - 5},
        ]
        idle, reading, decoding = engine.slots_view()
        self.assertEqual(idle, {"slot": 0, "state": "idle", "held_tokens": 500})
        self.assertEqual((reading["prompt_read"], reading["read_tok_s"], reading["context_tokens"]), (45000, 3100.5, 90000))
        self.assertIsNone(decoding["prompt_read"])     # a read's figures only while it reads
        self.assertIsNone(decoding["read_tok_s"])
        self.assertEqual((decoding["context_tokens"], decoding["reused"], decoding["read_ms"]), (70200, 68000, 900.0))
        self.assertAlmostEqual(decoding["tok_s"], 40.0, delta=1.0)


class MonitorPage(unittest.TestCase):
    """The Monitor tab: slots and a read's bar, no bar for decoding, request details in a drawer."""

    def setUp(self):
        web = Path(__file__).parent / "web"
        self.html = (web / "index.html").read_text(encoding="utf-8")
        self.js = (web / "app.js").read_text(encoding="utf-8")

    def test_request_details_drawer(self):
        for el in ("req-drawer", "req-title", "req-msg", "req-facts", "req-timings", "req-close", "req-scrim",
                   "req-input-text", "req-output-text", "req-reasoning-text", "req-response-text"):
            self.assertIn(f'id="{el}"', self.html)
        self.assertIn("api/requests?id=", self.js)
        self.assertIn("pre.textContent = text", self.js)   # a prompt is shown as text, never parsed as markup

    def test_no_decode_progress_bar(self):
        # decoding has no end to fill a bar towards: neither the answer's max_tokens nor the busy-slot share is one
        self.assertNotIn("live.max_tokens", self.js)
        self.assertNotIn("(100 * b.busy) / b.par", self.js)


if __name__ == "__main__":
    unittest.main()
