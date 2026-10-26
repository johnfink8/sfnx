"""The API surface through serve/ against an engine: chat completions (plain, streamed, tools, reasoning off, a JSON
format), /v1/responses (plain, strict tool + JSON format), /v1/messages (plain, streamed), /v1/models. Prints one line
per check and exits non-zero on any failure.

    api_smoke.py [BASE_URL]        (default http://127.0.0.1:8095)"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095"
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}
failures = []


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r), time.monotonic() - t0
        events = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                events.append(json.loads(line[5:]))
        return events, time.monotonic() - t0


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def run(name, f):
    try:
        f()
    except Exception as e:  # a failed request is a failed check, reported with what came back
        check(name, False, f"{type(e).__name__}: {e}")


def models():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        m = json.load(r)
    check("models", bool(m.get("data")), m["data"][0]["id"] if m.get("data") else m)


def chat_plain():
    r, s = post("/v1/chat/completions", {"model": "x", "max_tokens": 300, "messages": [{"role": "user", "content": "Say hello in French."}]})
    c = r["choices"][0]["message"].get("content") or ""
    check("chat plain", "bonjour" in c.lower(), f"{s:.1f} s: {c[:60]!r}")


def chat_stream():
    ev, s = post("/v1/chat/completions", {"model": "x", "max_tokens": 300, "stream": True,
                                          "messages": [{"role": "user", "content": "Count from 1 to 5, digits only."}]}, True)
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e.get("choices"))
    check("chat stream", "3" in text and len(ev) > 3, f"{len(ev)} events, {s:.1f} s: {text[:40]!r}")


def chat_tools():
    r, s = post("/v1/chat/completions", {"model": "x", "max_tokens": 400, "tools": [WEATHER],
                                         "messages": [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]})
    tc = r["choices"][0]["message"].get("tool_calls") or []
    ok = bool(tc) and json.loads(tc[0]["function"]["arguments"]).get("city", "").lower() == "oslo"
    check("chat tools", ok, f"{s:.1f} s: {tc[:1]}")


def chat_no_reasoning():
    r, s = post("/v1/chat/completions", {"model": "x", "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
                                         "messages": [{"role": "user", "content": "What is 2+2? Answer with the number."}]})
    m = r["choices"][0]["message"]
    check("chat reasoning off", "4" in (m.get("content") or "") and not m.get("reasoning_content"),
          f"{s:.1f} s: {(m.get('content') or '')[:40]!r}, reasoning {len(m.get('reasoning_content') or '')} chars")


def chat_json():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"],
              "additionalProperties": False}
    r, s = post("/v1/chat/completions", {"model": "x", "max_tokens": 600,
                                         "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": schema, "strict": True}},
                                         "messages": [{"role": "user", "content": "Make up a person with a name and an age."}]})
    c = r["choices"][0]["message"].get("content") or ""
    obj = json.loads(c)
    check("chat json schema", isinstance(obj.get("name"), str) and isinstance(obj.get("age"), int), f"{s:.1f} s: {c[:60]!r}")


def responses_plain():
    r, s = post("/v1/responses", {"model": "x", "max_output_tokens": 300, "input": "Name a primary color."})
    text = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
    check("responses plain", any(w in text.lower() for w in ("red", "blue", "yellow")), f"{s:.1f} s: {text[:50]!r}")


def responses_strict_tool_json():
    tool = {"type": "function", "name": "get_weather", "description": "Weather for a city", "strict": True,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}
    fmt = {"type": "json_schema", "name": "answer", "strict": True,
           "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}
    r, s = post("/v1/responses", {"model": "x", "max_output_tokens": 600, "tools": [tool], "text": {"format": fmt},
                                  "input": "What is the weather in Rome? Use the tool."})
    calls = [o for o in r.get("output", []) if o.get("type") == "function_call"]
    ok = bool(calls) and json.loads(calls[0]["arguments"]).get("city", "").lower() == "rome"
    check("responses strict tool + json format", ok, f"{s:.1f} s: {calls[:1] or r.get('output')}")


def messages_plain():
    r, s = post("/v1/messages", {"model": "x", "max_tokens": 300, "messages": [{"role": "user", "content": "Say hi."}]})
    text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
    check("messages plain", "hi" in text.lower() or "hello" in text.lower(), f"{s:.1f} s: {text[:50]!r}")


def messages_stream():
    ev, s = post("/v1/messages", {"model": "x", "max_tokens": 300, "stream": True, "messages": [{"role": "user", "content": "Say hi."}]}, True)
    kinds = {e.get("type") for e in ev}
    check("messages stream", "message_stop" in kinds and "content_block_delta" in kinds, f"{len(ev)} events, {s:.1f} s")


for name, f in [("models", models), ("chat plain", chat_plain), ("chat stream", chat_stream), ("chat tools", chat_tools),
                ("chat reasoning off", chat_no_reasoning), ("chat json schema", chat_json), ("responses plain", responses_plain),
                ("responses strict tool + json format", responses_strict_tool_json), ("messages plain", messages_plain),
                ("messages stream", messages_stream)]:
    run(name, f)
print(f"{len(failures)} failed" + (f": {failures}" if failures else ""))
sys.exit(1 if failures else 0)
