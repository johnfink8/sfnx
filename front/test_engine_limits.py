"""front/engine_limits.py against the engine's C++ sources: each mirrored constant must equal the engine's own value,
so a change on one side fails here instead of as a refused request or a wrong default.

    python -m pytest -q front/test_engine_limits.py
"""
import re
import unittest
from pathlib import Path

from front import engine_limits

FNX = Path(__file__).resolve().parents[1]


def engine_value(rel: str, pattern: str) -> str:
    """The one match of `pattern`'s group in FNX/rel; fails naming the file when the pattern no longer matches once."""
    text = (FNX / rel).read_text(encoding="utf-8")
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise AssertionError(f"{rel}: expected one match of {pattern!r}, found {len(found)} (the source changed: "
                             f"update test_engine_limits.py and front/engine_limits.py together)")
    return found[0]


class EngineLimits(unittest.TestCase):
    def check(self, name: str, rel: str, pattern: str, read=int):
        ours = getattr(engine_limits, name)
        theirs = read(engine_value(rel, pattern))
        self.assertEqual(ours, theirs, f"front/engine_limits.py {name} = {ours!r} but {rel} has {theirs!r}: "
                                       "change them together")

    def test_ctx_slack(self):
        self.check("CTX_SLACK", "tools/sfnx.cpp", r"r\.ids\.size\(\) \+ r\.max_new \+ (\d+) > a\.max_ctx")

    def test_batch_slots(self):
        self.check("PARALLEL_MAX", "tools/sfnx_input.hpp", r"constexpr int MAX_SLOTS = (\d+);")

    def test_eos_ids(self):
        self.check("EOS_IDS", "tools/sfnx.cpp", r"const std::vector<int> EOS\{([\d, ]+)\};",
                   read=lambda s: {int(x) for x in s.split(",")})

    def test_penalty_window_cap(self):
        self.check("PENALTY_LAST_N_MAX", "src/engine.hpp", r"static constexpr int PEN_CAP = (\d+);")

    def test_penalty_window_default(self):
        self.check("PENALTY_LAST_N_DEFAULT", "src/session.hpp", r"int pen_last_n = (\d+);")

    def test_top_k_cap(self):
        self.check("TOP_K_MAX", "src/kernels.hpp", r"constexpr int TOPK_N = (\d+);")

    def test_seed_range(self):
        self.check("SEED_MAX", "src/session.hpp", r"std::optional<(uint\d+)_t> seed;",
                   read=lambda s: 2 ** int(s[4:]) - 1)


if __name__ == "__main__":
    unittest.main()
