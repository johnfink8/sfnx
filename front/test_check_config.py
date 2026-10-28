"""front.server.check_config, the check `make install` runs: what it refuses, each with its reason, and what it takes.

    python -m pytest -q front/test_check_config.py
"""
import json
import tempfile
import unittest
from pathlib import Path

from front.server import check_config


class CheckConfig(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        tmp = Path(self.dir.name)
        (tmp / "tok").mkdir()
        for n in ("vocab.json", "merges.txt", "token_type.json"):
            (tmp / "tok" / n).write_text("")
        for n in ("engine", "model.gguf", "vocab.bin"):
            (tmp / n).write_text("")
        (tmp / "mtp").mkdir()
        self.tmp = tmp
        # absolute paths, so the files are this test's whatever the checkout holds
        self.engine_args = ["--native", str(tmp / "model.gguf"), "--mtp", str(tmp / "mtp"), "--draft-vocab", str(tmp / "vocab.bin")]
        self.base = {"exe": str(tmp / "engine"), "args": self.engine_args + ["--max-context", "262144"], "tokenizer": str(tmp / "tok"),
                     "sampling": {"temperature": 0.6}}

    def tearDown(self):
        self.dir.cleanup()

    def run_check(self, **changes):
        cfg = {**self.base, **changes}
        path = self.tmp / "cfg.json"
        path.write_text(json.dumps(cfg))
        check_config(path)

    def refused(self, why: str, **changes):
        with self.assertRaises(SystemExit) as e:
            self.run_check(**changes)
        self.assertIn(why, str(e.exception))

    def test_takes_a_good_config(self):
        self.run_check()
        self.run_check(args=self.engine_args + ["--max-context", "524288", "--rope-scale", "2"])
        self.run_check(env={"FNX_ROPE_SCALE": "1"})

    def test_refuses_what_the_start_would(self):
        self.refused("unknown config key", modle_name="x")
        self.refused("missing exe", exe=str(self.tmp / "nope"))
        self.refused("vocab.json", tokenizer=str(self.tmp / "none"))
        (self.tmp / "tok" / "merges.txt").unlink()
        self.refused("merges.txt")
        (self.tmp / "tok" / "merges.txt").write_text("")
        self.refused("--native", args=["--native", str(self.tmp / "gone.gguf")])
        self.refused("--mtp", args=["--mtp", str(self.tmp / "engine")])
        self.refused("--draft-vocab", args=["--draft-vocab", str(self.tmp / "gone.bin")])
        self.refused('needs "--vision"', vision={"exe": str(self.tmp / "engine")})
        self.refused("temperature", sampling={"temperature": "hot"})

    def test_refuses_bad_context_and_scale(self):
        a = self.engine_args
        self.refused("not a number of positions", args=a + ["--max-context", "lots"])
        self.refused("want 1 (off) or a number above 1", args=a + ["--rope-scale", "0.5"])
        self.refused("want 1 (off) or a number above 1", args=a + ["--rope-scale", "2x"])
        self.refused("want 1 (off) or a number above 1", env={"FNX_ROPE_SCALE": "inf"})
        self.refused("disagree", args=a + ["--rope-scale", "2"], env={"FNX_ROPE_SCALE": "4"})

    def test_refuses_what_is_not_a_config(self):
        path = self.tmp / "list.json"
        path.write_text("[]")
        with self.assertRaises(SystemExit) as e:
            check_config(path)
        self.assertIn("a JSON object", str(e.exception))


if __name__ == "__main__":
    unittest.main()
