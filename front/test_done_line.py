"""The engine's DONE and BDONE lines (tools/sfnx.cpp's layout) read into EngineDone / BatchDone: each field checked, a
line of another shape a ValueError naming the field, and through StrataEngine an engine that is ended (EngineDied).

    python -m pytest -q front/test_done_line.py
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from front.server import BatchDone, EngineDied, EngineDone, StrataEngine


class DoneLine(unittest.TestCase):
    def test_the_engine_s_line(self):
        d = EngineDone.parse("DONE 4 20 40.0 30.5 stop 7 12 5")
        self.assertEqual(d, EngineDone(generated=4, prompt_tokens=20, prompt_ms=40.0, decode_ms=30.5, finish="stop",
                                       drafts_accepted=7, drafts_offered=12, reused=5))
        self.assertEqual(d.figures(), {"generated": 4, "prompt_tokens": 20, "prompt_ms": 40.0, "decode_ms": 30.5,
                                       "finish": "stop", "drafts_accepted": 7, "drafts_offered": 12, "reused": 5})
        for finish in ("stop", "length", "cancel", "nonfinite"):
            self.assertEqual(EngineDone.parse(f"DONE 0 1 0 0 {finish} 0 0 0").finish, finish)

    def test_a_malformed_line_names_the_field(self):
        for line, field in (("DONE x 20 40.0 30.0 stop 7 12 5", "generated"),
                            ("DONE 4 -1 40.0 30.0 stop 7 12 5", "prompt_tokens"),
                            ("DONE 4 20 nan 30.0 stop 7 12 5", "prompt_ms"),
                            ("DONE 4 20 40.0 -3 stop 7 12 5", "decode_ms"),
                            ("DONE 4 20 40.0 30.0 error 7 12 5", "finish"),
                            ("DONE 4 20 40.0 30.0 stop 7.0 12 5", "drafts_accepted"),
                            ("DONE 4 20 40.0 30.0 stop 7 inf 5", "drafts_offered"),
                            ("DONE 4 20 40.0 30.0 stop 7 12 +5", "reused")):
            with self.subTest(line=line), self.assertRaises(ValueError) as cm:
                EngineDone.parse(line)
            self.assertIn(f"{field}=", str(cm.exception))

    def test_a_line_of_another_length(self):
        for line in ("DONE 0", "DONE 4 20 40.0 30.0 stop", "DONE 4 20 40.0 30.0 stop 3 5 0 9 10", "DONEX 1 2 3 4 stop 0 0 0"):
            with self.subTest(line=line), self.assertRaises(ValueError) as cm:
                EngineDone.parse(line)
            self.assertIn("not 8", str(cm.exception))

    def test_bdone(self):
        self.assertEqual(BatchDone.parse("BDONE 2 17 length 412.5 9 14 5"),
                         BatchDone(slot=2, produced=17, finish="length", ms=412.5, drafts_accepted=9, drafts_offered=14, rounds=5))
        for line, what in (("BDONE 2 17 length 1.0", "not 7"), ("BDONE 2 17 length 1.0 0 0 1 x", "not 7"),
                           ("BDONE x 17 length 1.0 0 0 1", "slot="), ("BDONE 2 -1 length 1.0 0 0 1", "produced="),
                           ("BDONE 2 17 0 1.0 0 0 1", "finish="), ("BDONE 2 17 stop inf 0 0 1", "ms="),
                           ("BDONE 2 17 stop 1.0 -1 0 1", "drafts_accepted="), ("BDONE 2 17 stop 1.0 0 0 x", "rounds=")):
            with self.subTest(line=line), self.assertRaises(ValueError) as cm:
                BatchDone.parse(line)
            self.assertIn(what, str(cm.exception))

    def test_the_engine_is_ended_on_a_malformed_line(self):
        """Through StrataEngine: `last` takes a good line's figures; a bad one kills the process and is an EngineDied
        (the next request starts the engine again), and `last` keeps what it had."""
        proc = mock.Mock()
        e = SimpleNamespace(proc=proc, ended=False, last={})
        e._out_of_step = lambda what: StrataEngine._out_of_step(e, what)
        done = StrataEngine._parse_done(e, "DONE 4 20 40.0 30.0 stop 7 12 5")
        self.assertEqual((done.generated, e.last["reused"]), (4, 5))
        with self.assertRaises(EngineDied) as cm:
            StrataEngine._parse_done(e, "DONE 4 20 40.0 30.0 stop")
        self.assertIn("not 8", str(cm.exception))
        self.assertIn("the server ended the engine", str(cm.exception))
        self.assertTrue(e.ended)
        proc.kill.assert_called_once()
        self.assertEqual(e.last["generated"], 4)
        with self.assertRaises(EngineDied):
            StrataEngine._parse_bdone(e, "BDONE 0 1 error 1.0 0 0 1")
        self.assertEqual(StrataEngine._parse_bdone(e, "BDONE 0 1 stop 1.0 0 0 1").produced, 1)


if __name__ == "__main__":
    unittest.main()
