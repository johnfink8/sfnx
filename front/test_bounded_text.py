"""The request monitor's text: the start and the end of a long prompt or answer, with its full length."""
import unittest

from front.bounded_text import BoundedText


class BoundedTextTest(unittest.TestCase):
    def test_short_text_is_whole(self):
        t = BoundedText("hello", head=5, tail=4)
        t.append(" you")
        self.assertEqual((t.text(), len(t), t.truncated), ("hello you", 9, False))

    def test_long_text_keeps_start_and_end_and_says_how_much_went(self):
        t = BoundedText(head=3, tail=4)
        for ch in "abcdefghijklmnopqrstuvwxyz":            # appended a character at a time, as tokens stream
            t.append(ch)
        self.assertTrue(t.truncated)
        self.assertEqual(len(t), 26)
        self.assertEqual(t.text(), "abc\n\n[... 19 of 26 characters left out here ...]\n\nwxyz")

    def test_one_big_append_equals_many_small_ones(self):
        text = "".join(chr(97 + i % 26) for i in range(10000))
        whole, parts = BoundedText(text, head=100, tail=300), BoundedText(head=100, tail=300)
        for i in range(0, len(text), 7):
            parts.append(text[i:i + 7])
        self.assertEqual(whole.text(), parts.text())
        self.assertTrue(whole.text().startswith(text[:100]))
        self.assertTrue(whole.text().endswith(text[-300:]))

    def test_exactly_full_is_not_truncated(self):
        t = BoundedText("x" * 7, head=3, tail=4)
        self.assertEqual((t.text(), t.truncated), ("x" * 7, False))

    def test_negative_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            BoundedText(head=-1)


if __name__ == "__main__":
    unittest.main()
