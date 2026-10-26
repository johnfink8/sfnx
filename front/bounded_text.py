"""front/bounded_text.py - the request monitor's text: a prompt or an answer kept as its start and its end.

Prompts reach 100k+ tokens, and what an agent's request is about sits at the end (the newest messages), so a record
keeps the first HEAD and the last TAIL characters with the full length, never only the start.
"""
from __future__ import annotations

HEAD = 16384
TAIL = 49152


class BoundedText:
    """Text that grows by append(); holds at most `head` + `tail` characters (plus a tail buffer it trims lazily)."""

    __slots__ = ("head_max", "tail_max", "head", "parts", "parts_len", "length")

    def __init__(self, text: str = "", head: int = HEAD, tail: int = TAIL):
        if head < 0 or tail < 0:
            raise ValueError(f"BoundedText: head ({head}) and tail ({tail}) must be >= 0")
        self.head_max, self.tail_max = head, tail
        self.head, self.parts, self.parts_len, self.length = "", [], 0, 0
        self.append(text)

    def append(self, s: str) -> None:
        if not s:
            return
        self.length += len(s)
        room = self.head_max - len(self.head)
        if room > 0:
            self.head += s[:room]
            s = s[room:]
            if not s:
                return
        self.parts.append(s)
        self.parts_len += len(s)
        if self.parts_len > 2 * self.tail_max:            # trimmed in batches: one join per tail's worth of text
            self._trim()

    def _trim(self) -> None:
        joined = "".join(self.parts)
        kept = joined[-self.tail_max:] if self.tail_max else ""
        self.parts, self.parts_len = ([kept] if kept else []), len(kept)

    @property
    def truncated(self) -> bool:
        return self.length > self.head_max + self.tail_max

    def text(self) -> str:
        """The whole text when it fits; else its start, a line saying how much was left out, and its end."""
        self._trim()
        tail = self.parts[0] if self.parts else ""
        if not self.truncated:
            return self.head + tail
        elided = self.length - len(self.head) - len(tail)
        return f"{self.head}\n\n[... {elided:,} of {self.length:,} characters left out here ...]\n\n{tail}"

    def __len__(self) -> int:
        return self.length
