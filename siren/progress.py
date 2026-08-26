"""
Progress reporting that survives a non-TTY pipe.

tqdm redraws in place with a carriage return. Colab runs `!python ...` in a
subprocess whose stdout is a pipe, and it discards those intermediate frames, so
a long stage shows "0%" and then jumps straight to "100%" on completion -- which
reads as a hang, and repeatedly did during this project's Colab runs.

So: a real tqdm bar when stdout is a terminal, and newline-terminated lines
otherwise, each one flushed. Lines are throttled by both count and time, so a
464-batch pooling stage reports about twenty times rather than 464.
"""

import sys
import time
from typing import Iterable, Optional, TextIO


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class _LineProgress:
    """Newline-per-update progress for pipes. Emits at most `max_lines` updates."""

    def __init__(self, total: Optional[int], desc: str, unit: str,
                 max_lines: int = 20, min_interval: float = 10.0,
                 stream: Optional[TextIO] = None):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.stream = stream or sys.stdout
        self.min_interval = min_interval
        self.step = max(1, -(-total // max_lines)) if total else 50
        self.start = time.time()
        self.last = 0.0
        self.n = 0
        print(f"{desc}: 开始，共 {total if total else '?'} {unit}", flush=True,
              file=self.stream)

    def update(self, final: bool = False) -> None:
        self.n += 0 if final else 1
        now = time.time()
        due = (self.n % self.step == 0) or (now - self.last >= self.min_interval)
        if not (final or due):
            return
        self.last = now
        elapsed = now - self.start
        if self.total:
            pct = 100.0 * self.n / self.total
            rate = self.n / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            tail = "完成" if final else f"剩余约 {_fmt(eta)}"
            print(f"{self.desc}: {self.n}/{self.total} ({pct:.0f}%)  "
                  f"已用 {_fmt(elapsed)}  {tail}", flush=True, file=self.stream)
        else:
            print(f"{self.desc}: {self.n} {self.unit}  已用 {_fmt(elapsed)}",
                  flush=True, file=self.stream)


def track(iterable: Iterable, desc: str, total: Optional[int] = None,
          unit: str = "it", stream: Optional[TextIO] = None) -> Iterable:
    """
    Wrap an iterable with progress output appropriate to where stdout goes.

    A terminal gets a tqdm bar; anything else (Colab's `!python`, a pipe, a log
    file) gets one flushed line per update, so progress is visible instead of
    buffered away until the stage ends.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    out = stream or sys.stdout
    if getattr(out, "isatty", lambda: False)():
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, desc=desc, total=total, unit=unit)
        except Exception:
            pass

    def _gen():
        bar = _LineProgress(total, desc, unit, stream=out)
        for item in iterable:
            yield item
            bar.update()
        bar.update(final=True)

    return _gen()
