"""
Progress reporting that survives a non-TTY pipe.

tqdm redraws in place with a carriage return, and Colab renders only the final
frame of such a sequence: a long stage shows "0%" and then jumps to "100%" once
it finishes, which reads as a hang and repeatedly did during this project's runs.

Detecting that case from inside the process turned out not to work. Colab runs
`!python ...` under a pseudo-terminal, so isatty() is True even though the
frontend still swallows the intermediate frames -- a first attempt keyed on
isatty() picked the bar and changed nothing.

So the default is now newline-terminated, flushed lines everywhere, which is
readable in every environment and cannot silently regress. Set SIREN_PROGRESS=bar
for a tqdm bar in a local terminal. Lines are throttled by both count and time,
so a 464-batch pooling stage reports about twenty times rather than 464.
"""

import os
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
    Wrap an iterable with progress that is visible in every environment.

    One flushed line per update by default -- Colab, pipes, log files and plain
    terminals all render those. SIREN_PROGRESS=bar opts into a tqdm bar locally.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    out = stream or sys.stdout
    if os.environ.get("SIREN_PROGRESS", "").strip().lower() == "bar":
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
