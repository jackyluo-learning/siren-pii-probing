"""
Progress reporting that stays visible in every environment.

tqdm redraws in place with a carriage return, and Colab renders only the final
frame of such a sequence: a long stage shows "0%" and then jumps to "100%" once
it finishes, which reads as a hang and repeatedly did during this project's runs.
Detecting that case from inside the process does not work either -- Colab runs
`!python ...` under a pseudo-terminal, so isatty() is True while the frontend
still swallows the frames. So the default is newline-terminated, flushed lines
everywhere; SIREN_PROGRESS=bar opts back into tqdm for a local terminal.

Lines alone are not enough when a single item is slow. The shuffled-label
control fits one layer roughly every 15 seconds, and progress can only be
emitted between items, so the stage announced itself and then said nothing for a
quarter minute -- long enough to look stuck again. A daemon heartbeat therefore
reports elapsed time whenever no line has appeared for a while, and `note` lets
a caller state the expected duration up front.
"""

import os
import sys
import threading
import time
from typing import Iterable, Optional, TextIO


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class _LineProgress:
    """Newline-per-update progress, with a heartbeat for slow items."""

    def __init__(self, total: Optional[int], desc: str, unit: str,
                 max_lines: int = 20, heartbeat: float = 20.0,
                 note: str = "", stream: Optional[TextIO] = None):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.stream = stream or sys.stdout
        self.heartbeat = heartbeat
        self.step = max(1, -(-total // max_lines)) if total else 50
        self.start = time.time()
        self.last = self.start
        self.n = 0
        self._first = True
        self._lock = threading.Lock()
        self._done = threading.Event()

        head = f"{desc}: 开始，共 {total if total else '?'} {unit}"
        self._emit(head + (f"（{note}）" if note else ""))

        self._beat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._beat.start()

    def _emit(self, line: str) -> None:
        print(line, flush=True, file=self.stream)
        self.last = time.time()

    def _heartbeat_loop(self) -> None:
        # Wakes twice per heartbeat window so a quiet stretch is reported near
        # the moment it becomes worth reporting, not a full window later.
        while not self._done.wait(self.heartbeat / 2.0):
            with self._lock:
                if time.time() - self.last >= self.heartbeat:
                    done = f"{self.n}/{self.total}" if self.total else str(self.n)
                    self._emit(f"{self.desc}: 运行中… 已完成 {done}，"
                               f"已用 {fmt_duration(time.time() - self.start)}")

    def update(self) -> None:
        with self._lock:
            self.n += 1
            due = self._first or self.n % self.step == 0
            self._first = False
            if not due:
                return
            self._emit(self._line())

    def _line(self, final: bool = False) -> str:
        elapsed = time.time() - self.start
        if not self.total:
            return f"{self.desc}: {self.n} {self.unit}  已用 {fmt_duration(elapsed)}"
        pct = 100.0 * self.n / self.total
        if final:
            tail = "完成"
        else:
            rate = self.n / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            tail = f"剩余约 {fmt_duration(eta)}"
        return (f"{self.desc}: {self.n}/{self.total} ({pct:.0f}%)  "
                f"已用 {fmt_duration(elapsed)}  {tail}")

    def close(self) -> None:
        self._done.set()
        with self._lock:
            self._emit(self._line(final=True))


def track(iterable: Iterable, desc: str, total: Optional[int] = None,
          unit: str = "it", note: str = "",
          stream: Optional[TextIO] = None) -> Iterable:
    """
    Wrap an iterable with progress that is visible in every environment.

    One flushed line per update by default -- Colab, pipes, log files and plain
    terminals all render those -- plus a heartbeat so a stage whose individual
    items take tens of seconds still reports that it is alive. `note` is printed
    with the opening line; use it to state an expected duration when one is
    known. SIREN_PROGRESS=bar opts into a tqdm bar locally.
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
        bar = _LineProgress(total, desc, unit, note=note, stream=out)
        try:
            for item in iterable:
                yield item
                bar.update()
        finally:
            bar.close()

    return _gen()
