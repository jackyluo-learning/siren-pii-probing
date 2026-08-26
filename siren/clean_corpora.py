"""
PII-free negatives pooled from several public corpora.

ai4privacy/pii-masking-200k has no PII-free rows -- measured over 3000, exactly
zero have an empty privacy_mask -- so a "does this text contain PII at all?" task
has to source its negatives elsewhere. That import is what this module does, and
it is not free: two corpora differ in vocabulary and register, not only in
whether an identifier is present, so a probe can separate them without ever
learning what an identifier is. Run `lexical_baseline` on any task built from
these and read the result before trusting the probe's score.

Three guards keep the gap as narrow as it can be made:

  * several sources, so "clean" is not defined by one house style;
  * a word-count band matched to the positive corpus, so length is not a cue;
  * a regex screen dropping any row that still looks like it carries an
    identifier, plus template markers such as "{{Order Number}}" and "[EMAIL]".

Each text is returned with the source it came from, so a task can hold one
source out of training entirely and test on it -- see `holdout_source` in
build_pii_presence_merged_task. That measures whether a probe learned "no
identifier" or merely "not that particular corpus".
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# (dataset path, split, column). Every one is parquet-backed: script-based
# datasets (PolyAI/banking77, daily_dialog) no longer load on current `datasets`.
CLEAN_SOURCES: Dict[str, Tuple[str, str, str]] = {
    "banking77":   ("mteb/banking77", "train", "text"),
    "dolly_instr": ("databricks/databricks-dolly-15k", "train", "instruction"),
    "dolly_resp":  ("databricks/databricks-dolly-15k", "train", "response"),
    "ag_news":     ("fancyzhx/ag_news", "train", "text"),
}

# Anything matching these is dropped: a leftover identifier would be a mislabeled
# negative, and a template marker is a giveaway token of exactly the kind this
# project keeps having to design around.
_REJECT = [re.compile(p) for p in (
    r"[\w.+-]+@[\w-]+\.[\w.]+",        # email
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN-shaped
    r"\b(?:\d[ -]?){13,19}\b",         # card-shaped
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",    # IPv4
    r"\b\d{6,}\b",                     # any long bare number
    r"\{\{",                           # {{Order Number}}
    r"\[[A-Z_]{3,}\]",                 # [FIRSTNAME]
)]


def normalise(text: str) -> str:
    """Collapse whitespace and strip the literal escapes ag_news carries."""
    return re.sub(r"\s+", " ", str(text).replace("\\b", " ").replace("\\", "")).strip()


def is_clean(text: str) -> bool:
    return not any(r.search(text) for r in _REJECT)


def load_clean_negatives(
    sources: Optional[Sequence[str]] = None,
    per_source: int = 1500,
    word_band: Tuple[int, int] = (13, 40),
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Return [(source_name, text)], length-banded and screened for stray PII."""
    from datasets import load_dataset

    names = list(sources) if sources else list(CLEAN_SOURCES)
    unknown = [n for n in names if n not in CLEAN_SOURCES]
    if unknown:
        raise ValueError(f"未知的干净语料来源 {unknown}；可选：{list(CLEAN_SOURCES)}")

    lo, hi = word_band
    out: List[Tuple[str, str]] = []
    for name in names:
        path, split, col = CLEAN_SOURCES[name]
        rows = load_dataset(path, split=split)[col]
        kept = 0
        for raw in rows:
            if kept >= per_source:
                break
            t = normalise(raw)
            if lo <= len(t.split()) <= hi and is_clean(t):
                out.append((name, t))
                kept += 1
        if verbose:
            print(f"    干净负样本 {name:12s}: {kept:5d} 条（词数 {lo}-{hi}，已过正则筛）")
    if not out:
        raise RuntimeError("没有取到任何干净负样本，检查 word_band 是否过窄。")
    return out
