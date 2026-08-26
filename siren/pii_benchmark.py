"""
Real PII benchmark built from ai4privacy/pii-masking-200k.

The dataset gives, per row, a natural sentence (`source_text`) plus a
`privacy_mask`: a list of {value, start, end, label} spans marking every piece of
personal data, across 40+ categories (SSN, EMAIL, CREDITCARDNUMBER, IPV4, ...).

Two experiment builders live here, and both draw positives AND negatives from
**the same corpus**. That is deliberate. An earlier iteration of this project
used hand-written templates where the positive said "SSN is 492-10-4921" and the
negative "Order ID is 492-10-4921"; the probe reached F1 = 1.000 at layer 1
because a single cue word separated the classes at the embedding layer. Drawing
both sides from one generator removes the domain/style shortcut: the classes
differ only in which kind of identifier the sentence carries, so a probe has to
read what the identifier *is*, not what neighbourhood the sentence came from.

  build_pii_binary_task     "does this text contain category X?"
                            negatives are texts carrying OTHER PII categories.

  build_pii_multiclass_task "which PII category does this text carry?"
                            restricted to texts carrying exactly one of the
                            selected categories, so the label is unambiguous.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

HF_DATASET = "ai4privacy/pii-masking-200k"


@dataclass
class PIITask:
    """A ready-to-probe split, with the label vocabulary kept alongside."""
    train_texts: List[str]
    y_train: np.ndarray
    val_texts: List[str]
    y_val: np.ndarray
    test_texts: List[str]
    y_test: np.ndarray
    class_names: List[str]
    meta: Dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _rows(cap: int, language: Optional[str] = "English", verbose: bool = True):
    """Stream rows, optionally restricted to one language."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    kept = 0
    for row in ds:
        if kept >= cap:
            break
        if language and str(row.get("language", "")).strip() != language:
            continue
        text = (row.get("source_text") or "").strip()
        mask = row.get("privacy_mask") or []
        if not text or not isinstance(mask, list) or not mask:
            continue
        labels = {str(m.get("label", "")).strip().upper() for m in mask if m.get("label")}
        labels.discard("")
        if not labels:
            continue
        kept += 1
        yield text, labels


def survey_categories(cap: int = 4000, language: Optional[str] = "English",
                      top: int = 25, verbose: bool = True) -> List[Tuple[str, int]]:
    """How often does each PII category appear? Use this to choose classes."""
    counts: Counter = Counter()
    n = 0
    for _, labels in _rows(cap, language, verbose):
        counts.update(labels)
        n += 1
    if verbose:
        print(f"扫描 {n} 条文本，共出现 {len(counts)} 个 PII 类别。前 {top} 个：")
        for name, c in counts.most_common(top):
            print(f"    {name:22s} {c:5d}  ({c/n*100:5.1f}% 的文本含有)")
    return counts.most_common()


def _split(texts: List[str], y: np.ndarray, val_frac: float, test_frac: float,
           seed: int) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    n_test = int(test_frac * len(idx))
    n_val = int(val_frac * len(idx))
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    pick = lambda ii: ([texts[i] for i in ii], y[ii])
    (trX, trY), (vaX, vaY), (teX, teY) = pick(tr), pick(va), pick(te)
    return trX, trY, vaX, vaY, teX, teY


def build_pii_binary_task(
    target: str = "SOCIALNUM",
    cap: int = 6000,
    language: Optional[str] = "English",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> PIITask:
    """
    "Does this text contain <target> PII?"

    Positives carry the target category. Negatives are texts from the SAME corpus
    that carry other PII but not the target -- same generator, same style, so the
    only systematic difference is the kind of identifier present.
    """
    pos, neg = [], []
    for text, labels in _rows(cap, language, verbose):
        (pos if target in labels else neg).append(text)

    if not pos:
        raise ValueError(f"类别 {target!r} 在前 {cap} 条中没有出现，换一个类别或加大 cap。")

    rng = np.random.RandomState(seed)
    k = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    texts = pos[:k] + neg[:k]
    y = np.array([1] * k + [0] * k, dtype=np.int64)

    if verbose:
        print(f"二分类任务「是否含 {target}」：正 {k} / 负 {k}（负样本均含其他 PII，非普通文本）")

    trX, trY, vaX, vaY, teX, teY = _split(texts, y, val_frac, test_frac, seed)
    return PIITask(trX, trY, vaX, vaY, teX, teY,
                   class_names=[f"no {target}", target],
                   meta={"task": "binary", "target": target, "per_class": k,
                         "negatives": "same corpus, other PII categories"})


def build_pii_multiclass_task(
    categories: Optional[Sequence[str]] = None,
    n_categories: int = 6,
    cap: int = 12000,
    language: Optional[str] = "English",
    min_per_class: int = 60,
    balance: bool = True,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> PIITask:
    """
    "Which PII category does this text carry?"

    Only texts carrying **exactly one** of the selected categories are kept, so
    each text has an unambiguous label. Texts mixing several selected categories
    are dropped rather than assigned arbitrarily -- the point is to test whether
    the representation separates identifier *types*, and a mixed text cannot
    answer that cleanly.

    Note this filter is why the usable corpus is much smaller than the raw one;
    the drop count is reported so the cost is visible.
    """
    rows = list(_rows(cap, language, verbose))
    if categories is None:
        counts: Counter = Counter()
        for _, labels in rows:
            counts.update(labels)
        categories = [c for c, _ in counts.most_common(n_categories)]
        if verbose:
            print(f"未指定类别，按频次自动选取 {n_categories} 个：{list(categories)}")
    cats: List[str] = list(categories)
    cat_set: Set[str] = set(cats)

    texts, labels_out, dropped_mixed, dropped_none = [], [], 0, 0
    for text, labels in rows:
        hit = labels & cat_set
        if len(hit) == 1:
            texts.append(text)
            labels_out.append(cats.index(next(iter(hit))))
        elif len(hit) > 1:
            dropped_mixed += 1
        else:
            dropped_none += 1

    y = np.array(labels_out, dtype=np.int64)
    counts_per = Counter(y.tolist())
    thin = [cats[c] for c in range(len(cats)) if counts_per.get(c, 0) < min_per_class]
    if thin:
        raise ValueError(
            f"这些类别样本不足 {min_per_class} 条：{thin}。"
            f"实际计数 { {cats[c]: counts_per.get(c,0) for c in range(len(cats))} }。"
            "请加大 cap、降低 min_per_class，或改选更高频的类别。")

    if balance:
        k = min(counts_per.values())
        rng = np.random.RandomState(seed)
        keep: List[int] = []
        for c in range(len(cats)):
            idx = np.nonzero(y == c)[0]
            keep.extend(rng.choice(idx, k, replace=False).tolist())
        rng.shuffle(keep)
        texts = [texts[i] for i in keep]
        y = y[keep]

    if verbose:
        print(f"多分类任务：{len(cats)} 类 × 每类 {Counter(y.tolist())[0]} 条 = {len(texts)} 条")
        print(f"  丢弃：同时含多个目标类别 {dropped_mixed} 条；不含任何目标类别 {dropped_none} 条")
        print(f"  类别：{cats}")

    trX, trY, vaX, vaY, teX, teY = _split(texts, y, val_frac, test_frac, seed)
    return PIITask(trX, trY, vaX, vaY, teX, teY, class_names=cats,
                   meta={"task": "multiclass", "dropped_mixed": dropped_mixed,
                         "dropped_none": dropped_none, "balanced": balance})
