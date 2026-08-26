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

# Label vocabulary observed in the dataset (verified against the HF dataset
# viewer). Recorded because the names are not the obvious ones: the social
# security label is SSN (not SOCIALNUM) and the phone label is PHONENUMBER
# (not TELEPHONENUM). Guessing a name silently yields an empty task.
KNOWN_LABELS = {
    "ACCOUNTNAME", "ACCOUNTNUMBER", "AGE", "AMOUNT", "BITCOINADDRESS",
    "BUILDINGNUMBER", "CITY", "COMPANYNAME", "COUNTY", "CREDITCARDCVV",
    "CREDITCARDISSUER", "CREDITCARDNUMBER", "CURRENCY", "CURRENCYCODE",
    "CURRENCYNAME", "CURRENCYSYMBOL", "DATE", "DOB", "EMAIL", "ETHEREUMADDRESS",
    "EYECOLOR", "FIRSTNAME", "GENDER", "HEIGHT", "IBAN", "IPV4", "IPV6",
    "JOBAREA", "JOBTITLE", "LASTNAME", "LITECOINADDRESS", "MAC", "MASKEDNUMBER",
    "MIDDLENAME", "NEARBYGPSCOORDINATE", "ORDINALDIRECTION", "PASSWORD",
    "PHONEIMEI", "PHONENUMBER", "PIN", "PREFIX", "SECONDARYADDRESS", "SEX",
    "SSN", "STATE", "STREET", "TIME", "URL", "USERNAME", "VEHICLEVIN",
    "VEHICLEVRM", "ZIPCODE",
}

# A default multiclass set: distinct identifier *types* that a representation
# should be able to tell apart, all frequent enough to fill a class.
SUGGESTED_MULTICLASS = ["EMAIL", "SSN", "CREDITCARDNUMBER", "IPV4",
                        "PHONENUMBER", "IBAN"]


# Labels that on their own (or nearly so) pin a message to one person or one
# account -- the things a PII guard actually blocks on. Everything else in
# KNOWN_LABELS is a quasi-identifier: a job title, a city, a date, a bare first
# name. Those are still personal data, but no deployed filter refuses a sentence
# for containing them, and treating them as positives would make the task
# unanswerable rather than hard.
DIRECT_IDENTIFIERS = {
    "SSN", "CREDITCARDNUMBER", "CREDITCARDCVV", "IBAN", "ACCOUNTNUMBER",
    "PASSWORD", "PIN", "BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS",
    "VEHICLEVIN", "VEHICLEVRM", "PHONEIMEI", "MAC", "IPV4", "IPV6", "EMAIL",
    "PHONENUMBER", "USERNAME", "DOB", "NEARBYGPSCOORDINATE", "MASKEDNUMBER",
}


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


# The `language` column holds ISO codes ("en"), not names ("English"). Accept
# either spelling so a natural argument does not silently filter everything out.
_LANG_ALIASES = {
    "en": {"en", "english"}, "fr": {"fr", "french"}, "de": {"de", "german"},
    "it": {"it", "italian"}, "es": {"es", "spanish"}, "nl": {"nl", "dutch"},
}


def _lang_matches(value: str, wanted: str) -> bool:
    v, w = str(value).strip().lower(), str(wanted).strip().lower()
    return v == w or v in _LANG_ALIASES.get(w, {w}) or w in _LANG_ALIASES.get(v, {v})


def _rows(cap: int, language: Optional[str] = "en", verbose: bool = True):
    """Stream rows, optionally restricted to one language."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    kept = seen = 0
    langs_seen: Counter = Counter()
    for row in ds:
        if kept >= cap:
            break
        seen += 1
        langs_seen[str(row.get("language", "")).strip()] += 1
        if language and not _lang_matches(row.get("language", ""), language):
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

    if kept == 0:
        raise RuntimeError(
            f"扫描了 {seen} 行，但 language={language!r} 一条都没匹配上。"
            f"该列出现过的取值：{dict(langs_seen.most_common(6))}。"
            "改用其中之一，或传 language=None 以不做语言过滤。")


def survey_categories(cap: int = 4000, language: Optional[str] = "en",
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


def lexical_baseline(task: "PIITask", verbose: bool = True) -> float:
    """
    Macro-F1 of a TF-IDF bag-of-words classifier on the same split.

    This is the floor any representation-level result has to clear to mean
    anything. A model with no notion of what an identifier is scores this purely
    on surface vocabulary, so a probe that merely matches it has demonstrated
    nothing about the representation. Measured floors: 0.755 for the same-corpus
    "direct identifier?" task, 0.783 for "contains SSN?", and 0.981 once the
    negatives come from other corpora -- which is why the merged task's headroom
    is thin and its score must always be reported next to this number.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(task.train_texts)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, task.y_train)
    f1 = float(f1_score(task.y_test, clf.predict(vec.transform(task.test_texts)),
                        average="macro", zero_division=0))
    if verbose:
        print(f"  TF-IDF 词袋基线（表面词汇下限）: Macro-F1 = {f1:.4f}")
    return f1


def build_pii_presence_merged_task(
    cap: int = 6000,
    per_source: int = 1500,
    sources: Optional[Sequence[str]] = None,
    word_band: Tuple[int, int] = (13, 40),
    holdout_source: Optional[str] = None,
    val_source: Optional[str] = None,
    language: Optional[str] = "en",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> PIITask:
    """
    "Does this text contain PII at all?", with negatives imported from other corpora.

    Positives are ai4privacy rows carrying any PII; negatives are PII-free text
    pooled from several public corpora (see siren.clean_corpora). Both sides are
    restricted to the same word-count band so length cannot stand in for the
    label.

    The honest caveat, measured rather than assumed: a TF-IDF bag-of-words model
    reaches Macro-F1 0.981 on this task, against 0.755 on the same-corpus "direct
    identifier?" task. The two corpora differ in vocabulary and register, not
    only in whether an identifier is present, so almost all of the separability
    here is available without understanding PII at all. Always report the score
    beside `lexical_baseline(task)`.

    `holdout_source` supplies every test negative and appears nowhere in
    training, which asks whether a model learned "no identifier" or just "not
    those particular corpora". `val_source` is a second corpus reserved for
    validation, defaulting to the first remaining source. Both are needed: with
    validation negatives drawn from the training corpora, every layer scored
    0.993-0.999 on val against 0.65-0.87 on test, so C was selected from
    fourth-decimal noise and the alpha_l weights behind SIREN's aggregation went
    uniform. Validating on its own unseen corpus restores the signal without
    tuning on the test corpus.
    """
    from .clean_corpora import load_clean_negatives, normalise

    lo, hi = word_band
    pos = [t for t in (normalise(x) for x, _ in _rows(cap, language, verbose))
           if lo <= len(t.split()) <= hi]
    if not pos:
        raise ValueError(f"词数带 {word_band} 内没有正样本，放宽 word_band 或加大 cap。")

    labelled = load_clean_negatives(sources, per_source, word_band, verbose)
    rng = np.random.RandomState(seed)

    def split_block(items, val_f, test_f):
        idx = np.arange(len(items)); rng.shuffle(idx)
        n_te, n_va = int(test_f * len(idx)), int(val_f * len(idx))
        return ([items[i] for i in idx[n_te + n_va:]],
                [items[i] for i in idx[n_te:n_te + n_va]],
                [items[i] for i in idx[:n_te]])

    if holdout_source:
        known = sorted({s for s, _ in labelled})
        if holdout_source not in known:
            raise ValueError(f"holdout_source={holdout_source!r} 不在负样本来源 {known} 中。")

        # A second corpus is held out for validation. With validation negatives
        # drawn from the training corpora, every layer scored 0.993-0.999 on val
        # while testing at 0.65-0.87: model selection then had nothing to
        # discriminate on, C was picked from fourth-decimal noise, and the alpha_l
        # weights that drive SIREN's cross-layer aggregation collapsed to uniform.
        # Validating on its own unseen corpus makes val representative of test
        # without tuning on the test corpus itself.
        if val_source is None:
            others = [s for s in known if s != holdout_source]
            if len(others) < 2:
                raise ValueError(
                    f"留出 {holdout_source!r} 后只剩 {others}，不足以再留一个做验证。"
                    "至少需要 3 个负样本来源。")
            val_source = others[0]
        if val_source == holdout_source:
            raise ValueError("val_source 不能与 holdout_source 相同，否则 C 会调到测试语料上。")
        if val_source not in known:
            raise ValueError(f"val_source={val_source!r} 不在负样本来源 {known} 中。")

        neg_tr = [t for s, t in labelled if s not in (holdout_source, val_source)]
        neg_va = [t for s, t in labelled if s == val_source]
        neg_te = [t for s, t in labelled if s == holdout_source]
        for name, block in (("训练", neg_tr), ("验证", neg_va), ("测试", neg_te)):
            if not block:
                raise ValueError(f"{name}集没有负样本，检查 sources / per_source。")
        rng.shuffle(neg_tr); rng.shuffle(neg_va); rng.shuffle(neg_te)
    else:
        neg_tr, neg_va, neg_te = split_block([t for _, t in labelled], val_frac, test_frac)

    pos_tr, pos_va, pos_te = split_block(pos, val_frac, test_frac)

    def pair(p, n):
        k = min(len(p), len(n))
        return p[:k] + n[:k], np.array([1] * k + [0] * k, dtype=np.int64)

    trX, trY = pair(pos_tr, neg_tr)
    vaX, vaY = pair(pos_va, neg_va)
    teX, teY = pair(pos_te, neg_te)
    if min(len(trX), len(vaX), len(teX)) == 0:
        raise ValueError(f"某个划分为空：train {len(trX)} / val {len(vaX)} / test {len(teX)}。"
                         "加大 cap 或 per_source。")

    if verbose:
        srcs = sorted({s for s, _ in labelled})
        print(f"二分类任务「是否含 PII」：正样本来自 ai4privacy，负样本来自 {srcs}")
        if holdout_source:
            print(f"  留出语料: 训练负样本={[s for s in srcs if s not in (holdout_source, val_source)]}"
                  f" | 验证负样本={val_source} | 测试负样本={holdout_source}")
            print(f"  验证与测试各用一个未见语料，C 与 alpha_l 才有区分度，且不会调到测试语料上")
        print(f"  ⚠️ 跨语料任务的 TF-IDF 词袋下限实测 0.981（同源任务仅 0.755），"
              f"请把探针分数与词袋基线并列解读")

    return PIITask(trX, trY, vaX, vaY, teX, teY,
                   class_names=["no PII", "contains PII"],
                   meta={"task": "presence_merged", "word_band": list(word_band),
                         "negative_sources": sorted({s for s, _ in labelled}),
                         "holdout_source": holdout_source,
                         "val_source": val_source,
                         "n_train": len(trX), "n_val": len(vaX), "n_test": len(teX),
                         "caveat": "negatives come from other corpora; "
                                   "read the score against lexical_baseline()"})


def build_pii_presence_task(
    direct_labels: Optional[Set[str]] = None,
    cap: int = 8000,
    language: Optional[str] = "en",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> PIITask:
    """
    "Does this text carry a direct identifier?"

    This is the binary framing a PII guard actually faces, and it is deliberately
    NOT called "contains PII vs contains none". Every row of this corpus carries
    some personal data -- measured over 3000 rows, exactly zero have an empty
    privacy_mask -- because it exists to train masking models. There is no
    PII-free text here to use as negatives.

    Two ways to manufacture PII-free negatives were rejected. Deleting the marked
    spans from source_text leaves "Dear , as per our records, your license  is
    still registered", so the probe can separate the classes on broken syntax
    alone; target_text replaces each span with a literal "[FIRSTNAME]" marker,
    which is an even plainer cue. Both rebuild the template shortcut this module
    exists to avoid.

    So both sides here are unmodified source_text from the same generator, split
    on which *kind* of personal data is present:

      positive: carries at least one direct identifier (SSN, card number, IBAN,
                password, email, IP, ...)
      negative: carries only quasi-identifiers (first name, job title, city,
                date, ...) and no direct identifier

    A probe cannot win this on sentence shape or topic; it has to tell an account
    number from a job title. Report it with that wording, not as PII detection in
    the "any personal data at all" sense.
    """
    direct = {s.strip().upper() for s in (direct_labels or DIRECT_IDENTIFIERS)}
    unknown = direct - KNOWN_LABELS
    if unknown:
        print(f"⚠️  这些标签不在已知词表里，可能永远匹配不到：{sorted(unknown)}")

    pos, neg, trig = [], [], Counter()
    for text, labels in _rows(cap, language, verbose):
        hit = labels & direct
        if hit:
            pos.append(text)
            trig.update(hit)
        else:
            neg.append(text)

    if not pos or not neg:
        raise ValueError(
            f"无法构建二分任务：正 {len(pos)} 条 / 负 {len(neg)} 条。"
            "扩大 cap，或调整 direct_labels 使两侧都有样本。")

    rng = np.random.RandomState(seed)
    k = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    texts = pos[:k] + neg[:k]
    y = np.array([1] * k + [0] * k, dtype=np.int64)

    if verbose:
        print(f"二分类任务「是否含直接标识符」：正 {k} / 负 {k}"
              f"（扫描到 正 {len(pos)} / 负 {len(neg)}，已平衡到较少的一侧）")
        print(f"  负样本并非无 PII 文本，而是只含准标识属性（姓名/职位/城市/日期等）")
        print(f"  正样本触发类别 Top8: {[f'{a}×{b}' for a, b in trig.most_common(8)]}")

    trX, trY, vaX, vaY, teX, teY = _split(texts, y, val_frac, test_frac, seed)
    return PIITask(trX, trY, vaX, vaY, teX, teY,
                   class_names=["quasi-identifiers only", "direct identifier"],
                   meta={"task": "presence", "per_class": k,
                         "n_pos_seen": len(pos), "n_neg_seen": len(neg),
                         "trigger_counts": dict(trig.most_common()),
                         "direct_labels": sorted(direct),
                         "negatives": "same corpus, quasi-identifiers only"})


def build_pii_binary_task(
    target: str = "SSN",
    cap: int = 6000,
    language: Optional[str] = "en",
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
    seen_labels: Counter = Counter()
    target = target.strip().upper()
    if target not in KNOWN_LABELS:
        print(f"⚠️  {target!r} 不在已知标签词表里。已知的例如："
              f"{sorted(KNOWN_LABELS)[:10]} …（先跑 --task survey 确认）")
    for text, labels in _rows(cap, language, verbose):
        seen_labels.update(labels)
        (pos if target in labels else neg).append(text)

    if not pos:
        raise ValueError(
            f"类别 {target!r} 在扫描的 {cap} 行里没有出现。"
            f"实际最常见的类别是：{[n for n, _ in seen_labels.most_common(12)]}。"
            "先跑 --task survey 看完整分布，再选一个存在的类别。")
    if not neg:
        raise ValueError(
            f"每一条文本都含 {target!r}，没有负样本可用。换一个不那么普遍的类别。")

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
    language: Optional[str] = "en",
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
    if not rows:
        raise RuntimeError("没有取到任何文本，检查 cap 与 language 参数。")
    if categories is None:
        counts: Counter = Counter()
        for _, labels in rows:
            counts.update(labels)
        categories = [c for c, _ in counts.most_common(n_categories)]
        if verbose:
            print(f"未指定类别，按频次自动选取 {n_categories} 个：{list(categories)}")
    cats: List[str] = [c.strip().upper() for c in categories]
    if not cats:
        raise ValueError("类别列表为空，无法构建多分类任务。")
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
    if not counts_per:
        avail: Counter = Counter()
        for _, labels in rows:
            avail.update(labels)
        raise ValueError(
            f"指定的类别 {cats} 没有匹配到任何「恰好含一个」的文本。"
            f"语料里实际存在的类别：{[n for n, _ in avail.most_common(15)]}。")
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
