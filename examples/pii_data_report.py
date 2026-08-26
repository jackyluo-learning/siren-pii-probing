"""
Data report for the PII experiments: what each task's split looks like, and how
much of it a bag-of-words model already solves.

Runs on CPU in under a minute and touches no LLM, so the floor is known before
any GPU time is committed. The point of the report is the last column: a probe
that merely matches the TF-IDF score has demonstrated nothing about the model's
representation, and the merged-corpus task starts with that floor very high
precisely because two corpora differ in vocabulary, not only in whether an
identifier is present.
"""

import argparse
import warnings

import numpy as np

from siren.pii_benchmark import (build_pii_binary_task, build_pii_multiclass_task,
                                 build_pii_presence_merged_task,
                                 build_pii_presence_task, lexical_baseline)

warnings.filterwarnings("ignore")


def _top_words(task, k: int = 14):
    """Which words a bag-of-words model leans on -- the tell for a corpus shortcut."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=20000)
    X = vec.fit_transform(task.train_texts)
    clf = LogisticRegression(max_iter=2000).fit(X, task.y_train)
    names = np.array(vec.get_feature_names_out())
    w = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[-1]
    return (", ".join(names[np.argsort(w)[::-1][:k]]),
            ", ".join(names[np.argsort(w)[:k]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--per-source", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 78)
    print("PII 数据体检 —— 各任务的规模，以及词袋模型已经能解决多少")
    print("=" * 78)

    builds = [
        ("含 PII?（合并语料）",
         lambda: build_pii_presence_merged_task(cap=args.cap, per_source=args.per_source,
                                                verbose=False)),
        ("含 PII?（留出 banking77）",
         lambda: build_pii_presence_merged_task(cap=args.cap, per_source=args.per_source,
                                                holdout_source="banking77", verbose=False)),
        ("含 PII?（留出 ag_news）",
         lambda: build_pii_presence_merged_task(cap=args.cap, per_source=args.per_source,
                                                holdout_source="ag_news", verbose=False)),
        ("含直接标识符?（同源）",
         lambda: build_pii_presence_task(cap=args.cap, verbose=False)),
        ("含 SSN?（同源）",
         lambda: build_pii_binary_task(target="SSN", cap=args.cap, verbose=False)),
        ("6 类标识符（多分类）",
         lambda: build_pii_multiclass_task(
             categories=["EMAIL", "CREDITCARDNUMBER", "IPV4", "BITCOINADDRESS",
                         "ACCOUNTNUMBER", "PASSWORD"],
             cap=args.cap * 4, min_per_class=60, verbose=False)),
    ]

    rows, keep = [], {}
    for name, build in builds:
        try:
            t = build()
        except Exception as exc:                      # a thin category should not stop the report
            print(f"  {name}: 跳过（{type(exc).__name__}: {str(exc)[:70]}）")
            continue
        keep[name] = t
        rows.append((name, len(t.train_texts), len(t.val_texts), len(t.test_texts),
                     t.n_classes, 1.0 / t.n_classes, lexical_baseline(t, verbose=False)))

    print(f"\n{'任务':26s} {'train':>6} {'val':>6} {'test':>6} {'类数':>4} "
          f"{'随机':>6} {'词袋基线':>8}")
    print("-" * 78)
    for name, ntr, nva, nte, k, ch, lex in rows:
        print(f"{name:26s} {ntr:>6} {nva:>6} {nte:>6} {k:>4} {ch:>6.3f} {lex:>8.4f}")

    print("\n词袋基线 = 只数词频、完全不懂语义的模型能拿到的分数。")
    print("探针必须明显高过这条线，结果才说明表示层真的编码了 PII。\n")

    for name in ("含 PII?（合并语料）", "含直接标识符?（同源）"):
        if name not in keep:
            continue
        t = keep[name]
        hi, lo = _top_words(t)
        print(f"── {name}  词袋倚重的词 ──")
        print(f"   判为「{t.class_names[-1]}」: {hi}")
        print(f"   判为「{t.class_names[0]}」: {lo}")
    print("\n合并语料那一组若出现 reuters / ap / what / how 这类词，说明它在认语料库风格，"
          "不是在认标识符——这正是要用 --holdout-source 的原因。")


if __name__ == "__main__":
    main()
