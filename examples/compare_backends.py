"""
Compare the HuggingFace and vLLM-Hook extraction paths layer by layer.

Lives in a script rather than a notebook cell for a practical reason: `git pull`
updates the .ipynb on disk, but Colab runs the copy already loaded in the
browser, so edits to a cell's own source never reach a running notebook. Every
other fix in this project took effect because it sat in a .py file that
`!python` re-reads on each run. This one should too.

The HuggingFace side comes from reference/pii_hf_presence_merged_banking77.json,
committed to the repo, because that run happened in a different notebook and
therefore a different Colab runtime -- the two never shared a filesystem, and
the original version of this comparison looked for a file that could not exist.
"""

import argparse
import json
import os

import numpy as np

REF = "reference/pii_hf_presence_merged_banking77.json"
PAIRS = [
    ("mean", "pii_vllm_presence-merged_results.json", "均值池化"),
    ("last", "pii_vllm_presence-merged_last_results.json", "末位词元池化"),
]


def compare(key: str, path: str, zh: str, ref: dict, plot: bool) -> bool:
    if not os.path.exists(path):
        print(f"[跳过] {zh}：缺少 {path}，先跑对应的实验 cell。\n")
        return False
    vl, hf = json.load(open(path)), ref[key]
    layers = sorted(int(k) for k in vl["test_f1"])
    a = np.array([hf["test_f1"][str(l)] for l in layers])
    b = np.array([vl["test_f1"][str(l)] for l in layers])
    d = np.abs(a - b)

    print(f"=== {zh} ===")
    print(f"  逐层测试 Macro-F1 绝对差: 均值 {d.mean():.4f}  最大 {d.max():.4f}"
          f"（L{layers[int(d.argmax())]}）   相关系数 r = {np.corrcoef(a, b)[0, 1]:+.3f}")
    print(f"  最佳单层   HF {a.max():.4f} (L{layers[int(a.argmax())]})"
          f"  |  vLLM {b.max():.4f} (L{layers[int(b.argmax())]})")
    print(f"  跨层融合   HF {hf['siren_test_f1']:.4f}  |  vLLM {vl['siren_test_f1']:.4f}")

    # The floor is computed on raw text with no model involved, so matching
    # floors are direct evidence both paths were handed the same texts and the
    # same split. A mismatch invalidates the comparison before any layer is read.
    #
    # Compare at 4 decimals, not exactly: the reference values were recovered by
    # parsing a "Macro-F1 = {:.4f}" printout, while the vLLM side reads a JSON
    # holding the full float. A 1e-6 tolerance called two identical 0.7851s
    # different and told the reader to distrust a comparison that was fine.
    same = abs(round(hf["lexical_baseline"], 4) - round(vl["lexical_baseline"], 4)) < 1e-9
    print(f"  词袋地板线 HF {hf['lexical_baseline']:.4f}  |  vLLM {vl['lexical_baseline']:.4f}"
          + ("   ← 相同，两边拿到的是同一批文本" if same
             else "   ← 不同！数据划分对不上，下面的比较无意义"))
    secs = vl.get("extract_seconds")
    if secs:
        print(f"  抽取耗时   vLLM {secs:.0f}s")
    print()

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]   # 图内文字全英文，避免缺字
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
        ax.plot(layers, a, label="HuggingFace forward hook", lw=2.2, color="#0E6A6E")
        ax.plot(layers, b, label="vLLM-Hook", lw=2.2, color="#B07A2E", ls="--")
        ax.axhline(hf["lexical_baseline"], color="#9A5518", ls=":", lw=1.4)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Same task, same probes -- only extraction differs [{key}]")
        ax.grid(alpha=.3)
        ax.legend()
        out = f"backend_compare_{key}.png"
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  对比图已保存: {out}\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        raise SystemExit(f"找不到参考数据 {args.ref}。先跑「同步代码」cell 拉取仓库。")
    ref = json.load(open(args.ref))

    print("=" * 74)
    print("抽取路径对比：HuggingFace forward hook  vs  vLLM-Hook")
    print(f"参考数据: {args.ref}（来自 {ref['mean']['source_notebook']}）")
    print("=" * 74 + "\n")

    done = sum(compare(k, p, zh, ref, not args.no_plot) for k, p, zh in PAIRS)
    if not done:
        raise SystemExit("没有任何 vLLM 结果可比。先跑步骤 4（和步骤 5）。")

    print("逐层差异应在 0.01 量级：fp16 与 fp32 的表征差异会改变 L1 选出哪些神经元，"
          "以及 C 网格选中哪个值。")
    print("跨层融合的差异通常更大，且方向不固定——MLP 分类头的初始化没有固定随机种子，"
          "z 向量上万维而训练集只有几千条。要区分抽取差异与随机性，"
          "把同一条路径跑两遍看波动幅度即可。")


if __name__ == "__main__":
    main()
