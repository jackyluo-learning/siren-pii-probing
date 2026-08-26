"""
Does the SIREN method work on real PII data — and can it tell PII *types* apart?

Runs the paper's pipeline unchanged (mean pooling, train/val/test, per-layer L1
probes with C picked on validation, adaptive alpha_l, MLP head, Macro-F1) on
ai4privacy/pii-masking-200k, in two modes:

  --task binary      "does this text contain <category>?"
  --task multiclass  "which PII category does this text carry?"

Both draw positives and negatives from the SAME corpus. An earlier version of
this experiment used hand-written templates ("SSN is 492-10-4921" vs "Order ID
is 492-10-4921") and hit F1 = 1.000 at layer 1, because one cue word separated
the classes at the embedding layer — the probe never had to read the hidden
layers at all. Same-corpus sampling removes that escape route.

Every run also fits a shuffled-label control probe. If the real-label curve is
genuine, the control must collapse to chance (Macro-F1 ~ 1/n_classes). If both
score high, the result is dimensionality overfitting, not signal.
"""

import argparse
import json
import warnings
from typing import Dict, List, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from siren import (AdaptiveNeuronAggregator, SafetyNeuronProbe,
                   SirenMLPHead, SirenTrainer)
from siren.probe import PAPER_C_GRID
from siren.pii_benchmark import (build_pii_binary_task, build_pii_multiclass_task,
                                 survey_categories, PIITask)
from run_real_layerwise_experiment import (load_model_and_extractor, pool_texts,
                                           _balanced_subsample)

warnings.filterwarnings("ignore")


def _fit_probe(tag: str, tr, y, va, y_va, layers, eta):
    """Fit one layer-wise probe set with a progress bar over layers."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k):
            return x
    probe = SafetyNeuronProbe(eta=eta, c_grid=PAPER_C_GRID, average="macro")
    neurons, val_f1 = {}, {}
    for l in tqdm(layers, desc=tag, unit="layer"):
        n, f = probe.fit_layer(l, tr[l], y, va[l], y_va)
        neurons[l], val_f1[l] = n, f
    return probe, neurons, val_f1


def run_task(task: PIITask, model_name: str, eta: float = 0.8,
             mlp_hidden_dim: int = 256, epochs: int = 20,
             probe_train_cap: int = 4000,
             device: str = "cpu", max_length: int = 128) -> Dict:
    """Paper pipeline on one PII task; returns per-layer and aggregated results."""
    tokenizer, _, extractor, num_layers = load_model_and_extractor(model_name, device)
    print(f"\n抽取 mean-pooled 表征（{num_layers} 层）...", flush=True)
    tr = pool_texts(tokenizer, extractor, task.train_texts, num_layers, max_length)
    va = pool_texts(tokenizer, extractor, task.val_texts, num_layers, max_length)
    te = pool_texts(tokenizer, extractor, task.test_texts, num_layers, max_length)
    extractor.remove_hooks()

    layers = list(range(1, num_layers + 1))
    chance = 1.0 / task.n_classes

    # L1 coordinate descent scales with n_samples, and multiclass multiplies the
    # cost by n_classes (one-vs-rest fits one probe per class). A class-balanced
    # subsample keeps probe fitting to minutes; val/test stay full-size and the
    # MLP head still trains on the full aggregated train set.
    sub = _balanced_subsample(task.y_train, probe_train_cap)
    tr_fit = {l: tr[l][sub] for l in layers}
    y_fit = task.y_train[sub]
    if len(sub) < len(task.y_train):
        print(f"\n探针拟合使用类别均衡子采样 {len(sub)}/{len(task.y_train)} 条"
              f"（{len(layers)} 层 × 4 个 C × {task.n_classes} 类 = "
              f"{len(layers)*4*task.n_classes} 次 L1 拟合）", flush=True)
    print("逐层 L1 探针（C 网格在验证集上选，Macro-F1）...", flush=True)
    probe, safety_neurons, val_f1 = _fit_probe(
        "probes", tr_fit, y_fit, va, task.y_val, layers, eta)
    test_f1 = {l: float(f1_score(task.y_test, probe.layer_probes[l].predict(te[l]),
                                 average="macro", zero_division=0)) for l in layers}
    for l in layers:
        print(f"  Layer {l:2d}: val={val_f1[l]:.4f}  test={test_f1[l]:.4f}  "
              f"C={probe.layer_best_c[l]:g}  neurons={len(safety_neurons[l])}")

    print("\nSIREN 跨层融合（alpha_l 来自验证集）+ MLP 分类头 ...", flush=True)
    agg = AdaptiveNeuronAggregator(safety_neurons, val_f1)
    z_tr, z_va, z_te = agg.transform(tr), agg.transform(va), agg.transform(te)
    mlp = SirenMLPHead(input_dim=z_tr.size(1), hidden_dim=mlp_hidden_dim,
                       num_classes=task.n_classes)
    trainer = SirenTrainer(model=mlp, lr=1e-3, device=device)
    trainer.fit(z_tr, torch.tensor(task.y_train), z_va, torch.tensor(task.y_val),
                epochs=epochs, batch_size=16)
    with torch.no_grad():
        pred = mlp(z_te.to(trainer.device)).argmax(dim=1).cpu().numpy()
    siren_f1 = float(f1_score(task.y_test, pred, average="macro", zero_division=0))

    # Same workload as the main probe — it used to run silently, which reads as
    # a hang. Progress bar and the same subsample apply here too.
    print("打乱标签对照探针（与上一步同等工作量）...", flush=True)
    rng = np.random.RandomState(0)
    y_shuf = y_fit.copy()
    rng.shuffle(y_shuf)
    ctrl, _, _ = _fit_probe("control", tr_fit, y_shuf, va, task.y_val, layers, eta)
    ctrl_f1 = {l: float(f1_score(task.y_test, ctrl.layer_probes[l].predict(te[l]),
                                 average="macro", zero_division=0)) for l in layers}

    return {
        "num_layers": num_layers, "chance": chance,
        "val_f1": val_f1, "test_f1": test_f1, "ctrl_f1": ctrl_f1,
        "best_c": dict(probe.layer_best_c),
        "neurons": {l: len(v) for l, v in safety_neurons.items()},
        "z_dim": int(z_tr.size(1)),
        "siren_test_f1": siren_f1,
        "y_true": task.y_test.tolist(), "y_pred": pred.tolist(),
        "class_names": task.class_names,
    }


def _fig_setup():
    """
    All in-figure text is English on purpose.

    Matplotlib's bundled DejaVu Sans carries no CJK glyphs and Colab ships no
    Chinese font, so Chinese axis labels and titles render as tofu boxes. Rather
    than depend on a font being installed, the figures stay English; the Chinese
    commentary lives in the console output and the notebook prose.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_layer_curve(res: Dict, title: str, out: str):
    _fig_setup()
    layers = sorted(res["test_f1"])
    y = np.array([res["test_f1"][l] for l in layers])
    c = np.array([res["ctrl_f1"][l] for l in layers])
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    ax.grid(True, linestyle="-", linewidth=.6, alpha=.35, color="#CCCCCC", zorder=1)
    ax.plot(x, y, color="#C88BA8", linewidth=1.1, alpha=.7, zorder=2)
    ax.scatter(x, y, s=32, facecolors="#D8A3BE", edgecolors="#5C2344", linewidth=.9, zorder=3)
    if len(layers) > 4:
        ax.plot(x, gaussian_filter1d(y, sigma=1.8), color="#8C2D62", linewidth=3,
                label="Layer-wise probes (test)", zorder=4)
    ax.axhline(res["siren_test_f1"], color="#1F77B4", linestyle="--", linewidth=2.4,
               label=f"SIREN aggregated ({res['siren_test_f1']:.3f})", zorder=5)
    ax.plot(x, c, color="#8A98A6", linewidth=1.5, linestyle=":", marker="x", markersize=4,
            label="Shuffled-label control", zorder=4)
    ax.axhline(res["chance"], color="#B0B0B0", linewidth=1, alpha=.8, zorder=1)
    ax.text(len(layers) - 1, res["chance"] + .012, f"chance {res['chance']:.2f}",
            fontsize=9, color="#8A98A6", ha="right")
    ax.set_xlim(-1, len(layers))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Layer index", fontsize=15)
    ax.set_ylabel("Macro-F1", fontsize=15)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower center", fontsize=10.5, frameon=True, facecolor="white",
              edgecolor="#CCCCCC", framealpha=.95)
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"层级曲线已保存: {out}")


def plot_confusion(res: Dict, out: str):
    _fig_setup()
    names = res["class_names"]
    cm = confusion_matrix(res["y_true"], res["y_pred"], labels=list(range(len(names))))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("s", ["#F7F9FA", "#A8CDE8", "#14456B"])

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.05 + 2), max(5, n * .92 + 1.6)), dpi=200)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    short = [x if len(x) <= 14 else x[:13] + "…" for x in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title("Confusion matrix (row-normalised, %)", fontsize=12, pad=10)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > .55 else "#16283D")
    plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"混淆矩阵已保存: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["survey", "binary", "multiclass"], default="multiclass")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--target", default="SSN", help="binary 模式的目标类别（注意是 SSN 不是 SOCIALNUM）")
    ap.add_argument("--categories", default=None, help="multiclass 模式的类别，逗号分隔")
    ap.add_argument("--n-categories", type=int, default=6)
    ap.add_argument("--cap", type=int, default=12000, help="流式扫描的原始行数上限")
    ap.add_argument("--min-per-class", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--probe-cap", type=int, default=4000,
                    help="探针拟合的类别均衡样本上限（0 = 用全部；L1 成本随样本数线性增长）")
    ap.add_argument("--out-prefix", default="pii")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 74)
    print(f"SIREN on real PII data — task={args.task} | model={args.model} | device={device}")
    print("=" * 74)

    if args.task == "survey":
        survey_categories(cap=args.cap, top=30)
        return

    if args.task == "binary":
        task = build_pii_binary_task(target=args.target, cap=args.cap)
        title = f"PII binary: contains {args.target}?  (negatives carry other PII)"
        tag = f"{args.out_prefix}_binary_{args.target.lower()}"
    else:
        cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
        task = build_pii_multiclass_task(categories=cats, n_categories=args.n_categories,
                                         cap=args.cap, min_per_class=args.min_per_class)
        title = f"PII multiclass: {task.n_classes} categories"
        tag = f"{args.out_prefix}_multiclass_{task.n_classes}way"

    print(f"\n数据划分: train {len(task.train_texts)} / val {len(task.val_texts)} "
          f"/ test {len(task.test_texts)} | 类别数 {task.n_classes}")
    print(f"示例: {task.train_texts[0][:110]!r} -> {task.class_names[task.y_train[0]]}")

    res = run_task(task, args.model, epochs=args.epochs, device=device,
                   max_length=args.max_length, probe_train_cap=args.probe_cap)

    best = max(res["test_f1"].values())
    best_l = max(res["test_f1"], key=res["test_f1"].get)
    ctrl_mean = float(np.mean(list(res["ctrl_f1"].values())))
    print("\n" + "=" * 74)
    print(f"结果 — Macro-F1（随机基线 {res['chance']:.3f}）")
    print("=" * 74)
    print(f"  最佳单层           : {best:.4f}  (L{best_l})")
    print(f"  SIREN 跨层融合     : {res['siren_test_f1']:.4f}")
    print(f"  相对最佳单层增益   : {res['siren_test_f1']-best:+.4f}")
    print(f"  z 维度             : {res['z_dim']}")
    print(f"  打乱标签对照(均值) : {ctrl_mean:.4f}   <- 应接近随机基线 {res['chance']:.3f}")
    verdict = "✅ 通过：真标签远高于对照，信号是真的" if best - ctrl_mean > 0.15 \
        else "⚠️ 存疑：真标签与对照差距过小，可能是维度过拟合"
    print(f"  对照判定           : {verdict}")

    if args.task == "multiclass":
        print("\n逐类别报告：")
        print(classification_report(res["y_true"], res["y_pred"],
                                    target_names=res["class_names"],
                                    digits=3, zero_division=0))
        plot_confusion(res, f"{tag}_confusion.png")

    plot_layer_curve(res, title, f"{tag}_layers.png")
    with open(f"{tag}_results.json", "w") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"结果已写入: {tag}_results.json")


if __name__ == "__main__":
    main()
