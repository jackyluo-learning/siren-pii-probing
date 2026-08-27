"""
The PII layer-wise experiment with activations taken through vLLM-Hook.

Everything after extraction is the code the other experiments already use --
the same L1 probes, the same C grid on validation, the same eta, the same
adaptive aggregation and MLP head, the same shuffled-label control and the same
TF-IDF floor. Only where the activations come from changes, which is the whole
point: a result from this path is comparable to the HuggingFace path only if
nothing downstream moved.

  --task parity   extract a handful of texts both ways and report the gap.
                  Run this first; a score from an unverified extraction path
                  means nothing.
  --task presence-merged / presence / binary / multiclass
                  the same tasks as run_pii_layerwise.py.
"""

import argparse
import json
import time
import warnings
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score

from siren import AdaptiveNeuronAggregator, SirenMLPHead, SirenTrainer
from siren.pii_benchmark import (build_pii_binary_task, build_pii_multiclass_task,
                                 build_pii_presence_merged_task,
                                 build_pii_presence_task, lexical_baseline, PIITask)
from siren.vllm_extractor import build_hook_llm, pool_layers, parity_report

warnings.filterwarnings("ignore")


def _num_layers(model_name: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", cfg), "num_hidden_layers", None)
    if not n:
        raise RuntimeError(f"读不出 {model_name} 的层数，请用 --num-layers 指定。")
    return int(n)


def build_task(args) -> PIITask:
    if args.task == "presence-merged":
        return build_pii_presence_merged_task(
            cap=args.cap, per_source=args.per_source,
            holdout_source=args.holdout_source, val_source=args.val_source)
    if args.task == "presence":
        return build_pii_presence_task(cap=args.cap)
    if args.task == "binary":
        return build_pii_binary_task(target=args.target, cap=args.cap)
    return build_pii_multiclass_task(
        categories=[c.strip() for c in args.categories.split(",")] if args.categories else None,
        n_categories=args.n_categories, cap=args.cap, min_per_class=args.min_per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="presence-merged",
                    choices=["parity", "presence-merged", "presence", "binary", "multiclass"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--num-layers", type=int, default=0, help="0 = 从模型配置读取")
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--per-source", type=int, default=1500)
    ap.add_argument("--holdout-source", default="banking77")
    ap.add_argument("--val-source", default=None)
    ap.add_argument("--target", default="SSN")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--n-categories", type=int, default=6)
    ap.add_argument("--min-per-class", type=int, default=60)
    ap.add_argument("--max-model-len", type=int, default=128,
                    help="必须与 HF 路径的 --max-length 一致，否则两边截断点不同")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="0 = 按空闲显存自动决定（hook 缓存驻留在 GPU 上）")
    ap.add_argument("--gpu-mem", type=float, default=0.62,
                    help="留低一些：vLLM 占掉的显存 hook 缓存就用不上了")
    ap.add_argument("--probe-cap", type=int, default=4000)
    ap.add_argument("--control-cap", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--parity-n", type=int, default=16)
    ap.add_argument("--out-prefix", default="pii_vllm")
    args = ap.parse_args()

    num_layers = args.num_layers or _num_layers(args.model)
    print("=" * 74)
    print(f"SIREN on PII via vLLM-Hook — task={args.task} | model={args.model} "
          f"| {num_layers} 层 | pooling={args.pooling}")
    print("=" * 74)

    # ------------------------------------------------------------ parity gate
    if args.task == "parity":
        from siren.pii_benchmark import build_pii_presence_merged_task
        t = build_pii_presence_merged_task(cap=800, per_source=200,
                                           holdout_source=args.holdout_source,
                                           verbose=False)
        # Deliberately include texts that fill the window exactly. The first
        # version sampled the head of the split, none of its 16 texts reached the
        # cap, and it passed while the real runs died on
        # "Total number of tokens: 129 > max_model_len: 128". A gate that cannot
        # see the boundary is not gating the boundary.
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        lens = [len(x) for x in tk(list(t.train_texts))["input_ids"]]
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        n_long = max(2, args.parity_n // 4)
        picked = order[:n_long] + [i for i in range(len(lens)) if i not in order[:n_long]]
        texts = [t.train_texts[i] for i in picked[:args.parity_n]]
        at_cap = sum(1 for i in picked[:args.parity_n]
                     if lens[i] >= args.max_model_len)
        print(f"\n用 {len(texts)} 条真实任务文本（其中 {at_cap} 条达到或超过 "
              f"{args.max_model_len} 词元的截断上限），两条路径各抽一次并对比 ...\n")
        if at_cap == 0:
            print("⚠️ 没有样本触及截断上限，这一轮检验覆盖不到边界情况。\n")
        rep = parity_report(args.model, texts, num_layers, pooling=args.pooling,
                            max_length=args.max_model_len,
                            gpu_memory_utilization=args.gpu_mem)
        print(f"{'层':>4} {'余弦相似度':>12} {'相对差':>12}")
        for r in rep["rows"]:
            flag = "" if r["cos"] > 0.999 and r["rel_diff"] < 0.02 else "   ← 偏差偏大"
            print(f"{r['layer']:>4} {r['cos']:>12.6f} {r['rel_diff']:>12.2e}{flag}")
        print(f"\n最差余弦 {rep['worst_cos']:.6f}   最大相对差 {rep['worst_rel']:.2e}")
        ok = rep["worst_cos"] > 0.999 and rep["worst_rel"] < 0.02
        print("✅ 两条路径数值一致，可以放心用 vLLM 路径跑实验。" if ok else
              "⚠️ 偏差超出容差。fp16 与 fp32 的差异约在 1e-3 量级；"
              "若远大于此，先查 dtype、max_model_len 截断、以及层号对齐，再跑正式实验。")
        json.dump(rep, open(f"{args.out_prefix}_parity.json", "w"), indent=1)
        return

    # ------------------------------------------------------------ full pipeline
    task = build_task(args)
    print(f"\n数据划分: train {len(task.train_texts)} / val {len(task.val_texts)} "
          f"/ test {len(task.test_texts)} | 类别数 {task.n_classes}")
    print("\n先量表面词汇下限（几秒，不用 GPU）...")
    lex = lexical_baseline(task)

    llm, mode = build_hook_llm(args.model, num_layers, args.pooling,
                               max_model_len=args.max_model_len,
                               gpu_memory_utilization=args.gpu_mem)
    t0 = time.time()
    pool = lambda xs: pool_layers(llm, xs, num_layers, mode, args.chunk_size,
                                  max_len=args.max_model_len)
    tr, va, te = pool(task.train_texts), pool(task.val_texts), pool(task.test_texts)
    extract_secs = time.time() - t0
    print(f"\n抽取用时 {extract_secs:.0f}s，共 "
          f"{len(task.train_texts)+len(task.val_texts)+len(task.test_texts)} 条文本")

    # Downstream is imported rather than reimplemented, so this run and the
    # HuggingFace runs differ in exactly one place.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_pii_layerwise import _fit_probe, plot_layer_curve, plot_confusion
    from run_real_layerwise_experiment import _balanced_subsample
    from siren.probe import PAPER_C_GRID

    layers = list(range(1, num_layers + 1))
    chance = 1.0 / task.n_classes
    sub = _balanced_subsample(task.y_train, args.probe_cap)
    tr_fit = {l: tr[l][sub] for l in layers}
    y_fit = task.y_train[sub]
    print(f"\n探针拟合子采样 {len(sub)}/{len(task.y_train)} 条", flush=True)

    probe, neurons, val_f1, _ = _fit_probe("逐层 L1 探针", tr_fit, y_fit, va,
                                           task.y_val, layers, 0.8)
    test_f1 = {l: float(f1_score(task.y_test, probe.layer_probes[l].predict(te[l]),
                                 average="macro", zero_division=0)) for l in layers}
    for l in layers:
        print(f"  Layer {l:2d}: val={val_f1[l]:.4f}  test={test_f1[l]:.4f}  "
              f"C={probe.layer_best_c[l]:g}  neurons={len(neurons[l])}")

    print("\nSIREN 跨层融合 + MLP 分类头 ...", flush=True)
    agg = AdaptiveNeuronAggregator(neurons, val_f1)
    z_tr, z_va, z_te = agg.transform(tr), agg.transform(va), agg.transform(te)
    mlp = SirenMLPHead(input_dim=z_tr.size(1), hidden_dim=256, num_classes=task.n_classes)
    trainer = SirenTrainer(model=mlp, lr=1e-3)
    trainer.fit(z_tr, torch.tensor(task.y_train), z_va, torch.tensor(task.y_val),
                epochs=args.epochs, batch_size=16)
    with torch.no_grad():
        pred = mlp(z_te.to(trainer.device)).argmax(dim=1).cpu().numpy()
    siren_f1 = float(f1_score(task.y_test, pred, average="macro", zero_division=0))

    print("\n打乱标签对照探针（每层独立打乱，用更小的子采样）...", flush=True)
    csub = _balanced_subsample(task.y_train, args.control_cap)
    ctr_fit = {l: tr[l][csub] for l in layers}
    ctrl, _, _, ctrl_y = _fit_probe("打乱标签对照探针", ctr_fit, task.y_train[csub], va,
                                    task.y_val, layers, 0.8, reshuffle_each_layer=True,
                                    note=f"用 {len(csub)} 条子采样")
    ctrl_f1 = {l: float(f1_score(task.y_test, ctrl.layer_probes[l].predict(te[l]),
                                 average="macro", zero_division=0)) for l in layers}
    ctrl_train = float(np.mean([
        f1_score(ctrl_y[l], ctrl.layer_probes[l].predict(ctr_fit[l]),
                 average="macro", zero_division=0) for l in layers]))

    best = max(test_f1.values()); best_l = max(test_f1, key=test_f1.get)
    print("\n" + "=" * 74)
    print(f"结果 — Macro-F1（随机基线 {chance:.3f}）　抽取路径: vLLM-Hook")
    print("=" * 74)
    print(f"  TF-IDF 词袋基线    : {lex:.4f}")
    print(f"  最佳单层           : {best:.4f}  (L{best_l})")
    print(f"  SIREN 跨层融合     : {siren_f1:.4f}")
    print(f"  相对最佳单层增益   : {siren_f1 - best:+.4f}")
    print(f"  打乱标签对照(均值) : {np.mean(list(ctrl_f1.values())):.4f}   <- 应接近 {chance:.3f}")
    print(f"    对照训练集 F1 = {ctrl_train:.4f}   <- 接近 1.0 才说明对照有效")
    print(f"  抽取用时           : {extract_secs:.0f}s")

    tag = f"{args.out_prefix}_{args.task}" + (f"_{args.pooling}" if args.pooling != "mean" else "")
    res = {"backend": "vllm-hook", "pooling": args.pooling, "num_layers": num_layers,
           "chance": chance, "lexical_baseline": lex, "val_f1": val_f1,
           "test_f1": test_f1, "ctrl_f1": ctrl_f1, "ctrl_train_f1": ctrl_train,
           "siren_test_f1": siren_f1, "extract_seconds": extract_secs,
           "z_dim": int(z_tr.size(1)), "neurons": {l: len(v) for l, v in neurons.items()},
           "y_true": task.y_test.tolist(), "y_pred": pred.tolist(),
           "class_names": task.class_names}
    json.dump(res, open(f"{tag}_results.json", "w"), ensure_ascii=False, indent=1)
    plot_layer_curve(res, f"vLLM-Hook extraction — {args.task} [{args.pooling}]",
                     f"{tag}_layers.png")
    if task.n_classes > 2:
        plot_confusion(res, f"{tag}_confusion.png")
    print(f"结果已写入: {tag}_results.json")


if __name__ == "__main__":
    main()
